"""Allocated metadata contracts for prospective chemistry production."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import production_contract as c


class ProductionContractTests(unittest.TestCase):
    def setUp(self):
        self.plan={'version':c.VERSION,'pilots':{
            arm:{**copy.deepcopy(c.PILOTS[arm]),'manifest_sha256':('a' if arm=='protenix' else 'c')*64,
                 'cpu_certificate_sha256':('b' if arm=='protenix' else 'd')*64,
                 'job_id':'29820001' if arm=='protenix' else '29820002'} for arm in c.PROFILES}}

    def test_exact_declared_plan(self):
        self.assertIs(c.validate_plan(self.plan),self.plan)

    def test_unknown_plan_field_rejected(self):
        self.plan['skip_gradient_checks']=True
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_missing_pilot_cpu_rejected(self):
        del self.plan['pilots']['protenix']['cpu_certificate_sha256']
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_old_plan_version_rejected(self):
        self.plan['version']='selected-nonchemistry'
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_duplicate_actual_pilots_rejected(self):
        self.plan['pilots']['boltz']['job_id']=self.plan['pilots']['protenix']['job_id']
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_unknown_arm_rejected(self):
        self.plan['pilots']['dbmol']={'job_id':'29820003'}
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_missing_arm_rejected(self):
        del self.plan['pilots']['boltz']
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_nonactual_job_identifier_rejected(self):
        self.plan['pilots']['protenix']['job_id']='pending'
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_bool_job_identifier_rejected(self):
        self.plan['pilots']['protenix']['job_id']=True
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_malformed_digest_rejected(self):
        self.plan['pilots']['protenix']['manifest_sha256']='x'*64
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_protenix_scope_exact(self):
        p=c.profile('protenix')
        self.assertEqual((p['arm'],p['rate'],p['target'],p['segments']),('random',.3,50000,5))

    def test_boltz_scope_exact(self):
        p=c.profile('boltz')
        self.assertEqual((p['arm'],p['rate'],p['target'],p['segments']),('boltz2_chemistry',.0025,30000,3))

    def test_native_only_arm_cannot_execute(self):
        with self.assertRaises(ValueError):c.profile('boltz2')

    def test_original_source_manifest_lengths(self):
        for spec in c.PROFILES.values():self.assertEqual(c.digest(spec['manifest']),spec['manifest'])

    def test_protenix_segment_zero(self):
        self.assertEqual(c.stage_label('protenix','segment',0),'random-segment-0000')

    def test_boltz_final_segment(self):
        self.assertEqual(c.stage_label('boltz','segment',2),'boltz2_chemistry-segment-0002')

    def test_initial_native_cannot_be_replayed(self):
        with self.assertRaises(ValueError):c.stage_label('protenix','native',0)

    def test_initial_native_callback_cannot_be_replayed(self):
        with self.assertRaises(ValueError):c.stage_label('boltz','after-native',0)

    def test_next_native_boundary(self):
        self.assertEqual(c.stage_label('protenix','native',1),'random-native-0001')

    def test_bool_index_cannot_select_boundary(self):
        with self.assertRaises(ValueError):c.stage_label('boltz','segment',True)

    def test_negative_index_rejected(self):
        with self.assertRaises(ValueError):c.stage_label('boltz','segment',-1)

    def test_out_of_budget_index_rejected(self):
        with self.assertRaises(ValueError):c.stage_label('protenix','segment',5)

    def test_unknown_phase_rejected(self):
        with self.assertRaises(ValueError):c.stage_label('boltz','restart',0)

    def test_protenix_sparse_projection(self):
        self.assertEqual(c.projection(1234,'protenix'),2*1234*5*503+5*1024**3)

    def test_boltz_sparse_projection(self):
        self.assertEqual(c.projection(1234,'boltz'),2*1234*3*503+3*1024**3)

    def test_projection_keeps_all_segment_allowances(self):
        self.assertGreater(c.projection(1,'protenix'),5*c.GIB)
        self.assertEqual(c.CONTROL_CAP,64*1024**2)
        self.assertLess(c.CONTROL_CAP,c.GIB)

    def test_empirical_projection_unit_required(self):
        for value in (None,0,-1,True,1.0):
            with self.assertRaises(ValueError):c.projection(value,'boltz')

    def test_protenix_single_process_gpu_mode(self):
        args=c.resources('segment','protenix')
        self.assertEqual(args[args.index('-gpu')+1],'num=1:mode=exclusive_process')
        self.assertEqual(args[args.index('-W')+1],'23:00')

    def test_boltz_exclusive_whole_job_multi_process_mode(self):
        args=c.resources('segment','boltz')
        self.assertEqual(args[args.index('-gpu')+1],'num=1:mode=shared:j_exclusive=yes:mps=no')

    def test_gpu_requires_eight_cpu_slots(self):
        for arm in c.PROFILES:
            args=c.resources('native',arm)
            self.assertEqual(args[args.index('-n')+1],'8')
            self.assertEqual(args[args.index('-W')+1],'02:00')

    def test_cpu_callback_resource_scope(self):
        args=c.resources('after-cpu')
        self.assertEqual(args[args.index('-q')+1],'hpc')
        self.assertNotIn('-gpu',args)

    def test_all_actual_dependencies(self):
        self.assertEqual(c.callback_dependency('29820003',self.plan['pilots']),
                         'ended(29820003) && ended(29820001) && ended(29820002)')

    def test_unknown_own_cpu_cannot_dispatch(self):
        with self.assertRaises(ValueError):c.callback_dependency('unknown',self.plan['pilots'])

    def test_current_complete_storage_pass(self):
        r={'status':'passed','original_gate_changed':False,'reserve_bytes':4*c.GIB,
           'file_reserve':100000,'required_bytes':1234,'available_bytes':1234}
        c.require_storage(r)

    def test_storage_reserve_cannot_be_reduced(self):
        r={'status':'passed','original_gate_changed':False,'reserve_bytes':3*c.GIB,
           'file_reserve':100000,'required_bytes':1234,'available_bytes':9999}
        with self.assertRaises(ValueError):c.require_storage(r)

    def test_storage_inode_reserve_cannot_be_reduced(self):
        r={'status':'passed','original_gate_changed':False,'reserve_bytes':4*c.GIB,
           'file_reserve':99999,'required_bytes':1234,'available_bytes':9999}
        with self.assertRaises(ValueError):c.require_storage(r)

    def test_failed_parent_storage_cannot_be_adopted(self):
        r={'status':'failed','original_gate_changed':False,'reserve_bytes':4*c.GIB,
           'file_reserve':100000,'required_bytes':1234,'available_bytes':9999}
        with self.assertRaises(ValueError):c.require_storage(r)

    def test_insufficient_whole_storage_rejected(self):
        r={'status':'passed','original_gate_changed':False,'reserve_bytes':4*c.GIB,
           'file_reserve':100000,'required_bytes':1234,'available_bytes':1233}
        with self.assertRaises(ValueError):c.require_storage(r)

    def test_nonfinite_or_bool_storage_not_capacity(self):
        for value in (float('inf'),float('nan'),True,None):
            r={'status':'passed','original_gate_changed':False,'reserve_bytes':4*c.GIB,
               'file_reserve':100000,'required_bytes':1234,'available_bytes':value}
            with self.assertRaises(ValueError):c.require_storage(r)

    def test_historical_guard_cannot_be_changed(self):
        r={'status':'passed','original_gate_changed':True,'reserve_bytes':4*c.GIB,
           'file_reserve':100000,'required_bytes':1234,'available_bytes':9999}
        with self.assertRaises(ValueError):c.require_storage(r)

    def test_canonical_absolute_input(self):
        self.assertEqual(c.canonical('/work3/s234818/mosaic/a'),'/work3/s234818/mosaic/a')

    def test_path_traversal_or_windows_separator_rejected(self):
        for value in ('relative','/work3/../etc/passwd','/work3\\other'):
            with self.assertRaises(ValueError):c.canonical(value)

    def test_protenix_cannot_reuse_failed_source02(self):
        self.plan['pilots']['protenix']['source']=c.PILOTS['boltz']['source']
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_boltz_cannot_use_protenix_source03(self):
        self.plan['pilots']['boltz']['source']=c.PILOTS['protenix']['source']
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_protenix_cannot_use_boltz_root02(self):
        self.plan['pilots']['protenix']['root']=c.PILOTS['boltz']['root']
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_boltz_cannot_use_protenix_root03(self):
        self.plan['pilots']['boltz']['root']=c.PILOTS['protenix']['root']
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_per_arm_missing_job_rejected(self):
        del self.plan['pilots']['boltz']['job_id']
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_per_arm_unknown_waiver_field_rejected(self):
        self.plan['pilots']['protenix']['skip_readiness']=True
        with self.assertRaises(ValueError):c.validate_plan(self.plan)

    def test_legacy_shared_source_plan_rejected(self):
        old={'version':'chemistry-accelerated-production-v1','pilot_manifest_sha256':'a'*64,
             'pilot_cpu_certificate_sha256':'b'*64,'pilot_jobs':{'protenix':'29820001','boltz':'29820002'}}
        with self.assertRaises(ValueError):c.validate_plan(old)

    def test_independent_cpu_certificate_pins_preserved(self):
        c.validate_plan(self.plan)
        self.assertNotEqual(self.plan['pilots']['protenix']['cpu_certificate_sha256'],
                            self.plan['pilots']['boltz']['cpu_certificate_sha256'])
