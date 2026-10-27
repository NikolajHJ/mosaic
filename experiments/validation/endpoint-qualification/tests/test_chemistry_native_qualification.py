"""Allocated controls for separately measured, immutable chemistry endpoints."""
import ast
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import chemistry_native_qualification as q
import chemistry_qualification_ops as ops


class ChemistryNativeQualificationTests(unittest.TestCase):
    def setUp(self):
        assert str(os.environ.get('LSB_JOBID','')).isdigit(),'Allocated DTU fixtures required'

    def test_version(self):self.assertEqual(q.VERSION,'chemistry-native-component-qualification-v1')
    def test_native_cap(self):self.assertEqual(q.CAP,512*1024**2)
    def test_whole_group_cap(self):self.assertEqual(ops.CAP,1088*1024**2)
    def test_distinct_new_roots(self):
        self.assertEqual(q.ROOT,ops.ROOT);self.assertNotEqual(ops.ROOT,ops.PARENT.parent)
    def test_cpu_actual_command(self):
        a=ops.inherited().command('cpu','a'*64);self.assertEqual(a[a.index('-q')+1],'hpc')
    def test_gpu_actual_command(self):
        a=ops.inherited().command('gpu-boltz','a'*64)
        self.assertEqual(a[a.index('-q')+1],'gpul40s');self.assertEqual(a[a.index('-W')+1],'02:00')
        self.assertEqual(a[a.index('-n')+1],'4');self.assertNotIn('-w',a)
    def test_new_command_points_to_new_source(self):
        a=ops.inherited().command('gpu-protenix','a'*64)
        self.assertEqual(a[a.index('-J')+1],'mosaic-chem-qualification03-gpu-protenix')
        self.assertIn(str(ops.SOURCE),a)
    def test_no_optimizer_phase(self):
        with self.assertRaises(ValueError):ops.inherited().command('segment','a'*64)
    def test_gpu_interarm_dependency_rejected(self):
        with self.assertRaises(ValueError):ops.inherited().command('gpu-protenix','a'*64,'done(1)')
    def test_own_callback_only(self):
        a=ops.inherited().command('after-cpu','a'*64,'ended(123)')
        self.assertEqual(a[a.index('-w')+1],'ended(123)')
    def test_protenix_logit_failure_not_invented_as_raw_original(self):
        d,scope=q.raw_directions({'id':'protenix'})
        self.assertEqual(d,{'X':[],'E_upper':[]});self.assertEqual(scope,'not_preexisting_at_new_endpoint')
    def test_import_no_eager_numerical_setup(self):
        tree=ast.parse(Path(q.__file__).read_text());names=[]
        for n in tree.body:
            if isinstance(n,ast.Import):names.extend(v.name for v in n.names)
        self.assertFalse({'numpy','jax','torch','mosaic'}&set(names))
    def test_parent_source_pin(self):
        self.assertEqual(ops.inherited().verify(ops.PARENT,ops.PARENT_PIN)['diagnostic_only'],True)
    def test_raw_policy_unchanged_version(self):
        import admission
        self.assertEqual(admission.VERSION,'prospective-endpoint-admission-v3')
        self.assertEqual(admission.RELATIVE_TOLERANCE,.15)
        self.assertEqual(admission.ORIGINAL_EPSILONS,(.01,.003,.001))
    def test_reference_policy_unchanged(self):
        import protenix_precision_admission as p
        self.assertEqual(p.VERSION,'protenix-separate-precision-admission-v1')
        self.assertEqual(p.RELATIVE_TOLERANCE,.15);self.assertEqual(p.LOCALITY_TOLERANCE,.15)
    def test_reference_installed_runtime_identity(self):
        import protenix_precision_reference as p
        self.assertEqual(p.authenticate_runtime()['functions_sha256'],p.FUNCTIONS_SHA256)
    def test_reference_native_four_precision_expressions(self):
        import protenix_precision_reference as p
        v=p.authenticate_native_runtime();self.assertEqual(v['native_sha256'],p.NATIVE_SHA256)
        self.assertEqual(len(v['mask_functions']),2)
        self.assertEqual(v['charge_function']['changed_expressions'],1)

    def linear(self,factor=1.):
        import numpy as np
        import admission as a
        v={'X':np.full((4,3),1/3,np.float32),'E_upper':np.full((6,4),.25,np.float32)}
        g={k:np.tile(np.arange(x.shape[1],dtype=np.float32),(x.shape[0],1)) for k,x in v.items()}
        p=a.plan(v,{'X':[],'E_upper':[]},original_direction_scope='not_preexisting_at_new_endpoint')
        def loss(x):return np.float32(sum(np.sum(x[k]*g[k]) for k in x))
        def ad(x):return loss(x),{k:y*np.float32(factor) for k,y in g.items()}
        return a.evaluate(p,loss,ad,lambda:{'frozen':'unchanged'})

    def test_native_linear_positive(self):self.assertTrue(self.linear()['can_admit_new_child'])
    def test_native_zero_rejected(self):self.assertFalse(self.linear(0.)['can_admit_new_child'])
    def test_native_reversed_rejected(self):self.assertFalse(self.linear(-1.)['can_admit_new_child'])
    def test_native_scaled_rejected(self):self.assertFalse(self.linear(1.5)['can_admit_new_child'])
    def test_no_local_reference_dtype_promotion(self):
        import protenix_precision_reference as p
        import jax
        before=bool(jax.config.jax_enable_x64)
        jax.config.update('jax_enable_x64',False)
        try:
            with self.assertRaises(ValueError):p.assert_reference_process()
        finally:jax.config.update('jax_enable_x64',before)
    def test_plan_boundaries_keep_targets_and_rates(self):
        p=q.read(Path(__file__).resolve().parents[1]/'plan.json')
        self.assertEqual((p['arms']['protenix']['learning_rate'],p['arms']['protenix']['target_updates']),(.3,50000))
        self.assertEqual((p['arms']['boltz']['learning_rate'],p['arms']['boltz']['target_updates']),(.0025,30000))
        self.assertFalse(p['can_certify_or_restart_original_study'])
    def test_new_suite_exact_dynamic_attribution(self):
        p=q.read(Path(__file__).resolve().parents[1]/'source_manifest.json')
        self.assertEqual(p['cpu_counts'],{'tests.test_chemistry_native_qualification.ChemistryNativeQualificationTests':24})


if __name__=='__main__':unittest.main()
