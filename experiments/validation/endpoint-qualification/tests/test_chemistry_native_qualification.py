"""Allocated controls for separately measured, immutable chemistry endpoints."""
import ast
import os
from pathlib import Path
import sys
import unittest
import tempfile
from unittest import mock

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
        self.assertEqual(a[a.index('-J')+1],'mosaic-chem-qualification08-gpu-protenix')
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
        count=sum(name.startswith('test_') and callable(getattr(type(self),name)) for name in dir(type(self)))
        self.assertEqual(p['cpu_counts'],{'tests.test_chemistry_native_qualification.ChemistryNativeQualificationTests':count})

    def test_atomic_race_repeats_whole_gate_and_preserves_attempt(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            parent=Path(directory);calls=[];records={}
            def operation(path,phase):
                calls.append((path,phase))
                if len(calls)==1:raise FileNotFoundError(2,'racing atomic reference',str(ops.ATOMIC_REFERENCE))
                return {'status':'passed','complete':True}
            result=ops.complete_inventory(operation,parent,'entry',lambda path,row:records.update({str(path):row}))
            self.assertEqual(result,{'status':'passed','complete':True});self.assertEqual(len(calls),2)
            self.assertNotEqual(calls[0][0],calls[1][0]);self.assertEqual(len(records),2)
            self.assertTrue(any(r.get('status')=='failed_incomplete_inventory_not_admission' for r in records.values()))

    def test_missing_protected_input_is_not_retried(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            calls=[]
            def operation(path,phase):
                calls.append(path);raise FileNotFoundError(2,'missing protected checkpoint','/work3/required-checkpoint.npz')
            with self.assertRaises(FileNotFoundError):ops.complete_inventory(operation,Path(directory),'entry',lambda *x:None)
            self.assertEqual(len(calls),1)

    def test_atomic_race_exhaustion_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            calls=[];records=[]
            def operation(path,phase):
                calls.append(path);raise FileNotFoundError(2,'racing atomic reference',str(ops.ATOMIC_REFERENCE))
            with self.assertRaises(FileNotFoundError):
                ops.complete_inventory(operation,Path(directory),'entry',lambda path,row:records.append(row))
            self.assertEqual(len(calls),3);self.assertEqual(len(records),3)
            self.assertTrue(all(r['partial_totals_used'] is False for r in records))

    def test_capacity_rejection_is_not_retried(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            calls=[]
            def operation(path,phase):
                calls.append(path);raise ValueError('Full budget does not fit')
            with self.assertRaises(ValueError):ops.complete_inventory(operation,Path(directory),'entry',lambda *x:None)
            self.assertEqual(len(calls),1)

    def test_failed_complete_result_is_not_admitted(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            with self.assertRaises(ValueError):
                ops.complete_inventory(lambda *x:{'status':'failed'},Path(directory),'entry',lambda *x:None)

    def reference_helpers(self):
        plan=q.read(Path(__file__).resolve().parents[1]/'plan.json')
        source=Path(plan['precision_source']['path'])
        self.assertEqual(plan['precision_source']['manifest_sha256'],'f2043af74b8fa9ff485a44b80fcf96454bf8585868859389828c37f0073a3664')
        science=Path(plan['arms']['protenix']['science_source']['path'])
        sys.path.insert(0,str(science/'src'))
        self.addCleanup(lambda:sys.path.remove(str(science/'src')))
        return q.module(source/'scripts/protenix_precision_native.py','_chem05_placement_fixture')

    def placed_inputs(self,helpers=None):
        import jax
        import numpy as np
        previous=bool(jax.config.jax_enable_x64);jax.config.update('jax_enable_x64',True)
        self.addCleanup(lambda:jax.config.update('jax_enable_x64',previous))
        helpers=helpers or self.reference_helpers()
        model={'weight':np.asarray([.25,.5],np.float64),'static':'unchanged'}
        context={'coords':np.asarray([[1.,2.,3.]],np.float64),'ids':np.asarray([1],np.int32)}
        fields={'X':np.full((25,6),1/6,np.float32),'E_upper':np.full((300,5),.2,np.float32)}
        return q.place_reference_inputs(helpers,model,context,jax.random.key(17),fields,jax.devices('cpu')[0]),fields

    def test_reference09_exact_source_is_adopted_not_its_other_endpoint_pass(self):
        plan=q.read(Path(__file__).resolve().parents[1]/'plan.json')
        self.assertEqual(plan['precision_source']['files'],20)
        self.assertEqual(plan['precision_source']['manifest_sha256'],'f2043af74b8fa9ff485a44b80fcf96454bf8585868859389828c37f0073a3664')
        self.assertIn('/protenix-endpoint-fix-09/gpu/payload/',plan['reference09_qualified_nonchem_endpoint_not_chemistry_admission']['path'])
        self.assertFalse(plan['can_submit_continuation'])

    def test_Boltz04_pass_is_preserved_and_only_failed_Pro_is_dispatched(self):
        plan=q.read(Path(__file__).resolve().parents[1]/'plan.json')
        self.assertEqual(ops.NATIVE_ARMS,('protenix',));self.assertEqual(plan['native_arms'],['protenix'])
        proof=plan['Boltz04_native_component'];self.assertEqual(q.binding(proof['path']),proof)
        value=q.read(proof['path']);self.assertEqual(value['status'],'native_component_qualified_not_continuation_admission')
        self.assertFalse(value['can_submit_continuation']);self.assertEqual(value['optimizer_updates'],0)

    def test_production_fp32_AST_is_identical_to_preserved04(self):
        plan=q.read(Path(__file__).resolve().parents[1]/'plan.json')
        prior=Path(plan['prior_failed_qualification']['path'])/'scripts/chemistry_native_qualification.py'
        def function(path):
            return next(node for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.FunctionDef) and node.name=='fp32')
        self.assertEqual(ast.dump(function(prior)),ast.dump(function(Path(q.__file__))))

    def test_reference_model_context_key_graph_have_one_device_and_preserve_bits(self):
        import jax
        import numpy as np
        (model,context,key,fields,receipt),original=self.placed_inputs()
        device=jax.devices('cpu')[0]
        for value in (model['weight'],context['coords'],context['ids'],key,*fields.values()):self.assertEqual(value.devices(),{device})
        for field in original:
            self.assertEqual(np.asarray(fields[field]).dtype,np.float64)
            self.assertEqual(np.asarray(fields[field]).astype(np.float32).tobytes(),original[field].tobytes())
        self.assertEqual(model['static'],'unchanged');self.assertEqual(context['ids'].dtype,np.int32)
        self.assertTrue(receipt['frozen_transfer_bits_unchanged']);self.assertFalse(receipt['production_execution_changed'])
        np.testing.assert_array_equal(np.asarray(jax.random.key_data(key)),np.asarray(jax.random.key_data(jax.random.key(17))))

    def test_actual_jitted_reference_graph_primal_AD_stays_FP64(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        (_,_,_,fields,_),_=self.placed_inputs()
        value,gradient=jax.jit(jax.value_and_grad(lambda v:jnp.sum(jnp.sin(v['X']))+jnp.sum(v['E_upper']**2)))(fields)
        self.assertEqual(value.dtype,np.float64)
        for field in fields:self.assertEqual(gradient[field].dtype,np.float64)

    def test_changed_reference_model_placement_is_rejected(self):
        helpers=self.reference_helpers();original=helpers.place_reference_tree
        def corrupt(tree,device):
            value,audit=original(tree,device)
            if 'weight' in value:value['weight']=value['weight'].at[0].add(.125)
            return value,audit
        with mock.patch.object(helpers,'place_reference_tree',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'changed promoted'):self.placed_inputs(helpers)

    def test_changed_reference_key_bits_are_rejected(self):
        import jax
        original=jax.device_put
        def corrupt(value,device):
            if str(getattr(value,'dtype','')).startswith('key<'):value=jax.random.fold_in(value,1)
            return original(value,device)
        with mock.patch.object(jax,'device_put',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'changed promoted'):self.placed_inputs()

    def test_narrowed_reference_graph_is_rejected(self):
        import jax
        import jax.numpy as jnp
        original=jax.device_put
        def narrow(value,device):
            result=original(value,device)
            return result.astype(jnp.float32) if result.shape==(25,6) else result
        with mock.patch.object(jax,'device_put',side_effect=narrow):
            with self.assertRaisesRegex(ValueError,'precision or shape'):self.placed_inputs()

    def test_reference_moves_inputs_before_loss_and_has_no_CPU_graph_branch(self):
        tree=ast.parse(Path(q.__file__).read_bytes())
        reference=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reference')
        calls=[n for n in ast.walk(reference) if isinstance(n,ast.Call)]
        place=next(n for n in calls if ast.unparse(n.func)=='place_reference_inputs')
        build=next(n for n in calls if ast.unparse(n.func)=='model.build_loss')
        self.assertLess(place.lineno,build.lineno)
        self.assertFalse(any(ast.unparse(n.func)=='jax.devices' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='cpu' for n in calls))
        self.assertIn("audit_module.reference_graph_arrays",ast.unparse(reference))

    def test_serial_production_exits_before_reference_start(self):
        tree=ast.parse(Path(q.__file__).read_bytes())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        text=ast.unparse(main)
        self.assertIn("for worker in ('production', 'reference')",text)
        self.assertIn('subprocess.run(command',text);self.assertNotIn('subprocess.Popen',text)

    def test_preparation_stops_without_current_resource_scope_before_SSH(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/manage_chemistry_qualification.py'
        self.assertTrue(path.is_file(),'Actual preparation operator must be in new immutable source')
        spec=importlib.util.spec_from_file_location('_chem05_manager_fixture',path);manager=importlib.util.module_from_spec(spec);spec.loader.exec_module(manager)
        with mock.patch.object(manager,'CURRENT_CAPACITY_SOURCE',None),mock.patch.object(manager,'transport',side_effect=AssertionError('Unexpected SSH path')):
            with self.assertRaisesRegex(ValueError,'preparation/staging blocked'):manager.prepare()


    def test_resource_source_and_child_are_in_one_funded_control_tree(self):
        parent=ops.BASE/'nonchem-precision-continuation-25'
        self.assertEqual(ops.ROOT.parent,parent)
        self.assertEqual(ops.SOURCE.parent.parent,parent)
        self.assertEqual(ops.CURRENT_CAPACITY_SOURCE['files'],50)

    def test_actual_commands_carry_current_capacity_identity(self):
        argv=ops.inherited().command('gpu-protenix','a'*64)
        self.assertEqual(argv[-2:],["--capacity-source",str(ops.BASE/'nonchem-precision-release-25/source')])
        self.assertEqual(argv.count('-env'),1)
        self.assertIn('~CUDA_VISIBLE_DEVICES',argv[argv.index('-env')+1])

    def test_old_lifetime_storage_is_not_executed_as_current_admission(self):
        tree=ast.parse(Path(ops.__file__).read_bytes())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inherited')
        text=ast.unparse(function)
        self.assertNotIn('oldstorage',text)
        self.assertIn('current_capacity(parent, phase)',text)
        self.assertIn('register_shared_job(op, phase, record)',text)


if __name__=='__main__':unittest.main()
