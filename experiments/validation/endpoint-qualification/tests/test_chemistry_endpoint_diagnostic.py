"""Allocated positive/negative decomposition and isolation controls."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import chemistry_endpoint_diagnostic as d


class ChemistryEndpointDiagnosticTests(unittest.TestCase):
    def test_import_has_no_numerical_setup(self):
        self.assertTrue(callable(d.np_module))
        self.assertEqual(d.VERSION,'chemistry-endpoint-diagnostic-v1')

    def test_nonallocated_numerics_rejected(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'Allocated'):d.np_module()

    def test_fixed_scales_keep_original_and_diagnostic_ladder(self):
        self.assertEqual(d.EPSILONS,(.01,.003,.001,.0003,.0001))

    def test_softmax_pullback_row_sum(self):
        np=d.np_module();p=np.asarray([[.2,.3,.5]]);g=np.asarray([[2.,-3.,4.]])
        np.testing.assert_allclose(d.pullback(p,g).sum(-1),0.,atol=1e-14)

    def test_softmax_pullback_constant_gradient_null(self):
        np=d.np_module();p=np.asarray([[.2,.3,.5]])
        np.testing.assert_array_equal(d.pullback(p,np.ones_like(p)),np.zeros_like(p))

    def test_softmax_pullback_rejects_nonfinite(self):
        np=d.np_module()
        with self.assertRaisesRegex(ValueError,'Invalid'):d.pullback(np.asarray([[1.,0.]]),np.asarray([[float('nan'),0.]]))

    def test_softmax_pullback_rejects_shape_change(self):
        np=d.np_module()
        with self.assertRaises(ValueError):d.pullback(np.ones((2,3)),np.ones((3,2)))

    def test_logit_probe_preserves_class_and_other_field(self):
        import equinox as eqx
        import jax.numpy as jnp
        class State(eqx.Module):
            atoms:object
            bonds:object
        state=State(jnp.asarray([[1.,2.,3.]]),jnp.asarray([[4.,5.]]))
        new=d.changed_logits(state,'atoms',0,0,2,.003)
        np=d.np_module();self.assertIs(type(new),State)
        np.testing.assert_array_equal(new.bonds,state.bonds)
        np.testing.assert_allclose(new.atoms,[[.997,2.,3.003]],atol=1e-6)

    def test_plan_rejects_restart_authority(self):
        plan={'version':d.VERSION,'optimizer_updates':0,'can_certify_or_restart_original_study':True,
              'epsilons':list(d.EPSILONS),'protected_inputs':[]}
        with self.assertRaisesRegex(ValueError,'policy'):d.verify_inputs(plan)

    def test_plan_rejects_optimizer_updates(self):
        plan={'version':d.VERSION,'optimizer_updates':1,'can_certify_or_restart_original_study':False,
              'epsilons':list(d.EPSILONS),'protected_inputs':[]}
        with self.assertRaises(ValueError):d.verify_inputs(plan)

    def test_plan_rejects_scale_change(self):
        plan={'version':d.VERSION,'optimizer_updates':0,'can_certify_or_restart_original_study':False,
              'epsilons':[.001],'protected_inputs':[]}
        with self.assertRaises(ValueError):d.verify_inputs(plan)

    def test_protected_input_byte_change_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'saved.json';path.write_text('{}')
            plan={'version':d.VERSION,'optimizer_updates':0,'can_certify_or_restart_original_study':False,
                  'epsilons':list(d.EPSILONS),'protected_inputs':[d.binding(path)]}
            self.assertTrue(d.verify_inputs(plan));path.write_text('{ }')
            with self.assertRaisesRegex(ValueError,'Protected'):d.verify_inputs(plan)

    def test_saved_decomposition_separates_native_and_chemistry(self):
        np=d.np_module()
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            root=Path(directory);probes={};eps=.01
            for sign in (-1,1):
                p=np.full((25,6),1/6);native=np.tile(np.arange(6),(25,1))
                total=d.pullback(p,native)+.001*np.tile(np.arange(6),(25,1))
                path=root/f'atoms-{eps:g}-{sign:+d}.npz'
                np.savez(path,X=p,contact_gradient_X=native,gradient_atom_logits=total,
                    contact_primal=1.+sign*eps*.2,combined_primal=1.5+sign*eps*.3,repair_weight=.001)
                probes[path.name]={'path':str(path)}
            report=root/'report.json';report.write_text(json.dumps({'status':'failed','directions':[
                {'field':'atoms','row':2,'source':1,'destination':3,'analytic':.3,
                 'probes':[{'epsilon':eps,'agrees':False,'finite_difference':.3}]}]}))
            plan={'arms':{'protenix':{'failed_report':{'path':str(report)},'saved_probes':probes}}}
            result=d.saved_protenix(plan);row=result['observations'][0]
            self.assertAlmostEqual(row['native_FD'],.2)
            self.assertAlmostEqual(row['weighted_chemistry_FD'],.1)
            self.assertEqual(result['original_status'],'failed')

    def test_saved_decomposition_rejects_different_direction(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'report.json';path.write_text(json.dumps({'status':'failed','directions':[
                {'field':'atoms','row':3,'source':1,'destination':3}]}))
            with self.assertRaisesRegex(ValueError,'direction'):d.saved_protenix({'arms':{'protenix':{'failed_report':{'path':str(path)}}}})

    def test_deadline_rejects_unbounded_budget(self):
        with self.assertRaises(ValueError):
            with d.deadline(5401):pass

    def test_write_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'report.json';d.write(path,{'status':'failed'})
            with self.assertRaises(FileExistsError):d.write(path,{'status':'passed'})
