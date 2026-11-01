"""Small local metadata/AST controls; no numerical backend, scheduler or model."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

p=Path(__file__).with_name('exploratory_continuation.py')
if not p.exists():p=Path(__file__).resolve().parents[1]/'scripts/exploratory_continuation.py'
s=importlib.util.spec_from_file_location('_exploratory_continuation_test',p)
tool=importlib.util.module_from_spec(s);s.loader.exec_module(tool)


class Tests(unittest.TestCase):
    def row(self,name):
        pro=name.startswith('protenix');start=10000
        return {'name':name,'research_profile':'exploratory','rate':.3 if pro else .0025,
            'start':start,'target':50000 if pro else 30000,'steps':40000 if pro else 20000,
            'chemistry':name.endswith('_chemistry'),'matrix_stride':20,'loss_stride':1,
            'acceptance_policy':'fixed_step','max_backtracks':1}

    def test_all_five_scopes(self):
        for name in ('protenix','protenix_chemistry','boltz','boltz_chemistry','dbmol'):
            self.assertEqual(tool.validate(self.row(name))['name'],name)

    def test_changed_science_or_partial_budget_rejected(self):
        for k,v in [('rate',.01),('target',10000),('steps',2),('chemistry',False),
                    ('max_backtracks',2),('acceptance_policy','best_primal'),('matrix_stride',1)]:
            row=self.row('protenix_chemistry');row[k]=v
            with self.assertRaises(ValueError):tool.validate(row)

    def test_only_duplicate_forward_assignments_change(self):
        raw='''def evaluate_combined():
    raw = evaluate_graph(loss, graph, key)
    primal, aux = contact_loss(graph, key=key)
    chemical_primal, _ = chemical(logits)
    gradients = pull(raw.gradients) + chemical_gradient
    proposal = logits - rate * gradients
    return proposal
'''
        changed=tool.backward_only_tree(raw)
        reverse=deepcopy(changed)
        reverse.body[0].body[1]=ast.parse('primal, aux = contact_loss(graph, key=key)').body[0]
        reverse.body[0].body[2]=ast.parse('chemical_primal, _ = chemical(logits)').body[0]
        self.assertEqual(ast.dump(reverse),ast.dump(ast.parse(raw)))
        self.assertIn('raw.value',ast.unparse(changed))

    def test_unrecognized_legacy_shape_is_not_silently_changed(self):
        with self.assertRaises(ValueError):tool.backward_only_tree('def evaluate_combined():\n    return 0')

    def test_native_child_inherits_the_original_science_and_scripts(self):
        import os
        original={'PYTHONPATH':'old','UNCHANGED':'yes'}
        result=tool.worker_environment(Path('science'),original)
        self.assertEqual(result['PYTHONPATH'].split(os.pathsep),[str(Path('science/src')),str(Path('science/scripts')),'old'])
        self.assertEqual(original,{'PYTHONPATH':'old','UNCHANGED':'yes'})
        self.assertEqual(result['UNCHANGED'],'yes')

    def test_noise_resume_requires_bound_matching_checkpoint_and_stream(self):
        from unittest.mock import patch
        noise_path=Path(__file__).with_name('gradient_scaled_noise.py')
        if not noise_path.exists():
            noise_path=Path(__file__).resolve().parents[4]/'experiments/brd4-jq1/protenix-random-chem/scripts/gradient_scaled_noise.py'
        spec=importlib.util.spec_from_file_location('_noise_resume_integration_test',noise_path)
        noise=importlib.util.module_from_spec(spec);spec.loader.exec_module(noise)
        row=self.row('protenix_chemistry');row.update(start=35300,steps=14700,
            snapshot={'sha256':'a'*64},noise_resume={'path':'bound-resume.json','sha256':'b'*64,'bytes':123})
        row['logit_noise']={'version':noise.VERSION,'ratio':.4,'seed':230017,
            'scope':'native_binding_logit_gradient_per_block','row_centered':True,'annealing':'none'}
        record={'version':noise.RESUME_VERSION,'global_start':35300,'snapshot_sha256':'a'*64,
            'predecessor_job_id':'29857358','committed_update':{'step':35299,'global_step':35299,
            'noise':{'version':noise.VERSION,'ratio':.4,'seed':230017,'step':35299,'rng':'numpy.PCG64',
                'model_key_changed':False,'gradient_normalized_or_clipped':False,
                'chemistry_weight_used_for_noise_scale':False,'rng_state_after':{
                    'bit_generator':'PCG64','state':{'state':123,'inc':45},'has_uint32':0,'uinteger':0}}}}
        with patch.dict('sys.modules',{'gradient_scaled_noise':noise}), \
                patch.object(tool,'bound',return_value=Path('bound-resume.json')) as bound, \
                patch.object(tool,'read',return_value=record):
            self.assertEqual(tool.validate(row)['steps'],14700)
            bound.assert_called_once_with(row['noise_resume'])
            missing=deepcopy(row);missing.pop('noise_resume')
            with self.assertRaises(ValueError):tool.validate(missing)
            changed=deepcopy(row);changed['snapshot']['sha256']='c'*64
            with self.assertRaises(ValueError):tool.validate(changed)
            fresh=deepcopy(row);fresh.update(start=0,steps=50000)
            with self.assertRaises(ValueError):tool.validate(fresh)


if __name__=='__main__':unittest.main()
