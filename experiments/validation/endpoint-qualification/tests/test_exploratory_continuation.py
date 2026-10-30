"""Small local metadata/AST controls; no numerical backend, scheduler or model."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

p=Path(__file__).resolve().parents[1]/'scripts/exploratory_continuation.py'
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


if __name__=='__main__':unittest.main()
