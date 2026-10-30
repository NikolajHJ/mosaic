"""Focused metadata and tiny numerical controls; no native weights/scorer."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).with_name('gradient_scaled_noise.py')
if not path.exists():
    path = Path(__file__).resolve().parents[1] / 'scripts/gradient_scaled_noise.py'
spec = importlib.util.spec_from_file_location('_gradient_scaled_noise_test', path)
noise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(noise)


def policy(ratio=.4):
    return {'version': noise.VERSION, 'ratio': ratio, 'seed': 230017,
            'scope': 'native_binding_logit_gradient_per_block',
            'row_centered': True, 'annealing': 'none'}


SOURCE = '''def run(evaluation, rate, step, prepared):
    record = {'preparation_identity': prepared.identity}
    proposal = GraphLogits(
        evaluation.logits.atoms - rate * evaluation.gradients.atoms,
        evaluation.logits.bonds - rate * evaluation.gradients.bonds)
    candidate = evaluate(proposal, prepared)
    return candidate, record
'''


class Metadata(unittest.TestCase):
    def test_config_is_explicit_and_immutable(self):
        p = policy(); result = noise.validate(p); result['ratio'] = .1
        self.assertEqual(p['ratio'], .4)

    def test_bad_config_rejected(self):
        for key, value in [('ratio', -1), ('ratio', 2), ('ratio', float('nan')),
                           ('seed', True), ('scope', 'combined_gradient'),
                           ('row_centered', False), ('annealing', 'implicit')]:
            p = policy(); p[key] = value
            with self.assertRaises(ValueError): noise.validate(p)

    def test_only_proposal_changes_and_teacher_is_unchanged(self):
        changed = noise.proposal_tree(SOURCE)
        body = changed.body[0].body
        self.assertEqual(ast.unparse(body[0]), "record = {'preparation_identity': prepared.identity}")
        self.assertEqual(ast.unparse(body[3]), 'candidate = evaluate(proposal, prepared)')
        body[1:3] = [deepcopy(ast.parse(SOURCE).body[0].body[1])]
        self.assertEqual(ast.dump(changed), ast.dump(ast.parse(SOURCE)))

    def test_unknown_original_shape_rejected(self):
        with self.assertRaises(ValueError): noise.proposal_tree('def run():\n    return None')

    def test_saved_protenix_concrete_logit_constructor_supported(self):
        source = SOURCE.replace('proposal = GraphLogits(', 'proposal = type(evaluation.logits)(')
        changed = noise.proposal_tree(source)
        self.assertIn('_binding_gradient_noise(evaluation, rate, step)', ast.unparse(changed))


class Numerical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import jax
        import jax.numpy as jnp
        cls.np, cls.jax, cls.jnp = np, jax, jnp

    def test_norm_and_softmax_gauge_for_different_dimensions(self):
        np = self.np
        for shape in ((25, 6), (300, 5), (2, 3)):
            g = np.ones(shape, dtype=np.float32)
            value, report = noise.direction_noise(g, .3, .4, np.random.default_rng(17))
            self.assertEqual(value.dtype, np.float32)
            self.assertTrue(np.allclose(value.sum(-1), 0, atol=2e-7))
            self.assertAlmostEqual(report['noise_norm'] / report['native_step_norm'], .4, places=6)

    def test_zero_gradient_and_ratio_zero(self):
        np = self.np
        for gradient, ratio in ((np.zeros((2, 3), np.float32), .4),
                                (np.ones((2, 3), np.float32), 0)):
            value, _ = noise.direction_noise(gradient, .3, ratio, np.random.default_rng(17))
            self.assertTrue(np.array_equal(value, np.zeros_like(value)))

    def test_seed_repeats_and_stream_advances(self):
        np = self.np; g = np.ones((25, 6), np.float32)
        a = np.random.default_rng(17); b = np.random.default_rng(17)
        first, _ = noise.direction_noise(g, .3, .4, a)
        same, _ = noise.direction_noise(g, .3, .4, b)
        second, _ = noise.direction_noise(g, .3, .4, a)
        self.assertTrue(np.array_equal(first, same)); self.assertFalse(np.array_equal(first, second))

    def test_nonfinite_and_wrong_shape_rejected(self):
        np = self.np
        for g in (np.array([[float('nan'), 1]]), np.array([1, 2]), np.ones((2, 1))):
            with self.assertRaises(ValueError):
                noise.direction_noise(g, .3, .4, np.random.default_rng(17))

    def evaluation(self, chemistry_scale=1):
        jnp, jax = self.jnp, self.jax
        class Logits:
            def __init__(self, atoms, bonds): self.atoms, self.bonds = atoms, bonds
            def graph(self):
                return SimpleNamespace(X=jax.nn.softmax(self.atoms, -1),
                                       E_upper=jax.nn.softmax(self.bonds, -1))
        x = jnp.asarray([[.1, -.2, .3], [.4, .1, -.1]], dtype=jnp.float32)
        e = jnp.asarray([[.2, -.2], [.1, .4], [-.3, .5]], dtype=jnp.float32)
        raw = SimpleNamespace(X=jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
                              E_upper=jnp.arange(6, dtype=jnp.float32).reshape(3, 2))
        evaluation = SimpleNamespace(logits=Logits(x, e), gradients=Logits(
            jnp.ones_like(x) * chemistry_scale, jnp.ones_like(e) * chemistry_scale),
            native=SimpleNamespace(evaluation=SimpleNamespace(gradients=raw)))
        return evaluation, Logits

    def test_chemistry_weight_does_not_scale_noise(self):
        np = self.np
        a, kind = self.evaluation(1); b, _ = self.evaluation(100)
        pa, ma = noise.Proposal(policy(), kind)(a, .3, 0)
        pb, mb = noise.Proposal(policy(), kind)(b, .3, 0)
        for name in ('atoms', 'bonds'):
            self.assertEqual(ma['blocks'][name]['noise_norm'], mb['blocks'][name]['noise_norm'])
            self.assertTrue(np.allclose(np.asarray(getattr(pa, name)) -
                np.asarray(getattr(a.logits, name) - .3 * getattr(a.gradients, name)),
                np.asarray(getattr(pb, name)) -
                np.asarray(getattr(b.logits, name) - .3 * getattr(b.gradients, name)), atol=3e-6))

    def test_zero_noise_preserves_exact_original_step(self):
        np = self.np; evaluation, kind = self.evaluation()
        proposal, report = noise.Proposal(policy(0), kind)(evaluation, .3, 0)
        for name in ('atoms', 'bonds'):
            self.assertTrue(np.array_equal(np.asarray(getattr(proposal, name)), np.asarray(
                getattr(evaluation.logits, name) - .3 * getattr(evaluation.gradients, name))))
        self.assertFalse(report['model_key_changed'])

    def test_one_proposal_only_and_native_inputs_unmodified(self):
        np = self.np; evaluation, kind = self.evaluation()
        before = np.asarray(evaluation.logits.atoms).copy()
        stream = noise.Proposal(policy(), kind)
        stream(evaluation, .3, 0)
        with self.assertRaises(ValueError): stream(evaluation, .3, 0)
        self.assertTrue(np.array_equal(before, np.asarray(evaluation.logits.atoms)))

    def test_later_retention_and_adapter_helpers_remain_visible(self):
        evaluation, kind = self.evaluation()
        namespace = {'GraphLogits': kind, 'evaluate': lambda *args: 'old helper'}
        exec(SOURCE, namespace)
        trajectory = SimpleNamespace(GraphLogits=kind, run_chemical_updates=namespace['run'])
        # exec() has no source file for inspect; provide the fixture explicitly.
        # Production instead uses the authenticated scientific source on disk.
        from unittest.mock import patch
        with patch.object(noise.inspect, 'getsource', return_value=SOURCE):
            noise.install(trajectory, policy())
        namespace['evaluate'] = lambda candidate, prepared: ('new helper', prepared.identity)
        value, record = trajectory.run_chemical_updates(evaluation, .3, 0,
                                                       SimpleNamespace(identity='fixed teacher'))
        self.assertEqual(value, ('new helper', 'fixed teacher'))
        self.assertEqual(record['preparation_identity'], 'fixed teacher')
        self.assertEqual(record['noise']['step'], 0)


if __name__ == '__main__': unittest.main()
