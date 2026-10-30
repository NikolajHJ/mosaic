"""Experimental logit perturbations scaled to the frozen binding gradient.

Only the random direction is centered/normalized. Native and chemistry gradients,
their weights, and the original deterministic displacement are not transformed.
Numerical imports are lazy so startup/configuration checks remain metadata-only.
"""
import ast
from copy import deepcopy
import inspect
import math
import textwrap


VERSION = 'binding-logit-gradient-relative-noise-v1'


def validate(policy):
    expected = {'version', 'ratio', 'seed', 'scope', 'row_centered', 'annealing'}
    if set(policy) != expected or policy['version'] != VERSION:
        raise ValueError('Unknown gradient-noise policy')
    if (type(policy['ratio']) not in (int, float) or not math.isfinite(policy['ratio'])
            or not 0 <= policy['ratio'] <= 1 or type(policy['seed']) is not int
            or not 0 <= policy['seed'] < 2**64):
        raise ValueError('Invalid noise ratio/seed')
    if (policy['scope'] != 'native_binding_logit_gradient_per_block'
            or policy['row_centered'] is not True or policy['annealing'] != 'none'):
        raise ValueError('Changed declared noise coordinates or schedule')
    return deepcopy(policy)


def direction_noise(gradient, rate, ratio, generator):
    """Return FP32 noise with L2 norm ratio * rate * ||gradient||.

    Gaussian directions are projected off each softmax row's constant-shift
    direction, then normalized. Zero native gradient means zero noise; no
    invented floor or alternate objective is introduced.
    """
    import numpy as np
    g = np.asarray(gradient)
    if (g.ndim != 2 or g.shape[-1] < 2 or not np.isfinite(g).all()
            or not math.isfinite(rate) or rate <= 0
            or not math.isfinite(ratio) or not 0 <= ratio <= 1):
        raise ValueError('Nonfinite/incompatible gradient-noise input')
    gradient_norm = float(np.linalg.norm(g.astype(np.float64)))
    z = generator.standard_normal(g.shape)
    z -= z.mean(axis=-1, keepdims=True)
    z_norm = float(np.linalg.norm(z))
    if not math.isfinite(z_norm) or z_norm == 0:
        raise ValueError('Degenerate random direction')
    requested_norm = ratio * rate * gradient_norm
    noise = np.asarray(z * (requested_norm / z_norm), dtype=np.float32)
    actual_norm = float(np.linalg.norm(noise.astype(np.float64)))
    if (not np.isfinite(noise).all()
            or not np.isclose(actual_norm, requested_norm, rtol=5e-6, atol=1e-35)):
        raise FloatingPointError('Nonfinite/unresolved requested noise')
    return noise, {'native_logit_gradient_norm': gradient_norm,
                   'native_step_norm': rate * gradient_norm,
                   'requested_noise_norm': requested_norm,
                   'noise_norm': actual_norm,
                   'flat_native_block': gradient_norm == 0}


class Proposal:
    def __init__(self, policy, logit_type):
        import numpy as np
        self.policy = validate(policy)
        self.generator = np.random.Generator(np.random.PCG64(policy['seed']))
        self.logit_type = logit_type
        self.next_step = 0

    def __call__(self, evaluation, rate, step):
        import jax
        import jax.numpy as jnp
        import numpy as np
        if step != self.next_step:
            raise ValueError('Noise stream requires exactly one proposal per update')
        before_rng = deepcopy(self.generator.bit_generator.state)
        graph = evaluation.logits.graph()
        raw = evaluation.native.evaluation.gradients
        # Preserve the original FP32 deterministic expression and placement.
        base = self.logit_type(
            evaluation.logits.atoms - rate * evaluation.gradients.atoms,
            evaluation.logits.bonds - rate * evaluation.gradients.bonds)
        result, metrics = {}, {}
        for name, probabilities, raw_gradient in (
                ('atoms', graph.X, raw.X), ('bonds', graph.E_upper, raw.E_upper)):
            if (probabilities.shape != raw_gradient.shape
                    or probabilities.shape != getattr(evaluation.logits, name).shape
                    or probabilities.shape != getattr(evaluation.gradients, name).shape):
                raise ValueError('Native/combined gradient shape differs')
            # Same T=1 softmax pullback used by the original combined evaluator.
            native = probabilities * (raw_gradient - jnp.sum(
                probabilities * raw_gradient, axis=-1, keepdims=True))
            noise, measurement = direction_noise(np.asarray(native), rate,
                                                 self.policy['ratio'], self.generator)
            deterministic = getattr(base, name)
            perturbation = jax.device_put(noise, deterministic.sharding)
            value = deterministic + perturbation
            if value.dtype != deterministic.dtype or not np.isfinite(np.asarray(value)).all():
                raise FloatingPointError('Nonfinite or changed-dtype noisy proposal')
            measurement['combined_gradient_norm'] = float(np.linalg.norm(
                np.asarray(getattr(evaluation.gradients, name), dtype=np.float64)))
            measurement['realized_FP32_noise_displacement_norm'] = float(np.linalg.norm(
                np.asarray(value, dtype=np.float64) - np.asarray(deterministic, dtype=np.float64)))
            result[name], metrics[name] = value, measurement
        self.next_step += 1
        return self.logit_type(result['atoms'], result['bonds']), {
            'version': VERSION, 'ratio': self.policy['ratio'], 'step': step,
            'seed': self.policy['seed'], 'rng': 'numpy.PCG64',
            'rng_state_before': before_rng, 'rng_state_after': deepcopy(self.generator.bit_generator.state),
            'blocks': metrics, 'model_key_changed': False,
            'gradient_normalized_or_clipped': False,
            'chemistry_weight_used_for_noise_scale': False}


def proposal_tree(source):
    """Replace only the original proposal, leaving teacher/trial policy intact."""
    tree = ast.parse(textwrap.dedent(source))
    expected = ast.parse('''proposal = GraphLogits(
        evaluation.logits.atoms - rate * evaluation.gradients.atoms,
        evaluation.logits.bonds - rate * evaluation.gradients.bonds)''').body[0]
    typed = deepcopy(expected)
    typed.value.func = ast.parse('type(evaluation.logits)', mode='eval').body
    admitted = {ast.dump(expected), ast.dump(typed)}
    replacements = 0

    class Change(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal replacements
            if ast.dump(node) in admitted:
                replacements += 1
                return [ast.copy_location(value, node) for value in ast.parse('''
proposal, noise_record = _binding_gradient_noise(evaluation, rate, step)
record['noise'] = noise_record
''').body]
            return self.generic_visit(node)

    tree = Change().visit(tree)
    if replacements != 1:
        raise ValueError('Original single-proposal shape differs')
    return ast.fix_missing_locations(tree)


def install(trajectory, policy):
    original = trajectory.run_chemical_updates
    if getattr(original, '_binding_noise_installed', False):
        raise ValueError('Noise proposal already installed')
    proposal = Proposal(policy, trajectory.GraphLogits)
    tree = proposal_tree(inspect.getsource(original))
    namespace = original.__globals__
    namespace['_binding_gradient_noise'] = proposal
    # This isolated producer's original globals must remain live: retention and
    # the existing acceleration context install their helpers after this hook.
    # No source file or another producer process is modified.
    exec(compile(tree, original.__code__.co_filename + '[binding-gradient-noise]', 'exec'), namespace)
    changed = namespace[original.__name__]
    changed._binding_noise_installed = True
    trajectory.run_chemical_updates = changed
    return proposal
