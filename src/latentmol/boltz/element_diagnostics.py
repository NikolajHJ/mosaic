"""Elemental response before RDKit conversion, including invalid hard graphs.

Gradients are of the signed minimized objective. A negative N-minus-C gradient
contrast favors transferring probability from carbon to nitrogen locally. These
are conditional fixed-pose signals, not evidence of useful binding chemistry.
"""
import torch


def element_record(decoded, graph, active, spec, temperature, gradients=None):
    """Detach only for reporting; preserve raw occupancy and conditional mass."""
    slots = active.nonzero().flatten()
    carbon = spec.atomic_numbers.index(6)
    others = [i for i, number in enumerate(spec.atomic_numbers) if number not in (0, 6)]
    raw = (decoded.atoms.detach() / temperature).softmax(-1)[0]
    probabilities = graph.atoms.detach()[0].index_select(0, slots)
    logits = (decoded.atoms.detach()[0] / temperature).index_select(0, slots)
    hard = probabilities.argmax(-1)
    row = {'active_slots': slots.tolist(), 'atomic_numbers': list(spec.atomic_numbers),
           'hard_atomic_numbers': [spec.atomic_numbers[i] for i in hard.tolist()],
           'conditional_probabilities': probabilities.tolist(),
           'conditional_expected_heteroatom_count': float(probabilities[:, others].sum()),
           'raw_expected_heteroatom_count_all_slots': float(raw[:, others].sum()),
           'raw_expected_heteroatom_count_active_slots': float(raw[slots][:, others].sum()),
           'carbon_logit_margin_over_best_heteroatom':
               (logits[:, carbon] - logits[:, others].max(-1).values).tolist(),
           'gradient_scope': 'signed loss; fixed occupancy and frozen native pose'}
    if gradients is not None:
        logit_gradient, probability_gradient = gradients
        if logit_gradient is None or probability_gradient is None:
            raise ValueError('Elemental diagnostic path is disconnected')
        for value in gradients:
            if not bool(value.isfinite().all()):
                raise ValueError('Nonfinite elemental diagnostic gradient')
        gl = logit_gradient.detach()[0].index_select(0, slots)
        gp = probability_gradient.detach()[0].index_select(0, slots)
        row.update(atom_logit_gradient_norm=float(gl.double().norm()),
                   atom_probability_gradient_norm=float(gp.double().norm()),
                   atom_logit_gradients=gl.tolist(),
                   probability_gradient_contrast_to_carbon=(gp - gp[:, carbon:carbon + 1]).tolist())
    return row
