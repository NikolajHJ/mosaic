"""Explicit final categorical arithmetic for the independent bridge comparison.

The historical default is unchanged. The new policy promotes only final decoder
logits while applying the original fixed-occupancy graph function, then restores
the decoder dtype before native adaptation. Decoder internals, learned mapping,
trained weights, hard labels and the common graph-policy implementation remain
unchanged. A diagnostic control motivated this candidate; fresh full-path gates
are still required. Numerical fixtures run only on allocated HPC.
"""

import hashlib
import json

import torch

from latentmol.graph_policy import fixed_occupancy_graph
from latentmol.representation import DecoderOutput, RelaxedGraph


DEFAULT_POLICY = "decoder_dtype_v1"
FINAL_FLOAT64_POLICY = "final_float64_cast_back_v1"
POLICIES = (DEFAULT_POLICY, FINAL_FLOAT64_POLICY)
_FIELDS = ("atoms", "charges", "bonds")


def validate_policy(policy):
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError("Unknown comparison final graph arithmetic policy")
    return policy


def policy_record(policy=DEFAULT_POLICY):
    policy = validate_policy(policy)
    return dict(version="embedding-comparison-final-graph-arithmetic-v1", policy=policy,
        scope="relaxed arm final fixed-occupancy categorical graph and comparison audits",
        categorical_dtype="torch.float64" if policy == FINAL_FLOAT64_POLICY else "decoder_atoms_dtype",
        output_dtype="decoder_atoms_dtype", temperature=1.,
        implementation="latentmol.graph_policy.fixed_occupancy_graph",
        decoder_internal_arithmetic="unchanged", learned_embedding_arithmetic="unchanged",
        hard_decoding="original logits; unchanged", frozen_weights="unchanged")


def policy_binding(policy=DEFAULT_POLICY):
    payload = json.dumps(policy_record(policy), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def comparison_graph(decoded, active, spec, *, policy=DEFAULT_POLICY, temperature=1.):
    """Apply the exact declared numerical boundary without changing parameters."""
    policy = validate_policy(policy)
    if type(temperature) not in (int, float) or temperature != 1.:
        raise ValueError("Comparison graph arithmetic is bound to temperature one")
    if policy == DEFAULT_POLICY:
        return fixed_occupancy_graph(decoded, active, spec, temperature)
    original_dtype = decoded.atoms.dtype
    if any(not isinstance(getattr(decoded, key), torch.Tensor)
            or not getattr(decoded, key).is_floating_point()
            or getattr(decoded, key).dtype != original_dtype
            or getattr(decoded, key).device != decoded.atoms.device for key in _FIELDS):
        raise ValueError("Widened comparison graph requires shared floating decoder dtype/device")
    promoted = DecoderOutput(*(getattr(decoded, key).double() for key in _FIELDS))
    graph = fixed_occupancy_graph(promoted, active, spec, temperature)
    return RelaxedGraph(*(getattr(graph, key).to(original_dtype) for key in _FIELDS))
