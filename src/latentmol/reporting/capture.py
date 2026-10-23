"""Detach the shared graph already evaluated by an allocated optimization run.

No re-decoding, sampling, probability modification or model imports occur here.
This persistence helper belongs outside autograd and outside scorer adapters.
"""

import json
from pathlib import Path


def capture_shared_graph(directory, step, graph, spec, *, active_slots=None, temperature=None):
    """Persist all three channels for later offline heatmaps and graph playback.

    The caller supplies its current canonical RelaxedGraph. These are the same
    fixed-mask-conditioned (or declared soft-occupancy) distributions sent to
    all scorers, not raw decoder logits and not a repaired DeFoG target.
    """
    path = Path(directory) / f"probabilities-{step:04d}.json"
    value = {"snapshot_version": "shared-relaxed-graph-v1", "step": step,
        "graph_spec": spec.to_dict(), "row_slots": list(range(spec.max_nodes)),
        "active_slots": active_slots, "temperature": temperature,
        "scope": "Saved canonical shared graph passed to scorers; occupancy policy is recorded in run configuration. Not a repaired target or raw-logit softmax."}
    for name in ("atoms", "charges", "bonds"):
        tensor = getattr(graph, name)
        if tensor.shape[0] != 1:
            raise ValueError("Trajectory snapshots require exactly one molecular state")
        value[name] = tensor[0].detach().cpu().tolist()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes((json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)
    return path
