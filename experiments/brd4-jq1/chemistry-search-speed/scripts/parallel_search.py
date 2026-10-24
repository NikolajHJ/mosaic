"""Experiment-only ordered parallel evaluation of detached repair layers.

The original candidate generators, budgets, native checks, costs, beam ranks,
tie order and witnesses are retained. Only independent checks within one layer
run concurrently. Depths and the original/progress passes remain sequential.
This module never installs a change into a saved or production run.
"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os


_ACTIVE = False


def _worker_initialize():
    """Keep detached RDKit workers away from GPUs and nested CPU threading."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def _base_point(payload):
    from mosaic.small_molecule.chemistry._reference.chemistry import encode_smiles
    from mosaic.small_molecule.chemistry._reference.probability_repair import (
        measure_probability_edits,
    )
    from mosaic.small_molecule.chemistry._reference.quality import diagnose_graph

    original, candidate, p, spec, config, depth = payload
    diagnostic = diagnose_graph(*candidate, spec)
    cost = measure_probability_edits(original, candidate, p, spec, config)
    if (
        cost["edit_count"] != depth
        or cost["edit_counts"]["atom"]
        or cost["edit_counts"]["charge"]
    ):
        raise RuntimeError("Bond-only generator violated the direct-cost invariant")
    rejected = False
    if diagnostic["valid"]:
        try:
            encoded = encode_smiles(diagnostic["smiles"], spec)
            if encoded["smiles"] != diagnostic["smiles"]:
                raise ValueError("Canonical schema roundtrip identity differs")
        except ValueError:
            rejected = True
            diagnostic = {
                **diagnostic,
                "valid": False,
                "problem_burden": max(1, diagnostic.get("problem_burden") or 0),
            }
    return diagnostic, cost, rejected


def _progress_diagnose(graph, spec):
    from mosaic.small_molecule.chemistry._reference.chemistry import (
        encode_smiles, inspect_graph,
    )
    from mosaic.small_molecule.chemistry._reference.quality import diagnose_graph
    from mosaic.small_molecule.chemistry._reference.repair_progress import (
        diagnostic_vector,
    )

    value = diagnose_graph(*graph, spec)
    inspected = inspect_graph(*graph, spec)
    valid = inspected.smiles is not None
    if valid:
        try:
            valid = encode_smiles(inspected.smiles, spec)["smiles"] == inspected.smiles
        except ValueError:
            valid = False
    value = {**value, "valid": valid}
    return value, diagnostic_vector(value)


def _progress_point(payload):
    from mosaic.small_molecule.chemistry._reference.probability_repair import (
        measure_probability_edits,
    )

    original, candidate, p, spec, cost_config, depth = payload
    diagnostic, after = _progress_diagnose(candidate, spec)
    measured = measure_probability_edits(original, candidate, p, spec, cost_config)
    if measured["edit_count"] != depth or measured["edit_counts"]["charge"]:
        raise RuntimeError("Progress search violated the depth/charge invariant")
    return diagnostic, after, measured


def _base_search(pool, raw, probabilities, spec, config):
    import numpy as np
    from mosaic.small_molecule.chemistry._reference.atom_preserving_chemistry import (
        _bond_edits, _key, repair_policy,
    )
    from mosaic.small_molecule.chemistry._reference.chemistry import encode_smiles
    from mosaic.small_molecule.chemistry._reference.probability_repair import (
        measure_probability_edits, probability_identity, validate_probabilities,
    )
    from mosaic.small_molecule.chemistry._reference.quality import diagnose_graph
    from mosaic.small_molecule.chemistry._reference.repair_search import (
        round_robin_candidates,
    )

    original = tuple(
        np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value).copy()
        for value in raw
    )
    p = validate_probabilities(probabilities, spec)
    diagnostic = diagnose_graph(*original, spec)
    raw_rdkit_valid = diagnostic["valid"]
    raw_roundtrip_error = None
    if raw_rdkit_valid:
        try:
            if encode_smiles(diagnostic["smiles"], spec)["smiles"] != diagnostic["smiles"]:
                raise ValueError("Canonical schema roundtrip identity differs")
        except ValueError as error:
            raw_roundtrip_error = f"{type(error).__name__}: {error}"
            diagnostic = {
                **diagnostic, "valid": False,
                "problem_burden": max(1, diagnostic.get("problem_burden") or 0),
            }
    result = {
        "policy": repair_policy(config), "raw_graph_id": diagnostic.get("graph_id"),
        "probability_sha256": probability_identity(p, spec),
        "raw_valid": diagnostic["valid"], "raw_rdkit_valid": raw_rdkit_valid,
        "raw_roundtrip_error": raw_roundtrip_error, "checks": 0,
        "checks_by_depth": [], "valid_witnesses_observed": 0,
        "roundtrip_rejected_candidates": 0, "weighted_witness": None,
        "best_observed_weighted_cost": None, "distance_is_exact": False,
    }
    if (
        not diagnostic.get("schema_supported")
        or not diagnostic.get("heavy_atoms")
        or not diagnostic.get("categorical_support_labels_ok")
    ):
        return {
            **result, "status": "unsupported",
            "reason": "Requires nonempty canonical hard-label support",
        }
    measure_probability_edits(original, original, p, spec, config)
    if diagnostic["valid"]:
        return {**result, "status": "already_valid", "best_observed_weighted_cost": 0.0}
    seen, beam, best, sequence = {_key(original)}, [(original, diagnostic)], None, 0
    for depth in range(1, config.max_edits + 1):
        frontier, capped, layer = [], False, []
        streams = [
            _bond_edits(graph, original, spec, diag, p, config)
            for graph, diag in beam
        ]
        # Gather only the exact ordered unique prefix the original loop checks.
        # The first extra unseen candidate still establishes its capped flag.
        for candidate in round_robin_candidates(streams):
            identity = _key(candidate)
            if identity in seen:
                continue
            if len(layer) >= config.checks_per_depth:
                capped = True
                break
            seen.add(identity)
            layer.append(candidate)
        checked = 0
        payloads = (
            (original, candidate, p, spec, config, depth) for candidate in layer
        )
        # map returns in input order, irrespective of worker completion order.
        points = pool.map(_base_point, payloads, chunksize=1)
        for candidate, (candidate_diag, cost, rejected) in zip(layer, points):
            checked += 1
            result["checks"] += 1
            result["roundtrip_rejected_candidates"] += int(rejected)
            if candidate_diag["valid"]:
                result["valid_witnesses_observed"] += 1
                witness = {
                    **cost, "graph_id": candidate_diag["graph_id"],
                    "smiles": candidate_diag["smiles"],
                    "labels": {
                        name: value.tolist()
                        for name, value in zip(("atoms", "charges", "bonds"), candidate)
                    },
                }
                if best is None or (cost["weighted_cost"], cost["edit_count"]) < (
                    best["weighted_cost"], best["edit_count"]
                ):
                    best = witness
                continue
            sequence += 1
            burden = candidate_diag.get("problem_burden")
            rank = (
                (1000000 if burden is None else burden)
                + max(0, candidate_diag.get("component_count", 1) - 1),
                cost["weighted_cost"], sequence,
            )
            frontier.append((rank, candidate, candidate_diag))
            frontier.sort(key=lambda value: value[0])
            del frontier[config.beam_width:]
        result["checks_by_depth"].append({"depth": depth, "checks": checked, "capped": capped})
        beam = [(item[1], item[2]) for item in frontier]
        if not beam:
            break
    if best is None:
        return {
            **result, "status": "censored",
            "reason": "No valid bond-only witness within the declared budget",
        }
    return {
        **result, "status": "witness_found", "weighted_witness": best,
        "best_observed_weighted_cost": best["weighted_cost"],
    }


def _progress_search(pool, raw, probabilities, spec, config=None):
    import numpy as np
    from mosaic.small_molecule.chemistry._reference.atom_preserving_chemistry import _bond_edits
    from mosaic.small_molecule.chemistry._reference.probability_repair import (
        ProbabilityRepairConfig, measure_probability_edits, validate_probabilities,
    )
    from mosaic.small_molecule.chemistry._reference.repair_progress import (
        ProgressSearchConfig, ProgressWitness, _atom_edits, _labels, _signature,
        strictly_improves,
    )
    from mosaic.small_molecule.chemistry._reference.repair_search import round_robin_candidates

    config = ProgressSearchConfig() if config is None else config
    if not isinstance(config, ProgressSearchConfig):
        raise ValueError("Require an explicit bounded progress search configuration")
    original = tuple(np.asarray(field).copy() for field in raw)
    p = validate_probabilities(probabilities, spec)
    cost_config = ProbabilityRepairConfig(
        max_edits=config.max_edits, checks_per_depth=config.checks_per_depth,
        beam_width=config.beam_width, probability_weight=config.probability_weight,
    )
    measure_probability_edits(original, original, p, spec, cost_config)
    initial, before = _progress_diagnose(original, spec)
    result = dict(
        policy=config.record(), status="no_progress", checks=0, before=before,
        witness=None, valid_witnesses=0, partial_witnesses=0,
        source_probability_sha256=_signature([v.tolist() for v in p]),
    )
    if initial["valid"]:
        return {**result, "status": "already_valid"}
    if before is None:
        return {**result, "status": "unsupported"}
    seen = {_labels(original)}
    beam = [(original, initial)]
    best_valid = best_partial = None
    for depth in range(1, config.max_edits + 1):
        streams = []
        for graph, diagnostic in beam:
            bonds = _bond_edits(graph, original, spec, diagnostic, p, cost_config)
            streams.append(
                round_robin_candidates([bonds, _atom_edits(graph, original, spec, config)])
                if config.allow_atom_edits else bonds
            )
        frontier, layer = [], []
        for candidate in round_robin_candidates(streams):
            key = _labels(candidate)
            if key in seen:
                continue
            if len(layer) >= config.checks_per_depth:
                break
            seen.add(key)
            layer.append(candidate)
        checked = 0
        payloads = (
            (original, candidate, p, spec, cost_config, depth) for candidate in layer
        )
        points = pool.map(_progress_point, payloads, chunksize=1)
        for candidate, (diagnostic, after, measured) in zip(layer, points):
            checked += 1
            result["checks"] += 1
            cost = measured["weighted_cost"] + config.atom_edit_penalty * measured["edit_counts"]["atom"]
            if diagnostic["valid"]:
                result["valid_witnesses"] += 1
                if best_valid is None or cost < best_valid.cost:
                    best_valid = ProgressWitness.create("valid", original, candidate, cost, before, after)
                continue
            if np.array_equal(original[0], candidate[0]) and strictly_improves(before, after):
                result["partial_witnesses"] += 1
                key = (sum(after.values()), cost)
                if best_partial is None or key < (sum(dict(best_partial.after).values()), best_partial.cost):
                    best_partial = ProgressWitness.create("partial", original, candidate, cost, before, after)
            if after is not None:
                frontier.append(((sum(after.values()), cost, checked), candidate, diagnostic))
                frontier.sort(key=lambda value: value[0])
                del frontier[config.beam_width:]
        beam = [(item[1], item[2]) for item in frontier]
        if not beam:
            break
    witness = best_valid or best_partial
    if witness is not None:
        witness.verify(spec)
        result.update(status=witness.kind, witness=witness)
    return result


@contextmanager
def install_parallel(workers):
    """Temporarily install only benchmark preparation calls with a spawn pool.

    Call from a guarded executable entry point (``if __name__ == '__main__'``).
    A persistent pool amortizes startup across preparations. Native worker errors
    propagate through ordered map; none become a censored or passed search.
    Context exit always restores both exact original function references.
    """
    global _ACTIVE
    if type(workers) is not int:
        raise TypeError("Parallel worker count must be an integer")
    if workers not in (2, 4, 8):
        raise ValueError("Declared benchmark worker counts are 2, 4 and 8")
    if _ACTIVE:
        raise RuntimeError("Parallel search contexts cannot be nested")
    from mosaic.small_molecule.chemistry import preparation

    original_base = preparation.search_atom_preserving_repair
    original_progress = preparation.search_progress_repair
    _ACTIVE = True
    pool = None
    try:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_initialize,
        )
        preparation.search_atom_preserving_repair = (
            lambda raw, probabilities, spec, config: _base_search(
                pool, raw, probabilities, spec, config
            )
        )
        preparation.search_progress_repair = (
            lambda raw, probabilities, spec, config=None: _progress_search(
                pool, raw, probabilities, spec, config
            )
        )
        yield {
            "workers": workers, "start_method": "spawn",
            "reduction_order": "original_candidate_order",
        }
    finally:
        preparation.search_atom_preserving_repair = original_base
        preparation.search_progress_repair = original_progress
        _ACTIVE = False
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
