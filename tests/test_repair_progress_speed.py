"""Allocated-HPC exact-search parity and timing fixtures, not efficacy claims.

Every numerical/RDKit execution belongs on allocated DTU HPC. The optimized
bond-only backend must retain the same scientific receipt and candidate trace;
its speed is measured without weakening the six-edit/32-check/four-beam policy.
"""
from time import perf_counter

import numpy as np
import pytest

from latentmol import repair_progress as progress
from latentmol.atom_preserving_diagnostics import BondOnlyDiagnostic
from latentmol.atom_preserving_fast import BondSearchContext
from latentmol.chemistry import encode_smiles
from latentmol.representation import GraphSpec


def _graph(smiles, *, mode="native", max_nodes=50):
    spec = GraphSpec(max_nodes=max_nodes)
    encoded = encode_smiles(smiles, spec)
    raw = tuple(np.asarray(encoded[name], dtype=np.int64)
                for name in ("atom_types", "charges", "bond_types"))
    if mode != "native":
        i, j = np.flatnonzero(raw[0])[:2]
        label = {"disconnected": 0, "triple": 3, "aromatic": 4}[mode]
        raw[2][i, j] = raw[2][j, i] = label
    return spec, raw


def _probabilities(raw, spec, *, regime="ties", seed=0):
    atoms = np.eye(len(spec.atomic_numbers))[raw[0]]
    charges = np.eye(len(spec.formal_charges))[raw[1]]
    if regime == "ties":
        bonds = np.full((*raw[2].shape, len(spec.bond_names)), 1 / len(spec.bond_names))
    elif regime == "zeros":
        bonds = np.eye(len(spec.bond_names))[raw[2]]
    else:
        generator = np.random.default_rng(seed)
        bonds = generator.random((*raw[2].shape, len(spec.bond_names)))
        bonds = bonds + bonds.transpose(1, 0, 2)
        if seed == 0:
            bonds[..., 0] = 0
        bonds /= bonds.sum(-1, keepdims=True)
    return atoms, charges, bonds


def _difficult_graph():
    spec, raw = _graph("CCCC(F)(F)F")
    # Deliberately make every present C-F label a triple. Atom identities,
    # formal charges and all 50 slots remain fixed for both search backends.
    fluorine = spec.atomic_numbers.index(9)
    for i in np.flatnonzero(raw[0] == fluorine):
        for j in np.flatnonzero(raw[2][i]):
            raw[2][i, j] = raw[2][j, i] = spec.bond_names.index("TRIPLE")
    return spec, raw


@pytest.mark.parametrize("smiles", ["CCC", "CCO", "C#N", "c1ccncc1", "CS(=O)(=O)C",
                                    "FS(F)(F)(F)(F)F", "C[NH2+]C", "CC(=O)[O-]"])
@pytest.mark.parametrize("mode", ["native", "disconnected", "triple", "aromatic"])
def test_progress_complete_receipt_identity_and_policy_match_reference(smiles, mode):
    spec, raw = _graph(smiles, mode=mode)
    probabilities = _probabilities(raw, spec)
    policy = progress.ProgressSearchConfig(max_edits=3, checks_per_depth=12, beam_width=4)
    expected = progress.search_progress_repair_reference(raw, probabilities, spec, policy)
    observed = progress.search_progress_repair(raw, probabilities, spec, policy)
    assert observed == expected
    assert observed["policy"] == policy.record()
    if observed["witness"] is not None:
        observed["witness"].verify(spec)
        assert observed["witness"].identity == expected["witness"].identity


@pytest.mark.parametrize("regime,seed", [("random", seed) for seed in range(4)] + [("zeros", 0)])
def test_full_six_edit_budget_probability_rankings_and_zero_support_are_exact(seed, regime):
    spec, raw = _difficult_graph()
    probabilities = _probabilities(raw, spec, regime=regime, seed=seed)
    original = tuple(value.copy() for value in (*raw, *probabilities))
    policy = progress.ProgressSearchConfig()  # Exactly six/32/four, not a speed shortcut.
    expected = progress.search_progress_repair_reference(raw, probabilities, spec, policy)
    observed = progress.search_progress_repair(raw, probabilities, spec, policy)
    assert observed == expected
    assert observed["checks"] <= 6 * 32
    assert all(np.array_equal(before, after) for before, after in zip(original, (*raw, *probabilities)))


def test_every_checked_candidate_order_and_scalar_cost_matches_reference(monkeypatch):
    spec, raw = _difficult_graph()
    probabilities = _probabilities(raw, spec, regime="random", seed=3)
    policy = progress.ProgressSearchConfig()
    reference_trace, optimized_trace = [], []
    authoritative = progress._diagnose_progress_graph
    native_diagnose = BondOnlyDiagnostic.diagnose
    scalar_cost = BondSearchContext.cost

    def reference_diagnose(graph, graph_spec):
        reference_trace.append(tuple(value.tobytes() for value in graph))
        return authoritative(graph, graph_spec)

    with monkeypatch.context() as context:
        context.setattr(progress, "_diagnose_progress_graph", reference_diagnose)
        expected = progress.search_progress_repair_reference(raw, probabilities, spec, policy)

    authoritative_calls = 0
    def initial_or_final(graph, graph_spec):
        nonlocal authoritative_calls
        authoritative_calls += 1
        if authoritative_calls == 1:
            optimized_trace.append(tuple(value.tobytes() for value in graph))
        return authoritative(graph, graph_spec)

    def trace_candidate(engine, bonds, **kwargs):
        optimized_trace.append((engine.raw[0].tobytes(), engine.raw[1].tobytes(), bonds.tobytes()))
        return native_diagnose(engine, bonds, **kwargs)

    checked_costs = 0
    def exact_cost(context, bonds):
        nonlocal checked_costs
        checked_costs += 1
        count, weighted = scalar_cost(context, bonds)
        measured = progress.measure_probability_edits(context.original,
            (context.original[0], context.original[1], bonds), context.probabilities,
            context.spec, context.config)
        assert measured["edit_count"] == count
        assert measured["weighted_cost"] == weighted
        assert measured["edit_counts"]["atom"] == measured["edit_counts"]["charge"] == 0
        return count, weighted

    with monkeypatch.context() as context:
        context.setattr(progress, "_diagnose_progress_graph", initial_or_final)
        context.setattr(BondOnlyDiagnostic, "diagnose", trace_candidate)
        context.setattr(BondSearchContext, "cost", exact_cost)
        observed = progress.search_progress_repair(raw, probabilities, spec, policy)
    assert observed == expected
    assert optimized_trace == reference_trace
    assert checked_costs == observed["checks"]
    assert authoritative_calls == 1 + int(observed["witness"] is not None)


def test_progress_roundtrip_rejected_intermediate_keeps_identical_beam(monkeypatch):
    spec, raw = _graph("CCC", max_nodes=3)
    raw[2][raw[2] == 1] = 3
    probabilities = _probabilities(raw, spec, regime="zeros")
    actual_encode = progress.encode_smiles
    def require_propane(smiles, graph_spec):
        if smiles != "CCC":
            raise ValueError("Fixture rejects an intermediate schema roundtrip")
        return actual_encode(smiles, graph_spec)
    monkeypatch.setattr(progress, "encode_smiles", require_propane)
    policy = progress.ProgressSearchConfig(max_edits=2, checks_per_depth=128, beam_width=32)
    observed = progress.search_progress_repair(raw, probabilities, spec, policy)
    assert observed == progress.search_progress_repair_reference(raw, probabilities, spec, policy)
    assert observed["witness"] is not None and observed["witness"].kind == "valid"


def test_progress_reuses_invariants_instead_of_full_receipts_for_rejected_candidates(monkeypatch):
    spec, raw = _difficult_graph()
    probabilities = _probabilities(raw, spec)
    policy = progress.ProgressSearchConfig()
    measure = progress.measure_probability_edits
    diagnose = progress._diagnose_progress_graph
    counts = {"reference_measure": 0, "reference_diagnose": 0,
              "optimized_measure": 0, "optimized_diagnose": 0}
    results = []
    for mode, search in (("reference", progress.search_progress_repair_reference),
                         ("optimized", progress.search_progress_repair)):
        def counted_measure(*args, **kwargs):
            counts[mode + "_measure"] += 1
            return measure(*args, **kwargs)
        def counted_diagnose(*args, **kwargs):
            counts[mode + "_diagnose"] += 1
            return diagnose(*args, **kwargs)
        with monkeypatch.context() as context:
            context.setattr(progress, "measure_probability_edits", counted_measure)
            context.setattr(progress, "_diagnose_progress_graph", counted_diagnose)
            results.append(search(raw, probabilities, spec, policy))
    assert results[0] == results[1]
    assert results[0]["checks"] > 2
    assert counts["reference_measure"] == counts["reference_diagnose"] == 1 + results[0]["checks"]
    assert counts["optimized_measure"] == counts["optimized_diagnose"] <= 2


def test_atom_edit_policy_retains_reference_execution(monkeypatch):
    sentinel = {"test": "atom-edited execution remains on its declared reference path"}
    called = []
    def reference(*args):
        called.append(args)
        return sentinel
    monkeypatch.setattr(progress, "search_progress_repair_reference", reference)
    policy = progress.ProgressSearchConfig(allow_atom_edits=True)
    assert progress.search_progress_repair("raw", "probabilities", "schema", policy) is sentinel
    assert called == [("raw", "probabilities", "schema", policy)]


@pytest.mark.parametrize("defect", ["asymmetric_bonds", "unnormalized_probabilities"])
def test_public_input_rejection_is_not_bypassed_by_optimized_execution(defect):
    spec, raw = _graph("CCO")
    probabilities = _probabilities(raw, spec)
    if defect == "asymmetric_bonds":
        raw[2][0, 1] = 3
    else:
        probabilities[2][0, 1] = 0
    for search in (progress.search_progress_repair_reference, progress.search_progress_repair):
        with pytest.raises(ValueError):
            search(raw, probabilities, spec)


@pytest.mark.parametrize("regime", ["ties", "random", "zeros"])
def test_paired_full_policy_walltimes_are_recorded_without_changing_search_budget(record_property, regime):
    spec, raw = _difficult_graph()
    probabilities = _probabilities(raw, spec, regime=regime, seed=2)
    policy = progress.ProgressSearchConfig()
    results, seconds = [], []
    for search in (progress.search_progress_repair_reference, progress.search_progress_repair):
        start = perf_counter()
        results.append(search(raw, probabilities, spec, policy))
        seconds.append(perf_counter() - start)
    assert results[0] == results[1]
    record_property("reference_seconds", seconds[0])
    record_property("optimized_seconds", seconds[1])
    record_property("reference_over_optimized", seconds[0] / max(seconds[1], 1e-12))
    record_property("checks", results[0]["checks"])
    record_property("policy", str(policy.record()))
