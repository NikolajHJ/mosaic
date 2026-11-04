"""Local tests for the Stage 2 operators and pitfall checks.

No GPU, no JAX, no Protenix, no RDKit: NumPy, SciPy and pytest only, so these
run on a laptop in a second or two. They cover the protocol 5.1 invariants, the
behaviour each operator claims, each improvement in isolation, and the pitfall
checks against trajectories built to fail in one known way.

    python -m pytest stage2/test_stage2.py -q
"""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from diagnostics import (
    check_pitfalls,
    per_step,
    score_metric,
    topology_reversals,
)
from operators import (
    BOND_STATES,
    ELEMENTS,
    NEUTRAL_VALENCE,
    Stage2Config,
    _pair_row,
    apply_fluidity_floor,
    apply_operators,
    argmax_valence_violations,
    check_invariants,
    components,
    connect_components,
    n_atoms_from_pairs,
    pair_indices,
    project_simplex_rows,
    project_valence,
    shrink_weak_edges,
    soft_valence,
    valence_ceiling,
)

TOL = 2e-6

EXPERIMENT_YAML = Path(__file__).resolve().parent / "experiment.yaml"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def one_hot(index, size):
    row = np.zeros(size)
    row[index] = 1.0
    return row


def atoms_of(symbols):
    return np.array([one_hot(ELEMENTS.index(s), len(ELEMENTS)) for s in symbols])


def edges_of(n, bonds):
    """``bonds`` maps ``(i, j)`` to a bond-state name; the rest are NO_BOND."""
    edges = np.tile(one_hot(0, len(BOND_STATES)), (n * (n - 1) // 2, 1))
    for (i, j), state in bonds.items():
        edges[_pair_row(n, i, j)] = one_hot(BOND_STATES.index(state), len(BOND_STATES))
    return edges


def chain(n, element="C", state="SINGLE"):
    return atoms_of([element] * n), edges_of(
        n, {(i, i + 1): state for i in range(n - 1)}
    )


def soften(rows, mixing=0.1):
    k = rows.shape[-1]
    return (1 - mixing) * rows + mixing / k


# --------------------------------------------------------------------------- #
# protocol 5.1 invariants
# --------------------------------------------------------------------------- #


def test_pair_row_matches_lexicographic_triu():
    for n in (2, 3, 5, 9):
        i, j = pair_indices(n)
        for row, (a, b) in enumerate(zip(i, j)):
            assert _pair_row(n, int(a), int(b)) == row
            assert _pair_row(n, int(b), int(a)) == row


def test_pair_count_inverts():
    for n in (1, 2, 4, 25):
        assert n_atoms_from_pairs(n * (n - 1) // 2) == n
    with pytest.raises(ValueError):
        n_atoms_from_pairs(4)


def test_check_invariants_accepts_a_valid_graph_and_returns_n():
    atoms, edges = chain(5)
    assert check_invariants(atoms, edges) == 5


@pytest.mark.parametrize("break_it", ["shape", "sum", "negative", "nonfinite"])
def test_check_invariants_rejects_each_violation(break_it):
    atoms, edges = chain(4)
    if break_it == "shape":
        edges = edges[:-1]
    elif break_it == "sum":
        edges = edges * 0.9
    elif break_it == "negative":
        edges = edges.copy()
        edges[0] = [1.5, -0.5, 0, 0, 0]
    else:
        atoms = atoms.copy()
        atoms[0, 0] = np.nan
    with pytest.raises((ValueError, TypeError)):
        check_invariants(atoms, edges)


def test_symmetry_and_diagonal_are_structural():
    """There is no variable that could hold an asymmetric or self bond."""
    _, edges = chain(6)
    assert len(edges) == 6 * 5 // 2
    i, j = pair_indices(6)
    assert (i < j).all()


def test_simplex_projection_returns_distributions():
    rows = np.array([[2.0, -1.0, 0.5, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]])
    out = project_simplex_rows(rows)
    assert (out >= 0).all()
    assert np.allclose(out.sum(-1), 1.0)
    assert np.allclose(out[1], rows[1]), "an in-set row is its own projection"


# --------------------------------------------------------------------------- #
# 5.2 valence projection
# --------------------------------------------------------------------------- #


def test_valence_projection_is_inactive_on_a_feasible_graph():
    atoms, edges = chain(6)
    out, receipt = project_valence(atoms, edges)
    assert receipt["active"] is False
    assert np.array_equal(out, edges)


def test_valence_projection_does_not_fill_toward_the_ceiling():
    """A carbon chain is under-coordinated; the operator leaves it alone."""
    atoms, edges = chain(4)
    before = soft_valence(edges, 4)
    out, _ = project_valence(atoms, edges)
    assert (before < atoms @ NEUTRAL_VALENCE).any()
    assert np.allclose(soft_valence(out, 4), before)


def test_valence_projection_removes_excess_and_satisfies_kkt():
    # One carbon with three TRIPLE bonds: valence 9 against a ceiling of 4.
    atoms = atoms_of(["C"] * 4)
    edges = edges_of(4, {(0, 1): "TRIPLE", (0, 2): "TRIPLE", (0, 3): "TRIPLE"})
    caps, _ = valence_ceiling(atoms)
    assert soft_valence(edges, 4)[0] == pytest.approx(9.0)
    out, receipt = project_valence(atoms, edges)
    assert receipt["active"] is True
    assert receipt["max_excess_before"] == pytest.approx(5.0)
    assert receipt["max_excess_after"] <= TOL
    assert (soft_valence(out, 4) <= caps + TOL).all()
    check_invariants(atoms, out)


def test_valence_projection_never_raises_an_atoms_valence():
    atoms = atoms_of(["C", "C", "O", "N"])
    edges = edges_of(4, {(0, 1): "TRIPLE", (0, 2): "DOUBLE", (2, 3): "TRIPLE"})
    before = soft_valence(edges, 4)
    out, receipt = project_valence(atoms, edges)
    assert receipt["active"] is True
    assert (soft_valence(out, 4) <= before + TOL).all()


def test_valence_projection_is_idempotent():
    atoms = atoms_of(["C"] * 5)
    edges = soften(edges_of(5, {(0, i): "TRIPLE" for i in range(1, 5)}))
    once, _ = project_valence(atoms, edges)
    twice, receipt = project_valence(atoms, once)
    assert receipt["active"] is False
    assert np.allclose(once, twice)


def test_readout_caps_only_tighten_the_ceiling():
    atoms = soften(atoms_of(["C", "F", "C"]), mixing=0.4)
    loose, tightened_none = valence_ceiling(atoms, readout_caps=False)
    tight, tightened = valence_ceiling(atoms, readout_caps=True)
    assert tightened_none == 0
    assert (tight <= loose + 1e-12).all()
    assert tightened >= 1, "the softened fluorine has a looser soft ceiling"


def test_readout_caps_catch_an_argmax_violation_the_soft_ceiling_allows():
    """The gap between a feasible relaxed state and a valid molecule.

    Atom 0 is 45% carbon and 55% fluorine, so its soft ceiling is 2.35 while
    the element its readout selects allows 1. Two partial bonds give it an
    expected valence of 1.6: inside the soft ceiling, outside fluorine's.
    """
    atoms = np.array(
        [
            [0.45, 0.0, 0.0, 0.0, 0.55, 0.0],
            *atoms_of(["C", "C"]),
        ]
    )
    edges = np.tile(one_hot(0, len(BOND_STATES)), (3, 1))
    edges[_pair_row(3, 0, 1)] = [0.1, 0.9, 0.0, 0.0, 0.0]
    edges[_pair_row(3, 0, 2)] = [0.3, 0.7, 0.0, 0.0, 0.0]
    loose, _ = valence_ceiling(atoms)
    tight, tightened = valence_ceiling(atoms, readout_caps=True)
    assert soft_valence(edges, 3)[0] == pytest.approx(1.6)
    assert soft_valence(edges, 3)[0] <= loose[0] + TOL, "soft ceiling is satisfied"
    assert tight[0] == pytest.approx(1.0) and tightened == 1
    before = argmax_valence_violations(atoms, edges)
    assert before == (1, pytest.approx(1.0)), "but the readout is over-valent"

    # Without the improvement the projection has nothing to do.
    untouched, receipt = project_valence(atoms, edges)
    assert receipt["active"] is False
    assert argmax_valence_violations(atoms, untouched) == before

    out, receipt = project_valence(atoms, edges, config=Stage2Config(readout_caps=True))
    assert receipt["active"] is True and receipt["tightened_cap_atoms"] == 1
    assert soft_valence(out, 3)[0] <= tight[0] + TOL
    assert argmax_valence_violations(atoms, out)[1] < before[1]


def test_valence_damping_is_partial_and_reports_its_residual():
    atoms = atoms_of(["C"] * 4)
    edges = edges_of(4, {(0, 1): "TRIPLE", (0, 2): "TRIPLE", (0, 3): "TRIPLE"})
    exact, _ = project_valence(atoms, edges)
    damped, receipt = project_valence(
        atoms, edges, config=Stage2Config(valence_damping=0.5)
    )
    assert receipt["damping"] == 0.5
    assert receipt["damped_excess"] > TOL, "a partial projection stays infeasible"
    assert np.allclose(damped, 0.5 * edges + 0.5 * exact)
    check_invariants(atoms, damped), "but every row is still a distribution"


def test_aromatic_weight_changes_the_bound():
    edges = edges_of(3, {(0, 1): "AROMATIC", (1, 2): "AROMATIC"})
    assert soft_valence(edges, 3, aromatic_order=1.5)[1] == pytest.approx(3.0)
    assert soft_valence(edges, 3, aromatic_order=1.0)[1] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 5.3 connectivity
# --------------------------------------------------------------------------- #


def test_components_counts_argmax_fragments():
    edges = edges_of(6, {(0, 1): "SINGLE", (1, 2): "SINGLE", (3, 4): "SINGLE"})
    assert len(np.unique(components(6, edges))) == 3  # {0,1,2} {3,4} {5}


def test_connectivity_bridges_fragments_without_prescribing_topology():
    atoms = atoms_of(["C"] * 6)
    edges = soften(edges_of(6, {(0, 1): "SINGLE", (1, 2): "SINGLE", (3, 4): "SINGLE"}))
    out, receipt = connect_components(atoms, edges)
    assert receipt["components_before"] == 3
    assert len(receipt["bridges"]) <= receipt["components_before"] - 1
    assert receipt["components_proposed"] == 1
    check_invariants(atoms, out)


def test_connectivity_only_adds_inter_component_edges():
    atoms = atoms_of(["C"] * 5)
    edges = edges_of(5, {(0, 1): "SINGLE", (2, 3): "SINGLE"})
    labels = components(5, edges)
    _, receipt = connect_components(atoms, edges)
    for a, b in receipt["bridges"]:
        assert labels[a] != labels[b]


def test_connectivity_respects_hard_valence_capacity():
    # Every fluorine is already saturated, so no bridge is possible.
    atoms = atoms_of(["F", "F", "F", "F"])
    edges = edges_of(4, {(0, 1): "SINGLE", (2, 3): "SINGLE"})
    out, receipt = connect_components(atoms, edges)
    assert receipt["bridges"] == []
    assert receipt["components_proposed"] == 2
    assert np.array_equal(out, edges)


def test_bridge_loss_is_reported_not_hidden():
    """A bridge the following valence projection removes is counted."""
    atoms = atoms_of(["C", "C", "F", "F"])
    edges = edges_of(4, {(0, 1): "TRIPLE", (2, 3): "SINGLE"})
    _, _, receipt = apply_operators(atoms, edges, Stage2Config(mode="connected"))
    connectivity = receipt["connectivity"]
    assert "bridges_surviving" in connectivity and "bridges_lost" in connectivity
    assert connectivity["bridges_lost"] == (
        len(connectivity["bridges"]) - connectivity["bridges_surviving"]
    )


def test_bridge_retry_is_bounded_to_one_attempt():
    atoms = atoms_of(["C", "C", "C", "C", "C", "C"])
    edges = soften(edges_of(6, {(0, 1): "SINGLE", (2, 3): "SINGLE", (4, 5): "SINGLE"}))
    _, _, receipt = apply_operators(
        atoms, edges, Stage2Config(mode="connected", bridge_retry=True)
    )
    retry = receipt["connectivity"].get("retry")
    assert retry is None or "bridges_surviving" in retry
    if retry is not None:
        assert retry["bridge_probability"] > 0.6
        assert "retry" not in retry, "no recursion; exactly one attempt"


# --------------------------------------------------------------------------- #
# 5.4 sparsity
# --------------------------------------------------------------------------- #


def test_sparsity_shrinks_only_weak_edges_and_keeps_the_simplex():
    atoms = atoms_of(["C"] * 4)
    edges = np.array(
        [
            [0.1, 0.9, 0.0, 0.0, 0.0],  # strong, untouched
            [0.7, 0.3, 0.0, 0.0, 0.0],  # weak, shrunk
            [1.0, 0.0, 0.0, 0.0, 0.0],  # absent, nothing to remove
            [0.6, 0.2, 0.2, 0.0, 0.0],
            [0.95, 0.05, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0, 0.0],
        ]
    )
    out, receipt = shrink_weak_edges(edges)
    check_invariants(atoms, out)
    assert np.allclose(out[0], edges[0])
    assert np.allclose(out[5], edges[5])
    assert out[1, 0] > edges[1, 0]
    assert receipt["removed_bond_mass"] > 0
    assert receipt["changed_pairs"] == 3


def test_sparsity_can_protect_selected_pairs():
    edges = np.tile(np.array([0.7, 0.3, 0.0, 0.0, 0.0]), (3, 1))
    protect = np.array([True, False, False])
    out, receipt = shrink_weak_edges(edges, protect=protect)
    assert np.allclose(out[0], edges[0])
    assert receipt["changed_pairs"] == 2


def test_sparsity_preserves_conditional_order_ratios():
    edges = np.array([[0.6, 0.2, 0.2, 0.0, 0.0]])
    out, _ = shrink_weak_edges(edges)
    assert out[0, 1] == pytest.approx(out[0, 2]), "ratios rescale, not reshuffle"


# --------------------------------------------------------------------------- #
# improvement: fluidity floor
# --------------------------------------------------------------------------- #


def test_fluidity_floor_is_a_no_op_at_its_defaults():
    atoms, _ = chain(4)
    out, receipt = apply_fluidity_floor(atoms, threshold=1.0, strength=0.0)
    assert receipt["rows_relaxed"] == 0
    assert np.array_equal(out, atoms)


def test_fluidity_floor_relaxes_only_collapsed_rows():
    rows = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # collapsed
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # ambiguous, untouched
        ]
    )
    out, receipt = apply_fluidity_floor(rows, threshold=0.95, strength=0.1)
    assert receipt["rows_relaxed"] == 1
    assert np.allclose(out[1], rows[1])
    assert out[0].max() < 1.0
    assert np.allclose(out.sum(-1), 1.0)
    assert (out[0] > 0).all(), "mass reaches every category"


def test_fluidity_floor_runs_before_valence_so_the_result_stays_feasible():
    atoms = atoms_of(["C", "C", "C"])
    edges = edges_of(3, {(0, 1): "TRIPLE", (0, 2): "TRIPLE"})
    config = Stage2Config(
        mode="valence", fluidity_threshold=0.95, fluidity_strength=0.1
    )
    out_atoms, out_edges, receipt = apply_operators(atoms, edges, config)
    assert receipt["fluidity_floor"]["edges"]["rows_relaxed"] > 0
    caps, _ = valence_ceiling(out_atoms)
    assert (soft_valence(out_edges, 3) <= caps + TOL).all()


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["none", "valence", "connected", "sparse"])
def test_every_mode_preserves_the_invariants(mode):
    atoms = soften(atoms_of(["C", "N", "O", "C", "F", "C"]))
    edges = soften(edges_of(6, {(0, 1): "TRIPLE", (0, 2): "DOUBLE", (3, 4): "SINGLE"}))
    out_atoms, out_edges, receipt = apply_operators(
        atoms, edges, Stage2Config(mode=mode)
    )
    check_invariants(out_atoms, out_edges)
    assert receipt["mode"] == mode
    assert "components_after" in receipt
    assert "argmax_valence_violations" in receipt


def test_mode_none_is_the_frozen_stage1_control():
    atoms, edges = chain(5)
    out_atoms, out_edges, receipt = apply_operators(
        atoms, edges, Stage2Config(mode="none")
    )
    assert np.array_equal(out_edges, edges)
    assert np.array_equal(out_atoms, atoms)
    assert "valence" not in receipt


def test_operator_order_leaves_a_feasible_state_in_every_mode():
    atoms = atoms_of(["C"] * 7)
    edges = soften(
        edges_of(
            7, {(0, 1): "TRIPLE", (0, 2): "TRIPLE", (0, 3): "TRIPLE", (4, 5): "SINGLE"}
        )
    )
    caps, _ = valence_ceiling(atoms)
    for mode in ("valence", "connected", "sparse"):
        _, out_edges, _ = apply_operators(atoms, edges, Stage2Config(mode=mode))
        excess = soft_valence(out_edges, 7) - caps
        assert excess.max() <= TOL, f"{mode} returned an infeasible state"


def test_defaults_record_no_active_improvements():
    config = Stage2Config()
    assert config.improvements == ()
    assert config.record()["active_improvements"] == []


@pytest.mark.parametrize(
    "field,value,name",
    [
        ("readout_caps", True, "readout_caps"),
        ("valence_damping", 0.5, "valence_damping"),
        ("bridge_retry", True, "bridge_retry"),
    ],
)
def test_each_improvement_reports_itself(field, value, name):
    config = Stage2Config(**{field: value})
    assert config.improvements == (name,)


def test_fluidity_floor_reports_itself_only_when_both_knobs_are_set():
    assert Stage2Config(fluidity_strength=0.1).improvements == ()
    assert Stage2Config(
        fluidity_threshold=0.95, fluidity_strength=0.1
    ).improvements == ("fluidity_floor",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "nonsense"},
        {"aromatic_order": 2.0},
        {"valence_damping": 0.0},
        {"valence_damping": 1.5},
        {"fluidity_strength": -0.1},
    ],
)
def test_configuration_rejects_undefined_settings(kwargs):
    with pytest.raises(ValueError):
        Stage2Config(**kwargs)


# --------------------------------------------------------------------------- #
# diagnostics, against trajectories built to fail in one known way
# --------------------------------------------------------------------------- #


def entropy(rows):
    positive = rows > 0
    terms = np.zeros_like(rows)
    terms[positive] = rows[positive] * np.log(rows[positive])
    return float(-terms.sum(-1).mean())


def write_run(directory, states, *, scores=None, receipts=None, metric="contact_score"):
    """Write the artifacts a real run writes, for a list of (atoms, edges).

    ``metric`` picks the backend convention: a Protenix contact score is
    negated into the loss because higher is better, while a Boltz-2 affinity
    prediction *is* the loss because lower is better.
    """
    trajectory = directory / "trajectory"
    (trajectory / "evaluations").mkdir(parents=True)
    (trajectory / "readouts").mkdir(parents=True)
    config = {
        "backend": "protenix" if metric == "contact_score" else "boltz2",
        "score_metric": metric,
    }
    (directory / "stage2_config.json").write_text(json.dumps(config))
    rows = []
    for step, (atoms, edges) in enumerate(states):
        label = f"step_{step:06d}"
        np.savez_compressed(
            trajectory / "evaluations" / f"{label}.npz", X=atoms, E_upper=edges
        )
        n = len(atoms)
        bond_mass = float((1 - edges[:, 0]).sum())
        readout = {
            "element_indices": atoms.argmax(-1).tolist(),
            "bond_upper_indices": edges.argmax(-1).tolist(),
            "sanitized": True,
            "fragment_count": len(np.unique(components(n, edges))),
            "canonical_smiles": f"C{step}",
        }
        (trajectory / "readouts" / f"{label}.json").write_text(json.dumps(readout))
        score = (scores or [0.1] * len(states))[step]
        row = {
            "evaluation": step,
            "evaluated_step": step,
            "loss": score if metric == "affinity_pred_value" else -score,
            "aux_metrics": {metric: score},
            "final": step == len(states) - 1,
            "graph": {
                "n_atoms": n,
                "atom_entropy": entropy(atoms),
                "bond_entropy": entropy(edges),
                "atom_max_probability": float(atoms.max(-1).mean()),
                "bond_max_probability": float(edges.max(-1).mean()),
                "bond_mass": bond_mass,
                "expected_degree": 2 * bond_mass / n,
            },
            "raw_gradient_norm_X": 1.0,
            "raw_gradient_norm_E_upper": 1.0,
            "evaluated_arrays": f"evaluations/{label}.npz",
            "readout": f"readouts/{label}.json",
        }
        if receipts:
            row["post_step_operators"] = receipts[step]
        rows.append(row)
    (trajectory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        "directory": str(directory),
        "name": directory.name,
        "trajectory": trajectory,
        "trace": rows,
        "stage2_config": config,
    }


def verdicts(run):
    rows = per_step(run)
    return {f["id"]: f for f in check_pitfalls(run, rows, topology_reversals(run))}


def test_a_healthy_run_flags_nothing_about_chemistry(tmp_path):
    atoms, edges = chain(6)
    soft = soften(edges, 0.3)
    states = [(soften(atoms, 0.3), soft), (soften(atoms, 0.2), edges), (atoms, edges)]
    run = write_run(tmp_path / "healthy", states, scores=[0.10, 0.12, 0.15])
    found = verdicts(run)
    assert found["valence"]["verdict"] == "ok"
    assert found["connectivity"]["verdict"] == "ok"
    assert found["predictor_exploitation"]["verdict"] == "ok"


def test_fragmentation_is_flagged(tmp_path):
    atoms = atoms_of(["C"] * 6)
    edges = edges_of(6, {(0, 1): "SINGLE", (2, 3): "SINGLE"})
    run = write_run(tmp_path / "fragmented", [(atoms, edges)] * 3)
    found = verdicts(run)
    assert found["connectivity"]["verdict"] == "flag"
    assert found["connectivity"]["final_components"] == 4


def test_over_valence_is_flagged(tmp_path):
    atoms = atoms_of(["C", "C", "C", "C"])
    edges = edges_of(4, {(0, 1): "TRIPLE", (0, 2): "TRIPLE", (0, 3): "TRIPLE"})
    run = write_run(tmp_path / "overvalent", [(atoms, edges)] * 3)
    found = verdicts(run)
    assert found["valence"]["verdict"] == "flag"
    assert found["valence"]["final_violations"] == 1


def test_frozen_topology_is_flagged_and_movement_is_not(tmp_path):
    atoms = atoms_of(["C"] * 4)
    fixed = soften(edges_of(4, {(0, 1): "SINGLE", (1, 2): "SINGLE", (2, 3): "SINGLE"}))
    frozen = write_run(tmp_path / "frozen", [(atoms, fixed)] * 4)
    assert verdicts(frozen)["topology"]["verdict"] == "flag"

    moved = soften(edges_of(4, {(0, 1): "SINGLE", (1, 2): "SINGLE"}))
    back = soften(edges_of(4, {(0, 1): "SINGLE", (1, 2): "SINGLE", (2, 3): "SINGLE"}))
    changing = write_run(
        tmp_path / "changing", [(atoms, fixed), (atoms, moved), (atoms, back)]
    )
    found = verdicts(changing)
    assert found["topology"]["verdict"] == "ok"
    assert found["topology"]["fell_then_rose"] == 1


def test_premature_collapse_is_flagged(tmp_path):
    atoms, edges = chain(5)
    states = [(atoms, edges)] * 6  # one-hot from the first step
    run = write_run(tmp_path / "collapsed", states)
    found = verdicts(run)
    assert found["fluidity"]["verdict"] == "flag"
    assert found["fluidity"]["collapse_step"] == 0


def test_predictor_exploitation_is_flagged_when_chemistry_does_not_follow(tmp_path):
    atoms = atoms_of(["C"] * 5)
    edges = edges_of(5, {(0, 1): "SINGLE", (2, 3): "SINGLE"})
    run = write_run(
        tmp_path / "exploited", [(atoms, edges)] * 3, scores=[0.10, 0.40, 0.90]
    )
    found = verdicts(run)
    assert found["predictor_exploitation"]["verdict"] == "flag"
    assert found["predictor_exploitation"]["final_score"] == 0.90


def test_relaxed_feasibility_is_assessed_apart_from_discrete_validity(tmp_path):
    """The plan requires these two to be reported separately, not conflated."""
    atoms = atoms_of(["C", "C", "C", "C"])
    edges = edges_of(4, {(0, 1): "TRIPLE", (0, 2): "TRIPLE", (0, 3): "TRIPLE"})
    run = write_run(tmp_path / "mismatch", [(atoms, edges)] * 2)
    found = verdicts(run)
    assert found["soft_versus_hard_feasibility"]["relaxed_feasible"] == 0
    assert found["valence"]["verdict"] == "flag"


def test_bridge_loss_surfaces_in_the_verdicts(tmp_path):
    atoms, edges = chain(4)
    receipt = {
        "mode": "connected",
        "valence": {
            "active": True,
            "displacement_l2": 0.2,
            "max_excess_before": 1.0,
            "max_excess_after": 0.0,
        },
        "connectivity": {
            "bridges": [[0, 2], [1, 3]],
            "bridges_surviving": 1,
            "bridges_lost": 1,
        },
    }
    run = write_run(
        tmp_path / "bridges",
        [(soften(atoms), soften(edges))] * 2,
        receipts=[receipt, receipt],
    )
    found = verdicts(run)
    assert found["bridge_survival"]["verdict"] == "flag"
    assert found["bridge_survival"]["bridges_lost"] == 2


def test_score_metric_comes_from_the_run_config(tmp_path):
    atoms, edges = chain(4)
    states = [(soften(atoms), soften(edges))] * 2
    boltz = write_run(tmp_path / "boltz", states, metric="affinity_pred_value")
    protenix = write_run(tmp_path / "protenix", states, metric="contact_score")
    assert score_metric(boltz) == "affinity_pred_value"
    assert score_metric(protenix) == "contact_score"
    assert per_step(boltz)[0]["score"] is not None, "the aux field is found"


def test_exploitation_direction_is_read_from_the_loss_not_the_score(tmp_path):
    """The two backends improve in opposite score directions.

    A Protenix contact score goes up when the objective improves; a Boltz-2
    affinity prediction goes down. Reading the trend off the minimized loss is
    the only way one check works for both.
    """
    atoms = atoms_of(["C"] * 5)
    broken = edges_of(5, {(0, 1): "SINGLE", (2, 3): "SINGLE"})  # 3 components

    # Boltz: affinity falling means improving, so this is exploitation.
    improving = write_run(
        tmp_path / "boltz-improving",
        [(atoms, broken)] * 3,
        scores=[-5.0, -6.0, -7.5],
        metric="affinity_pred_value",
    )
    found = verdicts(improving)
    assert found["predictor_exploitation"]["verdict"] == "flag"
    assert found["predictor_exploitation"]["final_score"] == -7.5

    # Same chemistry, affinity rising: the objective got worse, so not this.
    worsening = write_run(
        tmp_path / "boltz-worsening",
        [(atoms, broken)] * 3,
        scores=[-7.5, -6.0, -5.0],
        metric="affinity_pred_value",
    )
    assert verdicts(worsening)["predictor_exploitation"]["verdict"] == "ok"


def test_missing_artifacts_are_not_available_rather_than_passing(tmp_path):
    atoms, edges = chain(4)
    run = write_run(tmp_path / "bare", [(soften(atoms), soften(edges))] * 2)
    found = verdicts(run)
    assert found["soft_hard_gap"]["verdict"] == "n/a"
    assert found["projection_aggressiveness"]["verdict"] == "n/a"
    assert found["bridge_survival"]["verdict"] == "n/a"


# --------------------------------------------------------------------------- #
# the experiment file
# --------------------------------------------------------------------------- #


def test_every_schema_destination_is_a_real_run_option():
    """A setting that maps to no flag would silently do nothing."""
    import argparse

    import run as run_module
    from settings import SCHEMA

    options = vars(
        run_module.add_arguments(argparse.ArgumentParser()).parse_args(
            ["--out", "/tmp/unused"]
        )
    )
    missing = sorted(set(SCHEMA.values()) - set(options))
    assert not missing, f"schema maps to unknown options: {missing}"


def test_checked_in_experiment_file_loads_and_is_the_protocol_baseline():
    from settings import load

    values = load(EXPERIMENT_YAML)
    assert values["backend"] == "boltz2"
    assert values["mode"] == "valence"
    assert values["hardening"] is False
    assert values["hardening_gradient"] == "inherited"
    # Every improvement neutral, so the file is protocol 5.2 with nothing added.
    assert (
        Stage2Config(
            mode=values["mode"],
            aromatic_order=values["aromatic_order"],
            readout_caps=values["readout_caps"],
            valence_damping=values["valence_damping"],
            fluidity_threshold=values["fluidity_threshold"],
            fluidity_strength=values["fluidity_strength"],
            bridge_retry=values["bridge_retry"],
        ).improvements
        == ()
    )


def test_experiment_file_rejects_an_unknown_setting(tmp_path):
    from settings import load

    path = tmp_path / "typo.yaml"
    path.write_text("operators:\n  aromatic_ordr: 1.0\n")
    with pytest.raises(ValueError, match="aromatic_ordr"):
        load(path)


def test_experiment_file_paths_arrive_as_paths(tmp_path):
    from settings import load

    path = tmp_path / "paths.yaml"
    path.write_text("boltz2:\n  context: contexts/brd4-25\n  checkpoint: null\n")
    values = load(path)
    assert isinstance(values["boltz_context"], Path)
    assert values["boltz_checkpoint"] is None, "null stays null, not Path('None')"


def test_command_line_flags_beat_the_experiment_file(tmp_path):
    import argparse

    import run as run_module
    from settings import apply

    path = tmp_path / "e.yaml"
    path.write_text("operators:\n  mode: sparse\ntrajectory:\n  steps: 7\n")
    parser = run_module.add_arguments(argparse.ArgumentParser())
    args = apply(
        parser,
        ["--config", str(path), "--out", str(tmp_path / "o"), "--mode", "connected"],
    )
    assert args.mode == "connected", "explicit flag wins"
    assert args.steps == 7, "file still supplies what the flag did not"


# --------------------------------------------------------------------------- #
# hardening gradient compensation
# --------------------------------------------------------------------------- #


def test_attenuation_matches_the_documented_factor():
    from run import attenuation

    assert attenuation(0.0, 6) == pytest.approx(1.0), "no hardening, no shrinkage"
    assert attenuation(1.0, 6) == pytest.approx(1 / 6)
    assert attenuation(0.5, 6) == pytest.approx(0.5 + 0.5 / 6)
    assert attenuation(1.0, 5) == pytest.approx(0.2), "bonds have five categories"


def test_attenuation_is_monotone_and_never_amplifies():
    from run import attenuation

    values = [attenuation(h, 6) for h in np.linspace(0, 1, 11)]
    assert all(a >= b for a, b in pairwise(values)), "monotone decreasing"
    assert max(values) <= 1.0 and min(values) > 0.0


def test_rescaled_compensation_cancels_the_attenuation():
    """The point of the knob: attenuation times its correction is one."""
    from run import attenuation

    for categories in (5, 6):
        for hardness in (0.0, 0.25, 0.5, 0.75, 1.0):
            factor = attenuation(hardness, categories)
            assert factor * (1.0 / factor) == pytest.approx(1.0)


def test_hardening_gradient_modes_are_closed():
    from run import HARDENING_GRADIENTS

    assert HARDENING_GRADIENTS == ("inherited", "passthrough", "rescaled")


def test_missing_boltz_settings_fail_before_the_heavy_imports(tmp_path):
    """The message must be useful on a laptop, not a JAX import error."""
    import argparse

    import run as run_module
    from settings import apply

    parser = run_module.add_arguments(argparse.ArgumentParser())
    args = apply(parser, ["--config", str(EXPERIMENT_YAML), "--out", str(tmp_path)])
    assert args.backend == "boltz2" and args.boltz_context is None
    with pytest.raises(SystemExit, match="prepared context"):
        run_module.check_settings(args)


def test_check_settings_returns_the_validated_operator_config(tmp_path):
    import argparse

    import run as run_module
    from settings import apply

    parser = run_module.add_arguments(argparse.ArgumentParser())
    args = apply(
        parser,
        [
            "--backend",
            "protenix",
            "--out",
            str(tmp_path),
            "--mode",
            "connected",
            "--readout-caps",
        ],
    )
    config = run_module.check_settings(args)
    assert config.mode == "connected"
    assert config.improvements == ("readout_caps",)


def test_sparse_only_shrinks_without_bridging(tmp_path):
    """5.4 without 5.3, which the protocol's ladder cannot express."""
    atoms = atoms_of(["C"] * 6)
    edges = soften(
        edges_of(6, {(0, 1): "SINGLE", (1, 2): "SINGLE", (3, 4): "SINGLE"}), 0.2
    )
    _, sparse_only, receipt = apply_operators(
        atoms, edges, Stage2Config(mode="sparse_only")
    )
    assert "sparsity" in receipt, "shrinkage ran"
    assert "connectivity" not in receipt, "bridging did not"
    assert receipt["components_after"] > 1, "fragments are left alone"

    _, both, full = apply_operators(atoms, edges, Stage2Config(mode="sparse"))
    assert "connectivity" in full and "sparsity" in full
    assert full["components_after"] < receipt["components_after"]
    check_invariants(atoms, sparse_only)
    check_invariants(atoms, both)


def test_mode_ladder_covers_every_requested_arm():
    from operators import BRIDGING_MODES, MODES, SHRINKING_MODES

    assert MODES == ("none", "valence", "connected", "sparse_only", "sparse")
    # 5.2 alone, +5.3, +5.4, and both.
    assert "connected" in BRIDGING_MODES and "sparse" in BRIDGING_MODES
    assert "sparse_only" in SHRINKING_MODES and "sparse" in SHRINKING_MODES
    assert "sparse_only" not in BRIDGING_MODES
    assert "connected" not in SHRINKING_MODES


def test_prepared_seed_is_rebuilt_without_torch():
    """The JAX side has no torch, so the seed comes from preparation.json."""
    from backends import seed_arrays

    atoms, edges = seed_arrays(
        {"n_atoms": 5, "seed_smiles": "CCCCC", "atom_indices": [0, 1, 2, 3, 4]}
    )
    check_invariants(atoms, edges)
    assert (atoms.argmax(-1) == ELEMENTS.index("C")).all(), "a carbon chain"
    i, j = pair_indices(5)
    chain = j == i + 1
    assert (edges[chain].argmax(-1) == BOND_STATES.index("SINGLE")).all()
    assert (edges[~chain].argmax(-1) == BOND_STATES.index("NO_BOND")).all()
    assert len(np.unique(components(5, edges))) == 1, "one connected chain"


def test_prepared_seed_rejects_what_it_cannot_reconstruct():
    """A changed preparation must fail loudly, not mismatch the fixed pose."""
    from backends import seed_arrays

    with pytest.raises(ValueError, match="carbon chain"):
        seed_arrays({"n_atoms": 4, "seed_smiles": "CCCN", "atom_indices": [0, 1, 2, 3]})
    with pytest.raises(ValueError, match="atom slot count"):
        seed_arrays({"n_atoms": 4, "seed_smiles": "CCCC", "atom_indices": [0, 1, 2]})


def test_callback_platform_adds_cpu_without_dropping_cuda(monkeypatch):
    """JAX_PLATFORMS=cuda alone makes every Boltz evaluation fail."""
    from run import require_callback_platform

    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    assert require_callback_platform() == "cuda,cpu"

    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    assert require_callback_platform() == "cuda,cpu", "idempotent"

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    assert require_callback_platform() is None, "unset is left to JAX's own default"


def test_study_redoes_an_unfinished_arm_and_skips_a_finished_one(tmp_path):
    """An interrupted arm must be redone, not crash the rest of the study."""
    finished = tmp_path / "done"
    finished.mkdir()
    (finished / "run_summary.json").write_text("{}")
    partial = tmp_path / "killed"
    (partial / "trajectory").mkdir(parents=True)

    def classify(directory):
        if (directory / "run_summary.json").exists():
            return "skip"
        return "redo" if directory.exists() else "run"

    assert classify(finished) == "skip"
    assert classify(partial) == "redo"
    assert classify(tmp_path / "fresh") == "run"
