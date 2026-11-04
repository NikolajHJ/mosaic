"""Read a finished Stage 2 run and check the pitfalls the plans flag.

Model-free and GPU-free: everything here comes from artifacts the run already
wrote. NumPy only, so it works in the lightweight report environment and in the
tests. It never loads Protenix, never evaluates a graph and never repairs one.

Two things are produced. ``per_step`` is the trajectory as a flat table, and
``check_pitfalls`` turns that table into one verdict per documented pitfall,
each with the number behind it. Sources are the working protocol (sections 4.3
C and 5.2-5.4) and the submitted project plan's evaluation requirements.

A verdict is evidence about one recorded trajectory. ``flag`` means the failure
mode the plans told us to watch for is visibly present, which is information,
not a defect in the run.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from operators import (
    NEUTRAL_VALENCE,
    Stage2Config,
    argmax_valence_violations,
    components,
    soft_valence,
    valence_ceiling,
)

COLLAPSE_PROBABILITY = 0.99
REVERSAL_LEVEL = 0.5
DENSE_DEGREE = 4.0


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_run(run_dir):
    """Collect one run's artifacts. Raises if it has no committed trace."""
    run_dir = Path(run_dir)
    trajectory = run_dir / "trajectory"
    trace_path = trajectory / "trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"{run_dir} has no trajectory/trace.jsonl")
    optional = {}
    for name in ("stage2_config.json", "run_summary.json", "soft_hard_gap.json"):
        path = run_dir / name
        if path.exists():
            optional[name.removesuffix(".json")] = json.loads(path.read_text())
    return {
        "directory": str(run_dir),
        "name": run_dir.name,
        "trajectory": trajectory,
        "trace": _read_jsonl(trace_path),
        **optional,
    }


def score_metric(run):
    """Which aux field holds this backend's score.

    Recorded by ``run.py`` in ``stage2_config.json``. For a run made before
    backends existed, fall back to whichever field the trace actually carries,
    so old Protenix runs stay analysable.
    """
    recorded = (run.get("stage2_config") or {}).get("score_metric")
    if recorded:
        return recorded
    available = (run["trace"][0].get("aux_metrics") or {}) if run["trace"] else {}
    for candidate in ("affinity_pred_value", "contact_score"):
        if candidate in available:
            return candidate
    return "contact_score"


def per_step(run):
    """Flatten one run into a row per evaluated state.

    Most columns come straight from the trace. ``components`` and the discrete
    valence violations are recomputed from the saved readout so a run made
    before these diagnostics existed is still fully analysable.
    """
    trajectory = Path(run["trajectory"])
    metric = score_metric(run)
    rows = []
    for entry in run["trace"]:
        graph = entry.get("graph") or {}
        row = {
            "step": entry["evaluated_step"],
            "final": bool(entry.get("final")),
            "loss": entry.get("loss"),
            "score": entry.get("aux_metrics", {}).get(metric),
            "atom_entropy": graph.get("atom_entropy"),
            "bond_entropy": graph.get("bond_entropy"),
            "atom_max_probability": graph.get("atom_max_probability"),
            "bond_max_probability": graph.get("bond_max_probability"),
            "bond_mass": graph.get("bond_mass"),
            "expected_degree": graph.get("expected_degree"),
            "gradient_norm_X": entry.get("raw_gradient_norm_X"),
            "gradient_norm_E": entry.get("raw_gradient_norm_E_upper"),
        }
        readout_name = entry.get("readout")
        if readout_name:
            readout = json.loads((trajectory / readout_name).read_text())
            elements = np.asarray(readout["element_indices"])
            bonds = np.asarray(readout["bond_upper_indices"])
            n = len(elements)
            hard_edges = np.zeros((len(bonds), 5))
            hard_edges[np.arange(len(bonds)), bonds] = 1.0
            hard_atoms = np.zeros((n, len(NEUTRAL_VALENCE)))
            hard_atoms[np.arange(n), elements] = 1.0
            violations, worst = argmax_valence_violations(hard_atoms, hard_edges)
            row.update(
                components=len(np.unique(components(n, hard_edges))),
                argmax_valence_violations=violations,
                argmax_valence_max_excess=worst,
                sanitized=bool(readout.get("sanitized")),
                fragment_count=readout.get("fragment_count"),
                canonical_smiles=readout.get("canonical_smiles"),
            )
        operators = entry.get("post_step_operators") or {}
        valence = operators.get("valence") or {}
        connectivity = operators.get("connectivity") or {}
        sparsity = operators.get("sparsity") or {}
        fluidity = operators.get("fluidity_floor") or {}
        row.update(
            valence_active=bool(valence.get("active")),
            valence_displacement=valence.get("displacement_l2"),
            valence_excess_before=valence.get("max_excess_before"),
            valence_excess_after=valence.get("max_excess_after"),
            bridges=len(connectivity.get("bridges", [])),
            bridges_surviving=connectivity.get("bridges_surviving"),
            bridges_lost=connectivity.get("bridges_lost"),
            removed_bond_mass=sparsity.get("removed_bond_mass"),
            fluidity_rows_relaxed=(fluidity.get("atoms") or {}).get("rows_relaxed"),
        )
        rows.append(row)
    return rows


def soft_feasible_steps(run, config=None):
    """Count saved states whose *relaxed* graph satisfies its valence ceiling.

    The plan requires constraint satisfaction in the relaxed graph to be
    assessed separately from validity of the discrete readout, so this reads
    the saved X/E arrays rather than the readout.
    """
    config = config or Stage2Config()
    trajectory = Path(run["trajectory"])
    feasible = total = 0
    for entry in run["trace"]:
        name = entry.get("evaluated_arrays")
        if not name or not (trajectory / name).exists():
            continue
        with np.load(trajectory / name) as handle:
            atoms, edges = handle["X"], handle["E_upper"]
        caps, _ = valence_ceiling(atoms, readout_caps=config.readout_caps)
        valence = soft_valence(edges, len(atoms), aromatic_order=config.aromatic_order)
        total += 1
        feasible += int(np.max(valence - caps) <= 2e-6)
    return feasible, total


def topology_reversals(run):
    """Count edges that crossed the existence midpoint and then crossed back.

    Protocol 4.3 C.11 asks whether the optimizer can change its mind: whether
    strong early edges can later weaken and weak ones become dominant. An edge
    whose existence rises above 0.5 and later falls below it (or the reverse)
    is direct evidence of that, so zero reversals across a whole trajectory is
    the signal that topology has become irreversible.
    """
    trajectory = Path(run["trajectory"])
    series = []
    for entry in run["trace"]:
        name = entry.get("evaluated_arrays")
        if not name or not (trajectory / name).exists():
            continue
        with np.load(trajectory / name) as handle:
            series.append(1.0 - np.asarray(handle["E_upper"])[:, 0])
    if len(series) < 2:
        return {
            "states": len(series),
            "edges": 0,
            "rose_then_fell": 0,
            "fell_then_rose": 0,
            "reversals": 0,
        }
    existence = np.stack(series)
    present = existence > REVERSAL_LEVEL
    rose_then_fell = fell_then_rose = 0
    for column in present.T:
        changes = np.flatnonzero(np.diff(column.astype(np.int8)))
        if len(changes) < 2:
            continue
        directions = np.diff(column.astype(np.int8))[changes]
        rose_then_fell += int(np.any((directions[:-1] > 0) & (directions[1:] < 0)))
        fell_then_rose += int(np.any((directions[:-1] < 0) & (directions[1:] > 0)))
    return {
        "states": len(series),
        "edges": existence.shape[1],
        "rose_then_fell": rose_then_fell,
        "fell_then_rose": fell_then_rose,
        "reversals": rose_then_fell + fell_then_rose,
    }


def _first(rows, predicate):
    for row in rows:
        if predicate(row):
            return row["step"]
    return None


def check_pitfalls(run, rows=None, reversals=None):
    """One verdict per documented pitfall, each carrying the number behind it."""
    rows = rows if rows is not None else per_step(run)
    reversals = reversals if reversals is not None else topology_reversals(run)
    findings = []

    def add(ident, source, question, verdict, detail, **values):
        findings.append(
            {
                "id": ident,
                "source": source,
                "question": question,
                "verdict": verdict,
                "detail": detail,
                **values,
            }
        )

    def column(name):
        return [r[name] for r in rows if r.get(name) is not None]

    steps = len(rows)
    last = rows[-1] if rows else {}

    # 1. premature collapse of the relaxed state
    maxima = column("atom_max_probability")
    collapse = _first(
        rows,
        lambda r: (r.get("atom_max_probability") or 0) > COLLAPSE_PROBABILITY,
    )
    if not maxima:
        add(
            "fluidity",
            "protocol 4.3 C.10",
            "Do probabilities stay movable?",
            "n/a",
            "no atom probability recorded",
        )
    elif collapse is None:
        add(
            "fluidity",
            "protocol 4.3 C.10",
            "Do probabilities stay movable?",
            "ok",
            f"mean max atom probability peaked at {max(maxima):.3f}",
            peak_atom_probability=float(max(maxima)),
        )
    else:
        add(
            "fluidity",
            "protocol 4.3 C.10",
            "Do probabilities stay movable?",
            "flag" if collapse < steps / 2 else "ok",
            f"atom rows reached {COLLAPSE_PROBABILITY} at step {collapse} of {steps}",
            collapse_step=collapse,
        )

    # 2. irreversible topology
    if reversals["states"] < 2:
        add(
            "topology",
            "protocol 4.3 C.11",
            "Can the optimizer change its mind?",
            "n/a",
            "fewer than two saved states",
        )
    else:
        add(
            "topology",
            "protocol 4.3 C.11",
            "Can the optimizer change its mind?",
            "ok" if reversals["reversals"] else "flag",
            f"{reversals['rose_then_fell']} edges rose then fell, "
            f"{reversals['fell_then_rose']} fell then rose, "
            f"over {reversals['edges']} pairs",
            **{
                k: reversals[k]
                for k in ("reversals", "rose_then_fell", "fell_then_rose")
            },
        )

    # 3. discrete valence violations
    violations = column("argmax_valence_violations")
    if not violations:
        add(
            "valence",
            "protocol 4.3 C.12 / 5.2",
            "Is the readout over-valent?",
            "n/a",
            "no readout recorded",
        )
    else:
        affected = sum(1 for v in violations if v)
        add(
            "valence",
            "protocol 4.3 C.12 / 5.2",
            "Is the readout over-valent?",
            "ok" if not affected else "flag",
            f"{affected} of {len(violations)} readouts over-valent; "
            f"final state has {last.get('argmax_valence_violations')}",
            states_over_valent=affected,
            final_violations=last.get("argmax_valence_violations"),
        )

    # 4. fragmentation
    counts = column("components")
    if not counts:
        add(
            "connectivity",
            "protocol 4.3 C.13 / 5.3",
            "Is the graph fragmented?",
            "n/a",
            "no readout recorded",
        )
    else:
        connected = sum(1 for c in counts if c == 1)
        add(
            "connectivity",
            "protocol 4.3 C.13 / 5.3",
            "Is the graph fragmented?",
            "ok" if connected == len(counts) else "flag",
            f"{connected} of {len(counts)} readouts single-component; "
            f"final has {last.get('components')} components",
            single_component_states=connected,
            final_components=last.get("components"),
        )

    # 5. graph density
    degrees = column("expected_degree")
    if not degrees:
        add(
            "sparsity",
            "protocol 4.3 C.14 / 5.4",
            "Is edge mass diffuse?",
            "n/a",
            "no expected degree recorded",
        )
    else:
        add(
            "sparsity",
            "protocol 4.3 C.14 / 5.4",
            "Is edge mass diffuse?",
            "ok" if degrees[-1] <= DENSE_DEGREE else "flag",
            f"expected degree moved {degrees[0]:.2f} -> {degrees[-1]:.2f}, "
            f"threshold {DENSE_DEGREE}",
            initial_expected_degree=degrees[0],
            final_expected_degree=degrees[-1],
        )

    # 6. relaxed-to-discrete score gap
    gap = run.get("soft_hard_gap")
    if not gap:
        add(
            "soft_hard_gap",
            "protocol 4.3 C.15 / plan",
            "Does the score survive discretization?",
            "n/a",
            "soft_hard_gap.json absent; run did not complete its gap pass",
        )
    else:
        # Measured in loss space, where lower is better on either backend, so a
        # positive mean means discretization cost the objective something.
        values = [
            g["hard_minus_soft_loss"]
            for g in gap
            if g.get("hard_minus_soft_loss") is not None
        ]
        add(
            "soft_hard_gap",
            "protocol 4.3 C.15 / plan",
            "Does the objective survive discretization?",
            "ok" if values else "n/a",
            f"mean hard-minus-soft loss {np.mean(values):+.4f} over "
            f"{len(values)} checkpoints"
            if values
            else "gap file carries no loss-space comparison",
            mean_hard_minus_soft_loss=float(np.mean(values)) if values else None,
            final_hard_minus_soft_loss=values[-1] if values else None,
        )

    # 7. projection aggressiveness
    active = [r for r in rows if r.get("valence_active")]
    displacements = column("valence_displacement")
    if not displacements:
        add(
            "projection_aggressiveness",
            "protocol 5.2",
            "Does valence projection freeze topology?",
            "n/a",
            "no valence receipts; this arm has no operators",
        )
    else:
        always_on = len(active) == len([r for r in rows if "valence_active" in r])
        frozen = always_on and reversals["states"] >= 2 and not reversals["reversals"]
        add(
            "projection_aggressiveness",
            "protocol 5.2",
            "Does valence projection freeze topology?",
            "flag" if frozen else "ok",
            f"projection active on {len(active)} of {steps} steps, "
            f"mean displacement {np.mean(displacements):.4f}",
            active_steps=len(active),
            mean_displacement=float(np.mean(displacements)),
        )

    # 8. bridge survival
    attempted = sum(r.get("bridges") or 0 for r in rows)
    if not attempted:
        add(
            "bridge_survival",
            "protocol 5.3",
            "Do bridges survive reprojection?",
            "n/a",
            "no bridges attempted",
        )
    else:
        lost = sum(r.get("bridges_lost") or 0 for r in rows)
        add(
            "bridge_survival",
            "protocol 5.3",
            "Do bridges survive reprojection?",
            "ok" if lost == 0 else "flag",
            f"{attempted - lost} of {attempted} bridges survived valence reprojection",
            bridges_attempted=attempted,
            bridges_lost=lost,
        )

    # 9. relaxed feasibility against discrete validity
    feasible, total = soft_feasible_steps(run)
    sanitized = [r for r in rows if r.get("sanitized")]
    if not total:
        add(
            "soft_versus_hard_feasibility",
            "project plan",
            "Is relaxed satisfaction mistaken for molecular validity?",
            "n/a",
            "no saved X/E arrays",
        )
    else:
        mismatch = feasible == total and not sanitized
        add(
            "soft_versus_hard_feasibility",
            "project plan",
            "Is relaxed satisfaction mistaken for molecular validity?",
            "flag" if mismatch else "ok",
            f"{feasible} of {total} relaxed states satisfy valence; "
            f"{len(sanitized)} of {len(rows)} readouts sanitize",
            relaxed_feasible=feasible,
            relaxed_states=total,
            sanitized_states=len(sanitized),
        )

    # 10. predictor exploitation
    # Judged on loss, not score: the loss is minimized on either backend, while
    # "higher is better" holds for Protenix contact score and not for a Boltz-2
    # affinity prediction. Reading the direction off the loss is backend-proof.
    losses = column("loss")
    scores = column("score")
    if len(losses) < 2 or not counts:
        add(
            "predictor_exploitation",
            "project plan",
            "Does the objective improve while chemistry does not?",
            "n/a",
            "need a loss series and readouts",
        )
    else:
        improved = losses[-1] < losses[0]
        valid_end = bool(last.get("sanitized")) and last.get("components") == 1
        trend = f"loss {losses[0]:.5f} -> {losses[-1]:.5f}"
        if len(scores) >= 2:
            trend += f" ({score_metric(run)} {scores[0]:.5f} -> {scores[-1]:.5f})"
        add(
            "predictor_exploitation",
            "project plan",
            "Does the objective improve while chemistry does not?",
            "flag" if improved and not valid_end else "ok",
            f"{trend}; final readout "
            f"{'valid and connected' if valid_end else 'not valid and connected'}",
            initial_loss=losses[0],
            final_loss=losses[-1],
            final_score=scores[-1] if scores else None,
            final_valid=valid_end,
        )

    return findings


def summarize(run_dir):
    """Everything the report and the CLI need for one run."""
    run = load_run(run_dir)
    rows = per_step(run)
    reversals = topology_reversals(run)
    return {
        "run": run["name"],
        "directory": run["directory"],
        "config": run.get("stage2_config"),
        "states": len(rows),
        "reversals": reversals,
        "per_step": rows,
        "pitfalls": check_pitfalls(run, rows, reversals),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list({key: None for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_pitfalls(summary):
    marks = {"ok": "ok  ", "flag": "FLAG", "n/a": "--  "}
    print(f"\n{summary['run']}  ({summary['states']} saved states)")
    print("-" * 78)
    for finding in summary["pitfalls"]:
        print(f"{marks[finding['verdict']]}  {finding['id']:<28}  {finding['detail']}")
    flagged = [f["id"] for f in summary["pitfalls"] if f["verdict"] == "flag"]
    print("-" * 78)
    print(f"{len(flagged)} flagged: {', '.join(flagged) if flagged else 'none'}")
    print(f"sources: {', '.join(sorted({f['source'] for f in summary['pitfalls']}))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+", help="Stage 2 run directories")
    parser.add_argument("--json", type=Path, help="write the full summaries here")
    parser.add_argument("--csv", type=Path, help="write the per-step table here")
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.run]
    for summary in summaries:
        print_pitfalls(summary)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(
            args.csv,
            [{"run": s["run"], **row} for s in summaries for row in s["per_step"]],
        )
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
