"""Portable static HTML report and scientific SVG/PNG plots."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from .reporting import (
    ELEMENTS,
    analyse_run,
    load_runs,
    molecule_assets,
    read_json,
    readout_graph,
    sample_state,
    sha256,
)

ASSETS = Path(__file__).with_name("report_assets")
REPO = Path(__file__).resolve().parents[3]


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protein_assets(run, structure=None):
    """Reference coordinates only; the optimization does not produce a pose."""
    target = run.target
    if not target:
        return {"status": "No target reference metadata was saved"}
    if structure is None:
        structure = (
            REPO / "examples/data/stage1_targets/raw" / f"{target.get('pdb_id')}.cif"
        )
    structure = Path(structure)
    if not structure.exists():
        return {
            "status": "Reference coordinates unavailable; supply --protein-structure",
            "target": target,
        }
    import gemmi

    st = gemmi.read_structure(str(structure))
    while len(st) > 1:
        del st[-1]
    pocket = target.get("pocket", {})
    residues = pocket.get("residues", [])
    return {
        "status": "experimental reference",
        "pdb": st.make_pdb_string(),
        "source": str(structure),
        "sha256": sha256(structure),
        "target_name": target.get("name"),
        "pdb_id": target.get("pdb_id"),
        "pocket": pocket,
        "selections": [
            {"chain": r["auth_chain"], "resi": int(r["auth_seq_id"])}
            for r in residues
            if str(r["auth_seq_id"]).lstrip("-").isdigit()
        ],
        "reference_ligand": target.get("ligand"),
        "label": "Crystal reference and ligand-defined pocket; this is not the optimized ligand pose",
    }


def write_element_history(path, states):
    """Export every saved X row without rounding or changing its slot order."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "atom_slot", *ELEMENTS])
        for state in states:
            for slot, probabilities in enumerate(state.X, start=1):
                writer.writerow([state.step, slot, *probabilities.tolist()])


def argmax_trajectory(rows, states=None):
    """Compact, complete hard trajectory with cached 2D identity depictions."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    if states is not None and len(states) != len(rows):
        raise ValueError("Probability history must match every argmax evaluation")
    frames, drawings, errors = [], {}, {}
    for index, row in enumerate(rows):
        state = states[index] if states is not None else None
        if state is not None and (
            state.step != row["step"]
            or state.X.argmax(-1).tolist() != row["hard_readout"]["element_indices"]
        ):
            raise ValueError(
                "Probability history disagrees with argmax step or atom slots"
            )
        readout = row["hard_readout"]
        identity = readout["canonical_smiles"]
        if readout["sanitized"] and identity and identity not in drawings:
            try:
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(identity)
                    rdDepictor.Compute2DCoords(mol)
                    drawer = rdMolDraw2D.MolDraw2DSVG(700, 380)
                    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                    drawer.FinishDrawing()
                    svg = drawer.GetDrawingText()
                    drawings[identity] = svg[svg.index("<svg") :]
            except Exception as error:  # noqa: BLE001 -- retain hard graph on depiction failure
                drawings[identity] = None
                errors[identity] = f"{type(error).__name__}: {error}"
        frames.append(
            {
                "step": row["step"],
                "loss": row["loss"],
                "contact_score": row.get("metrics", {}).get("contact_score"),
                "readout": readout,
                "element_probabilities": state.X.tolist()
                if state is not None
                else None,
            }
        )
    return {
        "frames": frames,
        "element_order": list(ELEMENTS),
        "drawings": drawings,
        "depiction_errors": errors,
        "policy": "Every saved evaluation, in recorded order; independent atom/bond argmax; no graph repair or interpolation. Molecular drawings use canonical identity; fixed-slot diagrams preserve input slot numbering.",
    }


def _plot_series(ax, x, y, label, **kwargs):
    pairs = [
        (a, b) for a, b in zip(x, y, strict=True) if b is not None and np.isfinite(b)
    ]
    if pairs:
        xx, yy = zip(*pairs, strict=True)
        ax.plot(xx, yy, label=label, **kwargs)
        return True
    return False


def plots(analysis, samples, output, external):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "axes.prop_cycle": plt.cycler(
                color=["#087f8c", "#d77835", "#7257b4", "#a77800"]
            ),
        }
    )
    rows = analysis["rows"]
    x = [r["step"] for r in rows]
    outputs = []

    def finish(fig, name, caption):
        fig.tight_layout()
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        outputs.append(
            {
                "file": f"{name}.svg",
                "title": name.replace("_", " ").title(),
                "caption": caption,
            }
        )

    def legend(ax):
        ax.set_xlabel("Optimization update")
        ax.grid(alpha=0.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    _plot_series(
        axes[0], x, [r["loss"] for r in rows], "recorded optimization objective"
    )
    axes[0].set_ylabel("Objective (lower is better)")
    contact = [r["metrics"].get("contact_score") for r in rows]
    _plot_series(axes[1], x, contact, "optimization-forward contact score")
    gaps = analysis["source"]["metadata"].get("soft_hard_gap", [])
    if gaps:
        _plot_series(
            axes[1],
            [r["step"] for r in gaps],
            [r.get("soft_score") for r in gaps],
            "underlying relaxed contact score",
        )
        _plot_series(
            axes[1],
            [r["step"] for r in gaps],
            [r["hard_score"] for r in gaps],
            "raw argmax contact score",
        )
    axes[1].set_ylabel("Contact probability score (higher is better)")
    if any(v is not None for v in contact):
        axes[1].set_ylim(0, 1)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Contact score not recorded",
            ha="center",
            transform=axes[1].transAxes,
        )
    for ax in axes:
        legend(ax)
    finish(
        fig,
        "objective",
        "Contact score is mean best-target contact probability, not affinity or accuracy. Compare matched conditions; a small increase has no universal biological interpretation. Full 0–1 scale prevents magnifying tiny gains. With hardening, the optimization-forward graph differs from the underlying relaxed graph. A teacher/prior term changes the joint objective; its schedule can make loss values non-comparable across updates.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for key, label in (
        ("atom_step_tv", "atom movement"),
        ("bond_step_tv", "bond movement"),
    ):
        _plot_series(axes[0], x, [r[key] for r in rows], label)
    axes[0].set_ylabel("Mean consecutive-row TV [0,1]")
    windows = sorted({w for row in rows for w in row["windows"]}, key=int)
    for w in windows:
        _plot_series(
            axes[1],
            x,
            [r["windows"].get(w, {}).get("diameter") for r in rows],
            f"window {w} saved states apart",
        )
    axes[1].set_ylabel("Recent window diameter [0,1]")
    for ax in axes:
        legend(ax)
    finish(
        fig,
        "movement",
        "Small steps alone do not establish convergence. Window diameter measures the largest separation within a recent window; use several windows and inspect the update span when records are sparse.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for w in windows:
        _plot_series(
            axes[0],
            x,
            [r["windows"].get(w, {}).get("net_path_ratio") for r in rows],
            f"window {w}",
        )
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Net displacement / path length")
    legend(axes[0])
    msd = analysis["mean_squared_displacement"]
    if msd:
        axes[1].plot(
            [r["mean_update_span"] for r in msd],
            [r["mean_squared_displacement"] for r in msd],
            marker=".",
        )
    axes[1].set(xlabel="Mean lag in updates", ylabel="Euclidean squared displacement")
    axes[1].grid(alpha=0.18)
    finish(
        fig,
        "drift",
        "A high net/path ratio indicates persistent movement; a low ratio indicates cancellation or turns. MSD is descriptive: linear/quadratic lag scaling requires a justified random/directional null model, and bounded spaces eventually saturate. No random-walk classification is inferred.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for key, label in (
        ("atom_entropy", "atom entropy"),
        ("existence_entropy", "bond-existence entropy"),
        ("bond_order_entropy", "conditional bond-order entropy"),
    ):
        _plot_series(axes[0], x, [r[key] for r in rows], label)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Normalized categorical entropy [0,1]")
    for key, label in (
        ("projected_gradient_X_rms", "atoms"),
        ("projected_gradient_E_rms", "bonds"),
    ):
        _plot_series(axes[1], x, [r[key] for r in rows], label)
    axes[1].set_ylabel("Projected gradient RMS per row")
    if all(r["projected_gradient_X_rms"] is None for r in rows):
        axes[1].text(
            0.5, 0.5, "Gradients not saved", ha="center", transform=axes[1].transAxes
        )
    for ax in axes:
        legend(ax)
    finish(
        fig,
        "uncertainty_stationarity",
        "Entropy 0 means definite choices; 1 means uniform. This is representation ambiguity, not model confidence. Projected gradient uses a fixed reference step and respects simplex constraints; near zero indicates first-order stationarity, not global optimality.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    heat = axes[0].imshow(
        analysis["recurrence"],
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="auto",
    )
    rs = analysis["recurrence_steps"]
    ticks = np.unique(np.linspace(0, len(rs) - 1, min(6, len(rs))).astype(int))
    axes[0].set_xticks(ticks, [rs[t] for t in ticks])
    axes[0].set_yticks(ticks, [rs[t] for t in ticks])
    axes[0].set(
        xlabel="Update", ylabel="Update", title="Relaxed-state recurrence distance"
    )
    fig.colorbar(heat, ax=axes[0], label="TV-based distance [0,1]")
    h = axes[1].imshow(
        np.array([r["atom_entropies"] for r in rows]).T,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="magma",
    )
    ticks = np.unique(np.linspace(0, len(x) - 1, min(6, len(x))).astype(int))
    axes[1].set_xticks(ticks, [x[t] for t in ticks])
    axes[1].set(
        xlabel="Update",
        ylabel="Atom slot (zero based)",
        title="Where ambiguity remains",
    )
    fig.colorbar(h, ax=axes[1], label="Normalized entropy")
    finish(
        fig,
        "recurrence",
        "Dark terminal blocks indicate similar relaxed states. Repeated off-diagonal bands indicate returns. Pairwise distances use fixed atom slots and are not chemical graph-isomorphism distances. Recurrence may be downsampled; sampled update labels are retained.",
    )
    fig, ax = plt.subplots(figsize=(6.5, 5))
    observed_max = max(float(np.max(analysis["recurrence"])), 1e-12)
    zoom = ax.imshow(
        analysis["recurrence"],
        origin="lower",
        vmin=0,
        vmax=observed_max,
        cmap="viridis",
        aspect="auto",
    )
    rticks = np.unique(np.linspace(0, len(rs) - 1, min(6, len(rs))).astype(int))
    ax.set_xticks(rticks, [rs[t] for t in rticks])
    ax.set_yticks(rticks, [rs[t] for t in rticks])
    ax.set(xlabel="Optimization update", ylabel="Optimization update")
    fig.colorbar(
        zoom, ax=ax, label=f"Distance, zoomed color range 0–{observed_max:.3g}"
    )
    finish(
        fig,
        "recurrence_zoom",
        "Zoomed view of the same recurrence matrix. Colors span this run's observed range, explicitly shown on the colorbar; compare numeric distances across reports, not color intensity. The preceding recurrence plot keeps the absolute 0–1 range.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    sx = [s["step"] for s in samples]
    for key, label in (
        ("valid_connected_fraction", "valid and connected"),
        ("disconnected_fraction", "disconnected"),
    ):
        _plot_series(axes[0], sx, [s[key] for s in samples], label, marker=".")
    for key, label in (
        ("molecules", "largest molecule mass"),
        ("families", "largest scaffold mass"),
    ):
        _plot_series(
            axes[1],
            sx,
            [s[key][0]["probability"] if s[key] else 0 for s in samples],
            label,
            marker=".",
        )
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Fraction of ALL draws")
        legend(ax)
    finish(
        fig,
        "sampling",
        "Product-categorical sampling retains all invalid and disconnected draws. The leading molecule/family may change across checkpoints; inspect identities, not just the maximum mass. Empty acyclic Murcko scaffolds form a broad labelled group, not evidence of a narrow chemical family.",
    )

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for key, label in (
        ("molecule_entropy_nats", "molecular identity H(M)"),
        ("family_entropy_nats", "family H(F)"),
        ("within_family_entropy_nats", "within-family H(M|F)"),
    ):
        _plot_series(ax, sx, [s[key] for s in samples], label, marker=".")
    legend(ax)
    ax.set_ylabel("Plug-in entropy (nats)")
    finish(
        fig,
        "family_identity",
        "H(M)=H(F)+H(M|F) for fixed scaffold mapping. Invalid/disconnected outcomes share one OTHER category. These finite-sample entropy estimates are downward biased; zero entropy can mean every draw failed. Always read alongside validity and sample count.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ("molecules", "families"), strict=True):
        totals = {}
        for sample in samples:
            for item in sample[key]:
                totals[item["identity"]] = (
                    totals.get(item["identity"], 0) + item["count"]
                )
        leaders = sorted(totals, key=totals.get, reverse=True)[:4]
        for identity in leaders:
            values = [
                next(
                    (
                        v["probability"]
                        for v in sample[key]
                        if v["identity"] == identity
                    ),
                    0,
                )
                for sample in samples
            ]
            label = identity if len(identity) < 38 else identity[:35] + "..."
            _plot_series(ax, sx, values, label, marker=".")
        if not leaders:
            ax.text(
                0.5,
                0.5,
                "No valid connected identities sampled",
                ha="center",
                transform=ax.transAxes,
            )
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Fraction of ALL draws")
        ax.set_title(
            "Fixed molecule identities"
            if key == "molecules"
            else "Fixed scaffold identities"
        )
        legend(ax)
    finish(
        fig,
        "identity_mass",
        "The same identities are tracked across checkpoints, unlike a maximum-mass curve whose winner can change. Up to four leaders are selected descriptively across discovery batches; absent sampled identities have observed frequency zero, not proven zero probability. No inference is based on this post-selection alone.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    hard_ids = [r["hard_readout"]["identity_key"] for r in rows]
    seen = list(dict.fromkeys(hard_ids))
    axes[0].step(
        x,
        [seen.index(k) for k in hard_ids],
        where="post",
        label="canonical molecule / invalid labelled graph",
    )
    axes[0].set_ylabel("Identity index (categorical, not distance)")
    axes[1].step(
        x,
        [
            int(r["hard_readout"]["sanitized"] and r["hard_readout"]["connected"])
            for r in rows
        ],
        where="post",
        label="valid connected raw argmax",
    )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Validity indicator")
    for ax in axes:
        legend(ax)
    finish(
        fig,
        "discrete_identity",
        "Valid graph identity ignores atom numbering and unoptimized stereochemistry. Invalid graphs retain a labelled assignment hash; their stability is not convergence to a valid molecule. The numeric identity index has no structural distance interpretation.",
    )

    for i, series in enumerate(external):
        fig, ax = plt.subplots(figsize=(8, 3.8))
        _plot_series(
            ax,
            [p["step"] for p in series["values"]],
            [p["value"] for p in series["values"]],
            series["label"],
        )
        ax.set_ylabel(f"{series['label']} ({series['units']})")
        legend(ax)
        finish(
            fig,
            f"external_{i}",
            f"{series['kind']}; source: {series['source']}. Direction: {series['direction']}. {series['interpretation']}",
        )
    return outputs


def load_external(path):
    if path is None:
        return []
    data = read_json(path)
    if data.get("schema") != "mosaic_external_metrics_v1":
        raise ValueError("Unknown external metrics schema")
    required = {
        "run",
        "kind",
        "label",
        "units",
        "direction",
        "source",
        "interpretation",
        "values",
    }
    for series in data["series"]:
        if not required <= series.keys() or series["kind"] not in (
            "affinity",
            "confidence",
            "other",
        ):
            raise ValueError(
                "External metrics need explicit run, kind, units, direction, source, and interpretation"
            )
        for p in series["values"]:
            if not np.isfinite(p["value"]) or not isinstance(p["step"], int):
                raise ValueError(
                    "External metric points require finite values and integer steps"
                )
    return data["series"]


def render_report(
    source,
    output,
    *,
    samples=256,
    checkpoints=9,
    seed=17,
    windows=(5, 20, 50),
    reference_eta=0.01,
    protein_structure=None,
    external_metrics=None,
    max_browser_states=201,
):
    from .report_depiction import depict_graph

    runs = load_runs(source)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            "Report output already exists; choose a new directory to preserve provenance"
        )
    output.mkdir(parents=True)
    static = output / "assets"
    shutil.copytree(ASSETS, static)
    # Archive actual source as well as hashes to preserve this analyzer version.
    implementation = output / "implementation"
    package = implementation / "src/mosaic/small_molecule"
    package.mkdir(parents=True)
    for name in (
        "reporting.py",
        "report_render.py",
        "report_depiction.py",
        "__init__.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), package / name)
    shutil.copyfile(
        Path(__file__).parents[1] / "__init__.py", package.parent / "__init__.py"
    )
    shutil.copytree(ASSETS, package / "report_assets")
    (implementation / "scripts").mkdir()
    for name in (
        "render_small_molecule_report.py",
        "sample_relaxed_molecules.py",
        "requirements-report.txt",
    ):
        shutil.copyfile(REPO / "scripts" / name, implementation / "scripts" / name)
    guide = REPO / "docs/small_molecule_reporting.md"
    if guide.exists():
        shutil.copyfile(guide, output / "metric_guide.md")
        (implementation / "docs").mkdir()
        shutil.copyfile(guide, implementation / "docs/small_molecule_reporting.md")
    if Path(source).is_file() and runs[0].metadata.get("collection"):
        shutil.copyfile(source, output / "collection_source.json")
    ext = load_external(external_metrics)
    unknown = {s["run"] for s in ext} - {r.name for r in runs}
    if unknown:
        raise ValueError(f"External metric run does not match input: {unknown}")
    report = {
        "schema": "mosaic_report_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "runs": [],
        "collection": runs[0].metadata.get("collection", {}),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "matplotlib", "rdkit", "gemmi")
        },
        "analysis_parameters": {
            "samples": samples,
            "checkpoints": checkpoints,
            "seed": seed,
            "windows": list(windows),
            "reference_eta": reference_eta,
            "max_browser_states": max_browser_states,
        },
        "implementation": {
            p.name: sha256(p) for p in Path(__file__).parent.glob("report*.py")
        },
        "assets": {p.name: sha256(p) for p in ASSETS.iterdir() if p.is_file()},
        "external_metrics_sha256": sha256(external_metrics)
        if external_metrics
        else None,
    }
    if (
        samples < 1
        or checkpoints < 1
        or max_browser_states < 2
        or any(w < 1 for w in windows)
    ):
        raise ValueError("Samples, checkpoints, and windows must be positive")
    cache = {}
    for index, run in enumerate(runs):
        directory = output / f"run-{index}"
        directory.mkdir()
        analysis = analyse_run(run, windows=windows, reference_eta=reference_eta)
        browser_states = []
        last_identity = None
        display = set(
            np.unique(
                np.linspace(
                    0, len(run.states) - 1, min(max_browser_states, len(run.states))
                ).astype(int)
            )
        )
        for state_index, state in enumerate(run.states):
            ro = readout_graph(state.X.argmax(-1), state.E.argmax(-1))
            labelled = hashlib.sha256(
                state.X.argmax(-1).tobytes() + state.E.argmax(-1).tobytes()
            ).hexdigest()
            ro["identity_key"] = (
                ro["canonical_smiles"] or "INVALID_LABELLED:" + labelled
            )
            ro["identity_changed"] = (
                last_identity is not None and ro["identity_key"] != last_identity
            )
            last_identity = ro["identity_key"]
            analysis["rows"][state_index]["hard_readout"] = ro
            if state_index not in display:
                continue
            # Invalid hard graphs need labelled identity too; preserve raw assignment.
            state_svg = depict_graph(
                state.X, state.E, state.step, ro["sanitized"], svg=True
            )
            state_svg = state_svg[state_svg.index("<svg") :]
            browser_states.append(
                {
                    "step": state.step,
                    "loss": state.loss,
                    "readout": ro,
                    "svg": state_svg,
                    "uncertainty": analysis["rows"][state_index],
                    "probabilities": {
                        "X": state.X.tolist(),
                        "E_upper": state.E.tolist(),
                    },
                }
            )
        selected = np.unique(
            np.linspace(
                0, len(run.states) - 1, min(checkpoints, len(run.states))
            ).astype(int)
        )
        sample_summaries = []
        for j in selected:
            state = run.states[j]
            sample = sample_state(state, count=samples, seed=seed)
            _json(directory / f"samples-step-{state.step:06d}.json", sample)
            representatives = []
            for draw in sample["draws"][:3]:
                raw_svg = depict_graph(
                    np.eye(6)[draw["element_indices"]],
                    np.eye(5)[draw["bond_upper_indices"]],
                    state.step,
                    draw["sanitized"],
                    svg=True,
                )
                representatives.append(
                    {**draw, "svg": raw_svg[raw_svg.index("<svg") :]}
                )
            sample.pop("draws")
            sample["representative_raw_draws"] = representatives
            candidates = []
            for candidate in sample["molecules"][:8]:
                smiles = candidate["identity"]
                if smiles not in cache:
                    cache[smiles] = molecule_assets(smiles, seed=seed)
                candidates.append({**candidate, **cache[smiles]})
            sample["candidates"] = candidates
            sample["draws_file"] = f"samples-step-{state.step:06d}.json"
            # Independent validation batch for selected top identities/families.
            validation = sample_state(state, count=samples, seed=seed + 1)
            from .reporting import wilson

            for key in ("molecules", "families"):
                if sample[key]:
                    candidate = sample[key][0]["identity"]
                    k = next(
                        (
                            a["count"]
                            for a in validation[key]
                            if a["identity"] == candidate
                        ),
                        0,
                    )
                    sample[key + "_leader_validation"] = {
                        "identity": candidate,
                        "count": k,
                        "sample_count": samples,
                        "seed": seed + 1,
                        "probability": k / samples,
                        "pointwise_wilson_95": wilson(k, samples),
                    }
            _json(directory / f"validation-step-{state.step:06d}.json", validation)
            sample_summaries.append(sample)
        own_ext = [s for s in ext if s["run"] == run.name]
        steps = {s.step for s in run.states}
        if any(p["step"] not in steps for s in own_ext for p in s["values"]):
            raise ValueError(
                "External metric references an update absent from this run"
            )
        figures = plots(analysis, sample_summaries, directory, own_ext)
        protein = protein_assets(run, protein_structure)
        if protein.get("pdb"):
            (directory / "reference.pdb").write_text(protein["pdb"])
        write_element_history(directory / "element_probabilities.csv", run.states)
        _json(directory / "analysis.json", analysis)
        _json(directory / "sampling_summary.json", sample_summaries)
        data = {
            "name": run.name,
            "id": run.presentation.get(
                "id", hashlib.sha256(str(run.root).encode()).hexdigest()[:20]
            ),
            "starred": run.presentation.get("starred", True),
            "selection_reason": run.presentation.get(
                "reason", "Included in this report's explicit input selection"
            ),
            "tags": run.presentation.get("tags", []),
            "directory": directory.name,
            "status": run.status,
            "config": run.config,
            "target": run.target,
            "metadata": run.metadata,
            "warnings": run.warnings,
            "figures": figures,
            "states": browser_states,
            "argmax_trajectory": argmax_trajectory(analysis["rows"], run.states),
            "total_saved_states": len(run.states),
            "browser_downsampling": f"Argmax slider includes all {len(run.states)} saved evaluations. Probability details show at most {max_browser_states} evenly spaced states; metrics use all evaluations.",
            "samples": sample_summaries,
            "protein": protein,
            "external_metrics": own_ext,
            "assessment": analysis["assessment"],
        }
        contacts = [s.metrics.get("contact_score") for s in run.states]
        last = browser_states[-1]["readout"]
        deltas = [
            b - a for a, b in pairwise(contacts) if a is not None and b is not None
        ]
        data["final_argmax_candidate"] = None
        if last["sanitized"] and last["connected"]:
            smiles = last["canonical_smiles"]
            if smiles not in cache:
                cache[smiles] = molecule_assets(smiles, seed=seed)
            data["final_argmax_candidate"] = {
                "identity": smiles,
                "origin": "Final raw argmax (not a random sample)",
                "step": run.states[-1].step,
                **cache[smiles],
            }
        if any("prior_weight" in state.metrics for state in run.states):
            data["warnings"] = [
                *run.warnings,
                (
                    "Teacher/prior objectives may vary with the teacher query and schedule. "
                    "The simplex projected-gradient diagnostic does not include additional "
                    "valence/connectivity operators; it is not a full stationarity certificate."
                ),
            ]
        data["overview"] = {
            "first_step": run.states[0].step,
            "last_step": run.states[-1].step,
            "initial_contact": contacts[0],
            "final_contact": contacts[-1],
            "contact_gain": contacts[-1] - contacts[0]
            if contacts[-1] is not None and contacts[0] is not None
            else None,
            "largest_score_drop": float(max(0, -min(deltas))) if deltas else None,
            "fragments": last["fragment_count"],
            "sanitized": last["sanitized"],
            "valid_connected": last["sanitized"] and last["connected"],
        }
        report["runs"].append(data)
    _json(output / "report_data.json", report)
    # JSON in a non-executable script element: escape '<' to prevent closing it.
    embedded = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    template = (ASSETS / "report.html").read_text()
    (output / "report.html").write_text(template.replace("__REPORT_DATA__", embedded))
    _json(
        output / "report_manifest.json",
        {
            "schema": "mosaic_report_manifest_v1",
            "report_sha256": sha256(output / "report.html"),
            "files": {
                str(p.relative_to(output)): sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    return output / "report.html"
