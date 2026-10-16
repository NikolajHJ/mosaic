"""Plot and depict saved relaxed trajectories on allocated DTU HPC only.

Every drawing is the saved raw argmax heavy-atom graph. Invalid/disconnected
readouts are labelled, never repaired, and never fed back to the model.
"""

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path


CASE_COLORS = ("#087f8c", "#dc7139", "#7257b4", "#a77800", "#b84b70", "#2a6496")


def _load_studies(paths, *, allow_initialization_comparison=False):
    """Compare declared starts while requiring every other experiment input.

    Initialization replacement is an explicit display policy, never permission
    to combine different targets, weights, contexts, rates, budgets or seeds.
    Source configurations and summaries retain their own native hashes.
    """
    if not paths:
        raise ValueError("At least one completed study is required")
    studies, directories, receipts = [], [], []
    for path in paths:
        source = path / "summary.json"
        study = json.loads(source.read_text())
        if study["status"] != "passed":
            raise ValueError("Render completed authenticated trajectories only")
        for field in ("model_digest", "context_digest"):
            if study[field + "_before"] != study[field + "_after"]:
                raise ValueError("Frozen model/context changed within a study")
        if "initialization" in study["config"]:
            if (
                "mixing_values" in study["config"]
                or len(study["cases"]) != 1
                or study["cases"][0].get("initialization")
                != study["config"]["initialization"]
            ):
                raise ValueError(
                    "Saved initialization differs from the configured control"
                )
        elif [c["mixing"] for c in study["cases"]] != study["config"]["mixing_values"]:
            raise ValueError("Saved cases differ from the configured mixtures")
        if studies:
            reference = studies[0]
            for key in (
                "fixture_sha256",
                "hard_graph_sha256",
                "model",
                "numerical_profile",
                "model_digest_before",
                "model_digest_after",
                "context_digest_before",
                "context_digest_after",
            ):
                if study[key] != reference[key]:
                    raise ValueError(f"Cannot combine studies with different {key}")
            left, right = deepcopy(study["config"]), deepcopy(reference["config"])
            left.pop("mixing_values", None)
            right.pop("mixing_values", None)
            if allow_initialization_comparison:
                left.pop("initialization", None)
                right.pop("initialization", None)
            if left != right:
                raise ValueError(
                    "Combined study configurations changed an unapproved field"
                )
        studies.append(study)
        directories.extend(path / f"case-{i}" for i in range(len(study["cases"])))
        receipts.append(
            {
                "summary_path": str(source.resolve()),
                "summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "job_id": study["job_id"],
                "config_sha256": study["config_sha256"],
                "mixing_values": study["config"].get("mixing_values"),
                "initialization": study["config"].get("initialization"),
            }
        )
    combined = deepcopy(studies[0])
    combined["cases"] = [
        {**deepcopy(case), "source_job_id": study["job_id"]}
        for study in studies
        for case in study["cases"]
    ]
    mixtures = [case["mixing"] for case in combined["cases"]]
    recipes = [
        case.get("initialization")
        or {"kind": "hard_scaffold_mixture", "mixing": case["mixing"]}
        for case in combined["cases"]
    ]
    identities = [json.dumps(recipe, sort_keys=True) for recipe in recipes]
    if len(set(identities)) != len(identities):
        raise ValueError("Combined initialization recipes must be distinct")
    if allow_initialization_comparison and len(studies) > 1:
        combined["config"].pop("mixing_values", None)
        combined["config"].pop("initialization", None)
        combined["config"]["initialization_comparison"] = True
        combined["config"]["case_initializations"] = recipes
    elif "initialization" not in combined["config"]:
        combined["config"]["mixing_values"] = mixtures
    combined["job_id"] = "+".join(str(study["job_id"]) for study in studies)
    if len(studies) > 1:
        # The combined display config is derived; native hashes stay per receipt.
        combined["config_sha256"] = None
    return combined, directories, receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-initialization-comparison", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Trajectory analysis requires an allocated DTU job")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw
    from protenix_trajectory_depiction import DEPICTION_POLICY, depict_graph

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#495b64",
            "axes.edgecolor": "#d8e0e2",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=["#087f8c", "#dc7139"]),
            "grid.color": "#bac8cd",
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.4,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )
    study, case_directories, input_studies = _load_studies(
        args.study, allow_initialization_comparison=args.allow_initialization_comparison
    )
    source = args.study[0] / "summary.json"
    root = Path(__file__).resolve().parents[1]
    fixture_root = args.study[0].resolve() if "fixture_file" in study else root
    fixture_path = (
        fixture_root / study.get("fixture_file", study["config"]["fixture"])
    ).resolve()
    if not fixture_path.is_relative_to(fixture_root):
        raise ValueError("Fixture escaped the authenticated source")
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != study["fixture_sha256"]:
        raise ValueError("Report fixture differs from the completed experiment")
    fixture = json.loads(fixture_path.read_text())
    protein = fixture["protein_sequence"]
    pocket_ordinals = (
        study["config"].get(
            "pocket_protein_ordinals", fixture.get("pocket_protein_ordinals")
        )
        or ()
    )
    args.output.mkdir(parents=True, exist_ok=False)
    elements = ("C", "N", "O", "S", "F", "Cl")
    analysis = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["LSB_JOBID"],
        "cases": [],
        "initial_smiles": None
        if "initialization" in study["config"]
        or study["config"].get("initialization_comparison")
        else fixture["smiles"],
        "scaffold_smiles": fixture["smiles"],
        "initialization": study["config"].get("initialization"),
        "initialization_comparison": study["config"].get(
            "initialization_comparison", False
        ),
        "reference": study.get("reference"),
        "input_studies": input_studies,
        "config_identity_policy": "native configuration SHA per input study; combined display config has no native SHA",
        "summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": "raw hard argmax; fixed slot layout; probability-weighted sizes; no chemical repair",
        "depiction_policy": DEPICTION_POLICY,
        "depiction_sha256": hashlib.sha256(
            Path(__file__).with_name("protenix_trajectory_depiction.py").read_bytes()
        ).hexdigest(),
        "protein_context": {
            "sequence": protein,
            "name": fixture.get("protein_name", "synthetic peptide"),
            "pdb_id": fixture.get("pdb_id"),
            "source_url": fixture.get("source_url"),
            "residue_count": len(protein),
            "contact_partner_scope": fixture.get("contact_partner_scope", "pocket"),
            "pocket": [
                {"residue": i + 1, "residue_symbol": protein[i], "native_token": token}
                for i, token in zip(
                    pocket_ordinals, study["pocket_token_indices"], strict=True
                )
            ],
            "contact_cutoff_angstrom": study["config"]["contact_cutoff_angstrom"],
            "interpretation": fixture.get(
                "interpretation",
                "synthetic sequence context and selected engineering pocket; no experimental binding-site claim",
            ),
        },
        "plotting_packages": {
            name: importlib.metadata.version(name)
            for name in ("matplotlib", "numpy", "pillow", "rdkit")
        },
        "study_identity": {
            name: study[name]
            for name in (
                "job_id",
                "config_sha256",
                "fixture_sha256",
                "model",
                "numerical_profile",
                "model_digest_before",
                "model_digest_after",
                "context_digest_before",
                "context_digest_after",
            )
        },
    }
    template_path = Path(__file__).parent / "templates/protenix_trajectory_report.html"
    analysis["template_sha256"] = hashlib.sha256(template_path.read_bytes()).hexdigest()

    fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 4.3))
    fig_change, axes_change = plt.subplots(2, 2, figsize=(12, 7))
    overview = Image.new("RGB", (3600, 280 * len(study["cases"])), "white")
    overview_draw = ImageDraw.Draw(overview)
    browser = []
    for case_index, case in enumerate(study["cases"]):
        directory = case_directories[case_index]
        states = case["states"]
        expected = study["config"]["checks"]["steps"]
        if len(states) != expected + 1 or [s["step"] for s in states] != list(
            range(expected + 1)
        ):
            raise ValueError("Incomplete or noncontiguous saved trajectory")
        loaded = []
        for state in states:
            path = (directory / state["snapshot"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Snapshot escaped its case directory")
            with np.load(path, allow_pickle=False) as arrays:
                X, E = arrays["X"].copy(), arrays["E_upper"].copy()
            loaded.append((X, E))
        X0, E0 = loaded[0]
        n = X0.shape[0]
        upper = np.triu_indices(n, 1)
        atom_tv, edge_l1, atom_changes, bond_changes = [], [], [], []
        drawings, identities, change_steps = [], [], []
        previous = None
        for state, (X, E) in zip(states, loaded, strict=True):
            atom_tv.append(float(np.mean(0.5 * np.sum(np.abs(X - X0), axis=-1))))
            edge_l1.append(float(np.mean(np.abs(E[:, 0] - E0[:, 0]))))
            atom_changes.append(int(np.count_nonzero(X.argmax(-1) != X0.argmax(-1))))
            bond_changes.append(int(np.count_nonzero(E.argmax(-1) != E0.argmax(-1))))
            identity = (tuple(X.argmax(-1)), tuple(E.argmax(-1)))
            if previous is not None and identity != previous:
                change_steps.append(state["step"])
            identities.append(identity)
            previous = identity
            svg = depict_graph(
                X, E, state["step"], state["readout"]["sanitized"], svg=True
            )
            # Drop XML header so the same native SVG can be embedded inline.
            svg = svg[svg.index("<svg") :]
            drawings.append(
                {
                    "step": state["step"],
                    "svg": svg,
                    "X": X.tolist(),
                    "E_upper": E.tolist(),
                    "loss": state["primal_loss"],
                    "autodiff_loss": state["autodiff_loss"],
                    "readout": state["readout"],
                }
            )
        label = case.get("label") or f"{case['mixing']:.0%} uniform mixture"
        t = [s["step"] for s in states]
        values = np.array([s["primal_loss"] for s in states])
        color = CASE_COLORS[case_index % len(CASE_COLORS)]
        ax_loss[0].plot(t, values, label=label, color=color)
        ax_loss[1].plot(t, values - values[0], label=label, color=color)
        axes_change[0, 0].plot(t, atom_tv, label=label, color=color)
        axes_change[0, 1].plot(t, edge_l1, label=label, color=color)
        axes_change[1, 0].plot(t, atom_changes, label=label + " atoms", color=color)
        axes_change[1, 0].plot(
            t, bond_changes, label=label + " bonds", color=color, linestyle="--"
        )
        axes_change[1, 1].step(
            t,
            [int(s["readout"]["sanitized"]) for s in states],
            where="post",
            label=label,
            color=color,
        )
        selected = [
            0,
            int(expected * 0.1),
            int(expected * 0.25),
            int(expected * 0.5),
            int(expected * 0.75),
            expected,
        ]
        overview_draw.text((12, case_index * 280 + 2), label, fill="black")
        for column, step in enumerate(selected):
            X, E = loaded[step]
            png = depict_graph(X, E, step, states[step]["readout"]["sanitized"])
            overview.paste(
                Image.open(io.BytesIO(png)).convert("RGB"),
                (column * 600, case_index * 280 + 25),
            )

        def existence(E):
            B = np.zeros((n, n))
            B[upper] = 1 - E[:, 0]
            B += B.T
            return B

        Xf, Ef = loaded[-1]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for col, (array, title) in enumerate(
            (
                (X0, "Initial atom probabilities"),
                (Xf, "Final atom probabilities"),
                (Xf - X0, "Atom probability change"),
            )
        ):
            im = axes[0, col].imshow(
                array,
                cmap="coolwarm" if col == 2 else "viridis",
                vmin=-1 if col == 2 else 0,
                vmax=1,
                aspect="auto",
            )
            axes[0, col].set(title=title, xlabel="Element", ylabel="Atom slot")
            axes[0, col].set_xticks(range(6), elements)
            axes[0, col].set_yticks(range(n), range(1, n + 1))
            fig.colorbar(im, ax=axes[0, col], shrink=0.75)
        for col, (array, title) in enumerate(
            (
                (existence(E0), "Initial bond existence"),
                (existence(Ef), "Final bond existence"),
                (existence(Ef) - existence(E0), "Bond existence change"),
            )
        ):
            im = axes[1, col].imshow(
                array,
                cmap="coolwarm" if col == 2 else "viridis",
                vmin=-1 if col == 2 else 0,
                vmax=1,
            )
            axes[1, col].set(title=title, xlabel="Atom slot", ylabel="Atom slot")
            axes[1, col].set_xticks(range(n), range(1, n + 1))
            axes[1, col].set_yticks(range(n), range(1, n + 1))
            fig.colorbar(im, ax=axes[1, col], shrink=0.75)
        fig.suptitle(label + " — same occupied atom slots throughout")
        fig.tight_layout()
        fig.savefig(args.output / f"probabilities-{case_index}.png", dpi=150)
        plt.close(fig)
        browser.append(
            {
                "label": label,
                "color": color,
                "mixing": case["mixing"],
                "initialization": case.get("initialization"),
                "source_job_id": case["source_job_id"],
                "states": drawings,
                "selected_steps": selected,
            }
        )
        analysis["cases"].append(
            {
                "mixing": case["mixing"],
                "label": label,
                "initialization": case.get("initialization"),
                "source_job_id": case["source_job_id"],
                "updates": len(t) - 1,
                "initial_loss": float(values[0]),
                "final_loss": float(values[-1]),
                "unique_hard_graphs": len(set(identities)),
                "unique_sanitized_smiles": sorted(
                    {
                        s["readout"]["canonical_smiles"]
                        for s in states
                        if s["readout"]["sanitized"]
                    }
                ),
                "hard_graph_change_steps": change_steps,
                "sanitized_readouts": sum(s["readout"]["sanitized"] for s in states),
                "initial_readout": states[0]["readout"],
                "final_readout": states[-1]["readout"],
                "final_atom_tv": atom_tv[-1],
                "final_edge_existence_l1": edge_l1[-1],
                "atom_changes_from_start": atom_changes[-1],
                "bond_changes_from_start": bond_changes[-1],
            }
        )
        if study.get("reference"):
            from rdkit import Chem, DataStructs
            from rdkit.Chem import rdFingerprintGenerator

            reference_smiles = study["reference"]["readout"]["canonical_smiles"]
            reference_mol = Chem.MolFromSmiles(reference_smiles)
            from rdkit.Chem.Draw import rdMolDraw2D

            if "svg" not in analysis["reference"]:
                drawer = rdMolDraw2D.MolDraw2DSVG(300, 180)
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, reference_mol)
                drawer.FinishDrawing()
                reference_svg = drawer.GetDrawingText()
                analysis["reference"]["svg"] = reference_svg[
                    reference_svg.index("<svg") :
                ]
            final_readout = states[-1]["readout"]
            final_mol = (
                Chem.MolFromSmiles(final_readout["canonical_smiles"])
                if final_readout["sanitized"]
                else None
            )
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            comparison = {
                "reference_smiles": reference_smiles,
                "exact_smiles_match": final_mol is not None
                and Chem.MolToSmiles(final_mol) == reference_smiles,
                "morgan_r2_2048_tanimoto": (
                    DataStructs.TanimotoSimilarity(
                        generator.GetFingerprint(final_mol),
                        generator.GetFingerprint(reference_mol),
                    )
                    if final_mol is not None
                    else None
                ),
                "reference_primal_loss": study["reference"]["primal_loss"],
                "interpretation": "detached 2D identity/similarity, including all fragments; not binding or pose recovery",
            }
            analysis["cases"][-1]["reference_comparison"] = comparison
    for ax, title, ylabel in zip(
        ax_loss,
        ("Frozen Tiny contact loss", "Change from initial loss"),
        ("Contact loss (lower is better)", "Loss − initial loss"),
        strict=True,
    ):
        if study.get("reference") and ax is ax_loss[0]:
            ax.axhline(
                study["reference"]["primal_loss"],
                color="#3f4548",
                linestyle="--",
                label=study["reference"]["name"] + " (hard reference)",
            )
        ax.set(title=title, xlabel="Accepted projected updates", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig_loss.tight_layout()
    fig_loss.savefig(args.output / "loss.png", dpi=160)
    fig_loss.savefig(args.output / "loss.svg")
    plt.close(fig_loss)
    titles = (
        "Soft atom change",
        "Soft bond-existence change",
        "Hard argmax changes from initial state",
        "Raw graph sanitization",
    )
    labels = (
        "Mean atom total variation",
        "Mean absolute existence change",
        "Changed atoms / unordered bonds",
        "Sanitizes (1=yes, 0=no)",
    )
    for ax, title, ylabel in zip(axes_change.flat, titles, labels, strict=True):
        ax.set(title=title, xlabel="Accepted projected updates", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes_change[1, 1].set_ylim(-0.1, 1.1)
    fig_change.tight_layout()
    fig_change.savefig(args.output / "changes.png", dpi=150)
    plt.close(fig_change)
    overview.save(args.output / "molecules.png")
    endpoints = Image.new("RGB", (1200, overview.height), "white")
    endpoints.paste(overview.crop((0, 0, 600, overview.height)), (0, 0))
    endpoints.paste(overview.crop((3000, 0, 3600, overview.height)), (600, 0))
    endpoints.save(args.output / "molecule_endpoints.png")
    (args.output / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    payload = {"cases": browser, "analysis": analysis, "config": study["config"]}
    # Escape less-than in JSON so native SVG/XML cannot terminate the script tag.
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = template_path.read_text(encoding="utf-8").replace("__DATA__", data)
    html = html.replace("__UPDATES__", str(study["config"]["checks"]["steps"]))
    for name in ("loss", "changes", "molecules", "molecule_endpoints"):
        encoded = base64.b64encode((args.output / f"{name}.png").read_bytes()).decode()
        html = html.replace(f"__IMAGE_{name.upper()}__", encoded)
    probability_images = []
    for i in range(len(browser)):
        encoded = base64.b64encode(
            (args.output / f"probabilities-{i}.png").read_bytes()
        ).decode()
        probability_images.append("data:image/png;base64," + encoded)
    html = html.replace("__PROBABILITY_IMAGES__", json.dumps(probability_images))
    (args.output / "report.html").write_text(html, encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
