"""Allocated saved-only Boltz affinity report with authenticated raw trajectories.

This report has its own family: no Protenix controls or uncompleted random arm
are added. Rendering never calls a scorer, RDKit, a repair search or a sampler.
The shared viewer retains all saved probability frames and invalid raw readouts.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import base64
import json
import math
import os
from pathlib import Path

from assess_boltz_arm import (
    BOLTZ_COMMIT,
    CHECKPOINT_SHA,
    CONTEXT_DIGEST,
    CONTEXT_SHA,
    MODEL_DIGEST,
    REFERENCE_SHA,
    SOURCE_MANIFEST,
    verify_coupled,
    verify_raw_gradient,
)
from render_chemistry_trajectory import saved_frames, sha


SOURCE_EXECUTION_SHA = (
    "0ee4f1ca668b3255cb103fa92e8c95632e649b3910ebb625f3d94332c9834f77"
)
MIX50_CONFIG_SHA = "97faf3b192a7aca566ad86752c2f3efacad69e89b076c88e67e8d14b5435de0e"
RANDOM_CONFIG_SHA = "a9556cef734dee421a2c3146506f21da9716f5c5cf8d25874e367c432d15d932"
SUMMARY_SHA = "7f49087f12efc762a46691581f80bc42c43395c00b0cc4778a9e04a6679aae7c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def allocated():
    require(
        bool(os.environ.get("LSB_JOBID")),
        "Scientific rendering requires allocated DTU CPU",
    )


def verify_native(identity):
    for field, expected in {
        "checkpoint_sha256": CHECKPOINT_SHA,
        "context_digest": CONTEXT_DIGEST,
        "context_file_sha256": CONTEXT_SHA,
        "model_digest": MODEL_DIGEST,
        "optimized_output": "affinity_pred_value",
        "protocol": "mosaic-boltz-affinity-v1",
        "recycling_steps": 0,
        "source": BOLTZ_COMMIT,
        "status": "ready",
        "pose_policy": "fixed native predicted pose; no diffusion gradients",
    }.items():
        require(
            identity.get(field) == expected, "Native Boltz identity differs: " + field
        )


def verify_objective(identity, native):
    require(
        all(identity.get(k) == v for k, v in native.items()),
        "Affinity objective identity changed",
    )
    require(
        identity.get("coefficient") == 1.0
        and identity.get("context_sha256") == CONTEXT_SHA,
        "Affinity objective coefficient/context changed",
    )


def verify_readiness(path, expected_sha, config_path, config_sha, native):
    require(
        path is not None and expected_sha is not None,
        "Readiness path and SHA must be supplied together",
    )
    path = Path(path)
    require(sha(path) == expected_sha, "Native readiness assessment hash differs")
    assessment = read(path)
    require(
        assessment["status"] == "passed"
        and assessment["source_manifest_sha256"] == SOURCE_MANIFEST
        and assessment["context_sha256"] == CONTEXT_SHA
        and assessment["reference_sha256"] == REFERENCE_SHA
        and assessment["model_digest"] == MODEL_DIGEST
        and assessment["native_identity"] == native
        and assessment["configs"].get(config_path.name) == config_sha,
        "Native readiness did not certify this configuration/model/context",
    )
    if "certified_arms" in assessment:
        arm = "mix50" if config_sha == MIX50_CONFIG_SHA else "random"
        require(
            arm in assessment["certified_arms"],
            "Native readiness does not certify the included arm",
        )
    require(bool(assessment["proofs"]), "Readiness proof list is empty")
    for proof in assessment["proofs"]:
        require(
            sha(Path(proof["path"])) == proof["sha256"],
            "Native readiness proof bytes changed",
        )
    require(
        bool(assessment["runtime_gates"])
        and all(
            gate["status"] == "passed" for gate in assessment["runtime_gates"].values()
        ),
        "Native allocation-fit gate failed",
    )
    return {
        "path": str(path),
        "sha256": expected_sha,
        "source_manifest_sha256": assessment["source_manifest_sha256"],
        "version": assessment.get("version"),
        "certified_arms": assessment.get("certified_arms"),
        "scope": assessment.get("scope"),
        "proof_count": len(assessment["proofs"]),
    }


def load_study(
    root,
    config_path,
    *,
    expected_job,
    expected_summary_sha256,
    expected_config_sha256,
    expected_budget=10000,
    color="#008c95",
    readiness_path=None,
    readiness_sha256=None,
):
    """Authenticate one completed saved study; a smaller library budget is for fixtures.

    The CLI always fixes the real 10,000-update budget and known configuration
    pins. Independent tests may supply an explicit small fixture budget, keeping
    the same schema, native identities and all persistence/completion gates.
    """
    allocated()
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule

    root, config_path = Path(root), Path(config_path)
    summary_path, case_path = root / "summary.json", root / "case-0/report.json"
    require(sha(summary_path) == expected_summary_sha256, "Native summary hash differs")
    require(sha(config_path) == expected_config_sha256, "Configuration hash differs")
    summary, config, case = read(summary_path), read(config_path), read(case_path)
    require(
        summary["status"] == case["status"] == "passed" and summary["mode"] == "run",
        "Only completed native Boltz runs can enter this report",
    )
    require(str(summary["job_id"]) == str(expected_job), "Native job identity differs")
    require(
        summary["config"] == config
        and summary["config_sha256"] == expected_config_sha256,
        "Native/report configuration identity differs",
    )
    require(
        config["version"] == "brd4-boltz-affinity-chemistry-v1"
        and config["n_atoms"] == 6
        and config["steps"] == expected_budget
        and config["seed"] == 17
        and config["initial_step_size"] == 0.03
        and config["max_backtracks"] == 10
        and config["element_js_weight"] == 0.1,
        "Fixed six-slot Boltz experiment settings changed",
    )
    require(
        config["initialization"]
        in (
            {"kind": "chain_uniform", "mixing": 0.5},
            {"kind": "dirichlet", "concentration": 1.0, "seed": 20261019},
        ),
        "Unsupported or changed initialization",
    )
    require(
        summary["source"]["execution_content_sha256"] == SOURCE_EXECUTION_SHA,
        "Native scientific source identity differs",
    )
    if expected_budget == 10000:
        # Production configurations live in the original immutable release;
        # the explicit small library-fixture budget does not imitate that tree.
        require(
            config_path.parent.name == "configs"
            and config_path.parent.parent.name == "examples",
            "Production configuration is outside its scientific release",
        )
        scientific_source = config_path.resolve().parents[2]
        require(
            sha(scientific_source / "source_manifest.json") == SOURCE_MANIFEST,
            "Original scientific release manifest differs",
        )
        manifest = read(scientific_source / "source_manifest.json")
        relative_config = (
            config_path.resolve().relative_to(scientific_source).as_posix()
        )
        require(
            manifest["files"].get(relative_config) == expected_config_sha256,
            "Configuration differs from original scientific release",
        )
        source_files = summary["source"].get("files")
        require(
            bool(source_files)
            and all(
                manifest["files"].get(name) == value
                for name, value in source_files.items()
            ),
            "Saved execution file map differs from original scientific release",
        )
    require(
        summary["context_sha256"] == CONTEXT_SHA
        and summary["reference_sha256"] == REFERENCE_SHA,
        "Frozen context/composition reference changed",
    )
    native = summary["native_identity"]
    verify_native(native)
    after = summary["frozen_identity_after"]
    require(
        after.get("status") == "passed"
        and after["model_digest"] == MODEL_DIGEST
        and after["context_digest"] == CONTEXT_DIGEST,
        "Frozen native terminal identity failed",
    )
    parity = summary["native_hard_parity"]
    require(
        parity["status"] == "passed"
        and parity["atol"] == 2e-5
        and len(parity["native"]) == len(parity["attached"]) == 3
        and all(
            math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 2e-5
            for a, b in zip(parity["native"], parity["attached"])
        ),
        "Integer-native hard-seed forward parity failed",
    )
    for field in ("native_gradient_checks", "terminal_gradient_checks"):
        actual = read(root / field / "report.json")
        require(actual == summary[field], "Saved native gradient report differs")
        verify_raw_gradient(actual)
    require(
        sha(root / "initialization.npz") == summary["initialization_sha256"],
        "Saved native initialization changed",
    )
    require(
        len(summary["cases"]) == 1
        and all(summary["cases"][0].get(k) == v for k, v in case.items()),
        "Native terminal summary and saved case differ",
    )
    require(
        case["requested_updates"] == expected_budget
        and case["parameterization"] == "categorical_logits_v1"
        and case["acceptance_policy"] == "best_primal",
        "Native optimizer/budget changed",
    )
    verify_objective(case["affinity_objective_identity"], native)
    require(
        case["initial_checks_file"] == "combined_initial_checks/report.json",
        "Unexpected initial combined-check artifact",
    )
    combined = read(root / "case-0" / case["initial_checks_file"])
    require(
        combined["status"] == "passed"
        and combined["directions"] == case["initial_checks"],
        "Initial coupled native/chemistry check did not pass",
    )
    verify_objective(combined["affinity_objective_identity"], native)
    verify_coupled(combined)
    schedule = RepairWeightSchedule(**config["repair_schedule"])
    require(
        case["repair_weight_schedule"] == schedule.record()
        and schedule.total_updates == expected_budget
        and schedule.initial == 0.001
        and schedule.final == 50.0
        and (expected_budget != 10000 or schedule.hold_updates == 5000),
        "Saved delayed chemistry schedule differs",
    )
    states, updates = case["states"], case["updates"]
    require(
        len(states) == expected_budget + 1
        and [s["step"] for s in states] == list(range(expected_budget + 1))
        and len(updates) == expected_budget
        and [u["step"] for u in updates] == list(range(expected_budget)),
        "Incomplete or noncontiguous saved trajectory",
    )
    for name, expected in (("states", states), ("updates", updates)):
        persisted = [
            json.loads(line)
            for line in (root / "case-0" / (name + ".jsonl")).read_text().splitlines()
            if line.strip()
        ]
        require(persisted == expected, "Saved JSONL and terminal case differ: " + name)
    counts = case["counts"]
    require(
        counts["applied"] == expected_budget
        and counts["strict_improving"] == sum(not u["fallback_used"] for u in updates)
        and counts["forced_fallback"] == sum(u["fallback_used"] for u in updates)
        and counts["no_op"] == sum(u["no_op"] for u in updates),
        "Saved update counts differ",
    )
    frames = saved_frames(root / "case-0", states, native_name="affinity")
    for frame in frames:
        components = frame["components"]
        require(
            frame["repair_weight"] == schedule.for_state(frame["step"]),
            "Saved state coefficient differs from next-update schedule",
        )
        require(
            components["affinity_loss"]
            == components["affinity_pred_value"]
            == frame["affinity"],
            "Saved affinity metric identity differs",
        )
        require(
            all(
                isinstance(v, (int, float)) and math.isfinite(v)
                for v in components.values()
            )
            and 0 <= components["affinity_probability_binary"] <= 1,
            "Saved affinity/chemistry metrics are nonfinite or malformed",
        )
        require(
            abs(components["affinity_autodiff_loss"] - frame["affinity"]) <= 2e-5,
            "Saved native AD/primal parity failed",
        )
    for update in updates:
        step = update["step"]
        weight = schedule.for_update(step + 1)
        require(
            update["repair_weight"] == weight
            and 1 <= len(update["attempts"]) <= 10
            and [a["trial"] for a in update["attempts"]]
            == list(range(len(update["attempts"])))
            and all(a["repair_weight"] == weight for a in update["attempts"]),
            "Trial coefficient changed or attempts are missing/reordered",
        )
        require(
            update["preparation_identity"] == states[step]["preparation_identity"]
            and update["fixed_before"] == states[step]["refreshed_surrogate"]
            and update["fixed_after"]
            == states[step + 1]["previous_fixed_surrogate_after"]
            and update["affinity_before"] == states[step]["affinity_loss"]
            and update["affinity_after"] == states[step + 1]["affinity_loss"],
            "Within-update fixed-reference or native scalar linkage differs",
        )
        require(
            math.isfinite(update["fixed_before"])
            and math.isfinite(update["fixed_after"])
            and update["fixed_increased"]
            is (update["fixed_after"] > update["fixed_before"]),
            "Saved fixed-reference delta is nonfinite or mislabeled",
        )
        frames[step + 1]["previous_update_fixed_delta"] = {
            "delta": update["fixed_after"] - update["fixed_before"],
            "fallback_used": update["fallback_used"],
            "no_op": update["no_op"],
            "repair_weight": weight,
            "preparation_identity": update["preparation_identity"],
        }
    best = case["best_observed_affinity"]
    require(0 <= best["step"] < len(states), "Best affinity state is missing")
    best_state = states[best["step"]]
    require(
        best["loss"] == min(s["affinity_loss"] for s in states)
        and best["loss"] == best_state["affinity_loss"]
        and best["snapshot"] == best_state["snapshot"]
        and best["snapshot_sha256"] == best_state["snapshot_sha256"],
        "Best observed affinity record differs from saved states",
    )
    verify_objective(best["objective_identity"], native)
    readiness = None
    if readiness_path is not None or readiness_sha256 is not None:
        readiness = verify_readiness(
            readiness_path,
            readiness_sha256,
            config_path,
            expected_config_sha256,
            native,
        )
    return {
        "label": config["label"],
        "color": color,
        "job_id": str(summary["job_id"]),
        "frames": frames,
        "counts": counts,
        "status": "passed",
        "complete": True,
        "updates": updates,
        "best_observed_affinity": best,
        "summary_sha256": expected_summary_sha256,
        "config_sha256": expected_config_sha256,
        "schedule": schedule.record(),
        "initialization": config["initialization"],
        "initialization_sha256": summary["initialization_sha256"],
        "model_digest": MODEL_DIGEST,
        "context_digest": CONTEXT_DIGEST,
        "context_sha256": CONTEXT_SHA,
        "reference_sha256": REFERENCE_SHA,
        "source_execution_sha256": SOURCE_EXECUTION_SHA,
        "native_identity": native,
        "readiness": readiness,
    }


def render_report(studies, output):
    """Render previously verified studies using the shared offline HTML viewer."""
    allocated()
    require(
        bool(studies) and len({s["job_id"] for s in studies}) == len(studies),
        "Need distinct included completed native studies",
    )
    for study in studies:
        require(
            study["complete"] is True and study["status"] == "passed",
            "Incomplete study",
        )
        for field in ("model_digest", "context_digest", "context_sha256", "schedule"):
            require(
                study[field] == studies[0][field],
                "Included studies have different frozen contexts/schedules",
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(5, 1, figsize=(12, 17), constrained_layout=True)
    for study in studies:
        frames, updates, color = study["frames"], study["updates"], study["color"]
        steps = [f["step"] for f in frames]
        axes[0].plot(
            steps, [f["affinity"] for f in frames], color=color, label=study["label"]
        )
        axes[1].plot(
            steps,
            [f["components"]["affinity_probability_binary"] for f in frames],
            color=color,
            label=study["label"] + " · logged probability",
        )
        axes[2].plot(
            steps,
            [f["components"]["chemistry_loss"] for f in frames],
            color=color,
            label=study["label"] + " · raw repair",
        )
        axes[2].plot(
            steps,
            [f["repair_weight"] * f["components"]["chemistry_loss"] for f in frames],
            color=color,
            linestyle=":",
            label=study["label"] + " · weighted repair",
        )
        axes[3].plot(steps, [f["repair_weight"] for f in frames], color=color)
        axes[4].scatter(
            [u["step"] + 1 for u in updates],
            [u["fixed_after"] - u["fixed_before"] for u in updates],
            s=4,
            color=color,
            label=study["label"],
        )
        forced = [u for u in updates if u["fallback_used"]]
        if forced:
            axes[4].scatter(
                [u["step"] + 1 for u in forced],
                [u["fixed_after"] - u["fixed_before"] for u in forced],
                s=12,
                marker="x",
                color="#cc2457",
                label="Forced fallback",
            )
    axes[0].set(
        title="Fixed-pose Boltz-2 affinity_pred_value · lower is better",
        ylabel="Predicted log10(IC50 / µM)",
    )
    axes[1].set(
        title="Binder probability · logged separately, excluded from objective",
        ylabel="Probability",
        ylim=(0, 1),
    )
    axes[2].set(
        title="Saved raw repair and coefficient × repair · changing targets",
        ylabel="Repair diagnostic",
    )
    axes[3].set(
        title="Saved-state chemistry coefficient · next update, terminal held",
        ylabel="Coefficient",
        yscale="log",
    )
    axes[4].set(
        title="Accepted minus before · same targets and coefficient within each update",
        ylabel="Fixed-reference delta",
        xlabel="Applied graph updates",
    )
    axes[4].axhline(0, color="#777", linewidth=0.7)
    for index, ax in enumerate(axes):
        ax.axvline(
            studies[0]["schedule"]["hold_updates"],
            linestyle="--",
            color="#777",
            linewidth=0.8,
        )
        if index != 3:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    figure.savefig(output / "loss_components.svg")
    figure.savefig(output / "loss_components.png", dpi=160)
    plt.close(figure)
    budget = studies[0]["schedule"]["total_updates"]
    pins = [
        {
            k: study[k]
            for k in (
                "job_id",
                "summary_sha256",
                "config_sha256",
                "initialization",
                "initialization_sha256",
                "schedule",
                "model_digest",
                "context_digest",
                "context_sha256",
                "reference_sha256",
                "source_execution_sha256",
                "native_identity",
                "readiness",
                "counts",
            )
        }
        for study in studies
    ]
    caption = (
        "Quantitative affinity is predicted log10(IC50 / µM), lower is better, for the saved relaxed graph in one fixed native pose/context. "
        "It is not a fresh prediction or experimental affinity of the displayed hard argmax graph. Binder probability is a separate logged metric. "
        "Chemistry targets and coefficients change: only each within-update delta compares one fixed surrogate. Red crosses mark forced fallbacks. "
        "No model, repair, resampling or chemical conversion is run by this viewer."
    )
    data = {
        "version": "mosaic-boltz-affinity-report-v1",
        "primary_metric": "affinity",
        "complete": True,
        "status": "passed",
        "pending_studies": [],
        "title": "BRD4: Boltz-2 affinity + delayed chemistry",
        "status_text": f"{len(studies)} completed native stud{'y' if len(studies) == 1 else 'ies'} · {budget:,} updates each · all saved states retained",
        "intro": (
            "Fresh 50% chain/uniform initialization"
            if len(studies) == 1
            else "Independent completed initializations"
        )
        + " · six occupied neutral slots · frozen Boltz-2 · fixed native BRD4/hexane pose · LR 0.03. "
        + "Chemistry weight 0.001 through update 5,000, then geometric ramp to 50 at update 10,000. Raw invalid graphs remain visible.",
        "plot_caption": caption,
        "plot_alt": "Boltz-2 affinity, separately logged binder probability, raw and weighted chemistry, coefficient schedule and within-update fixed-reference loss changes",
        "model_digest": MODEL_DIGEST,
        "context_digest": CONTEXT_DIGEST,
        "studies": [
            {k: v for k, v in study.items() if k != "updates"} for study in studies
        ],
        "historical_contact": None,
        "new_counts": [s["counts"] for s in studies],
        "best_observed_affinity": [s["best_observed_affinity"] for s in studies],
        "provenance": pins,
        "limitations": caption
        + " Zero recycles, no molecular-weight correction and fixed neutral occupancy define a partial chemistry gradient. "
        + "A detached repaired teacher is supervision, never the current molecule. No validity, binding or selectivity guarantee is established.",
    }
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    template = (
        (_layout_path(_repository_root(__file__) / 'scripts' / "chemistry_report_template.html"))
        .read_text(encoding="utf-8")
    )
    plot = base64.b64encode((output / "loss_components.svg").read_bytes()).decode(
        "ascii"
    )
    (output / "report.html").write_text(
        template.replace("__SAVED_TRAJECTORIES__", payload).replace(
            "__LOSS_PLOT_BASE64__", plot
        ),
        encoding="utf-8",
    )
    receipt = {
        "version": "mosaic-boltz-affinity-report-receipt-v1",
        "render_job_id": os.environ["LSB_JOBID"],
        "complete": True,
        "primary_metric": "affinity",
        "input_jobs": [s["job_id"] for s in studies],
        "states": [len(s["frames"]) for s in studies],
        "inputs": pins,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    }
    (output / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "config", "readiness"):
        parser.add_argument("--" + name, type=Path, action="append", required=True)
    for name in ("job", "summary-sha256", "config-sha256", "readiness-sha256"):
        parser.add_argument("--" + name, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocated()
    aligned = (
        args.study,
        args.config,
        args.job,
        args.summary_sha256,
        args.config_sha256,
        args.readiness,
        args.readiness_sha256,
    )
    require(
        len({len(items) for items in aligned}) == 1,
        "Each study requires exact job/config/summary/readiness pins",
    )
    require(
        all(pin in (MIX50_CONFIG_SHA, RANDOM_CONFIG_SHA) for pin in args.config_sha256),
        "Unknown production configuration pin",
    )
    studies = [
        load_study(
            root,
            config,
            expected_job=job,
            expected_summary_sha256=summary_sha,
            expected_config_sha256=config_sha,
            readiness_path=readiness,
            readiness_sha256=readiness_sha,
            color=("#008c95", "#b55e38", "#7754bb")[i % 3],
        )
        for i, (
            root,
            config,
            job,
            summary_sha,
            config_sha,
            readiness,
            readiness_sha,
        ) in enumerate(zip(*aligned))
    ]
    print(json.dumps(render_report(studies, args.output)))


if __name__ == "__main__":
    main()
