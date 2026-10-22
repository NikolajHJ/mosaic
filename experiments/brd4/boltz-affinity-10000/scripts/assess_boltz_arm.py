"""Adopt a passed arm from a failed aggregate readiness, without rerunning science.

This deliberately scoped adapter preserves the failed parent job and random arm.
Its output certifies only the already validated mix50 configuration. Execute on
an allocated CPU: the original immutable scientific source remains unchanged.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


SOURCE_MANIFEST = "5eb6e17ea8d19e52c99a589243228710994dcf137f59a6b9a6659a4fe719a7f4"
PARENT_JOB = "29759069"
CPU_JOB = "29759068"
CONTEXT_SHA = "254d5b7f11605fb41db6f5abf7015f9cc856a19dfe1d4b506ccaea9007200770"
MODEL_DIGEST = "395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16"
CONTEXT_DIGEST = "495002ffa2a52cc85b58ef34424622763dfc4ca556139a98297e8789ae287aa5"
BOLTZ_COMMIT = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"
CHECKPOINT_SHA = "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"
REFERENCE_SHA = "37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb"
SUMMARY_SHA = "9e8121a2c7e3de93ab886244372aadc303928c6ddb2402d797172c1fcd667cee"
CPU_FINAL_SHA = "4ebda873e1cca12e2905aae19326f52c98364ebba3d616c20bc45d7e3017971f"
CPU_STDOUT_SHA = "a97fcbb15c897f8ea1e1ae7e04e8b439526bc068c2d64d7c8634ae9b0cff608f"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_release(release, manifest):
    """Verify every original source byte before importing its CPU verifier."""
    require(manifest == SOURCE_MANIFEST, "Only original source02 adoption is supported")
    require(
        sha(release / "source_manifest.json") == manifest, "Source manifest changed"
    )
    recorded = read(release / "source_manifest.json")["files"]
    for name, expected in recorded.items():
        path = (release / name).resolve()
        require(path.is_relative_to(release), "Unsafe source manifest path")
        require(
            not path.is_symlink() and sha(path) == expected,
            "Source bytes changed: " + name,
        )
    for folder in ("src", "tests", "examples", "scripts"):
        for path in (release / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                require(
                    path.relative_to(release).as_posix() in recorded,
                    "Unmanifested source input",
                )
    return recorded


def verify_cpu(release, cpu, manifest):
    path = _layout_path(release / "scripts/check_boltz_cpu_proof.py")
    spec = importlib.util.spec_from_file_location("original_boltz_cpu_proof", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify(cpu, manifest)


def scheduler_evidence(cpu):
    """Require actual final outcomes; an EXIT parent is never called a pass."""
    result = subprocess.run(
        ["bjobs", "-noheader", "-o", "jobid stat queue", PARENT_JOB],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    require(len(rows) == 1 and len(rows[0]) == 3, "Ambiguous scheduler outcomes")
    require(
        tuple(rows[0]) == (PARENT_JOB, "EXIT", "gpul40s"),
        "Expected actual native EXIT/gpul40s",
    )
    # The CPU job aged out of live bjobs. Authenticate its original final
    # scheduler capture and LSF success footer instead of inventing a result.
    final = cpu.parent / "scheduler-final.txt"
    stdout = cpu.parent / ("stdout." + CPU_JOB + ".log")
    require(
        sha(final) == CPU_FINAL_SHA and sha(stdout) == CPU_STDOUT_SHA,
        "Original CPU final scheduler/LSF evidence changed",
    )
    compact = " ".join(final.read_text(encoding="utf-8").split())
    log = stdout.read_text(encoding="utf-8")
    require(
        "Job <" + CPU_JOB + ">" in compact
        and "Status <DONE>" in compact
        and "Done successfully." in compact
        and "Successfully completed." in log
        and "Subject: Job " + CPU_JOB + ":" in log,
        "Original CPU did not actually finish successfully",
    )
    return result.stdout, [final, stdout]


def finite_positive(value):
    return isinstance(value, (float, int)) and math.isfinite(value) and value > 0


def verify_raw_gradient(report):
    require(report["status"] == "passed", "Raw atom/bond finite differences failed")
    require(
        report["typed_bond_contrast_nonzero"] is True, "Typed-bond contrast is missing"
    )
    require(
        [row["field"] for row in report["rows"]] == ["X", "E_upper"],
        "Raw derivative fields differ",
    )
    for row in report["rows"]:
        probes = row["probes"]
        require(
            [p["epsilon"] for p in probes] == [0.01, 0.003, 0.001],
            "Derivative scales changed",
        )
        checked = []
        for probe in probes:
            analytic, fd = probe["analytic"], probe["finite_difference"]
            endpoints = probe["primal_endpoints"]
            require(
                len(endpoints) == 2 and all(math.isfinite(v) for v in endpoints),
                "Nonfinite primal endpoint",
            )
            reconstructed = (endpoints[1] - endpoints[0]) / (2 * probe["epsilon"])
            require(
                math.isclose(fd, reconstructed, rel_tol=1e-12, abs_tol=1e-12),
                "Recorded FD differs from primal endpoints",
            )
            agrees = (
                math.isfinite(fd)
                and math.isfinite(analytic)
                and abs(analytic) > 1e-7
                and analytic * fd > 0
                and abs(fd - analytic) <= 0.15 * max(abs(fd), abs(analytic))
            )
            require(
                probe["agrees"] is agrees,
                "Recorded derivative agreement is inconsistent",
            )
            checked.append(agrees)
        require(
            any(a and b for a, b in zip(checked, checked[1:])),
            "Two adjacent derivative scales are required",
        )


def objective_identity(report, native):
    identity = report["affinity_objective_identity"]
    require(
        all(identity.get(k) == value for k, value in native.items()),
        "Pilot/coupled native identity changed",
    )
    require(
        identity["coefficient"] == 1.0 and identity["context_sha256"] == CONTEXT_SHA,
        "Pilot objective/context changed",
    )


def verify_coupled(report):
    """Recheck saved agreement values at the unchanged native tolerances."""
    directions = report["directions"]
    require(
        [row["field"] for row in directions] == ["atoms", "bonds"],
        "Coupled derivative fields differ",
    )
    for row in directions:
        probes = row["probes"]
        require(
            [p["epsilon"] for p in probes] == [0.01, 0.003, 0.001],
            "Coupled derivative scales changed",
        )
        agreements = []
        for probe in probes:
            analytic, fd = probe["analytic"], probe["finite_difference"]
            require(
                analytic == row["analytic"]
                and probe["preparation_identity"] == report["preparation_identity"],
                "Coupled derivative target/analytic changed within probes",
            )
            agrees = (
                math.isfinite(analytic)
                and math.isfinite(fd)
                and abs(analytic) > 1e-7
                and analytic * fd > 0
                and abs(fd - analytic) <= 0.15 * max(abs(fd), abs(analytic))
            )
            require(
                probe["agrees"] is agrees,
                "Coupled derivative agreement is inconsistent",
            )
            agreements.append(agrees)
        require(
            any(a and b for a, b in zip(agreements, agreements[1:])),
            "Two adjacent coupled derivative scales are required",
        )


def verify_arm(release, run, recorded, cpu):
    root = run / "mix50"
    summary = read(root / "summary.json")
    require(
        summary["status"] == "passed" and summary["mode"] == "validate",
        "Selected arm did not pass validation",
    )
    require(summary["job_id"] == PARENT_JOB, "Selected arm parent job differs")
    config_name = "brd4_boltz_affinity_mix50_10000.json"
    config_path = _layout_path(release / "examples/configs" / config_name)
    config_sha = sha(config_path)
    require(
        summary["config_sha256"] == config_sha
        and summary["config"] == read(config_path),
        "Selected arm configuration changed",
    )
    require(
        summary["source"]["files"]
        == {
            name: digest
            for name, digest in recorded.items()
            if name == "pyproject.toml"
            or (
                name.split("/")[0] in {"src", "tests", "examples", "scripts"}
                and Path(name).suffix
                in {".py", ".json", ".lsf", ".sh", ".toml", ".txt"}
                and "__pycache__" not in Path(name).parts
            )
        },
        "Selected arm execution source differs",
    )
    native = summary["native_identity"]
    expected = {
        "model_digest": MODEL_DIGEST,
        "context_digest": CONTEXT_DIGEST,
        "context_file_sha256": CONTEXT_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "source": BOLTZ_COMMIT,
        "optimized_output": "affinity_pred_value",
        "recycling_steps": 0,
        "protocol": "mosaic-boltz-affinity-v1",
        "status": "ready",
        "pose_policy": "fixed native predicted pose; no diffusion gradients",
    }
    require(
        native == expected, "Native model/context/checkpoint/protocol identity changed"
    )
    frozen = summary["frozen_identity_after"]
    require(
        frozen["status"] == "passed"
        and frozen["model_digest"] == MODEL_DIGEST
        and frozen["context_digest"] == CONTEXT_DIGEST,
        "Frozen model/context verification failed",
    )
    require(
        summary["context_sha256"] == CONTEXT_SHA
        and summary["reference_sha256"] == REFERENCE_SHA
        and cpu["reference_sha256"] == REFERENCE_SHA,
        "Context/reference binding changed",
    )
    parity = summary["native_hard_parity"]
    require(
        parity["status"] == "passed" and parity["atol"] == 2e-5,
        "Hard-seed parity gate changed/failed",
    )
    require(
        len(parity["native"]) == len(parity["attached"]) == 3
        and all(
            math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 2e-5
            for a, b in zip(parity["native"], parity["attached"])
        ),
        "Hard-seed native forward parity failed",
    )
    gradient = read(root / "native_gradient_checks/report.json")
    require(
        gradient == summary["native_gradient_checks"],
        "Raw gradient summary/proof differ",
    )
    verify_raw_gradient(gradient)
    assay = summary["chemistry_assay"]
    require(
        assay["status"] == "passed"
        and finite_positive(assay["loss"])
        and finite_positive(assay["gradient_norm"]),
        "Missing nonzero chemistry signal",
    )
    runtime = summary["runtime_gate"]
    means = runtime["mean_update_seconds"]
    require(
        len(means) == 2 and all(finite_positive(v) for v in means),
        "Invalid pilot timing",
    )
    projection = 1.5 * max(means) * 10000 + 1800
    require(
        runtime["status"] == "passed"
        and runtime["allocation_seconds"] == 82800
        and math.isclose(
            runtime["conservative_projected_seconds"], projection, rel_tol=1e-12
        )
        and projection < 82800,
        "Runtime fit gate failed/changed",
    )
    for phase, weight in (("early", 0.001), ("high", 50.0)):
        coupled = read(root / (phase + "_coupled_checks/report.json"))
        require(
            coupled["status"] == "passed" and coupled["repair_weight"] == weight,
            "Coupled chemistry gate failed/changed",
        )
        objective_identity(coupled, native)
        verify_coupled(coupled)
        pilot_root = root / (phase + "_pilot")
        pilot = read(pilot_root / "report.json")
        require(
            pilot["status"] == "passed"
            and pilot["requested_updates"] == 10
            and pilot["acceptance_policy"] == "best_primal",
            "Disposable pilot failed/changed",
        )
        objective_identity(pilot, native)
        require(
            [s["step"] for s in pilot["states"]] == list(range(11))
            and [u["step"] for u in pilot["updates"]] == list(range(10)),
            "Incomplete pilot trajectory",
        )
        for state in pilot["states"]:
            snapshot = (pilot_root / state["snapshot"]).resolve()
            require(
                snapshot.is_relative_to(pilot_root.resolve())
                and sha(snapshot) == state["snapshot_sha256"],
                "Pilot snapshot changed",
            )
            require(
                state["repair_weight"] == weight, "Pilot chemistry coefficient changed"
            )
    require(
        sha(root / "initialization.npz") == summary["initialization_sha256"],
        "Initialization bytes changed",
    )
    require(
        sha(root / "summary.json") == SUMMARY_SHA,
        "Original selected summary bytes changed",
    )
    context = run / "prepared/context.pt"
    preparation = read(run / "prepared/preparation.json")
    require(
        sha(context) == CONTEXT_SHA
        and preparation["context_sha256"] == CONTEXT_SHA
        and preparation["job_id"] == PARENT_JOB
        and preparation["status"] == "prepared"
        and preparation["boltz_commit"] == BOLTZ_COMMIT
        and preparation["n_atoms"] == 6
        and preparation["seed_smiles"] == "CCCCCC",
        "Prepared context identity changed",
    )
    return summary, config_name, config_sha


def adopt(release, run, cpu_root, manifest, arm, output):
    require(
        bool(os.environ.get("LSB_JOBID")),
        "Evidence adoption requires allocated DTU HPC CPU",
    )
    require(arm == "mix50", "Only passed mix50 is authorized; random is not certified")
    require(not output.exists(), "Adoption output already exists")
    release, run, cpu_root = release.resolve(), run.resolve(), cpu_root.resolve()
    recorded = verify_release(release, manifest)
    cpu = verify_cpu(release, cpu_root, manifest)
    summary, config_name, config_sha = verify_arm(release, run, recorded, cpu)
    random_summary = read(run / "random/summary.json")
    require(
        random_summary["status"] == "failed" and random_summary["job_id"] == PARENT_JOB,
        "Historical failed random arm is not preserved as expected",
    )
    scheduler, scheduler_proofs = scheduler_evidence(cpu_root)
    proof_paths = [p for p in (run / arm).rglob("*") if p.is_file()]
    proof_paths += [
        run / "prepared/context.pt",
        run / "prepared/preparation.json",
        run / "random/summary.json",
        cpu_root / "assessment.json",
        cpu_root / "composition_reference.json",
        release / "source_manifest.json",
    ]
    proof_paths += [Path(p["path"]) for p in cpu["proofs"]]
    proof_paths += scheduler_proofs
    proofs = [
        {"path": str(path), "sha256": sha(path)} for path in sorted(set(proof_paths))
    ]
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared"
    prepared.mkdir()
    for name in ("context.pt", "preparation.json"):
        shutil.copyfile(run / "prepared" / name, prepared / name)
        require(
            sha(prepared / name) == sha(run / "prepared" / name),
            "Copied context/preparation differs",
        )
        proofs.append({"path": str(prepared / name), "sha256": sha(prepared / name)})
    scheduler_path = output / "scheduler-parent-cpu.txt"
    scheduler_path.write_text(scheduler, encoding="utf-8")
    proofs.append({"path": str(scheduler_path), "sha256": sha(scheduler_path)})
    result = {
        "version": "mosaic-boltz-passed-arm-adoption-v1",
        "status": "passed",
        "scope": "selected passed mix50 evidence only; parent/global readiness remains failed; random is not certified",
        "certified_arms": [arm],
        "job_id": os.environ["LSB_JOBID"],
        "parent_job_id": PARENT_JOB,
        "parent_scheduler_outcome": "EXIT",
        "parent_global_readiness_status": "failed",
        "uncertified_arms": {"random": "failed"},
        "cpu_job_id": CPU_JOB,
        "source_manifest_sha256": manifest,
        "original_summary_sha256": sha(run / arm / "summary.json"),
        "model_digest": summary["native_identity"]["model_digest"],
        "native_identity": summary["native_identity"],
        "context_sha256": CONTEXT_SHA,
        "reference_sha256": REFERENCE_SHA,
        "configs": {config_name: config_sha},
        "runtime_gates": {arm: summary["runtime_gate"]},
        "proofs": proofs,
        "cpu_assessment_sha256": sha(cpu_root / "assessment.json"),
    }
    (output / "assessment.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "run", "cpu", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm", choices=("mix50",), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            adopt(
                args.release, args.run, args.cpu, args.manifest, args.arm, args.output
            )
        )
    )


if __name__ == "__main__":
    main()
