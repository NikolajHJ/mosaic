"""Evidence-only adoption contracts; run on an allocated DTU HPC CPU."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


REPO = _repository_root(__file__)
SPEC = importlib.util.spec_from_file_location(
    "boltz_arm_assessment", _layout_path(REPO / "scripts/assess_boltz_arm.py")
)
ARM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARM)


class ArmAdoptionTests(unittest.TestCase):
    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.release, self.run, self.cpu = [
            self.base / p for p in ("source", "native", "cpu")
        ]
        self.output = self.base / "adopted"
        self.config = (
            _layout_path(self.release / "examples/configs/brd4_boltz_affinity_mix50_10000.json")
        )
        self.put(
            self.config,
            {"steps": 10000, "n_atoms": 6, "initialization": {"mixing": 0.5}},
        )
        self.put(self.release / "pyproject.toml", {})
        reference = self.cpu / "composition_reference.json"
        self.put(
            reference, {"status": "passed", "fixture": "unchanged empirical reference"}
        )
        self.reference_sha = ARM.sha(reference)
        verifier = (_layout_path(REPO / "scripts/check_boltz_cpu_proof.py")).read_text(
            encoding="utf-8"
        )
        verifier = verifier.replace(ARM.REFERENCE_SHA, self.reference_sha)
        verifier_path = _layout_path(self.release / "scripts/check_boltz_cpu_proof.py")
        verifier_path.parent.mkdir(parents=True)
        verifier_path.write_text(verifier, encoding="utf-8")
        self.files = {
            p.relative_to(self.release).as_posix(): ARM.sha(p)
            for p in self.release.rglob("*")
            if p.is_file()
        }
        self.put(self.release / "source_manifest.json", {"files": self.files})
        self.manifest = ARM.sha(self.release / "source_manifest.json")
        context = self.run / "prepared/context.pt"
        context.parent.mkdir(parents=True)
        context.write_bytes(
            b"opaque context bytes; never load tensors in evidence tests"
        )
        self.context_sha = ARM.sha(context)
        self.put(
            self.run / "prepared/preparation.json",
            {
                "context_sha256": self.context_sha,
                "job_id": ARM.PARENT_JOB,
                "status": "prepared",
                "boltz_commit": ARM.BOLTZ_COMMIT,
                "n_atoms": 6,
                "seed_smiles": "CCCCCC",
            },
        )
        self.native = {
            "model_digest": ARM.MODEL_DIGEST,
            "context_digest": ARM.CONTEXT_DIGEST,
            "context_file_sha256": self.context_sha,
            "checkpoint_sha256": ARM.CHECKPOINT_SHA,
            "source": ARM.BOLTZ_COMMIT,
            "optimized_output": "affinity_pred_value",
            "recycling_steps": 0,
            "protocol": "mosaic-boltz-affinity-v1",
            "status": "ready",
            "pose_policy": "fixed native predicted pose; no diffusion gradients",
        }
        self.root = self.run / "mix50"
        self.root.mkdir()
        (self.root / "initialization.npz").write_bytes(b"opaque saved initialization")
        self.raw = {"status": "passed", "typed_bond_contrast_nonzero": True, "rows": []}
        for field in ("X", "E_upper"):
            self.raw["rows"].append(
                {
                    "field": field,
                    "probes": [
                        {
                            "epsilon": e,
                            "analytic": 1.0,
                            "finite_difference": 1.0,
                            "primal_endpoints": [-e, e],
                            "agrees": True,
                        }
                        for e in (0.01, 0.003, 0.001)
                    ],
                }
            )
        self.put(self.root / "native_gradient_checks/report.json", self.raw)
        self.summary = {
            "status": "passed",
            "mode": "validate",
            "job_id": ARM.PARENT_JOB,
            "config": json.loads(self.config.read_text()),
            "config_sha256": ARM.sha(self.config),
            "source": {"files": self.files},
            "native_identity": self.native,
            "frozen_identity_after": {
                "status": "passed",
                "model_digest": ARM.MODEL_DIGEST,
                "context_digest": ARM.CONTEXT_DIGEST,
            },
            "context_sha256": self.context_sha,
            "reference_sha256": self.reference_sha,
            "native_hard_parity": {
                "status": "passed",
                "atol": 2e-5,
                "native": [1.0, 0.2, -0.3],
                "attached": [1.0, 0.2, -0.3],
            },
            "native_gradient_checks": self.raw,
            "chemistry_assay": {"status": "passed", "loss": 0.5, "gradient_norm": 0.1},
            "runtime_gate": {
                "mean_update_seconds": [1.0, 2.0],
                "allocation_seconds": 82800,
                "conservative_projected_seconds": 31800.0,
                "status": "passed",
            },
            "initialization_sha256": ARM.sha(self.root / "initialization.npz"),
        }
        objective = {
            **self.native,
            "coefficient": 1.0,
            "context_sha256": self.context_sha,
        }
        for phase, weight in (("early", 0.001), ("high", 50.0)):
            self.put(
                self.root / (phase + "_coupled_checks/report.json"),
                {
                    "status": "passed",
                    "repair_weight": weight,
                    "affinity_objective_identity": objective,
                    "preparation_identity": phase + "-fixed-target",
                    "directions": [
                        {
                            "field": field,
                            "analytic": 1.0,
                            "probes": [
                                {
                                    "epsilon": e,
                                    "analytic": 1.0,
                                    "finite_difference": 1.0,
                                    "agrees": True,
                                    "preparation_identity": phase + "-fixed-target",
                                }
                                for e in (0.01, 0.003, 0.001)
                            ],
                        }
                        for field in ("atoms", "bonds")
                    ],
                },
            )
            pilot_root = self.root / (phase + "_pilot")
            pilot_root.mkdir()
            states = []
            for step in range(11):
                snapshot = pilot_root / f"step-{step:04d}.npz"
                snapshot.write_bytes(f"opaque snapshot {step}".encode())
                states.append(
                    {
                        "step": step,
                        "snapshot": snapshot.name,
                        "snapshot_sha256": ARM.sha(snapshot),
                        "repair_weight": weight,
                    }
                )
            self.put(
                pilot_root / "report.json",
                {
                    "status": "passed",
                    "requested_updates": 10,
                    "acceptance_policy": "best_primal",
                    "affinity_objective_identity": objective,
                    "states": states,
                    "updates": [{"step": i} for i in range(10)],
                },
            )
        self.random = {
            "status": "failed",
            "job_id": ARM.PARENT_JOB,
            "error": "Native E_upper FD failed",
        }
        self.put(self.run / "random/summary.json", self.random)
        cpu_proof = self.cpu / "tests.xml"
        cpu_proof.write_text("original complete CPU proof", encoding="utf-8")
        self.cpu_assessment = {
            "status": "passed",
            "passed": 299,
            "source_manifest_sha256": self.manifest,
            "reference_sha256": self.reference_sha,
            "proofs": [{"path": str(cpu_proof), "sha256": ARM.sha(cpu_proof)}],
        }
        self.put(self.cpu / "assessment.json", self.cpu_assessment)
        self.scheduler = f"{ARM.PARENT_JOB} EXIT gpul40s\n"
        self.cpu_final = self.cpu.parent / "scheduler-final.txt"
        self.cpu_final.write_text(
            f"Job <{ARM.CPU_JOB}>, Status <DONE>\nDone successfully.\n",
            encoding="utf-8",
        )
        self.cpu_stdout = self.cpu.parent / f"stdout.{ARM.CPU_JOB}.log"
        self.cpu_stdout.write_text(
            f"Subject: Job {ARM.CPU_JOB}: <fixture> Done\nSuccessfully completed.\n",
            encoding="utf-8",
        )
        self.cpu_final_sha, self.cpu_stdout_sha = (
            ARM.sha(self.cpu_final),
            ARM.sha(self.cpu_stdout),
        )

    def invoke(self, arm="mix50", environment=True):
        self.put(self.root / "summary.json", self.summary)
        with (
            mock.patch.object(ARM, "SOURCE_MANIFEST", self.manifest),
            mock.patch.object(ARM, "CONTEXT_SHA", self.context_sha),
            mock.patch.object(ARM, "REFERENCE_SHA", self.reference_sha),
            mock.patch.object(ARM, "SUMMARY_SHA", ARM.sha(self.root / "summary.json")),
            mock.patch.object(ARM, "CPU_FINAL_SHA", self.cpu_final_sha),
            mock.patch.object(ARM, "CPU_STDOUT_SHA", self.cpu_stdout_sha),
            mock.patch.dict(
                os.environ,
                {"LSB_JOBID": "adoption-cpu"} if environment else {},
                clear=True,
            ),
            mock.patch.object(
                ARM.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, self.scheduler, ""),
            ),
        ):
            return ARM.adopt(
                self.release, self.run, self.cpu, self.manifest, arm, self.output
            )

    def reject(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_adopts_only_selected_pass_and_keeps_failed_parent_and_random(self):
        original_random = (self.run / "random/summary.json").read_bytes()
        result = self.invoke()
        self.assertEqual(result["certified_arms"], ["mix50"])
        self.assertEqual(result["parent_scheduler_outcome"], "EXIT")
        self.assertEqual(result["parent_global_readiness_status"], "failed")
        self.assertEqual(list(result["configs"]), [self.config.name])
        self.assertEqual(result["uncertified_arms"], {"random": "failed"})
        self.assertEqual(
            (self.run / "random/summary.json").read_bytes(), original_random
        )
        self.assertEqual(
            (self.output / "prepared/context.pt").read_bytes(),
            (self.run / "prepared/context.pt").read_bytes(),
        )
        for proof in result["proofs"]:
            self.assertEqual(ARM.sha(proof["path"]), proof["sha256"])
        self.assertIn(
            str(self.root / "initialization.npz"), [p["path"] for p in result["proofs"]]
        )

    def test_random_cannot_be_certified(self):
        with self.assertRaisesRegex(ValueError, "random is not certified"):
            self.invoke(arm="random")
        self.assertFalse(self.output.exists())

    def test_failed_selected_arm_cannot_be_adopted(self):
        self.summary["status"] = "failed"
        self.reject("Selected arm did not pass")

    def test_changed_config_or_execution_source_rejected(self):
        self.summary["config_sha256"] = "changed"
        self.reject("configuration changed")
        self.summary["config_sha256"] = ARM.sha(self.config)
        self.summary["source"] = {"files": {}}
        self.reject("execution source differs")

    def test_changed_native_model_or_context_rejected(self):
        self.summary["native_identity"] = {**self.native, "model_digest": "other"}
        self.reject("Native model/context")
        self.summary["native_identity"] = self.native
        (self.run / "prepared/context.pt").write_bytes(b"different context")
        self.reject("Prepared context identity")

    def test_frozen_identity_or_runtime_rejected(self):
        self.summary["frozen_identity_after"]["context_digest"] = "changed"
        self.reject("Frozen model/context")
        self.summary["frozen_identity_after"]["context_digest"] = ARM.CONTEXT_DIGEST
        self.summary["runtime_gate"]["conservative_projected_seconds"] = 1.0
        self.reject("Runtime fit gate")

    def test_missing_or_failed_coupled_pilot_rejected(self):
        path = self.root / "high_coupled_checks/report.json"
        report = json.loads(path.read_text())
        report["status"] = "failed"
        self.put(path, report)
        self.reject("Coupled chemistry gate")
        report["status"] = "passed"
        self.put(path, report)
        path = self.root / "high_pilot/report.json"
        report = json.loads(path.read_text())
        report["states"].pop()
        self.put(path, report)
        self.reject("Incomplete pilot")

    def test_changed_initialization_or_cpu_proof_rejected(self):
        (self.root / "initialization.npz").write_bytes(b"changed")
        self.reject("Initialization bytes")
        self.summary["initialization_sha256"] = ARM.sha(
            self.root / "initialization.npz"
        )
        (self.cpu / "tests.xml").write_text("changed CPU result")
        self.reject("CPU test evidence changed")

    def test_native_fd_cannot_be_waived_by_passed_label(self):
        changed = copy.deepcopy(self.raw)
        changed["rows"][1]["probes"][1]["agrees"] = False
        self.summary["native_gradient_checks"] = changed
        self.put(self.root / "native_gradient_checks/report.json", changed)
        self.reject("derivative agreement is inconsistent")

    def test_coupled_fd_keeps_the_same_target_and_agreement_gate(self):
        path = self.root / "early_coupled_checks/report.json"
        report = json.loads(path.read_text())
        report["directions"][1]["probes"][1]["preparation_identity"] = "different-target"
        self.put(path, report)
        self.reject("target/analytic changed within probes")

    def test_native_and_cpu_actual_scheduler_states_required(self):
        self.scheduler = f"{ARM.PARENT_JOB} RUN gpul40s\n"
        self.reject("actual native EXIT")
        self.scheduler = f"{ARM.PARENT_JOB} EXIT gpul40s\n"
        self.cpu_final.write_text("different final CPU outcome", encoding="utf-8")
        self.reject("CPU final scheduler/LSF evidence changed")

    def test_duplicate_output_is_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.invoke()

    def test_allocation_required_even_for_adoption(self):
        with self.assertRaisesRegex(ValueError, "allocated DTU HPC"):
            self.invoke(environment=False)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
