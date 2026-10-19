"""Stdlib launch-contract checks; execute only on allocated DTU HPC CPU."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/submit_boltz_arms.py"
SPEC = importlib.util.spec_from_file_location("mosaic_boltz_controller", SCRIPT)
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)


class SubmissionControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.release = self.base / "sealed-source"
        self.ready = self.base / "ready"
        self.ready.mkdir()
        self.proof = self.ready / "proof.json"
        self.proof.write_text('{"status":"passed"}\n')
        self.assessment = {
            "status": "passed",
            "source_manifest_sha256": "authenticated-source-manifest",
            "proofs": [
                {
                    "path": str(self.proof),
                    "sha256": hashlib.sha256(self.proof.read_bytes()).hexdigest(),
                }
            ],
        }
        self.root = self.base / "long-arms"
        self.scheduler_stdout = "DONE gpul40s\n"
        self.commands = []
        self.submitted = 0

    def subprocess_run(self, command, **kwargs):
        self.commands.append(command)
        if command[:4] == ["bjobs", "-noheader", "-o", "stat queue"]:
            self.assertEqual(command[4], "ready-job")
            self.assertTrue(kwargs["check"])
            return subprocess.CompletedProcess(command, 0, self.scheduler_stdout, "")
        if command[0] == "bsub":
            self.submitted += 1
            return subprocess.CompletedProcess(
                command, 0, f"Job <{7000 + self.submitted}> is submitted.\n", ""
            )
        if command[:2] == ["bjobs", "-l"]:
            return subprocess.CompletedProcess(command, 0, "PEND\n", "")
        if str(command[1]).endswith("zhome_quota_preflight.py"):
            return subprocess.CompletedProcess(command, 0, "quota passed\n", "")
        raise AssertionError(f"Unexpected controller command: {command}")

    def invoke(self, queue="gpul40s"):
        (self.ready / "assessment.json").write_text(json.dumps(self.assessment))
        argv = [
            str(SCRIPT),
            "--release",
            str(self.release),
            "--root",
            str(self.root),
            "--cpu",
            str(self.base / "cpu"),
            "--ready",
            str(self.ready),
            "--ready-job",
            "ready-job",
            "--manifest",
            "authenticated-source-manifest",
        ]
        if queue is not None:
            argv.extend(["--queue", queue])
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(os.environ, {"LSB_JOBID": "controller-job"}),
            mock.patch.object(
                CONTROLLER.subprocess, "run", side_effect=self.subprocess_run
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            CONTROLLER.main()

    def assert_no_submissions(self):
        self.assertFalse(any(command[0] == "bsub" for command in self.commands))
        self.assertFalse(self.root.exists())

    def test_l40s_pass_submits_two_independent_unchanged_resource_arms(self):
        self.invoke()
        commands = [command for command in self.commands if command[0] == "bsub"]
        self.assertEqual(len(commands), 2)
        self.assertEqual([command[-1] for command in commands], ["mix50", "random"])
        for command in commands:
            self.assertEqual(command[command.index("-q") + 1], "gpul40s")
            self.assertEqual(command[command.index("-n") + 1], "4")
            self.assertEqual(
                command[command.index("-gpu") + 1], "num=1:mode=exclusive_process"
            )
            self.assertEqual(command[command.index("-W") + 1], "23:00")
            self.assertEqual(
                [
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "-R"
                ],
                ["span[hosts=1]", "rusage[mem=8GB]"],
            )
            self.assertNotIn("-w", command)
            self.assertNotIn("-hold", command)
            self.assertIn(str(self.release / "scripts/hpc/brd4_boltz_run.lsf"), command)
        quota = [
            command
            for command in self.commands
            if len(command) > 1 and str(command[1]).endswith("zhome_quota_preflight.py")
        ]
        self.assertEqual(len(quota), 2)
        for command in quota:
            self.assertEqual(command[-2:], ["--minimum-free-gib", "8"])
        saved = json.loads((self.root / "submissions.json").read_text())
        self.assertEqual(saved["status"], "submitted")
        self.assertEqual(saved["launch_queue"], "gpul40s")
        self.assertEqual(
            saved["controller_script_sha256"],
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        )
        self.assertEqual(saved["arms"]["mix50"]["job_id"], "7001")
        self.assertEqual(saved["arms"]["random"]["job_id"], "7002")
        for arm in ("mix50", "random"):
            self.assertTrue((self.root / arm / "intent.json").is_file())
            self.assertTrue((self.root / arm / "submission.txt").is_file())
            self.assertTrue((self.root / arm / "scheduler-submitted.txt").is_file())

    def test_default_retains_a10_and_requires_matching_readiness(self):
        self.scheduler_stdout = "DONE gpua10\n"
        self.invoke(queue=None)
        self.assertEqual(self.submitted, 2)
        saved = json.loads((self.root / "submissions.json").read_text())
        self.assertEqual(saved["launch_queue"], "gpua10")
        for command in self.commands:
            if command[0] == "bsub":
                self.assertEqual(command[command.index("-q") + 1], "gpua10")

    def test_different_readiness_queue_prevents_submission(self):
        self.scheduler_stdout = "DONE gpua10\n"
        with self.assertRaisesRegex(ValueError, "queue differs"):
            self.invoke()
        self.assert_no_submissions()

    def test_non_done_readiness_prevents_submission(self):
        for state in ("PEND", "RUN", "EXIT"):
            with self.subTest(state=state):
                self.scheduler_stdout = state + " gpul40s\n"
                with self.assertRaisesRegex(ValueError, "not DONE"):
                    self.invoke()
                self.assert_no_submissions()

    def test_ambiguous_scheduler_response_prevents_submission(self):
        self.scheduler_stdout = "DONE gpul40s\nDONE gpul40s\n"
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.invoke()
        self.assert_no_submissions()

    def test_failed_assessment_prevents_submission(self):
        self.assessment["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "does not certify"):
            self.invoke()
        self.assert_no_submissions()

    def test_wrong_source_prevents_submission(self):
        self.assessment["source_manifest_sha256"] = "other-source"
        with self.assertRaisesRegex(ValueError, "does not certify"):
            self.invoke()
        self.assert_no_submissions()

    def test_changed_proof_prevents_submission(self):
        self.proof.write_text('{"status":"changed"}\n')
        with self.assertRaisesRegex(ValueError, "evidence bytes changed"):
            self.invoke()
        self.assert_no_submissions()


if __name__ == "__main__":
    unittest.main()
