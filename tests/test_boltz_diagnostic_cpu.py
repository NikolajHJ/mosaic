"""Allocated stdlib fixtures for persisted diagnostic producer certificates."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "diagnostic_cpu_verifier",
    Path(__file__).resolve().parents[1] / "scripts/check_boltz_diagnostic_cpu.py",
)
CPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CPU)


class PersistedDiagnosticCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("LSB_JOBID"):
            raise RuntimeError("Producer contract tests require allocated DTU HPC")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / "payload"
        self.root.mkdir()
        self.xml = self.root / "tests.xml"
        self.xml.write_text(
            '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
            '<testcase name="one"/><testcase name="two"/></testsuite></testsuites>'
        )
        self.final = self.parent / "scheduler-final.txt"
        self.final.write_text(
            "Job <200>, Job Name <fixture>, Status <DONE>, Queue\n <hpc>\nDone successfully.\n"
        )
        self.stdout = self.parent / "stdout.200.log"
        self.stdout.write_text(
            "Sender: LSF System\nSubject: Job 200: <fixture> in cluster <dcc> Done\n"
            "Terminated at fixed timestamp\nSuccessfully completed.\n"
        )
        self.record = {
            "status": "passed",
            "job_id": "200",
            "tests": 2,
            "failures": 0,
            "errors": 0,
            "skips": 0,
            "tools_manifest_sha256": "a" * 64,
            "scientific_manifest_sha256": "b" * 64,
            "proofs": [{"path": str(self.xml), "sha256": CPU.sha(self.xml)}],
        }
        self.persist()

    def persist(self):
        (self.root / "assessment.json").write_text(json.dumps(self.record))

    def invoke(self, **overrides):
        options = {
            "assessment_sha256": CPU.sha(self.root / "assessment.json"),
            "job_id": "200",
            "tools_manifest_sha256": "a" * 64,
            "scientific_manifest_sha256": "b" * 64,
            "tests": 2,
            "scheduler_sha256": overrides.get("scheduler_sha256")
            or CPU.sha(self.final),
            "stdout_sha256": overrides.get("stdout_sha256") or CPU.sha(self.stdout),
        }
        options.update(overrides)
        return CPU.verify(self.root, **options)

    def test_verified_persisted_done_needs_no_scheduler_command(self):
        result = self.invoke()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["job_id"], "200")
        self.assertEqual(result["tests"], 2)
        self.assertEqual(result["proofs"][0]["sha256"], CPU.sha(self.xml))

    def test_scheduler_word_wrap_does_not_change_the_bound_queue(self):
        self.final.write_text(
            "Job <200>, Status <DONE>, Que\n ue <hpc>\nDone successfully.\n"
        )
        self.assertEqual(self.invoke()["job_id"], "200")

    def test_wrapped_done_and_queue_values_keep_exact_terminal_identity(self):
        self.final.write_text(
            "Job <200>, Job Name <mosaic-boltz-diagnostic-launch-cpu-long-name>, Status <D\n"
            "                          ONE>, Queue <h\n"
            "                          pc>\nDone successfully.\n"
        )
        self.assertEqual(self.invoke()["status"], "passed")

    def test_changed_assessment_pin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "assessment hash"):
            self.invoke(assessment_sha256="0" * 64)

    def test_changed_declared_proof_is_rejected(self):
        self.xml.write_text("changed raw test evidence")
        with self.assertRaisesRegex(ValueError, "declared proof hash"):
            self.invoke()

    def test_source_and_job_bindings_are_not_interchangeable(self):
        for field, value in (
            ("job_id", "201"),
            ("tools_manifest_sha256", "c" * 64),
            ("scientific_manifest_sha256", "d" * 64),
            ("tests", 3),
        ):
            with self.assertRaises(ValueError):
                self.invoke(**{field: value})

    def test_passed_label_cannot_hide_raw_xml_failure(self):
        self.xml.write_text(
            '<testsuites><testsuite tests="2" failures="0">'
            '<testcase name="one"><failure>failed</failure></testcase>'
            '<testcase name="two"/></testsuite></testsuites>'
        )
        self.record["proofs"][0]["sha256"] = CPU.sha(self.xml)
        self.persist()
        with self.assertRaisesRegex(ValueError, "Raw XML contains"):
            self.invoke()

    def test_raw_xml_count_is_checked(self):
        self.xml.write_text(
            '<testsuites><testsuite tests="1"><testcase/></testsuite></testsuites>'
        )
        self.record["proofs"][0]["sha256"] = CPU.sha(self.xml)
        self.persist()
        with self.assertRaisesRegex(ValueError, "testcase count"):
            self.invoke()

    def test_inflated_suite_counter_is_not_accepted_as_additional_testcases(self):
        self.xml.write_text(
            '<testsuites><testsuite tests="13" failures="0" errors="0" skipped="0">'
            '<testcase name="one"/><testcase name="two"/></testsuite></testsuites>'
        )
        self.record["proofs"][0]["sha256"] = CPU.sha(self.xml)
        self.persist()
        with self.assertRaisesRegex(ValueError, "declared count differs"):
            self.invoke()

    def test_raw_suite_failures_cannot_be_hidden_without_case_children(self):
        self.xml.write_text(
            '<testsuites><testsuite tests="2" errors="1"><testcase/><testcase/></testsuite></testsuites>'
        )
        self.record["proofs"][0]["sha256"] = CPU.sha(self.xml)
        self.persist()
        with self.assertRaisesRegex(ValueError, "suite records"):
            self.invoke()

    def test_unsafe_proof_path_is_rejected_even_with_matching_hash(self):
        outside = self.parent / "outside.xml"
        outside.write_bytes(self.xml.read_bytes())
        self.record["proofs"][0]["path"] = str(outside)
        self.persist()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.invoke()

    def test_xml_must_be_an_authenticated_declared_proof(self):
        other = self.root / "other.txt"
        other.write_text("unrelated proof")
        self.record["proofs"] = [{"path": str(other), "sha256": CPU.sha(other)}]
        self.persist()
        with self.assertRaisesRegex(ValueError, "not an authenticated"):
            self.invoke()

    def test_persisted_scheduler_must_match_job_done_and_cpu_queue(self):
        for content in (
            "Job <201>, Status <DONE>, Queue <hpc> Done successfully.",
            "Job <200>, Status <EXIT>, Queue <hpc> Done successfully.",
            "Job <200>, Status <DONE>, Queue <gpul40s> Done successfully.",
        ):
            self.final.write_text(content)
            with self.assertRaisesRegex(ValueError, "scheduler did not certify"):
                self.invoke()

    def test_missing_final_evidence_and_bad_external_final_pin_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "scheduler/LSF hash"):
            self.invoke(scheduler_sha256="0" * 64)
        original_pin = CPU.sha(self.final)
        self.final.unlink()
        with self.assertRaisesRegex(ValueError, "proof missing"):
            self.invoke(scheduler_sha256=original_pin)

    def test_lsf_subject_success_and_absence_of_exit_footer_are_required(self):
        for content in (
            "Subject: Job 201: <fixture> Done\nSuccessfully completed.\n",
            "Subject: Job 200: <fixture> Exited\nSuccessfully completed.\n",
            "Subject: Job 200: <fixture> Done\nExited with exit code 1.\nSuccessfully completed.\n",
            "Subject: Job 200: <fixture> Done\n",
        ):
            self.stdout.write_text(content)
            with self.assertRaisesRegex(ValueError, "LSF output did not certify"):
                self.invoke()


if __name__ == "__main__":
    unittest.main()
