"""Allocated CPU metadata contracts; no numerical/scientific imports or execution."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "scripts/softmax_replay_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("softmax_replay_metadata", MODULE)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def plan_fixture():
    points = []
    for variant in ("baseline", "combined"):
        for label in diagnostic.POINT_LABELS:
            paths = diagnostic.point_paths(variant, label)
            points.append({"variant": variant, "label": label,
                **{key: {"path": str(path), "bytes": 17, "sha256": "a" * 64}
                   for key, path in zip(("snapshot", "replay"), paths)}})
    return {"version": diagnostic.VERSION,
        "scientific_source": {"path": str(diagnostic.SCIENCE),
                              "manifest_sha256": diagnostic.SCIENCE_PIN, "files": 643},
        "failed_pilot_source": {"path": str(diagnostic.FAILED_SOURCE),
                                "manifest_sha256": diagnostic.FAILED_PIN, "files": 30},
        "points": points}


class SoftmaxReplayMetadataTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.environ.get("LSB_JOBID", "").isdigit(), "Fixtures require actual allocated HPC")
        self.assertIn(os.environ.get("LSB_QUEUE"), ("hpc", "gpul40s"))
        self.temporary = tempfile.TemporaryDirectory(prefix="mosaic-replay-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def file_binding(self, name="input.bin", raw=b"original bytes"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def source_fixture(self):
        root = self.root / "source"
        root.mkdir()
        (root / "original.py").write_bytes(b"original = True\n")
        manifest = {"files": {"original.py": diagnostic.sha(root / "original.py")}}
        diagnostic.write(root / "source_manifest.json", manifest)
        return {"path": str(root), "manifest_sha256": diagnostic.sha(root / "source_manifest.json"), "files": 1}

    def test_01_complete_ordered_plan_is_accepted(self):
        self.assertEqual(diagnostic.validate_plan(plan_fixture()), plan_fixture())

    def test_02_wrong_diagnostic_version_is_rejected(self):
        value = plan_fixture(); value["version"] = "other"
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_03_extra_plan_fields_are_rejected(self):
        value = plan_fixture(); value["can_restart"] = True
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_04_changed_original_source_pin_is_rejected(self):
        value = plan_fixture(); value["scientific_source"]["manifest_sha256"] = "b" * 64
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_05_changed_failed_source_count_is_rejected(self):
        value = plan_fixture(); value["failed_pilot_source"]["files"] = 29
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_06_missing_boundary_is_rejected(self):
        value = plan_fixture(); value["points"].pop()
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_07_reordered_boundaries_are_rejected(self):
        value = plan_fixture(); value["points"].reverse()
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_08_duplicate_boundary_is_rejected(self):
        value = plan_fixture(); value["points"][1] = deepcopy(value["points"][0])
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_09_redirected_snapshot_is_rejected(self):
        value = plan_fixture(); value["points"][0]["snapshot"]["path"] = "/tmp/other.npz"
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_10_redirected_replay_receipt_is_rejected(self):
        value = plan_fixture(); value["points"][0]["replay"]["path"] = "/tmp/other.json"
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_11_extra_point_fields_are_rejected(self):
        value = plan_fixture(); value["points"][0]["tolerance"] = 1e-6
        with self.assertRaises(ValueError): diagnostic.validate_plan(value)

    def test_12_undeclared_point_identity_is_rejected(self):
        with self.assertRaises(ValueError): diagnostic.point_paths("baseline", "early-9")
        with self.assertRaises(ValueError): diagnostic.point_paths("replacement", "early-0")

    def test_13_regular_input_binding_passes(self):
        row = self.file_binding()
        self.assertEqual(diagnostic.check_binding(row), row)

    def test_14_changed_input_bytes_fail(self):
        row = self.file_binding(); Path(row["path"]).write_bytes(b"changed! bytes")
        with self.assertRaises(ValueError): diagnostic.check_binding(row)

    def test_15_changed_length_fails(self):
        row = self.file_binding(); row["bytes"] += 1
        with self.assertRaises(ValueError): diagnostic.check_binding(row)

    def test_16_invalid_digest_fails(self):
        row = self.file_binding(); row["sha256"] = "G" * 64
        with self.assertRaises(ValueError): diagnostic.binding_schema(row)

    def test_17_boolean_or_nonpositive_lengths_fail(self):
        for size in (True, False, 0, -1, 1.5):
            row = self.file_binding("size-" + str(size)); row["bytes"] = size
            with self.assertRaises(ValueError): diagnostic.binding_schema(row)

    def test_18_relative_or_parent_paths_fail(self):
        for path in ("relative.bin", "/tmp/../other.bin", "/tmp/other\\file.bin"):
            with self.assertRaises(ValueError):
                diagnostic.binding_schema({"path": path, "bytes": 1, "sha256": "a" * 64})

    def test_19_symlinked_input_fails(self):
        row = self.file_binding(); linked = self.root / "linked.bin"
        linked.symlink_to(row["path"]); row["path"] = str(linked)
        with self.assertRaises(ValueError): diagnostic.check_binding(row)

    def test_20_symlinked_input_parent_fails(self):
        row = self.file_binding("original/file.bin"); link = self.root / "linked"
        link.symlink_to(self.root / "original", target_is_directory=True)
        row["path"] = str(link / "file.bin")
        with self.assertRaises(ValueError): diagnostic.check_binding(row)

    def test_21_exact_source_membership_passes(self):
        row = self.source_fixture()
        self.assertEqual(diagnostic.verify_source(row), row)

    def test_22_changed_source_bytes_fail(self):
        row = self.source_fixture(); (Path(row["path"]) / "original.py").write_bytes(b"changed = True\n")
        with self.assertRaises(ValueError): diagnostic.verify_source(row)

    def test_23_extra_source_member_fails(self):
        row = self.source_fixture(); (Path(row["path"]) / "extra.py").write_bytes(b"extra = True\n")
        with self.assertRaises(ValueError): diagnostic.verify_source(row)

    def test_24_source_count_or_manifest_pin_changes_fail(self):
        row = self.source_fixture(); changed = deepcopy(row); changed["files"] += 1
        with self.assertRaises(ValueError): diagnostic.verify_source(changed)
        changed = deepcopy(row); changed["manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError): diagnostic.verify_source(changed)

    def test_25_output_count_and_link_rejection(self):
        row = self.file_binding(); self.assertEqual(diagnostic.output_bytes(self.root), row["bytes"])
        linked = self.root / "symlink.bin"; linked.symlink_to(row["path"])
        with self.assertRaises(ValueError): diagnostic.output_bytes(self.root)

    def test_26_output_root_link_rejected(self):
        folder = self.root / "output"; folder.mkdir()
        linked = self.root / "linked"; linked.symlink_to(folder, target_is_directory=True)
        with self.assertRaises(ValueError): diagnostic.output_bytes(linked)

    def test_27_json_is_finite_and_fresh_only(self):
        path = self.root / "receipt.json"; diagnostic.write(path, {"status": "diagnostic_only"})
        self.assertEqual(diagnostic.read(path), {"status": "diagnostic_only"})
        with self.assertRaises(FileExistsError): diagnostic.write(path, {})
        nonfinite = self.root / "nonfinite.json"; nonfinite.write_text('{"value":NaN}', encoding="utf8")
        with self.assertRaises(ValueError): diagnostic.read(nonfinite)

    def test_28_allocation_rejects_non_gpu_or_wrong_slot_count(self):
        valid = {"LSB_JOBID": "123", "LSB_QUEUE": "gpul40s", "LSB_DJOB_NUMPROC": "8"}
        diagnostic.allocation_guard(valid)
        for key, value in (("LSB_JOBID", "0"), ("LSB_QUEUE", "hpc"), ("LSB_DJOB_NUMPROC", "4")):
            changed = dict(valid); changed[key] = value
            with self.assertRaises(ValueError): diagnostic.allocation_guard(changed)

    def test_29_cli_rejects_unpaired_private_controls(self):
        common = ["--plan", "/tmp/plan.json", "--plan-sha256", "a" * 64, "--output", "/tmp/output"]
        with self.assertRaises(ValueError): diagnostic.parse_args(common + ["--control", "plain"])
        with self.assertRaises(ValueError): diagnostic.parse_args(common + ["--repeat", "0"])
        args = diagnostic.parse_args(common + ["--control", "combined", "--repeat", "1"])
        self.assertEqual((args.control, args.repeat), ("combined", 1))
        complete = [{"control": control, "repeat": repeat, "returncode": 0,
            "status": "diagnostic_control_completed_original_failure_preserved", "assessment_sha256": "a" * 64}
            for control in diagnostic.CONTROLS for repeat in (0, 1)]
        self.assertTrue(diagnostic.validate_completed_controls(complete))
        for failure in ("returncode", "status", "assessment_sha256"):
            incomplete = deepcopy(complete); incomplete[0][failure] = 1 if failure == "returncode" else "failed"
            with self.assertRaises(ValueError): diagnostic.validate_completed_controls(incomplete)
        for incomplete in (complete[:-1], complete[::-1], []):
            with self.assertRaises(ValueError): diagnostic.validate_completed_controls(incomplete)

    def test_30_no_model_loss_optimizer_or_checkpoint_calls(self):
        for name in ("load_model", "evaluate_combined", "run_chemical_updates", "forward_trunk"):
            frame = SimpleNamespace(f_code=SimpleNamespace(co_name=name, co_filename="/original/module.py"))
            with self.assertRaises(RuntimeError): diagnostic.forbid_scientific_execution(frame, "call", None)
        frame = SimpleNamespace(f_code=SimpleNamespace(co_name="load", co_filename="/original/mosaic/small_molecule/protenix.py"))
        with self.assertRaises(RuntimeError): diagnostic.forbid_scientific_execution(frame, "call", None)
        for name in ("model.eqx", "context.pt", "model.safetensors", "model.skeleton.pkl"):
            with self.assertRaises(RuntimeError): diagnostic.forbid_weight_open("open", (name, "rb", 0))
        diagnostic.forbid_weight_open("open", ("original-snapshot.npz", "rb", 0))
        self.assertEqual(diagnostic.MAX_BYTES, 64 * 1024**2)
        self.assertEqual(diagnostic.INTERNAL_SECONDS, 600)
        self.assertEqual(len(diagnostic.CONTROLS), 5)
        self.assertEqual(set(diagnostic.FIELD_NAMES), {"atom_logits", "bond_logits", "X", "E_upper", "key_data"})


if __name__ == "__main__": unittest.main()
