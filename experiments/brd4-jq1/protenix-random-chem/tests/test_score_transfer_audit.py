"""Portable metadata controls: no model, numerical framework or GPU imports."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_transfer_audit.py"
spec = importlib.util.spec_from_file_location("score_transfer_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def seal(repair):
    receipt = {k: v for k, v in repair.items() if k not in audit.REFERENCE_EXTRAS}
    repair["identity"] = hashlib.sha256(json.dumps(
        {"arrays": repair["arrays"], "receipt": receipt}, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    return {"repair": repair, "neighbor": None}


def reference(route=1):
    return seal({"attempted_rows": 1, "source_update": 120,
                 "rows": [{"route": route}], "schema_sha256": "test-schema",
                 "preparation_seconds": 0.2, "search_seconds": [],
                 "arrays": {"route": [route], "target_atoms": [[1, 2, 6]],
                            "target_charges": [[0, 0, 0]], "active": [[True, True, True]],
                            "target_bonds": [[[0, 1, 0], [1, 0, 2], [0, 2, 0]]]}})


def plan():
    return {"version": audit.VERSION, "points": [
        {"id": "arm-120", "arm": "arm", "step": 120, "temperatures": [0.7, 0.4]}]}


class SavedTeacherTests(unittest.TestCase):
    def test_candidate_producer_report_and_raw_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples.npz"
            samples.write_bytes(b"opaque-existing-array-bytes-not-loaded-locally")
            raw = audit.binding(samples)
            report = root / "report.json"
            report.write_text(json.dumps({"last_sampled_validity": {"raw_samples": raw, "draws": 128,
                                                                     "strict_valid_connected": 5}}))
            report_binding = audit.binding(report)
            frozen = {"model_digest": "model", "context_digest": "context"}
            result = root / "result.json"
            result.write_text(json.dumps({"job_id": "123", "status": "completed_paired_exploratory_refinement",
                "frozen_unchanged": True, "inputs_unchanged": True, "frozen_before": frozen,
                "frozen_after": frozen, "arms": [{"report": report_binding}]}))
            group = {"id": "saved", "source_job_id": "123", "expected_draws": 128,
                     "raw_samples": raw, "source_report": report_binding, "source_result": audit.binding(result)}
            self.assertEqual(audit.candidate_source(group, frozen)["strict_valid_connected"], 5)
            wrong = copy.deepcopy(group)
            wrong["source_job_id"] = "124"
            with self.assertRaisesRegex(ValueError, "producer"):
                audit.candidate_source(wrong, frozen)
            with self.assertRaisesRegex(ValueError, "scorer"):
                audit.candidate_source(group, {"model_digest": "other", "context_digest": "context"})
            wrong = copy.deepcopy(group)
            wrong["raw_samples"]["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "sample bytes"):
                audit.candidate_source(wrong, frozen)
            samples.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Input bytes"):
                audit.candidate_source(group, frozen)

    def test_candidate_budget_and_all_draw_denominators_must_be_declared(self):
        value = plan()
        group = {"id": "one", "expected_draws": 128, "source_job_id": "123"}
        value.update(candidate_samples=[group], maximum_unique_samples=64)
        self.assertIs(audit.validate_plan(value), value)
        for field, bad in (("maximum_unique_samples", 0), ("maximum_unique_samples", 129),
                           ("candidate_samples", [group, group])):
            changed = copy.deepcopy(value)
            changed[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate_plan(changed)
        value["candidate_samples"][0]["expected_draws"] = 100
        with self.assertRaisesRegex(ValueError, "unfiltered"):
            audit.validate_plan(value)

    def test_complete_target_maps_absent_offset_and_unordered_pairs(self):
        result = audit.teacher_labels(reference(), step=120, n_atoms=3)
        self.assertEqual(result["candidate"], {"element_indices": [0, 1, 5], "bond_upper_indices": [1, 0, 2]})
        self.assertEqual(result["status"], "saved_complete_repair")

    def test_already_valid_remains_explicit_raw_candidate(self):
        result = audit.teacher_labels(reference(0), step=120, n_atoms=3)
        self.assertEqual(result["status"], "saved_raw_already_valid")

    def test_partial_and_absent_witness_never_become_valid_candidates(self):
        for route in (3, 4):
            with self.subTest(route=route):
                result = audit.teacher_labels(reference(route), step=120, n_atoms=3)
                self.assertIsNone(result["candidate"])

    def test_target_tampering_rejected_before_mapping(self):
        value = reference()
        value["repair"]["arrays"]["target_atoms"][0][0] = 2
        with self.assertRaisesRegex(ValueError, "identity"):
            audit.teacher_labels(value, step=120, n_atoms=3)

    def test_reference_from_another_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different saved point"):
            audit.teacher_labels(reference(), step=121, n_atoms=3)

    def test_unsupported_charge_and_occupancy_are_rejected(self):
        for field, value in (("target_charges", [[0, 1, 0]]),
                             ("target_atoms", [[1, 0, 6]]),
                             ("active", [[True, False, True]])):
            with self.subTest(field=field):
                current = reference()["repair"]
                current["arrays"][field] = value
                with self.assertRaises(ValueError):
                    audit.teacher_labels(seal(current), step=120, n_atoms=3)

    def test_asymmetric_and_diagonal_bonds_are_rejected(self):
        for i, j, value in ((0, 1, 2), (0, 0, 1), (0, 2, 9)):
            current = reference()["repair"]
            current["arrays"]["target_bonds"][0][i][j] = value
            with self.subTest(i=i, j=j, value=value), self.assertRaisesRegex(ValueError, "bond labels"):
                audit.teacher_labels(seal(current), step=120, n_atoms=3)


class MetadataControls(unittest.TestCase):
    def test_temperature_preserves_argmax_including_first_tie(self):
        for row in ([2.0, -3.0, 0.0], [-100.0, -99.0, -99.0], [0.0, 0.0, 0.0]):
            expected = max(range(len(row)), key=row.__getitem__)
            for temperature in (0.4, 0.7, 1.0, 2.0):
                self.assertEqual(audit.positive_temperature_argmax(row, temperature), expected)

    def test_zero_negative_and_nonfinite_temperature_rejected(self):
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                audit.positive_temperature_argmax([1., 2.], temperature)

    def test_input_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_bytes(b"original")
            record = audit.binding(path)
            self.assertEqual(audit.bound(record), path)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Input bytes differ"):
                audit.bound(record)

    def test_duplicate_ids_and_unbounded_requests_rejected(self):
        value = plan()
        self.assertIs(audit.validate_plan(value), value)
        invalid = copy.deepcopy(value)
        invalid["points"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.validate_plan(invalid)
        invalid = copy.deepcopy(value)
        invalid["optimizer_updates"] = 1
        with self.assertRaisesRegex(ValueError, "must not request updates"):
            audit.validate_plan(invalid)
        invalid = copy.deepcopy(value)
        invalid["points"][0]["temperatures"] = [0.]
        with self.assertRaisesRegex(ValueError, "Temperatures"):
            audit.validate_plan(invalid)


if __name__ == "__main__":
    unittest.main()
