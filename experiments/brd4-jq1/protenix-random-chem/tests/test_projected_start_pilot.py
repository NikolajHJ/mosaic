"""Portable controls for the paired protocol and finite target initialization."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
import unittest


DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("projected_start_pilot", DIRECTORY / "projected_start_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def plan():
    return {"version": pilot.VERSION, "parent_step": 30000, "updates": 2000,
            "temperature": .4, "chemistry_coefficient": 30., "learning_rate": .3,
            "smoothing": .02, "internal_seconds": 4800}


def parent_configuration():
    return {"version": "protenix-chemistry-screen-v1", "updates": 50000,
            "init_seed": 17, "initialization": {"mode": "sparse_random"},
            "temperature": .7, "temperature_schedule": {"initial": .7, "final": .4},
            "chemistry": {"mode": "ramp_hold", "initial": 1., "final": 30.},
            "logit_noise": {"initial_ratio": .4}, "binding": {"mode": "always"},
            "structural_repair": {"version": "connected-backbone-repair-v1", "max_candidates": 6},
            "sampling_interval": 1000, "sampling_draws": 128}


class PairedPilotControls(unittest.TestCase):
    def test_binding_contrast_keeps_initialization_identical_and_changes_only_forward(self):
        declared = plan()
        declared["comparison"] = "binding_forward"
        arms = pilot.arm_protocol(declared)
        self.assertEqual(arms, (("relaxed_forward", "repair", "relaxed"),
                                ("hard_forward", "repair", "hard_forward_soft_backward")))
        self.assertEqual(pilot.fixed_configuration(parent_configuration(), declared),
                         pilot.fixed_configuration(parent_configuration(), plan()))

    def test_historical_projection_contrast_is_retained(self):
        self.assertEqual(pilot.arm_protocol(plan()),
                         (("original_logits", "original", "relaxed"),
                          ("saved_repair", "repair", "relaxed")))

    def test_unknown_comparison_or_forward_mode_rejected_before_scientific_imports(self):
        declared = plan()
        declared["comparison"] = "unknown"
        with self.assertRaisesRegex(ValueError, "causal comparison"):
            pilot.validate_plan(declared)
        with self.assertRaisesRegex(ValueError, "forward mode"):
            pilot.binding_graph(object(), "unknown")

    def test_declared_fixed_protocol_preserves_other_settings(self):
        original = parent_configuration()
        saved = deepcopy(original)
        result = pilot.fixed_configuration(original, plan())
        self.assertEqual(original, saved)
        self.assertEqual(result["updates"], 2000)
        self.assertEqual(result["temperature"], .4)
        self.assertEqual(result["chemistry"], {"mode": "constant", "coefficient": 30.})
        self.assertNotIn("temperature_schedule", result)
        self.assertNotIn("logit_noise", result)
        for key in ("structural_repair", "init_seed", "initialization", "sampling_interval", "sampling_draws"):
            self.assertEqual(result[key], saved[key])

    def test_extra_guidance_cannot_silently_enter_this_contrast(self):
        for key in ("neighbor_environment", "severity_progress"):
            config = parent_configuration()
            config[key] = {"weight": 1.}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "other guidance"):
                pilot.fixed_configuration(config, plan())

    def test_mixture_has_positive_support_and_preserves_target(self):
        for categories in (5, 6):
            for label in range(categories):
                probabilities = pilot.smoothed_row(label, categories)
                self.assertAlmostEqual(sum(probabilities), 1.)
                self.assertTrue(all(p > 0 for p in probabilities))
                self.assertEqual(max(range(categories), key=probabilities.__getitem__), label)
                self.assertAlmostEqual(probabilities[label], .98 + .02/categories)

    def test_temperature_scaled_log_probabilities_represent_requested_mixture(self):
        probabilities = pilot.smoothed_row(3, 6)
        logits = [.4 * math.log(p) for p in probabilities]
        exponentials = [math.exp(z/.4) for z in logits]
        normalized = [v/sum(exponentials) for v in exponentials]
        for expected, actual in zip(probabilities, normalized):
            self.assertAlmostEqual(expected, actual)

    def test_invalid_mixture_and_protocol_are_rejected(self):
        for value in (0., 1., float("nan"), -1.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.smoothed_row(0, 5, smoothing=value)
        for key, value in (("learning_rate", .03), ("temperature", .7), ("parent_step", 20000), ("updates", 0)):
            changed = plan()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.validate_plan(changed)

    def test_confirmation_budget_can_reuse_same_runner_without_math_changes(self):
        value = plan()
        value["updates"] = 20000
        value["internal_seconds"] = 6600
        self.assertIs(pilot.validate_plan(value), value)
        value["updates"] = 20001
        with self.assertRaises(ValueError):
            pilot.validate_plan(value)


if __name__ == "__main__":
    unittest.main()
