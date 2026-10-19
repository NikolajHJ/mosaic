"""Allocated stdlib contracts: diagnostic replay cannot weaken readiness gates."""

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/diagnose_boltz_random_gradient.py"
)
SPEC = importlib.util.spec_from_file_location("boltz_gradient_diagnostic", SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


class BoltzGradientDiagnosticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("LSB_JOBID"):
            raise RuntimeError("These contracts require allocated DTU HPC")

    def endpoints(self, derivatives):
        return {
            epsilon: (-epsilon * derivative, epsilon * derivative)
            for epsilon, derivative in zip(DIAGNOSTIC.ORIGINAL_EPSILONS, derivatives)
        }

    def test_original_failed_native_values_remain_failed(self):
        result = DIAGNOSTIC.original_gate(
            9.497333526611328,
            {
                0.01: (0.8627460598945618, 1.0147287845611572),
                0.003: (0.8750758171081543, 0.9488521814346313),
                0.001: (0.8834429979324341, 0.9028720855712891),
            },
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual([p["agrees"] for p in result["probes"]], [False, False, True])
        self.assertFalse(result["readiness_certificate"])

    def test_nonadjacent_passes_do_not_satisfy_original_gate(self):
        result = DIAGNOSTIC.original_gate(1.0, self.endpoints((1.0, 2.0, 1.0)))
        self.assertEqual(result["status"], "failed")

    def test_adjacent_original_scales_record_observation_without_readiness(self):
        result = DIAGNOSTIC.original_gate(1.0, self.endpoints((2.0, 1.0, 1.0)))
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["readiness_certificate"])
        self.assertEqual(result["relative_tolerance"], 0.15)
        self.assertEqual(result["required_adjacent_scales"], 2)

    def test_extra_scales_cannot_replace_original_failures(self):
        endpoints = self.endpoints((2.0, 2.0, 2.0))
        endpoints.update(
            {epsilon: (-epsilon, epsilon) for epsilon in DIAGNOSTIC.EXTRA_EPSILONS}
        )
        result = DIAGNOSTIC.original_gate(1.0, endpoints)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([p["epsilon"] for p in result["probes"]], [0.01, 0.003, 0.001])

    def test_nonfinite_and_zero_analytic_derivatives_fail(self):
        for derivative in (math.nan, math.inf, 0.0):
            result = DIAGNOSTIC.original_gate(
                derivative, self.endpoints((1.0, 1.0, 1.0))
            )
            self.assertEqual(result["status"], "failed")

    def test_modes_reject_unallocated_execution_before_scientific_imports(self):
        environment = dict(os.environ)
        environment.pop("LSB_JOBID", None)
        for mode, arguments in (
            ("replay", ["--initialization", "/missing"]),
            (
                "diagnose",
                [
                    "--replay",
                    "/missing",
                    "--replay-sha256",
                    "0" * 64,
                    "--context",
                    "/missing",
                    "--checkpoint",
                    "/missing",
                ],
            ),
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    mode,
                    "--scientific-source",
                    "/missing",
                    "--output",
                    "/missing",
                    *arguments,
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("require allocated DTU HPC", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
