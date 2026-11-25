"""Focused checks for numerical recovery in the Stage 2 valence projection."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from mosaic.small_molecule.Stage2 import operators


class ProjectionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.atoms = np.zeros((8, 6), dtype=np.float64)
        self.atoms[:, 0] = 1
        self.edges = np.zeros((28, 5), dtype=np.float64)
        self.edges[:, 1] = 1

    @staticmethod
    def premature(*args, **kwargs):
        return OptimizeResult(x=np.zeros(8), nit=0, success=True,
                              message="simulated premature convergence")

    def test_retry_finds_feasible_projection(self):
        real_minimize = operators.minimize
        calls = []

        def minimize(*args, **kwargs):
            calls.append(kwargs["method"])
            return self.premature() if len(calls) == 1 else real_minimize(*args, **kwargs)

        with patch.object(operators, "minimize", side_effect=minimize):
            edges, note = operators.project_valence(self.atoms, self.edges)
        np.testing.assert_allclose(edges.sum(axis=1), 1, atol=1e-10)
        self.assertGreaterEqual(edges.min(), -1e-10)
        self.assertLessEqual(note["kkt_excess"], 2e-6)
        self.assertLessEqual(note["complementarity"], 2e-6)
        self.assertFalse(note["kkt_warning"])
        self.assertGreaterEqual(len(calls), 2)

    def test_unresolved_residual_is_recorded(self):
        with patch.object(operators, "minimize", side_effect=self.premature):
            edges, note = operators.project_valence(self.atoms, self.edges)
        self.assertTrue(np.isfinite(edges).all())
        self.assertTrue(note["kkt_warning"])
        self.assertGreater(note["kkt_excess"], 2e-6)

    def test_nonfinite_dual_is_rejected(self):
        bad = OptimizeResult(x=np.full(8, np.nan), nit=0, success=False,
                             message="nonfinite")
        with patch.object(operators, "minimize", return_value=bad):
            with self.assertRaises(FloatingPointError):
                operators.project_valence(self.atoms, self.edges)


if __name__ == "__main__":
    unittest.main()
