import json
from types import SimpleNamespace

import numpy as np
import pytest

from combined_search import BindingBackend, FreshReadouts, calibrate_search
from repair_frontier import ParentFrontier
from repair_search import project, run_arm
from repair_signal import evaluate_repair
from study import PAIR_I, PAIR_J, hard_smiles


class ElementScorer:
    def __init__(self, scale):
        self.scale = scale
        self.fresh_calls = []

    def gradient(self, atoms, bonds):
        da = np.zeros_like(atoms)
        da[0, 1] = -self.scale
        return -float(atoms[0, 1]) * self.scale, da, np.zeros_like(bonds)

    def hard_value(self, atoms, bonds):
        return -float(atoms[0, 1]) * self.scale

    def affinity_confidence_gradients(self, atoms, bonds):
        da = np.zeros_like(atoms)
        da[0, 1] = -.5
        confidence = 2 - float(atoms[0, 1]) * .5
        return self.gradient(atoms, bonds), (confidence, da, np.zeros_like(bonds))

    def weighted_affinity_confidence_gradient(self, atoms, bonds, aw, cw):
        (a, aa, ab), (c, ca, cb) = self.affinity_confidence_gradients(atoms, bonds)
        return a, c, aw*aa + cw*ca, aw*ab + cw*cb

    def fresh(self, smiles):
        self.fresh_calls.append(smiles)
        return dict(status="scored", has_interface=True, affinity_pred_value=-float("N" in smiles),
                    affinity_probability_binary=.8 if "N" in smiles else .2)


@pytest.mark.parametrize("policy", ["fixed", "adaptive", "protenix", "confidence"])
def test_search_uses_same_graph_and_reports_hard_readouts(tmp_path, policy):
    p, n = ElementScorer(1), ElementScorer(4)
    cfg = dict(learning_rate=60., max_step_norm=.5, probability_floor=.0001, calibration_steps=[0, 1],
               repair_weights=[2.], founder_updates=2, child_updates=1, child_generations=1, parents=1,
               children_per_parent=2, search_versions={policy: {"vina_weight": 0}}, compare_repair_weights=False,
               restart_mass=.002, restart_mass_after_merge=.01)
    atoms = project(np.eye(6)[np.zeros(25, int)], .0001)
    atoms[0] = project(np.array([[.53, .46, .003, .003, .002, .002]]), .0001)[0]
    bonds = project(np.eye(5)[(PAIR_J == PAIR_I + 1).astype(int)], .0001)
    start = atoms, bonds
    oracle = SimpleNamespace(evaluate=evaluate_repair)
    if policy == "confidence":
        cfg["search_versions"][policy]["gradient_shares"] = dict(protenix=.35, affinity=.35, confidence=.30)
    calibration = calibrate_search(p, n, oracle, start, cfg, policy)
    fitted = calibration["coefficients"]
    if policy == "confidence":
        assert (fitted["protenix_weight"], fitted["nesso_weight"], fitted["confidence_weight"]) == pytest.approx((.35, .0875, .6))
        assert calibration["weighted_backward_check"]["absolute_gradient_l2_error"] == pytest.approx(0)
    else:
        assert (fitted["protenix_weight"], fitted["nesso_weight"]) == pytest.approx((.5, .125))
    backend = BindingBackend(p, n, policy, fitted, cfg)
    archive = FreshReadouts(n, hard_smiles(start), tmp_path)
    run_arm(backend, oracle, ParentFrontier(p), start, cfg, 2., 123, policy, tmp_path / "search",
            score_archive=archive.record)
    assert archive.last_summary["improved_both"] == 1
    assert archive.last_summary["improved_all_three"] == 1
    assert len(n.fresh_calls) == len(set(n.fresh_calls)) == 2
    first = json.loads((tmp_path/"search/generation-00/child-00/trajectory.jsonl").read_text().splitlines()[0])
    assert first["binding"]["protenix_contact_loss"] == first["relaxed_contact_loss"]
    assert first["binding"]["combined_to_protenix"] == pytest.approx(1)
    if policy == "adaptive":
        assert first["stepwise_weighted_loss"] == pytest.approx(first["binding"]["weighted_model_value"])
        assert "total_loss" not in first
    else:
        assert first["total_loss"] == pytest.approx(first["binding"]["weighted_model_value"])
    if policy == "confidence":
        second = json.loads((tmp_path/"search/generation-00/child-00/trajectory.jsonl").read_text().splitlines()[1])
        assert not second["binding"]["measured_component_gradients"]
        assert second["binding"]["weighted_model_value"] == pytest.approx(
            .35*second["relaxed_contact_loss"] + .0875*second["binding"]["nesso_fixed_crop_affinity"]
            + .6*second["binding"]["nesso_binder_loss"])
