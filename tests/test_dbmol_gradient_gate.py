"""HPC-only orchestration fixtures; no native pretrained-model evidence."""

from dataclasses import replace

import pytest
import torch

from latentmol.boltz.active_set import ContextBank, make_branch, select_slots, validate_branch
from latentmol.boltz.dbmol_objective import DBMolObjectiveConfig
from latentmol.multimodel.config import Objective
from latentmol.multimodel.engine import Counters
from test_boltz_active_set_runtime import SPEC, _config, _Decoder, _Scorer, _WrongDerivative


class CompositeFixture(_Scorer):
    """One smooth non-native feature drives three attached scalar components."""

    def __init__(self, profile, wrong_contact=False):
        super().__init__()
        self.manifest_config = profile.record()
        self.wrong_contact = wrong_contact

    def forward(self, features, types, coordinates):
        x = super().forward(features, types, coordinates)["affinity_pred_value"]
        contact = _WrongDerivative.apply(x * 2) if self.wrong_contact else x * 2
        confidence = x * 0 + .3  # Attached but stationary; report uninformative.
        return {"dbmol_affinity_loss": x, "dbmol_contact_loss": contact,
                "dbmol_confidence_loss": confidence,
                "dbmol_total_loss": x + 2.5 * contact + 2.5 * confidence}


@pytest.mark.parametrize("wrong_contact", [False, True])
def test_component_and_total_gate_rejects_wrong_backward_and_preserves_stationary_term(tmp_path, wrong_contact):
    profile = DBMolObjectiveConfig.from_mapping({
        "confidence_checkpoint": {"path": "conf.ckpt", "sha256": "a" * 64},
        "pocket_residues": [{"asym_id": 0, "residue_index": 0}]}, tmp_path)
    cfg = replace(_config(tmp_path), dbmol=profile, fd_policy="local_v1",
                  objective=Objective("dbmol_total_loss", direction="minimize"))
    decoder = _Decoder()
    point = torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(point), temperature=1., threshold=.5)
    branch = make_branch(CompositeFixture(profile, wrong_contact), ContextBank(cfg, SPEC), initial,
                         cfg, "c" * 64)
    path = tmp_path / "validation.json"
    if wrong_contact:
        with pytest.raises(ValueError, match="component finite-difference gate failed"):
            validate_branch(decoder, point, branch, cfg, Counters(), report_path=path)
        import json
        report = json.loads(path.read_text())
        assert report["passed"] is False
        assert report["component_gradients"]["dbmol_contact_loss"]["passed"] is False
    else:
        report = validate_branch(decoder, point, branch, cfg, Counters(), report_path=path)
        assert report["passed"] is True
        assert report["component_gradients"]["dbmol_affinity_loss"]["status"] == "validated_nonzero"
        assert report["component_gradients"]["dbmol_contact_loss"]["status"] == "validated_nonzero"
        assert report["component_gradients"]["dbmol_confidence_loss"]["status"] == "zero_at_current_point_uninformative"
        assert report["local_fd_validation"]["passed"] is True
