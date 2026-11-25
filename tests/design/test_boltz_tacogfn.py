import json
from types import SimpleNamespace

import pytest
import torch

from latentmol.design import boltz_tacogfn as study
from latentmol.experiments import file_sha256


@pytest.mark.parametrize("fail_close", [False, True])
def test_native_pair_publishes_after_frozen_check(tmp_path, monkeypatch, fail_close):
    def store(name, content):
        path = tmp_path / name
        path.write_text(json.dumps(content))
        return path

    output = tmp_path / "pair"
    weights = {"checkpoint": "aff", "checkpoint_sha256": "aff-hash", "confidence_checkpoint": {"path": "conf"},
               "recycling_steps": 0, "allow_lossy_charge": True}
    assets = store("assets.json", weights)
    prepared = store("native.json", weights | {"dbmol": {"confidence_checkpoint": weights["confidence_checkpoint"]}})
    cfg = dict(prepared_config={"path": str(prepared), "sha256": file_sha256(prepared)},
               tasks={"founder-1": {"seed": 17}}, scorer={"target_id": "brd4-jq1", "pocket_pdb_id": "3MXF",
               "execution": {"tf32": False, "autocast": "none"}})
    config_path = store("config.json", cfg)
    decoder_ckpt = store("decoder.json", {})
    plan = store("plan.json", {"config_sha256": file_sha256(config_path), "arms": {"founder-1": str(output)},
                 "planned_updates": {"founder-1": 1000}, "planned_fresh_attempts": {"founder-1": 12},
                 "planned_native_vina_attempts": {"founder-1": 12}})
    args = SimpleNamespace(config=config_path, plan=plan, output=output, checkpoint=decoder_ckpt,
             arm="founder-1", plan_sha256=file_sha256(plan), checkpoint_sha256=file_sha256(decoder_ckpt),
             boltz_assets=assets)
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "fixture")
    decoder = SimpleNamespace(spec=object())
    monkeypatch.setattr(study, "load_model", lambda *_: (decoder, None))
    pocket = {"pdb_id": "3MXF", "protein_chain": "A"}
    active = torch.ones(31, dtype=torch.bool)
    scorer = SimpleNamespace(prepared=SimpleNamespace(active=active, metadata={
        "target_id": "brd4-jq1", "target": {"pocket": pocket}}), identity={"fixture": True},
        score_valid_records=lambda *_: [])
    def close():
        assert not (output / "result.json").exists()
        if fail_close:
            raise ValueError("Frozen model changed")
        return {"frozen_unchanged": True}
    scorer.close = close
    monkeypatch.setattr(study.BoltzSearchScorer, "load", lambda *args, **kwargs: scorer)
    def run_pair(actual_decoder, factory, hard_score, config, arm, directory, digest):
        assert actual_decoder is decoder and factory() is scorer and hard_score is scorer.score_valid_records
        directory.mkdir()
        return {"status": "completed"}
    monkeypatch.setattr(study, "run_pair", run_pair)
    call = (args, cfg, SimpleNamespace(occupied_atoms=31),
            {"updates": 1000, "fresh_attempts": 12, "native_vina_attempts": 12})
    if fail_close:
        with pytest.raises(ValueError, match="Frozen model"):
            study.run(*call)
        assert not (output / "result.json").exists()
    else:
        study.run(*call)
        assert json.loads((output / "result.json").read_text())["status"] == "completed"
        assert json.loads((output / "boltz-context.json").read_text())["frozen"]["frozen_unchanged"]
