"""The new epoch driver retains the historical health baselines and thresholds."""
import json
from copy import deepcopy

from latentmol import vae_neighborhood_continuation as worker


def test_fixed_panel_uses_original_baselines_and_creates_saved_health_parent(tmp_path, monkeypatch):
    from test_chemistry_continuation import _tiny_models
    model, reference, config, _, _ = _tiny_models()
    population=dict(attempts=1024,raw_argmax_validity=.8,expected_size=20.,nonempty_fraction=1.,
                    sulfur_frequency=.01,element_js=0.,charge_js=0.)
    rec=dict(molecules=256,exact_reconstruction=1.,validity=1.)
    observation=dict(population=population,reconstruction=rec,
                     classification=dict(attempts=1024,counts=dict(strict_teacher_valid=819)))
    old=tmp_path/"old";old.mkdir()
    (old/"early-health-baseline.json").write_text(json.dumps(dict(observation=observation)))
    (old/"failure_readiness.json").write_text(json.dumps(dict(native_panel=dict(
        starting_retention_population=population,starting_reconstruction=rec))))
    from latentmol.chemistry_short_continuation import ORIGINAL_PROTOCOL, EARLY_POLICY
    protocol=dict(deepcopy(ORIGINAL_PROTOCOL),
        baseline_continuation=dict(version="epoch36-original-repair-v1"),
        early_health=deepcopy(EARLY_POLICY))
    base=dict(campaign=str(old),protocol=protocol)
    folder=tmp_path/"new";folder.mkdir()
    protocol["evaluation_seed"]=904017
    monkeypatch.setattr(worker.c,"observe_early_health",lambda *a,**k:observation)
    monkeypatch.setattr(worker.c,"population_panel",lambda *a,**k:population)
    monkeypatch.setattr(worker.c,"validation",lambda *a,**k:rec)
    worker.observe(model,reference,None,config,{"val":None},{},{},base,folder,283628)
    saved=json.loads((folder/"early_health/update_000283628.json").read_text())
    assert saved["status"]=="passed" and all(saved["checks"].values())
    assert saved["starting_strict_teacher_validity"]==819/1024

