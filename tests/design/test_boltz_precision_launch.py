"""Local staging checks: correct Boltz assets, paired budgets and GPU requests."""
import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py"


def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("_precision_test_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(module, "remote", lambda _script: {"status": "passed"})
    monkeypatch.setattr(module, "upload", lambda *_args: None)
    return module


def options(**overrides):
    return SimpleNamespace(config="restart-boltz-precision.json", arm=["seed-a", "seed-f"],
        name="boltz-restart-precision-unit", source_receipt=None, queue="gpul40s",
        a100_80gb=False, hold=False, hours=4, max_updates=400, max_fresh_attempts=402, **overrides)


def test_staging_keeps_pair_on_one_gpu_and_includes_common_fp32_scoring(tmp_path, monkeypatch):
    module = launcher(monkeypatch)
    output = tmp_path / "staged"
    module.stage(options(), output, "/control/unit", "/release/unit")
    plan = json.loads((output / "gpu-plan.json").read_text())
    assert plan["planned_updates"] == {"seed-a": 400, "seed-f": 400}
    assert plan["planned_fresh_attempts"] == {"seed-a": 402, "seed-f": 402}
    assert plan["relaxed_gradient_observations"] == {"seed-a": 402, "seed-f": 402}
    assert plan["evaluation_execution"] == dict(tf32=False, autocast="none", checkpoint_every=1,
                                               checkpoint_preserve_rng_state=True)
    commands = json.loads((output / "commands.json").read_text())
    for argv in commands.values():
        assert argv[argv.index("-n") + 1] == "4"
        assert "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=20G" == argv[argv.index("-gpu") + 1]
        assert "--boltz-assets" in argv and "--fresh-config" not in argv
        assert argv[argv.index("-m") + 1] == "latentmol.design.boltz_speed"
        assert "select[gpu80gb]" not in argv
    with tarfile.open(output / "source.tar.gz") as archive:
        names = set(archive.getnames())
        assert "src/latentmol/design/boltz_speed.py" in names
        assert "experiments/latent-design/protenix-vae/restart-multitarget.json" in names
        assert "experiments/latent-design/protenix-vae/restart-boltz-dtu.json" in names


@pytest.mark.parametrize("field,value", [("max_updates", 399), ("max_fresh_attempts", 401)])
def test_insufficient_pair_budget_stops_before_any_remote_write(tmp_path, monkeypatch, field, value):
    module = launcher(monkeypatch)
    args = options()
    setattr(args, field, value)
    monkeypatch.setattr(module, "remote", lambda _script: pytest.fail("Budget validation must precede remote mutation"))
    with pytest.raises(ValueError, match="common FP32 scoring caps"):
        module.stage(args, tmp_path / "staged", "/control/unit", "/release/unit")
    assert not (tmp_path / "staged").exists()
