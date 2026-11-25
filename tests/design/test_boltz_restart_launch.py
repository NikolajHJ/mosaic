"""Local-only staging checks for the native 31-slot BRD4 search."""
import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/latent-design/protenix-vae"
LAUNCHER = EXPERIMENT / "restart-operations/submit_campaign.py"
CONFIG = EXPERIMENT / "restart-boltz-brd4.json"


def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("_boltz_restart_test_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(ROOT)
    calls = []

    def remote(script):
        calls.append(script)
        return {"status": "passed"}

    monkeypatch.setattr(module, "remote", remote)
    monkeypatch.setattr(module, "upload", lambda *_args: None)
    return module, calls


def options():
    return SimpleNamespace(config=CONFIG.name, arm=["brd4-jq1"],
        name="boltz-restart-brd4-unit", source_receipt=None, queue="gpuv100",
        a100_80gb=False, hold=False, hours=24, max_updates=48_000,
        max_fresh_attempts=None)


def test_boltz_taco_pairs_stage_native_scorer_and_matched_redocking(tmp_path, monkeypatch):
    module, preflight = launcher(monkeypatch)
    args = options()
    args.config = "restart-boltz-taco-guidance.json"
    args.arm = ["founder-1", "founder-2", "founder-3"]
    args.max_updates, args.max_fresh_attempts = 1000, 12
    output = tmp_path / "staged"
    module.stage(args, output, "/control/unit", "/release/unit")
    plan = json.loads((output / "gpu-plan.json").read_text())
    assert plan["planned_updates"] == dict.fromkeys(args.arm, 1000)
    assert plan["planned_fresh_attempts"] == plan["planned_native_vina_attempts"] == dict.fromkeys(args.arm, 12)
    assert plan["host_resources"] == dict.fromkeys(args.arm, "gpu32gb")
    cfg = json.loads((module.EXPERIMENT / args.config).read_text())
    for source in (cfg["prepared_config"], cfg["parent_state"], cfg["tacogfn"]["pocket"],
                   cfg["tacogfn"]["topologies"], cfg["vina_parent_panel"]["receptor"]):
        assert source["sha256"] in preflight[0]
    for command in json.loads((output / "commands.json").read_text()).values():
        assert command[command.index("-m") + 1] == "latentmol.design.boltz_tacogfn"
        assert "--boltz-assets" in command and "--mosaic-python" not in command


def test_brd4_stage_seals_31_slot_full_scorer_budget_host_and_new_structure(tmp_path, monkeypatch):
    from latentmol.design.protenix_search import resolve_config

    module, calls = launcher(monkeypatch)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    design, search, settings = resolve_config(config, "brd4-jq1")
    assert settings["budget"] == dict(branches=120, updates=48_000, fresh_attempts=720)
    assert (design.occupied_atoms, design.learning_rate, design.updates) == (31, 1., 400)
    assert design.noise_initial == 0 and design.chemistry_initial == design.chemistry_final == 1
    assert (search.parents, search.children, search.bootstrap_branches, search.waves) == (3, 2, 6, 20)
    assert search.time_budget_seconds == 79_200 and search.hard_scoring == "fixed_pose_dbmol"
    assert search.profile_steps and search.duplicate_refill_attempts == 8
    assert config["scorer"]["kind"] == "boltz2_dbmol"
    assert config["scorer"]["execution"] == dict(tf32=False, autocast="none", checkpoint_every=1,
                                               checkpoint_preserve_rng_state=True)
    assert config["nominal_budget"]["waves"] == 6
    assert config["nominal_budget"]["updates"] == 14_400

    output = tmp_path / "staged"
    module.stage(options(), output, "/control/unit", "/release/unit")
    plan = json.loads((output / "gpu-plan.json").read_text())
    assert plan["planned_updates"] == {"brd4-jq1": 48_000}
    assert plan["planned_fresh_attempts"] == {"brd4-jq1": 720}
    assert plan["host_resources"] == {"brd4-jq1": "gpu32gb"}
    argv = json.loads((output / "commands.json").read_text())["brd4-jq1"]
    assert argv[argv.index("-q") + 1] == "gpuv100"
    assert "select[gpu32gb]" in argv and "select[gpu80gb]" not in argv
    assert argv[argv.index("-gpu") + 1] == "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G"
    assert argv[argv.index("-W") + 1] == "24:00"
    assert argv[argv.index("-m") + 1] == "latentmol.design.protenix_search"
    assert "--boltz-assets" in argv and "--fresh-config" not in argv
    arm = config["arms"]["brd4-jq1"]
    with tarfile.open(output / "source.tar.gz") as archive:
        assert json.load(archive.extractfile(CONFIG.relative_to(ROOT).as_posix())) == config
        assert archive.extractfile(arm["target"]["structure_path"]).read() == (ROOT / arm["structure_source"]).read_bytes()
        assert archive.getmember("src/latentmol/guidance/boltz_search.py").isfile()
    assert arm["prepared_config"]["path"] in calls[0]
    assert arm["prepared_config"]["sha256"] in calls[0]


@pytest.mark.parametrize("resource,queue", [
    ("gpu80gb", "gpuv100"), ("gpu32gb", "gpua100"), ("gpu16gb", "gpuv100"),
])
def test_invalid_restart_host_fails_before_local_or_remote_writes(tmp_path, monkeypatch, resource, queue):
    module, _ = launcher(monkeypatch)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["arms"]["brd4-jq1"].update(host_resource=resource, queue=queue)
    original_read = module.read
    monkeypatch.setattr(module, "read", lambda path: config if path.name == CONFIG.name else original_read(path))
    monkeypatch.setattr(module, "remote", lambda _script: pytest.fail("Host validation must precede remote mutation"))
    monkeypatch.setattr(module, "upload", lambda *_args: pytest.fail("Invalid host must not upload"))
    output = tmp_path / "staged"
    with pytest.raises(ValueError, match="Incompatible restart GPU host requirement"):
        module.stage(options(), output, "/control/unit", "/release/unit")
    assert not output.exists()
