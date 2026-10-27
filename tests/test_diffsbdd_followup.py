"""Independent native confirmation contracts; execute only on allocated HPC."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmol.diffsbdd import followup
from latentmol.diffsbdd.transport import MolecularGraph


def settings():
    return dict(recycling_steps=3, sampling_steps=200, diffusion_samples=1,
        sampling_steps_affinity=200, diffusion_samples_affinity=5, max_msa_seqs=8192)


def test_native_cli_flags_are_explicit_and_no_server_or_selection_is_added():
    assets = dict(affinity={"path": "/a/aff.ckpt"}, confidence={"path": "/a/conf.ckpt"})
    command = followup.prediction_command("/env/python", "/own/input.yaml", "/own/out", "/own/cache",
                                         17, assets, settings())
    assert command[:4] == ["/env/python", "-m", "boltz.main", "predict"]
    for flag in ("--checkpoint", "--affinity_checkpoint", "--seed", "--cache", "--no_kernels"):
        assert flag in command
    assert command[command.index("--seed") + 1] == "17"
    assert "--use_msa_server" not in command and "--override" not in command
    with pytest.raises(ValueError, match="exactly 17 and 23"):
        followup.prediction_command("p", "i", "o", "c", 19, assets, settings())
    with pytest.raises(ValueError, match="One model per fresh seed"):
        followup.prediction_command("p", "i", "o", "c", 17, assets, dict(settings(), diffusion_samples=5))


def test_forecast_keeps_slowest_observation_and_all_remaining_cases():
    assert followup.runtime_forecast([10, 40, 20], 5, 30) == 330
    with pytest.raises(ValueError, match="measured positive"):
        followup.runtime_forecast([], 5, 30)


def test_parent_cuda_ownership_check_does_not_import_or_initialize_cuda(monkeypatch):
    monkeypatch.setitem(followup.sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(_initialized=True)))
    with pytest.raises(RuntimeError, match="never initialized CUDA"):
        followup._parent_cuda_uninitialized()
    monkeypatch.setitem(followup.sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(_initialized=False)))
    followup._parent_cuda_uninitialized()


def test_owned_cache_uses_symlinks_and_preserves_original_bytes(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    for name in ("aff.ckpt", "conf.ckpt", "mols.tar"):
        (original / name).write_bytes(name.encode())
    molecules = original / "mols"
    molecules.mkdir()
    (molecules / "ALA.pkl").write_bytes(b"native molecule bytes")
    catalog = original / "molecule-catalog.json"
    catalog.write_text(json.dumps({"members": {"ALA.pkl": followup._sha(molecules / "ALA.pkl")}}))
    asset = lambda path: {"path": str(path), "sha256": followup._sha(path)}
    assets = dict(affinity=asset(original / "aff.ckpt"), confidence=asset(original / "conf.ckpt"),
        mols_tar=asset(original / "mols.tar"), mols_dir={"path": str(molecules), "manifest": asset(catalog)})
    monkeypatch.setattr(followup, "AFFINITY_SHA256", assets["affinity"]["sha256"])
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in original.rglob("*") if path.is_file()}
    own = tmp_path / "owned"
    receipt = followup.make_owned_cache(own, assets)
    assert (own / "mols").is_symlink() and (own / "boltz2_conf.ckpt").is_symlink()
    assert followup._own_bytes(own) == (own / "ready.json").stat().st_size
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())
    assert receipt["molecules"]["members"] == 1
    with pytest.raises(ValueError, match="authenticated ready"):
        followup.make_owned_cache(own, assets)
    reused = followup.make_owned_cache(own, assets, expected_ready_sha256=receipt["ready_sha256"])
    assert reused["reused_authenticated_cache"]


def test_native_skip_or_retry_log_is_failure_even_with_successful_process_exit(tmp_path):
    with pytest.raises(RuntimeError, match="failed/skipped/retried"):
        followup.validate_prediction_output(tmp_path, "case", "CCC",
            "Tokenizer failed on case with error example. Skipping.\nNumber of failed examples: 0")


def test_all_valid_terminal_memberships_and_D_baselines_survive_deduplication(tmp_path):
    decoder = ("C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F")
    cohort = dict(stage="pilot", seeds=list(range(31, 39)), arms=["D", "H0", "S", "A1", "A5", "A5-exact"],
                  status="completed_cohort_not_fresh_pose_validation")
    (tmp_path / "cohort.json").write_text(json.dumps(cohort))
    features = torch.zeros((25, len(decoder)))
    features[:, 0] = 1
    for seed in cohort["seeds"]:
        for arm in cohort["arms"]:
            directory = tmp_path / f"seed-{seed:03d}" / arm
            directory.mkdir(parents=True)
            terminal = directory / "terminal.pt"
            torch.save(dict(seed=seed, arm=arm,
                tensors=dict(positions=torch.zeros((25, 3)), one_hot=features)), terminal)
            (directory / "arm.json").write_text(json.dumps(dict(seed=seed, arm=arm,
                status="completed_sampler_arm", terminal_sha256=followup._sha(terminal))))

    def builder(candidate):
        bonds = [[0] * 25 for _ in range(25)]
        for index in range(24):
            bonds[index][index + 1] = bonds[index + 1][index] = 1
        return MolecularGraph(candidate.graph.elements, (0,) * 25, tuple(tuple(row) for row in bonds),
                              tuple(range(25))), {"synthetic_fixture": True}

    identities, attempts = followup.collect_terminal_identities(tmp_path, builder, decoder)
    assert len(identities) == 1 and len(identities[0]["memberships"]) == 48
    assert len(attempts) == 48 and all(row["valid"] for row in attempts)
    assert sum(row["arm"] == "D" for row in identities[0]["memberships"]) == 8
    assert all(row["terminal_sha256"] for row in identities[0]["memberships"])

    def disconnected_builder(candidate):
        return candidate.graph, {"synthetic_fixture": "two_or_more_fragments", "raw_bonds": []}

    disconnected, rejected = followup.collect_terminal_identities(tmp_path, disconnected_builder, decoder)
    assert disconnected == []
    assert len(rejected) == 48 and all(not row["valid"] for row in rejected)
    assert all(row["failure"] == "disconnected_terminal" for row in rejected)
    assert all(row["builder"]["synthetic_fixture"] == "two_or_more_fragments" for row in rejected)
    assert all(row["failure_receipt"] == row["builder"] for row in rejected)


def test_catalog_rejects_unbound_extra_native_molecule_files(tmp_path):
    molecules = tmp_path / "mols"
    molecules.mkdir()
    (molecules / "ALA.pkl").write_bytes(b"a")
    (molecules / "unbound.pkl").write_bytes(b"extra")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(members={"ALA.pkl": followup._sha(molecules / "ALA.pkl")})))
    asset = dict(path=str(molecules), manifest=dict(path=str(catalog), sha256=followup._sha(catalog)))
    with pytest.raises(ValueError, match="omits or adds"):
        followup._catalog_verified(asset)
