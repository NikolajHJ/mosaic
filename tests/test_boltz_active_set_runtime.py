"""Small CPU fixtures for active-set orchestration, not native model evidence.

No pretrained assets, Boltz installation, GPU, network or molecular data are
used. The arithmetic scorer below only makes controlled transitions observable.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from latentmol.boltz.active_set import (
    ActiveSetConfig, ContextBank, bind_slots, run_arm, select_slots,
)
from latentmol.boltz.contracts import BOLTZ_COMMIT, PreparedComplex
from latentmol.experiments import file_sha256
from latentmol.multimodel.config import Diagnostics, Objective, Optimizer
from latentmol.multimodel.engine import Counters
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


SPEC = GraphSpec(max_nodes=3, atomic_numbers=(0, 6, 7),
                 formal_charges=(0,), bond_names=("NONE", "SINGLE"))
TARGET = "a" * 64
MSA = "b" * 64


def _carbon_context(count):
    """Exact one/two-row native tensor fixture with complete carbon seed parity."""
    atoms = torch.zeros(1, SPEC.max_nodes, dtype=torch.long)
    atoms[:, :count] = 1
    bonds = torch.zeros(1, SPEC.max_nodes, SPEC.max_nodes, dtype=torch.long)
    for index in range(count - 1):
        bonds[0, index, index + 1] = bonds[0, index + 1, index] = 1
    native_bonds = bonds[:, :count, :count] * 2
    pool = torch.eye(count)[None]
    features = {
        "ref_element": F.one_hot(torch.full((1, count), 6), 128),
        "ref_charge": torch.zeros(1, count),
        "token_pad_mask": torch.ones(1, count),
        "atom_pad_mask": torch.ones(1, count),
        "affinity_token_mask": torch.ones(1, count),
        "mol_type": torch.full((1, count), 3),
        "type_bonds": native_bonds,
        "token_bonds": (native_bonds > 0).float()[..., None],
        "atom_to_token": pool, "token_to_rep_atom": pool,
        "ref_pos": torch.zeros(1, count, 3),
    }
    mapping = torch.full((SPEC.max_nodes,), -1, dtype=torch.long)
    mapping[:count] = torch.arange(count)
    return PreparedComplex(
        features, torch.zeros(1, 1, count, 3), mapping, mapping.clone(),
        HardGraphBatch(atoms, torch.zeros_like(atoms), bonds), SPEC,
        {"boltz_commit": BOLTZ_COMMIT, "seed_smiles": "C" * count,
         "target_sequence_sha256": TARGET, "msa_sha256": MSA,
         "fixed_msa_row_indices": [0],
         "slot_map": [{"slot": i, "native_atom": i, "native_token": i}
                      for i in range(count)]},
    )


def _config(tmp_path, counts=(1, 2)):
    contexts = {}
    for count in counts:
        path = tmp_path / f"context-{count}.pt"
        _carbon_context(count).save(path)
        contexts[count] = {"path": str(path), "sha256": file_sha256(path)}
    return ActiveSetConfig(
        name="fixture", mode="vae", decoder={}, affinity_checkpoint={},
        context_manifest=None, contexts=contexts, anchor_count=1,
        output_root=tmp_path, device="cpu", temperature=1.0, threshold=0.5,
        recycling_steps=0,
        objective=Objective("affinity_pred_value", direction="maximize"),
        optimizer=Optimizer(steps=2, learning_rate=1.0, trust_radius=1.0),
        diagnostics=Diagnostics(hard_samples=0, fd_rtol=0.005, fd_atol=1e-7),
        allow_lossy_charge=False,
    )


class _Decoder(nn.Module):
    """One scalar drives a smooth scored feature and a separate slot threshold."""

    def __init__(self, transition="grow"):
        super().__init__()
        self.spec = SPEC
        self.transition = transition
        self.eval()

    def decode(self, z):
        x = z[:, 0]
        zero = x * 0
        if self.transition == "empty":
            absent = torch.stack((-2 + 40 * x, zero + 5, zero + 5), -1)
        else:
            absent = torch.stack((zero - 5, 2 - 20 * x, zero + 5), -1)
        atoms = torch.stack((absent, zero[:, None].expand(-1, 3),
                             x[:, None].expand(-1, 3)), -1)
        charges = zero[:, None, None].expand(-1, 3, 1)
        bonds = torch.stack((zero - 2, zero + 2), -1)[:, None, None].expand(-1, 3, 3, -1)
        return DecoderOutput(atoms, charges, bonds)


class _WrongDerivative(torch.autograd.Function):
    """Keep the forward identical but intentionally violate the derivative gate."""

    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, derivative):
        return 2 * derivative


class _Scorer(nn.Module):
    """Frozen arithmetic on expected nitrogen features; never an affinity model."""

    def __init__(self, *, bad_parity=False, bad_derivative=False):
        super().__init__()
        self.recycling_steps = 0
        self.bad_parity = bad_parity
        self.bad_derivative = bad_derivative
        self.seen_counts = []
        self.register_buffer("constant", torch.tensor(1.0, dtype=torch.float64))
        self.eval()

    def forward(self, features, types, coordinates):
        self.seen_counts.append(features["ref_element"].shape[1])
        value = features["ref_element"].to(torch.float64)[..., 7].sum() * self.constant
        if self.bad_derivative:
            value = _WrongDerivative.apply(value)
        return {"affinity_pred_value": value,
                "affinity_probability_binary": value.sigmoid()}

    @torch.no_grad()
    def discrete_reference(self, features, coordinates):
        result = self(features, F.one_hot(features["type_bonds"], 7), coordinates)
        if self.bad_parity:
            result["affinity_pred_value"] = result["affinity_pred_value"] + 1
        return result


class _Run:
    """Capture orchestration events without an HPC ExperimentRun claim."""

    def __init__(self, path):
        self.path = path
        self.events = []

    def log(self, item):
        self.events.append(item)


def _invoke(arm, decoder, scorer, cfg, directory):
    start = torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(start), temperature=1.0, threshold=0.5)
    assert initial.count == 1
    return run_arm(arm, decoder, start, initial, ContextBank(cfg, SPEC), scorer,
                   cfg, "c" * 64, _Run(directory), Counters())


def test_real_count_bank_switches_and_fixed_control_retains_start_mask(tmp_path):
    cfg = _config(tmp_path)
    decoder, scorer = _Decoder(), _Scorer()
    active = _invoke("active_set", decoder, scorer, cfg, tmp_path)
    fixed = _invoke("fixed_initial_mask", decoder, scorer, cfg, tmp_path)
    assert active["context_switches"] == 1
    assert active["gated_contexts"] == 2
    assert active["endpoint"]["selection"]["count"] == 2
    assert fixed["context_switches"] == 0
    assert fixed["gated_contexts"] == 1
    assert fixed["endpoint"]["selection"]["count"] == 1
    assert fixed["endpoint"]["natural_selection"]["count"] == 2
    assert {1, 2} <= set(scorer.seen_counts)
    transition = json.loads((tmp_path / "active_set" / "transition-0001.json").read_text())
    assert transition["status"] == "completed"
    assert transition["score_jump_may_change_pose_and_native_features"]
    assert transition["old_context_sha256"] != transition["new_context_sha256"]
    assert transition["old_score"] is not None and transition["new_score"] is not None
    # Both arms begin with exactly the same state, before their occupancy differs.
    starts = [torch.load(tmp_path / arm / "state-0000.pt", weights_only=True)
              for arm in ("active_set", "fixed_initial_mask")]
    assert torch.equal(starts[0]["z"], starts[1]["z"])
    assert torch.equal(starts[0]["active"], starts[1]["active"])


@pytest.mark.parametrize("failure", ["parity", "derivative"])
def test_failed_gate_retains_partial_report_transition_and_latent(tmp_path, failure):
    cfg = _config(tmp_path)
    scorer = _Scorer(bad_parity=failure == "parity", bad_derivative=failure == "derivative")
    with pytest.raises((ValueError, AssertionError)):
        _invoke("active_set", _Decoder(), scorer, cfg, tmp_path)
    folder = tmp_path / "active_set"
    report = json.loads((folder / "gate-0000.json").read_text())
    assert not report["passed"] and report["status"] == "failed"
    assert report["error"] and report["context_sha256"]
    assert report["active_slots"] == [0]
    if failure == "derivative":
        assert report["native_seed_parity"]
        assert len(report["finite_differences"]) == 6
        assert any(not row["passed"] for row in report["finite_differences"])
    else:
        assert not report["native_seed_parity"]
        assert not report["parity_heads"]["affinity_pred_value"]["passed"]
    assert json.loads((folder / "gates.json").read_text())["step-0000"] == report
    transition = json.loads((folder / "transition-0000.json").read_text())
    assert transition["status"] == "failed"
    state = torch.load(folder / transition["state_path"], weights_only=True)
    assert state["context_sha256"] == report["context_sha256"]
    assert torch.equal(state["z"], torch.zeros(1, 1, dtype=torch.float64))
    assert not (folder / "summary.json").exists()


@pytest.mark.parametrize("transition,count,reason", [
    ("grow", 2, "native_count_context_unavailable"),
    ("empty", 0, "empty_ligand"),
])
def test_rejected_transition_retains_proposed_state(tmp_path, transition, count, reason):
    cfg = _config(tmp_path, counts=(1,))
    with pytest.raises(ValueError, match=f"selected count {count}"):
        _invoke("active_set", _Decoder(transition), _Scorer(), cfg, tmp_path)
    folder = tmp_path / "active_set"
    rejected = json.loads((folder / "rejected-transition-0000.json").read_text())
    assert rejected["reason"] == reason
    assert rejected["selection"]["count"] == count
    assert rejected["available_counts"] == [1]
    state = torch.load(folder / rejected["proposal_state_path"], weights_only=True)
    assert int(state["active"].sum()) == count
    assert state["z"].item() > 0
    assert state["no_atom"].tolist() == rejected["selection"]["no_atom_probabilities"]
    assert not (folder / "summary.json").exists()


def test_same_count_binding_relabels_metadata_without_mutating_template():
    template = _carbon_context(2)
    bound = bind_slots(template, torch.tensor([False, True, True]))
    assert [row["slot"] for row in bound.metadata["slot_map"]] == [1, 2]
    assert [row["slot"] for row in bound.metadata["template_slot_map"]] == [0, 1]
    assert [row["slot"] for row in template.metadata["slot_map"]] == [0, 1]
    assert bound.slot_to_atom.tolist() == [-1, 0, 1]


def test_bank_rejects_catalog_identity_disagreeing_with_snapshot(tmp_path):
    cfg = _config(tmp_path)
    identity = {"target_sequence_sha256": "0" * 64, "msa_sha256": MSA,
                "fixed_msa_row_indices": [0]}
    with pytest.raises(ValueError, match="header differs"):
        ContextBank(replace(cfg, context_identity=identity), SPEC)
    identity["target_sequence_sha256"] = TARGET
    assert ContextBank(replace(cfg, context_identity=identity), SPEC).target_identity[:2] == (TARGET, MSA)


def _yaml_configuration():
    asset = {"path": "unloaded.pt", "sha256": "a" * 64}
    return {"version": 1, "name": "fixture", "mode": "vae",
            "decoder": dict(asset), "affinity_checkpoint": dict(asset),
            "contexts": {1: dict(asset)}, "anchor_count": 1,
            "output_root": "runs", "device": "cuda"}


@pytest.mark.parametrize("field,value", [
    ("temperature", float("inf")), ("temperature", float("nan")),
    ("temperature", True), ("threshold", float("inf")),
    ("recycling_steps", 0.5), ("recycling_steps", True),
    ("allow_lossy_charge", "false"), ("anchor_count", True),
    ("device", "cuda:invalid"), ("unknown_setting", 42),
])
def test_config_rejects_invalid_settings_before_loading_assets(tmp_path, field, value):
    raw = _yaml_configuration()
    raw[field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        ActiveSetConfig.load(path)


def test_config_requires_one_context_source_and_rejects_silent_asset_extras(tmp_path):
    raw = _yaml_configuration()
    raw["context_manifest"] = {"path": "never-opened.json", "sha256": "f" * 64}
    path = tmp_path / "ambiguous.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exactly one"):
        ActiveSetConfig.load(path)
    raw.pop("context_manifest")
    raw["contexts"][1]["ignored_field"] = "must not disappear"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exactly path"):
        ActiveSetConfig.load(path)


def test_prior_policy_is_explicit_and_requires_vae(tmp_path):
    raw = _yaml_configuration()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert ActiveSetConfig.load(path).start.kind == "encoded_seed"
    raw["start"] = {"kind": "standard_normal", "seed": 7}
    path.write_text(yaml.safe_dump(raw))
    assert ActiveSetConfig.load(path).start.seed == 7
    raw["mode"] = "ae"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="requires a VAE"):
        ActiveSetConfig.load(path)


def test_mixed_catalog_reuses_exact_snapshots_and_retains_failures(tmp_path):
    from latentmol.boltz.prepare import seal_count_catalog
    cfg = _config(tmp_path)
    paths = {count: Path(asset["path"]) for count, asset in cfg.contexts.items()}
    output = tmp_path / "new" / "catalog.json"
    seal_count_catalog(paths, output, failed_counts={"3": {"status": "failed"}})
    value = json.loads(output.read_text())
    assert value["contexts"] == {str(n): asset for n, asset in cfg.contexts.items()}
    assert set(value["failed_counts"]) == {"3"}
    assert value["target_sequence_sha256"] == TARGET
    with pytest.raises(ValueError, match="disjoint"):
        seal_count_catalog(paths, tmp_path / "bad.json", failed_counts={"1": {"status": "failed"}})


def test_random_run_never_encodes_chain_and_saves_rejected_draw(tmp_path, monkeypatch):
    from latentmol.boltz import active_set
    from latentmol.initialization import LatentStart, sample_prior

    class EmptyPriorDecoder(nn.Module):
        spec = SPEC
        config = SimpleNamespace(latent=4)

        def encode(self, graph):
            raise AssertionError("A random prior start must never encode a carbon chain")

        def decode(self, z):
            atoms = torch.full((1, 3, 3), -20.)
            atoms[..., 0] = 20.
            return DecoderOutput(atoms, torch.zeros(1, 3, 1), torch.zeros(1, 3, 3, 2))

    checkpoint = tmp_path / "fixture-checkpoint"
    checkpoint.write_text("unloaded fixture weights")
    asset = {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture-only")
    cfg = replace(_config(tmp_path), decoder=asset, affinity_checkpoint=asset,
                  output_root=tmp_path / "runs", start=LatentStart("standard_normal", 7))
    monkeypatch.setenv("LSB_JOBID", "fixture")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "fixture")
    monkeypatch.setattr(active_set.ActiveSetConfig, "load", lambda path: cfg)
    monkeypatch.setattr(active_set, "load_model", lambda *args: (EmptyPriorDecoder(), {"train_config": {"mode": "vae"}}))
    with pytest.raises(ValueError, match="Random start selected count 0"):
        active_set.run(config_path)
    saved, = (tmp_path / "runs").glob("*/start.pt")
    record = torch.load(saved, weights_only=True)
    assert torch.equal(record["z"], sample_prior(4, 7))
    assert record["initialization"] == {"kind": "standard_normal", "seed": 7}
    assert record["selection"]["count"] == 0
    assert json.loads((saved.parent / "run.json").read_text())["status"] == "failed"
