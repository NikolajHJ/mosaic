"""Allocated small-model observer contracts; no scientific validity claim."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmol import chemistry_epoch45_generation as generation
from latentmol.chemistry_epoch45_extension import POLICY
from latentmol.chemistry_valid_reference_training_policy import POLICY as ORIGINAL
from latentmol.experiments import file_sha256
from latentmol.train import capture_rng_state, state_digest
from latentmol.chemistry_continuation import nested_equal
from test_chemistry_valid_self_guidance import device
from test_chemistry_valid_self_training import _models


def test_structural_diversity_handles_empty_and_one_identity_without_fabricated_perfection():
    empty=generation.structural_diversity([])
    assert empty["internal_diversity"] is None and empty["fingerprint_pairs"] == 0
    one=generation.structural_diversity([dict(valid=True,smiles="CC"),dict(valid=True,smiles="CC")])
    assert one["unique_valid"] == 1 and one["internal_diversity"] is None


def test_diversity_is_order_invariant_and_retains_distinct_structures():
    rows=[dict(valid=True,smiles=s) for s in ("CC", "c1ccccc1", "CCO")]
    first=generation.structural_diversity(rows)
    assert first == generation.structural_diversity(rows[::-1])
    assert first["unique_valid"] == 3 and first["fingerprint_pairs"] == 3
    assert first["unique_scaffolds"] == 2 and 0 <= first["internal_diversity"] <= 1


@pytest.mark.parametrize("path_kind", [Path, str], ids=["path-object", "serialized-path"])
@pytest.mark.parametrize("epoch,updates", [(40,263840), (41,270436)], ids=["initial40", "saved41"])
def test_checkpoint_observer_retains_every_draw_and_preserves_weights_modes_and_all_rng(
        device,tmp_path,monkeypatch,path_kind,epoch,updates):
    model,reference,config,protocol,composition=_models(device)
    model.train()
    generators={"fresh_prior":torch.Generator().manual_seed(71),"train":torch.Generator().manual_seed(72)}
    before=capture_rng_state(generators,torch.device(device).type=="cuda")
    modes=[m.training for m in model.modules()];weights=state_digest(model.state_dict())
    # Small engineering cohort; production policy remains exactly2048 fresh
    # latents/eight192 categorical attempts over its two declared seeds.
    small=deepcopy(POLICY);small["generation"].update(prior_count=4,seeds=[140041,140042],categorical_samples_per_latent=2)
    monkeypatch.setattr(generation,"POLICY",small)
    checkpoint=tmp_path/f"epoch{epoch}.pt";torch.save(dict(model_state=model.state_dict(),epoch=epoch,
        continuation=dict(updates=updates)),checkpoint)
    plan=dict(epoch45_extension=deepcopy(POLICY),valid_reference_continuation=deepcopy(ORIGINAL),
              _plan_sha256="a"*64,archive_sha256="b"*64,manifest_sha256="c"*64)
    result=generation.observe_checkpoint(plan,model,generators,tmp_path/"observation",
        checkpoint=path_kind(checkpoint),checkpoint_sha256=file_sha256(checkpoint),epoch=epoch,updates=updates)
    assert state_digest(model.state_dict()) == weights and [m.training for m in model.modules()] == modes
    assert nested_equal(before,capture_rng_state(generators,torch.device(device).type=="cuda"))
    record=json.loads(open(result["path"]).read())
    assert record["epoch"] == epoch and record["updates"] == updates and len(record["seeds"]) == 2
    assert record["checkpoint"] == str(checkpoint)
    for seed in record["seeds"]:
        with gzip.open(seed["raw_graphs"]["path"],"rt") as stream:rows=[json.loads(line) for line in stream]
        assert len(rows) == 12 and all("graph" in row for row in rows)
        assert seed["metrics"]["argmax"]["attempts"] == 4
        assert seed["metrics"]["categorical"]["attempts"] == 8
        assert seed["detailed"]["argmax"]["classification"]["attempts"] == 4
    with pytest.raises(ValueError, match="model/epoch/update"):
        generation.observe_checkpoint(plan,model,generators,tmp_path/"wrong-update",
            checkpoint=path_kind(checkpoint),checkpoint_sha256=file_sha256(checkpoint),epoch=epoch,updates=updates+1)
    with pytest.raises(FileExistsError):
        generation.observe_checkpoint(plan,model,generators,tmp_path/"observation",
            checkpoint=path_kind(checkpoint),checkpoint_sha256=file_sha256(checkpoint),epoch=epoch,updates=updates)


def test_saved_epoch_observer_receipt_flows_through_real_callback_integrity_checks(device,tmp_path,monkeypatch):
    """Exercise the string path returned by the observer, not a mocked observer.

    Only the final LSF advancement is replaced. The model decode, raw graph
    output, native classification, report serialization and callback hashes all
    execute their production implementations on a bounded allocated fixture.
    """
    model,*_=_models(device)
    small=deepcopy(POLICY);small["generation"].update(prior_count=2,seeds=[140041],categorical_samples_per_latent=1)
    monkeypatch.setattr(generation,"POLICY",small)
    checkpoint=tmp_path/"epoch41.pt"
    torch.save(dict(model_state=model.state_dict(),epoch=41,continuation=dict(updates=270436)),checkpoint)
    plan=dict(epoch45_extension=deepcopy(POLICY),valid_reference_continuation=deepcopy(ORIGINAL),
        _plan_sha256="a"*64,archive_sha256="b"*64,manifest_sha256="c"*64)
    observation=generation.observe_checkpoint(plan,model,{},tmp_path/"generation/epoch0041",
        checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),epoch=41,updates=270436)
    assert isinstance(observation["path"],str)

    scripts=Path(__file__).resolve().parents[1]/"scripts/hpc"
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location("epoch45_observer_callback_fixture",scripts/"chemistry_epoch45_campaign.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    campaign=object.__new__(module.Epoch45Campaign);campaign.root=tmp_path;campaign.sha=plan["_plan_sha256"]
    completion=tmp_path/"segment-001-completion.json"
    completion.write_text(json.dumps(dict(completed_epoch=41)))
    (tmp_path/"generation-segment-001.json").write_text(json.dumps(dict(
        status="passed_observation_integrity",plan_sha256=campaign.sha,
        completion_sha256=file_sha256(completion),observed_epochs=[41],observations=[observation])))
    advanced=[]
    def advance(self,stage):
        advanced.append(stage)
        return "inherited-advance-checked"
    monkeypatch.setattr(module.ValidReferenceCampaign,"advance",advance)
    assert campaign.advance("segment-001") == "inherited-advance-checked"
    assert advanced == ["segment-001"]

    # A parseable but changed report must still stop before scheduler advance.
    with Path(observation["path"]).open("a") as stream:stream.write(" ")
    with pytest.raises(ValueError,match="Saved per-epoch observation changed"):
        campaign.advance("segment-001")
    assert advanced == ["segment-001"]


def test_observer_exception_restores_modes_and_rng(device,tmp_path,monkeypatch):
    model,*_= _models(device)
    model.train()
    generators={"fresh_prior":torch.Generator().manual_seed(19)}
    before=capture_rng_state(generators,torch.device(device).type=="cuda")
    modes=[m.training for m in model.modules()]
    checkpoint=tmp_path/"checkpoint.pt"
    torch.save(dict(model_state=model.state_dict(),epoch=40,continuation=dict(updates=263840)),checkpoint)
    plan=dict(epoch45_extension=deepcopy(POLICY),valid_reference_continuation=deepcopy(ORIGINAL),
              _plan_sha256="a"*64,archive_sha256="b"*64,manifest_sha256="c"*64)
    def fail(*args,**kwargs):
        torch.rand(5);torch.rand(5,generator=generators["fresh_prior"])
        raise RuntimeError("expected observer failure")
    monkeypatch.setattr(generation,"evaluate_codes",fail)
    with pytest.raises(RuntimeError,match="expected observer failure"):
        generation.observe_checkpoint(plan,model,generators,tmp_path/"observation",
            checkpoint=checkpoint,checkpoint_sha256=file_sha256(checkpoint),epoch=40,updates=263840)
    assert [m.training for m in model.modules()] == modes
    assert nested_equal(before,capture_rng_state(generators,torch.device(device).type=="cuda"))


def test_actual_worker_initial_and_saved_observation_phases_use_the_real_observer(device,tmp_path,monkeypatch):
    """Run both production phase functions with bounded real model artifacts.

    Frozen-source/data readiness and model setup are fixture inputs, and the
    hardware precondition is bypassed for the CPU parameter. The observer,
    exact model/moment/RNG comparisons, raw decoder evaluation, serialization,
    checkpoint loader and post-observation receipts are their real functions.
    The production allocation separately requires actual sequential CUDA
    ownership probes after its full CUDA pytest process has exited.
    """
    from dataclasses import asdict
    from latentmol import chemistry_continuation as continuation
    from latentmol import chemistry_epoch45_extension as extension
    from latentmol import evaluate
    scripts=Path(__file__).resolve().parents[1]/"scripts/hpc"
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location("epoch45_real_phase_fixture",scripts/"chemistry_epoch45_worker.py")
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    model,reference,config,*_=_models(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    generators={"fresh_prior":torch.Generator().manual_seed(91)}
    model._valid_reference_anchor=SimpleNamespace(assert_unchanged=lambda:{"reference_epoch":36})
    small=deepcopy(POLICY);small["generation"].update(prior_count=2,seeds=[140041],categorical_samples_per_latent=1)
    monkeypatch.setattr(generation,"POLICY",small)
    checkpoint=tmp_path/"epoch40.pt"
    torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),epoch=40,
        continuation=dict(updates=263840,rng=capture_rng_state(generators,True))),checkpoint)
    monkeypatch.setattr(extension,"CHECKPOINT",str(checkpoint))
    monkeypatch.setattr(extension,"CHECKPOINT_SHA",file_sha256(checkpoint))
    plan=dict(campaign=str(tmp_path),protocol=dict(segment_wall_seconds=23*3600),
        epoch45_extension=deepcopy(POLICY),valid_reference_continuation=deepcopy(ORIGINAL),
        _plan_sha256="a"*64,archive_sha256="b"*64,manifest_sha256="c"*64)
    for name in ("benchmark_gate.json","failure_readiness.json"):
        (tmp_path/name).write_text("{}")
    monkeypatch.setattr(continuation,"read_plan",lambda *args:deepcopy(plan))
    monkeypatch.setattr(continuation,"require_failure_readiness",lambda plan:None)
    # Hardware/LSF authentication is exercised by the native production probe;
    # keeping it isolated here lets the same real observer run on allocated CPU.
    monkeypatch.setattr(worker,"_require_cuda_phase",lambda *args:None)
    closed=[]
    teacher=SimpleNamespace(close=lambda:closed.append(True))
    def setup(current,*,resume):
        assert resume==(checkpoint,file_sha256(checkpoint))
        return model,reference,optimizer,config,None,None,generators,teacher,40,263840,None,None
    monkeypatch.setattr(continuation,"setup",setup)
    args=SimpleNamespace(plan=tmp_path/"plan.json",plan_sha256=plan["_plan_sha256"],segment=1,deadline=1e9)
    weights=state_digest(model.state_dict());before=capture_rng_state(generators,True)
    worker._observe_baseline(args)
    handoff=json.loads((tmp_path/"epoch40-handoff.json").read_text())
    assert handoff["status"]=="passed" and all(handoff["checks"].values()) and closed==[True]
    assert state_digest(model.state_dict())==weights and nested_equal(before,capture_rng_state(generators,True))
    assert json.loads(Path(handoff["observation"]["path"]).read_text())["checkpoint"]==str(checkpoint)

    saved=tmp_path/"epoch41.pt"
    torch.save(dict(format_version=1,graph_spec=model.spec.to_dict(),model_config=asdict(model.config),
        model_state=model.state_dict(),epoch=41,continuation=dict(updates=270436),
        chemistry_phase=dict(epoch45_extension=deepcopy(POLICY))),saved)
    real_load_model=evaluate.load_model;loads=[]
    def load(path,requested):
        loads.append((path,requested))
        return real_load_model(path,device)
    monkeypatch.setattr(evaluate,"load_model",load)
    args.checkpoint=str(saved);args.checkpoint_sha256=file_sha256(saved);args.epoch=41;args.updates=270436
    worker._observe_saved(args)
    receipt=json.loads((tmp_path/"observation-receipts/epoch0041.json").read_text())
    observed=json.loads(Path(receipt["path"]).read_text())
    assert loads==[(saved,"cuda")] and observed["epoch"]==41 and observed["updates"]==270436
    assert file_sha256(Path(receipt["path"]))==receipt["sha256"]
    assert observed["checkpoint_sha256"]==file_sha256(saved)
    assert observed["seeds"][0]["metrics"]["argmax"]["attempts"]==2
    assert observed["seeds"][0]["metrics"]["categorical"]["attempts"]==2
