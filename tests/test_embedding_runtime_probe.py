"""Allocated-only probe integrity, matched-input and timing contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import time

import pytest


path=Path(__file__).resolve().parents[1]/"scripts/hpc/embedding_runtime_probe.py"
spec=importlib.util.spec_from_file_location("runtime_probe",path)
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_nested_timer_does_not_double_count_host_exclusive_work():
    spans=probe.Spans()
    with spans.span("outer"):
        with spans.span("child"):
            time.sleep(.002)
    rows=spans.report()
    assert rows["outer"]["wall_seconds"]>=rows["child"]["wall_seconds"]>0
    assert rows["outer"]["exclusive_wall_seconds"]+rows["child"]["wall_seconds"]==pytest.approx(rows["outer"]["wall_seconds"])
    assert all(v["calls"]==1 and v["cuda_seconds"]==0 for v in rows.values())


def test_failed_timed_component_retains_its_receipt_and_unwinds_stack():
    spans=probe.Spans()
    with pytest.raises(RuntimeError):
        with spans.span("failed"):
            raise RuntimeError("kernel failed")
    assert spans.stack==[] and spans.report()["failed"]["calls"]==1


@pytest.mark.parametrize("values",[[],[-1],[float("nan")],[float("inf")]])
def test_invalid_timing_cannot_be_reported_as_a_success(values):
    with pytest.raises(ValueError): probe.quantiles(values)


def test_nearest_rank_timing_summary_preserves_samples():
    values=[4.,1.,2.,3.,5.]
    row=probe.quantiles(values)
    assert row["n"]==5 and row["mean"]==3 and row["median"]==3 and row["p90"]==5
    assert row["samples"]==values


def test_asset_and_exclusive_writer_reject_mutation(tmp_path):
    p=tmp_path/"receipt.json"; probe.write(p,{"passed":True}); asset=probe.artifact(p)
    assert probe.verify(asset)==p
    with pytest.raises(FileExistsError): probe.write(p,{"passed":False})
    p.write_text("changed")
    with pytest.raises(ValueError): probe.verify(asset)


@pytest.mark.parametrize("stage",["cpu","gpu"])
def test_output_cannot_enter_running_scientific_campaign(stage):
    release=Path("/work3/s234818/latentmol/releases/source-144-embedding-runtime-probe-attempt001")
    with pytest.raises(ValueError):
        probe.validate_output(release,Path("/work3/s234818/latentmol/runs/source142-epoch36-guarded-embedding-long-attempt001")/stage,stage)
    probe.validate_output(release,Path("/work3/s234818/latentmol/runs/source144-embedding-runtime-probe-attempt001")/stage,stage)


@pytest.mark.parametrize("field,value",[("fixed_learning_rate",400.),("step_candidates",16),("line_search",True),("repeats",100)])
def test_probe_cannot_reintroduce_step_search(field,value):
    config={"version":probe.VERSION,"fixed_learning_rate":.0025,"step_candidates":1,"line_search":False,"warmups":2,"repeats":5}
    config[field]=value
    with pytest.raises(ValueError): probe.validate_config(config)


def test_detaching_native_inputs_then_chaining_their_vjp_preserves_latent_gradient():
    import torch
    z=torch.tensor([[.2,-.4]],requires_grad=True)
    weight=torch.tensor([[.7,-.3],[.2,.8]],requires_grad=False)
    mapped=torch.tanh(z@weight)
    loss=(mapped.square()+mapped.sin()).sum()
    full=torch.autograd.grad(loss,z)[0]
    native=mapped.detach().requires_grad_()
    cotangent=torch.autograd.grad((native.square()+native.sin()).sum(),native)[0]
    again=z.detach().requires_grad_(); rebuilt=torch.tanh(again@weight)
    split=torch.autograd.grad(rebuilt,again,grad_outputs=cotangent)[0]
    assert torch.equal(split,full)
    wrong=z.detach().requires_grad_()
    bad=torch.autograd.grad(torch.tanh(wrong@weight),wrong,grad_outputs=cotangent.flip(-1))[0]
    assert not torch.allclose(bad,full)
    assert weight.grad is None and not weight.requires_grad


@pytest.mark.parametrize("corruption",["primal","gradient","dtype","direct_primal","relaxed_primal"])
def test_matched_baseline_requires_exact_primal_and_latent_derivative(corruption):
    import torch
    gold=dict(scores={"dbmol_total_loss":1.},gradient=torch.tensor([[1.,2.]]))
    raw=copy.deepcopy(gold); direct={"scores":dict(gold["scores"])}; relaxed={"scores":dict(gold["scores"])}; graph=copy.deepcopy(relaxed)
    probe.require_matched(torch,raw,gold,direct,relaxed,graph)
    if corruption=="primal": raw["scores"]["dbmol_total_loss"]=2.
    elif corruption=="gradient": raw["gradient"][0,0]=2.
    elif corruption=="dtype": raw["gradient"]=raw["gradient"].double()
    elif corruption=="direct_primal": direct["scores"]["dbmol_total_loss"]=2.
    else: graph["scores"]["dbmol_total_loss"]=2.
    with pytest.raises(ValueError): probe.require_matched(torch,raw,gold,direct,relaxed,graph)


def test_raw_decoder_container_matches_real_original_tensor_contract():
    import torch
    from latentmol.representation import DecoderOutput
    leaves={"atoms":torch.zeros(1,3,4),"charges":torch.zeros(1,3,3),"bonds":torch.zeros(1,3,3,5)}
    decoded=DecoderOutput(**leaves)
    assert all(getattr(decoded,k) is v for k,v in leaves.items())
