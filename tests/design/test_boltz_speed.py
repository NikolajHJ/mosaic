import json
from pathlib import Path

import pytest

from latentmol.design.boltz_speed import compare_children, group_device, group_workers


def group(gradient=(1.,2.), loss=3., terminal="same"):
    child=dict(child=0,initial_binding_gradient=list(gradient),initial_combined_gradient=list(gradient),
        initial_binding_loss=loss,terminal_hard_graph=terminal,terminal_latent=[1.,2.],initial_point_sha256="point")
    return dict(results={"0":dict(result=dict(children=[child]))})


def test_matched_child_comparison_detects_direction_loss_and_graph_changes():
    reference=group()
    same=compare_children(reference,group())[0]
    assert same["initial_loss_delta"]==0 and same["binding_gradient_cosine"]==pytest.approx(1)
    assert same["binding_gradient_relative_l2"]==0 and same["terminal_hard_graph_equal"]
    changed=compare_children(reference,group(gradient=(-1.,-2.),loss=5.,terminal="changed"))[0]
    assert changed["binding_gradient_cosine"]==pytest.approx(-1) and changed["initial_loss_delta"]==2
    assert not changed["terminal_hard_graph_equal"]


def test_rejects_mismatched_initial_points_and_physical_gpu_identity():
    wrong=group()
    wrong["results"]["0"]["result"]["children"][0]["initial_point_sha256"]="different"
    with pytest.raises(ValueError,match="initial latent"):
        compare_children(group(),wrong)
    devices=dict(results={str(i):dict(result=dict(device=dict(device_count=1,device_name="GPU",
        device_uuid=uuid,environment={"allocation":"one"}))) for i,uuid in enumerate(("uuid-one","uuid-two"))})
    with pytest.raises(ValueError,match="different physical"):
        group_device(devices)
    devices["results"]["1"]["result"]["device"]["device_uuid"]="uuid-one"
    assert group_device(devices)["scope"]=="matched_gpu_uuid"


def test_speed_cases_have_same_child_update_workload_and_serial_references():
    path=Path(__file__).resolve().parents[2]/"experiments/latent-design/protenix-vae/restart-boltz-speed.json"
    config=json.loads(path.read_text())
    assert config["children"]==6 and config["updates"]==2
    for task in config["tasks"].values():
        assert task["cases"][0]=="fp32-c1"
        assert task["managed_mps"]
        for name in task["cases"]:
            assert config["children"]%config["cases"][name]["parallel"]==0
    assert len(group_workers(group()))==1
