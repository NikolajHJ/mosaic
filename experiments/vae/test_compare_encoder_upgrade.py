"""Exercise the CPU report with real, small checkpoints, including an unequal-duration finish."""

from dataclasses import asdict
import json
import sys

import pytest
import torch

from experiments.vae import compare_encoder_upgrade as comparison
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


def fixture_checkpoint(root, model, update, protocol):
    cursor = dict(updates=update, epoch=47, row=update*2, real_points=update*2)
    path = root / f"checkpoint-{update:09d}.pt"
    torch.save(dict(format_version=1, model_config=asdict(model.config), graph_spec=model.spec.to_dict(),
                    model_state=model.state_dict(), config_sha256=protocol, continuation=cursor), path)
    return dict(cursor, checkpoint=str(path), sha256=comparison.file_sha256(path))


def test_complete_cpu_report_with_unequal_endpoints(tmp_path, monkeypatch):
    model = GraphAutoencoder(GraphSpec(max_nodes=2), ModelConfig(hidden=8, latent=4, layers=1,
                                                                 encoder_pooling="canonical")).eval()
    # Every input decodes to the same two-carbon parent. Exact recovery should
    # be perfect and create zero valid edits on the four test directions.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.decoder.atom_head.bias[1] = 10
        model.decoder.charge_head.bias[2] = 10
        model.decoder.bond_head[-1].bias[1] = 10
    source = fixture_checkpoint(tmp_path, model, 8, "production")
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    for arm, directory, end in (("production",control,11), ("expanded",candidate,10)):
        directory.mkdir()
        fork = dict(parent={"epoch": 47}, resume=source, config_sha256=arm, source_config_sha256="production",
                    production_control=str(control), generated_roundtrip={"weight": 0.1})
        put(directory / "fork.json", fork)
        latest = fixture_checkpoint(directory, model, end, arm)
        put(directory / "latest.json", latest)
        put(directory / "summary.json", dict(latest=latest, updates=end, segment_updates=end-8, elapsed_seconds=50,
            stop_reason="allocation_budget_reached", dataset_sha256="x", training_molecules=24, batch_size=2,
            microbatch_size=2))
        with (directory / "training.jsonl").open("w", encoding="utf8") as f:
            for u in range(9,end+1):
                obj = dict(updates=u, seconds=1.5, fresh_prior_points=2)
                if arm=="expanded" and u==10:
                    obj["generated_roundtrip"] = dict(excess_confidence=dict(weighted_loss=0.2,
                                                      selected=3, exact_pairs=2, loss=2.0))
                f.write(json.dumps(obj)+"\n")
        with (directory / "quality-panel.jsonl").open("w", encoding="utf8") as f:
            for u in range(8,end+1):
                fixed = dict(prior_attempts=16, eligible_parents=16, mean_exact=4, mean_valid=6,
                             extra_canonical_slot_edges=1, missing_canonical_slot_edges=0)
                obj = dict(updates=u, seed=9, validation_rows=[1], validation_points=1, dataset_sha256="x",
                           prior_points=16, prior_valid=16, prior_unique_valid=2, reconstruction_exact=1,
                           reconstruction_valid=1, generated_roundtrip=dict(leg_start_update=8,fixed_at_leg_start=fixed))
                f.write(json.dumps(obj)+"\n")
    target = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["compare","--control",str(control),"--candidate",str(candidate),"--output",str(target)])
    comparison.main()
    result = comparison.read(target / "comparison.json")
    assert not result["endpoint_updates_matched"]
    assert [q["updates"] for q in result["curves"]["matched"]] == [8,9,10]
    exposure = result["curves"]["candidate_margin_exposure"]
    assert (exposure["scheduled_batches"], exposure["exact_pairs"], exposure["batches_with_positive_loss"]) == (1,2,1)
    assert exposure["mean_effective_loss_per_training_update"] == pytest.approx(.01)
    for s in result["geometric_edits"]["all_eligible_parents"].values():
        assert (s["parents"],s["exact_centers"],s["parents_with_valid_edit"]) == (128,128,0)


def test_failed_centers_stay_in_denominator_and_repeated_valid_edits_count_once():
    rows = [dict(parent="CC", a=dict(center="CC", neighbors=["CN","CN",None,"CC"])),
            dict(parent="CCC", a=dict(center="CC", neighbors=[]))]
    result = comparison.score_edits(rows,"a")
    assert result["distinct_edits_summed_by_parent"] == 1
    assert result["fraction_parents_with_edit"] == result["distinct_edits_per_attempted_parent"] == .5
