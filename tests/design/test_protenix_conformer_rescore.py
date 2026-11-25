from argparse import Namespace
from hashlib import sha256
import json

import pytest

from latentmol.design.protenix_conformer_rescore import run_rescore, verify_panel


def sealed_json(path, payload):
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": sha256(raw).hexdigest()}


def test_rescore_keeps_panel_and_ranks_complete_held_out_pairs(tmp_path):
    panel = [
        dict(id="challenger", group="early", wave=7, branch="cycle-07/branch-01", step=40,
             smiles="CCO", score=1.8),
        dict(id="incumbent", group="early", wave=3, branch="cycle-03/branch-01", step=30,
             smiles="CCN", score=1.9),
        dict(id="late-1", group="late", wave=8, branch="cycle-08/branch-01", step=60,
             smiles="CCCl", score=2.0),
        dict(id="late-2", group="late", wave=9, branch="cycle-09/branch-01", step=70,
             smiles="CCBr", score=2.1),
    ]
    config = dict(study="conformer_rescore", geometry_seeds=[1201, 2203, 3209],
        historical_seed=1201, source_job="42", panel=panel, max_pairwise_similarity=1.0,
        anchors=dict(incumbent="incumbent", challenger="challenger"), tasks={"panel": {}},
        selection=dict(source_waves={"early": [1, 7], "late": [8, 20]}, count_per_group=2),
        source_result=sealed_json(tmp_path / "original-result.json", {"status": "completed"}),
        sources={str(row["wave"]): sealed_json(tmp_path / f"source-{row['wave']}.json",
            dict(status="completed", geometry_seed=1201, job_id="42",
                 candidates=[dict(arm=row["branch"], step=row["step"], smiles=row["smiles"],
                                  loss=row["score"], status="scored")])) for row in panel})
    assets_path = tmp_path / "assets.json"
    assets_path.write_text("{}")
    protocol_path = tmp_path / "fresh.json"
    protocol_path.write_text(json.dumps(dict(geometry_seed=1201, assets_config=assets_path.name,
                                             candidate_limit=850, pocket={"sha256": "fixture"})))
    expected_ids = [row["id"] for row in panel]

    def score(args, _root, records):
        protocol = json.loads(args.fresh_config.read_text())
        assert protocol["assets_config"] == str(assets_path)
        assert protocol["pocket"] == {"sha256": "fixture"}
        assert protocol["candidate_limit"] == 4
        assert [row["start"] for row in records] == expected_ids
        losses = {1201: [1.8, 1.9, 2.0, 2.1], 2203: [2.1, 1.9, 1.8, None],
                  3209: [2.0, 1.95, 1.81, 1.0]}[protocol["geometry_seed"]]
        return [row | dict(loss=loss, status="scored" if loss is not None else "embedding_failed")
                for row, loss in zip(records, losses, strict=True)]

    args = Namespace(arm="panel", fresh_config=protocol_path, output=tmp_path / "rescore")
    result = run_rescore(args, config, fresh_score=score)
    assert result["fresh_attempts"] == 12
    assert result["optimization_updates"] == 0
    assert result["held_out_mean_ranking"] == ["late-1", "incumbent", "challenger"]
    assert result["original_winner_held_out_rank"] == 3
    assert result["by_seed"][1]["successful"] == 3
    assert result["anchor_comparison"]["original_gain"] == pytest.approx(0.1)
    assert result["anchor_comparison"]["held_out_mean_gain"] == pytest.approx(-0.125)
    assert result["anchor_comparison"]["held_out_challenger_wins"] == 0
    assert result["candidates"][-1]["held_out_mean"] is None
    assert result["candidates"][-1]["held_out_successful"] == 1
    (tmp_path / "source-7.json").write_text("{}")
    with pytest.raises(ValueError, match="Source scoring artifact changed"):
        verify_panel(config)


def saved_synthesis_config(tmp_path):
    source = dict(branches=[dict(name="parent-01-wave-pareto-00", parent_smiles="CC",
        parent_fresh_loss=2.1673, parent_source={"source": "archive_pareto"},
        candidates={"mixed": [dict(smiles="CCO", branch="cycle-03/parent-01-wave-pareto-00",
            step=278, loss=2.2028, status="scored")]},
        endpoint=dict(smiles="CCN", branch="cycle-03/parent-01-wave-pareto-00", step=500))])
    return dict(study="conformer_rescore", panel_kind="synthesis_saved", tasks={"panel": {}},
        geometry_seeds=[1201, 2203, 3209], historical_seed=1201, source_job="30043389",
        source_result=sealed_json(tmp_path / "source-result.json", {"status": "completed"}),
        sources={"wave3": sealed_json(tmp_path / "wave3-batch.json", source)},
        panel=[dict(id="joint-intermediate", smiles="CCO", source_key="wave3",
                    branch="cycle-03/parent-01-wave-pareto-00", step=278, location="candidate",
                    score=2.2028, provenance="Wave-three selected intermediate, update278"),
               dict(id="terminal", smiles="CCN", source_key="wave3",
                    branch="cycle-03/parent-01-wave-pareto-00", step=500,
                    location="endpoint", score=None)])


def test_saved_synthesis_rescore_authenticates_unscored_endpoint_and_retains_provenance(tmp_path):
    config = saved_synthesis_config(tmp_path)
    panel, seeds = verify_panel(config)
    assert panel[0]["source_parent"]["fresh_loss"] == 2.1673
    assert panel[1]["source_parent"]["smiles"] == "CC"
    assert panel[1]["score"] is None
    assets = tmp_path / "assets.json"
    assets.write_text("{}")
    protocol = tmp_path / "fresh.json"
    protocol.write_text(json.dumps(dict(geometry_seed=1201, assets_config=assets.name, candidate_limit=850)))

    def score(args, _root, records):
        fresh = json.loads(args.fresh_config.read_text())
        assert fresh["candidate_limit"] == 2
        assert all(row["role"] == "saved_synthesis_point" for row in records)
        assert records[0]["provenance"] == config["panel"][0]["provenance"]
        assert records[1]["source_parent"]["fresh_loss"] == 2.1673
        losses = {1201: [2.2028, 2.4], 2203: [2.25, 2.35], 3209: [None, 2.3]}[fresh["geometry_seed"]]
        return [row | dict(loss=loss, status="scored" if loss is not None else "embedding_failed")
                for row, loss in zip(records, losses, strict=True)]

    result = run_rescore(Namespace(arm="panel", fresh_config=protocol, output=tmp_path / "rescored"),
                         config, fresh_score=score)
    assert result["status"] == "completed" and result["panel_kind"] == "synthesis_saved"
    assert result["optimization_updates"] == 0 and result["fresh_attempts"] == 6
    assert result["candidates"][0]["historical_repeat_delta"] == 0
    assert result["candidates"][0]["mean_loss"] == pytest.approx((2.2028 + 2.25) / 2)
    assert result["candidates"][0]["held_out_mean"] is None
    endpoint = result["candidates"][1]
    assert endpoint["historical_repeat_delta"] is None
    assert endpoint["complete"] and endpoint["successful"] == 3
    assert endpoint["mean_loss"] == pytest.approx(2.35)
    assert endpoint["median_loss"] == pytest.approx(2.35)
    assert endpoint["loss_range"] == [2.3, 2.4]
    assert endpoint["chemistry"]["valid"] and endpoint["chemistry"]["sa_score"] > 0
    assert result["held_out_mean_ranking"] == ["terminal"]
    assert "anchor_comparison" not in result


@pytest.mark.parametrize("change", [dict(step=501), dict(score=2.1),
                                    dict(branch="cycle-99/parent-01-wave-pareto-00"),
                                    dict(source_key="missing"), dict(location="invalid")])
def test_saved_synthesis_rejects_endpoint_provenance_changes(tmp_path, change):
    config = saved_synthesis_config(tmp_path)
    config["panel"][1].update(change)
    with pytest.raises(ValueError):
        verify_panel(config)


def test_saved_synthesis_checks_sealed_batches_and_completed_source(tmp_path):
    config = saved_synthesis_config(tmp_path)
    (tmp_path / "wave3-batch.json").write_text("{}")
    with pytest.raises(ValueError, match="Source scoring artifact changed"):
        verify_panel(config)
    config = saved_synthesis_config(tmp_path)
    config["source_result"] = sealed_json(tmp_path / "incomplete.json", {"status": "running"})
    with pytest.raises(ValueError, match="completed synthesis"):
        verify_panel(config)


def test_saved_synthesis_supports_unambiguous_short_branch_names(tmp_path):
    config = saved_synthesis_config(tmp_path)
    config["panel"][1]["branch"] = "parent-01-wave-pareto-00"
    panel, _ = verify_panel(config)
    assert panel[1]["branch"] == "parent-01-wave-pareto-00"
