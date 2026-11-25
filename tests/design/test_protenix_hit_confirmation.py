from argparse import Namespace
import json

import pytest

from latentmol.design.protenix_hit_confirmation import confirm_search_hits


def test_held_out_budget_is_shared_and_frozen_before_scoring(tmp_path):
    (tmp_path / "assets.json").write_text("{}")
    fresh = tmp_path / "fresh.json"
    fresh.write_text(json.dumps(dict(geometry_seed=1201, assets_config="assets.json")))
    founders = tmp_path / "panel.json"
    founders.write_text(json.dumps(dict(historical_valid_smiles=["CCN"], points=[{"smiles": "CCCl"}])))
    cycle = tmp_path / "cycle-01"
    cycle.mkdir()

    def candidate(smiles, loss, sa=4.2):
        return dict(smiles=smiles, status="scored", is_founder=False, step=34,
                    loss=loss, chemistry=dict(valid=True, sa_score=sa, sa_mode="pre_smoothing"))

    branches = [dict(name=arm, arm=arm, candidates={"mixed": [
        candidate("CCN", 2.0), candidate("CCCl", 2.1), candidate("CCC", 2.2, 7.0),
        candidate("CCO", 2.3), candidate(extra, loss)]})
        for arm, extra, loss in [("baseline", "CCF", 2.28), ("vina", "CCBr", 2.30001)]]
    (cycle / "batch.json").write_text(json.dumps(dict(branches=branches)))
    args = Namespace(output=tmp_path, fresh_config=fresh)
    config = dict(parent_panel=dict(path=str(founders)), confirm_top_hits=2, sa_mode="pre_smoothing")
    result = dict(status="completed", arms=dict(baseline={}, vina={}), cycles=[dict(cycle=1)])

    def score(geometry_args, _root, rows):
        assert (tmp_path / "confirmation" / "panel.json").is_file()
        seed = json.loads(geometry_args.fresh_config.read_text())["geometry_seed"]
        assert {r["smiles"] for r in rows} == {"CCO", "CCF"}
        return [r | (dict(loss=None, status="embedding_failed") if r["smiles"] == "CCF" and seed == 3209
                     else dict(loss=2.29, status="scored")) for r in rows]

    summary = confirm_search_hits(args, config, result, score)
    assert summary["fresh_attempts"] == 4 and summary["fresh_successes"] == 3
    hits = {r["smiles"]: r for r in summary["candidates"]}
    assert hits["CCF"]["confirmed"] is None and hits["CCF"]["held_out_mean"] is None
    assert hits["CCO"]["confirmed"] and set(hits["CCO"]["discovery_sources"]) == {"baseline", "vina"}
    assert summary["confirmed_by_arm"] == {"baseline": 1, "vina": 1}
    with pytest.raises(FileExistsError):
        confirm_search_hits(args, config, result, score)
