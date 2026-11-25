import json

import pytest

from latentmol.design.protenix_nesso_readout import write_nesso_readout


def put(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value))


def candidate(smiles, loss, role="edit"):
    return dict(smiles=smiles, loss=loss, arm="branch-01", step=0, role=role,
                status="scored" if loss is not None else "embedding_failed")


@pytest.mark.parametrize("has_interface", [True, False])
def test_reports_fresh_descendants_relative_to_same_founder(tmp_path, has_interface):
    founder = candidate("CCC", 1.4, role="initial")
    rows = [candidate("CCO", 1.1), candidate("CCN", 1.0), candidate("CCS", 1.3), candidate("COC", None)]
    put(tmp_path / "run.json", dict(parents=[dict(smiles="CCC", fresh_loss=1.4)],
        job_id="1", checkpoint_sha256="old", scoring_configs=dict(fresh="fresh")))
    put(tmp_path / "result.json", dict(status="completed", completed_updates=1500, planned={},
        unique_fresh_attempts=4, cycles=[dict(cycle=1, fresh_attempts=4, fresh_scored=3)]))
    for folder, records in [("founder-score", [founder]), ("cycle-01", rows)]:
        put(tmp_path / folder / "fresh" / "summary.json",
            dict(status="completed", geometry_seed=1201, candidates=records))
    def fresh(smiles):
        if smiles == "CCS":
            return dict(status="feature_failed", reason="no conformer")
        affinity, prob, interface = {"CCC": (1., .2, has_interface), "CCO": (.7, .3, True),
                                     "CCN": (.6, .4, False)}[smiles]
        return dict(status="scored", affinity_pred_value=affinity, affinity_probability_binary=prob,
                    has_interface=interface)

    report = write_nesso_readout(tmp_path, fresh, seed=1, model_sha256="model")
    assert report["protenix_attempts_new"] == 4
    assert report["summary"]["distinct_new"] == 3 and report["summary"]["nesso_scored_new"] == 2
    assert report["summary"]["joint_hits"] == (1 if has_interface else None)
    assert report["summary"]["triple_hits"] == (1 if has_interface else None)
    assert report["summary"]["best_joint_by_contact"] == ("CCO" if has_interface else None)
    assert report["molecules"][2]["improved_both"] is None
    assert report == json.loads((tmp_path / "nesso-readout.json").read_bytes())
    with pytest.raises(FileExistsError):
        write_nesso_readout(tmp_path, fresh, seed=1, model_sha256="model")
