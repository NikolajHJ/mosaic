import json
from pathlib import Path
import runpy

import pytest


OPERATIONS = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae/restart-operations"


def write(path, data):
    path.write_text(json.dumps(data))
    return path


def test_completed_matched_panel_and_read_only_counts(tmp_path):
    extend = runpy.run_path(str(OPERATIONS / "prepare_synthesis_panel.py"))["extend_panel"]
    report = runpy.run_path(str(OPERATIONS / "report_synthesis_novelty.py"))["report"]
    model = dict(checkpoint_sha256="a", decoder_sha256="b", graph_spec={})
    original = dict(model, version=1, pocket_sha256="brd4", points=[{"smiles": "CC"}],
                    historical_valid_smiles=["CC"])
    matched = tmp_path / "matched"
    child = matched / "cycle-01/child"
    child.mkdir(parents=True)
    write(matched / "run.json", dict(model, runner="matched_search", job_id="123"))
    write(matched / "result.json", dict(status="completed", planned={"branches": 1}, cycles=[{"cycle": 1}]))
    write(child.parent / "batch.json", dict(native_identity={"pocket_sha256": "brd4"},
                                            branches=[{"name": "child", "summary": {"updates": 1}}]))
    motif = "NC(O)(O)N"
    (child / "metrics.jsonl").write_text("\n".join(json.dumps(dict(step=i, raw_valid=True, smiles=s))
                                               for i, s in enumerate(("CC", motif))))
    expanded = extend(original, [], [matched], "brd4")
    assert expanded["history_sources"][0]["added_to_history"] == 1
    assert expanded["points"] == original["points"]
    with pytest.raises(ValueError, match="already includes"):
        extend(expanded, [], [matched], "brd4")
    original_file, expanded_file = write(tmp_path / "old.json", original), write(tmp_path / "new.json", expanded)
    pilot = tmp_path / "pilot"
    cycle = pilot / "cycle-01"
    cycle.mkdir(parents=True)
    write(pilot / "run.json", dict(model, parents=[{"smiles": "CC"}], runner="synthesis_waves", job_id="456",
                                  sa_mode="pre_smoothing", historical_identity_exclusions=1))
    metrics = {f"sa<={k},loss<=2.3": dict(distinct=2, diverse=2) for k in (5, 6)}
    write(pilot / "result.json", dict(status="completed", cycles=[{"cycle": 1}],
                                     arms={"standard": {"joint_quality_new_to_history": metrics}}))
    rows = [dict(smiles=s, step=20, status="scored", loss=2.2, is_founder=False,
                 chemistry=dict(valid=True, sa_mode="pre_smoothing", sa_score=4.0)) for s in ("c1ccccc1", motif)]
    write(cycle / "batch.json", dict(native_identity={"pocket_sha256": "brd4"},
         branches=[dict(name="candidate", arm="standard", candidates={"mixed": rows})]))
    (pilot / "confirmation").mkdir()
    write(pilot / "confirmation/result.json", dict(status="completed", confirmed_by_arm={"standard": 1},
          candidates=[dict(smiles=motif, discovery_sources={"standard": {}}, confirmed=True)]))
    result = report(pilot, original_file, expanded_file)
    output = result["arms"]["standard"]["sa<=5,loss<=2.3"]
    assert output["original"]["distinct"] == 2 and output["original"]["with_motif"] == 1
    assert output["expanded"]["distinct"] == 1 and output["expanded"]["with_motif"] == 0
    assert result["held_out_confirmed_smiles"]["standard"] == dict(original=[motif], expanded=[])
    expanded["history_sources"][0]["job_id"] = "456"
    with pytest.raises(ValueError, match="query run itself"):
        report(pilot, original_file, write(expanded_file, expanded))
