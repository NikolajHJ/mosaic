import json

from repair_report import collect, markdown


def arm(seed, weight, loss):
    hit = None if loss is None else dict(smiles="CCN", branch="generation-01/child-02", step=3,
                                         selected_hard_contact_loss=loss, score_change=loss - 2)
    return dict(seed=seed, repair_weight=weight, starting_smiles="CCC", starting_hard_contact_loss=2.,
                selected=hit, updates=2100, fraction_valid_steps=.6, distinct_valid=9, distinct_new_valid=8,
                mean_sulfur_drift=-.2, mean_final_leaf_endpoint_sulfur_drift=-1.,
                endpoint_sulfur_drifts={"generation-02/child-00": -1}, founder_only={"best": None})


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_summary_keeps_partial_and_missing_tasks_out_of_denominators(tmp_path):
    weights = [0., .1, 1.]
    tasks = {f"balanced-{s}": dict(seed=s, policy="balanced", output=f"job-{s}") for s in (1, 2, 3)}
    plan = tmp_path / "gpu-plan.json"
    write(plan, dict(repair_weights=weights, tasks=tasks))
    arms = {str(w): arm(1, w, {0.: None, .1: 1.4, 1.: 1.2}[w]) for w in weights}
    write(tmp_path / "job-1/result.json", dict(seed=1, status="completed", policy="balanced", arms=arms))
    write(tmp_path / "job-2/repair-0/result.json", arm(2, 0., .9))
    write(tmp_path / "job-2/repair-0.1/generation-00/progress.json", dict(generation=0, updates=500))
    r = collect(plan)
    assert [s["completed_seeds"] for s in r["summary"]] == [[1], [1], [1]]
    assert [s["improved_seeds"] for s in r["summary"]] == [[], [1], [1]]
    assert r["summary"][0]["provisional_seeds"] == [2]
    assert r["summary"][1]["unresolved_seeds"] == [2, 3]
    assert r["best_per_start"][0]["winners"][0]["repair_weight"] == 1
    assert [t["status"] for t in r["tasks"]] == ["completed", "partial", "awaiting_output"]
    table = markdown(r)
    assert "0 / 1" in table and "1 / 1" in table and "through generation 0" in table and "CCN" in table


def test_damaged_receipts_are_explicit(tmp_path):
    plan = tmp_path / "gpu-plan.json"
    write(plan, dict(repair_weights=[0.], tasks={"balanced-1": dict(seed=1, policy="balanced", output="job-1")}))
    bad = arm(1, 0., 2.4)
    write(tmp_path / "job-1/result.json", dict(seed=1, status="completed", policy="balanced", arms={"0.0": bad}))
    report = collect(plan)
    assert report["tasks"][0]["status"] == "receipt_error"
    assert not report["summary"][0]["completed_seeds"]
    assert "does not improve" in markdown(report)


def test_single_weight_jobs_share_a_seed_and_complete_independently(tmp_path):
    weights = [1., 5., 10.]
    task = lambda w: f"balanced-1-w{w:g}"
    plan = tmp_path / "gpu-plan.json"
    write(plan, dict(repair_weights=weights, tasks={task(w): dict(seed=1, policy="balanced", output=task(w),
                                                                 repair_weights=[w]) for w in weights}))

    def finish(weight, digest, offset=0.):
        directory = tmp_path / task(weight)
        row = arm(1, weight, 1.4)
        row["starting_hard_contact_loss"] += offset
        row["selected"]["score_change"] -= offset
        write(directory / "founder.json", dict(seed=1, smiles="CCC", start_sha256=digest))
        write(directory / "result.json", dict(status="completed", seed=1, policy="balanced", arms={str(weight): row}))

    for w in weights[:2]:
        finish(w, "1" * 64)
    partial = collect(plan)
    assert len(partial["seeds"]) == 1 and len(partial["tasks"]) == 3
    assert [s["completed_seeds"] for s in partial["summary"]] == [[1], [1], []]
    assert not partial["best_per_start"][0]["paired_complete"]
    assert partial["summary"][0]["paired_completed_seeds"] == []

    finish(10., "1" * 64, 3.e-7)
    complete = collect(plan)
    assert complete["summary"][1]["paired_completed_seeds"] == [1]
    assert complete["best_per_start"][0]["paired_complete"]
    assert markdown(complete).count("## balanced · seed 1") == 1

    for digest, offset in [("2" * 64, 0.), ("1" * 64, .001)]:
        finish(10., digest, offset)
        damaged = collect(plan)
        assert not any(row["completed_seeds"] for row in damaged["summary"])
        assert damaged["seeds"][0]["status"] == "receipt_error"
        assert "different founder" in markdown(damaged)
