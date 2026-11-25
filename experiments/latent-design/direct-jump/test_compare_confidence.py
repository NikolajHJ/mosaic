import json

import pytest

from compare_confidence import compare


FOUNDER = dict(smiles="CC", seed=111310, start_sha256="a" * 64)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def arm(root, policy, last, completed=False):
    write(root/"founder.json", FOUNDER)
    def row(smiles, contact, affinity, prob, interface=True):
        return dict(smiles=smiles, step=3, branch="test", selected_hard_contact_loss=contact,
                    score_change=contact + .5, native_nesso=dict(status="scored", affinity_pred_value=affinity,
                        affinity_probability_binary=prob, has_interface=interface))
    rows = [row("CC", -.5, .5, .4), row("CCC", -.7, .3, .5),
            row("CCO", -.6, .4, .2), row("CCN", -.8, .2, .9, False)]
    rows.append(dict(smiles="C", selected_hard_contact_loss=-.1, score_change=.4,
                     native_nesso=dict(status="feature_failed", reason="embedding")))
    summary = dict(distinct_new=4, native_scored=4, improved_both=2)
    if policy == "confidence":
        summary["improved_all_three"] = 1
        for r in rows:
            r["improved_all_three"] = r["smiles"] == "CCC"
    for i in range(last+1):
        g = root/"search"/f"generation-{i:02d}"
        write(g/"progress.json", dict(generation=i, updates=700+8400*i))
        write(g/"dual-readouts.json", dict(summary=summary, molecules=rows))
    if completed:
        write(root/"result.json", dict(status="completed", policy=policy, seed=111310, founder_sha256=FOUNDER["start_sha256"],
              planned_search_updates=700+last*8400, wall_seconds=999,
              config=dict(child_generations=last, founder_updates=700, child_updates=700, parents=4,
                          children_per_parent=3, repair_weights=[2], learning_rate=60)))


def test_pair_complete_old_and_new_native_schemas_and_separate_extra_wave(tmp_path):
    f, c = tmp_path/"fixed", tmp_path/"confidence"
    arm(f, "fixed", 5, True)
    arm(c, "confidence", 6, True)
    report = compare(f, c)
    assert report["primary_available"] and list(report["matched_generations"]) == list("012345")
    assert report["confidence_only_seventh_generation"]["updates"] == 51100
    for policy in ("fixed", "confidence"):
        s = report["matched_generations"]["5"][policy]
        assert s["updates"] == 42700 and s["distinct_new"] == 4
        assert s["fresh_attempts_including_founder"] == 5 and s["fresh_scored_new"] == 3
        assert (s["improved_both"], s["improved_all_three"]) == (2, 1)
        assert s["median_binder_probability_joint_hits"] == pytest.approx(.35)
        assert s["median_binder_probability_new"] == .5
        assert s["highest_probability_triple_hit"]["smiles"] == "CCC"
        assert report["arms"][policy]["wall_seconds"] == 999


def test_require_same_founder_probabilities_and_equal_update_checkpoints(tmp_path):
    f, c = tmp_path/"fixed", tmp_path/"confidence"
    arm(f, "fixed", 0)
    arm(c, "confidence", 0)
    write(c/"founder.json", dict(FOUNDER, start_sha256="b" * 64))
    with pytest.raises(ValueError, match="different founder"):
        compare(f, c)
    write(c/"founder.json", FOUNDER)
    write(c/"search/generation-00/progress.json", dict(generation=0, updates=1400))
    with pytest.raises(ValueError, match="update budget"):
        compare(f, c)


def test_distinguish_missing_live_and_explicit_failed_receipts(tmp_path):
    f, c = tmp_path/"fixed", tmp_path/"confidence"
    arm(f, "fixed", 1)
    partial = compare(f, c)
    assert partial["arms"]["confidence"]["status"] == "missing"
    assert partial["arms"]["fixed"]["status"] == "partial"
    assert not partial["primary_available"] and partial["matched_generations"] == {}
    arm(c, "confidence", 1)
    (c/"search/generation-01/dual-readouts.json").write_text("{")
    assert list(compare(f, c)["matched_generations"]) == ["0"]
    launch = tmp_path/"launch.json"
    write(launch, dict(status="worker_exited", returncode=1))
    failed = compare(f, c, confidence_launch=launch)
    assert failed["arms"]["confidence"]["status"] == "failed"
    assert list(failed["matched_generations"]) == ["0"]
