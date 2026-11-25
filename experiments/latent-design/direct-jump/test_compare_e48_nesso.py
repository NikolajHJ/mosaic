from copy import deepcopy
import hashlib
import json
import sys

import pytest

from compare_e48_nesso import check_search_pair, compare, main


def readout(job, *, joint):
    return dict(seed=9312401, model_sha256="a"*64, hard_recycles=5, geometry_seed=1201,
                checkpoint_sha256="b"*64, protenix_fresh_config_sha256="c"*64,
                planned=dict(updates=9000, fresh_attempts=108), founder=dict(smiles="CC"),
                completed_updates=9000, protenix_attempts_new=2, search_job_id=job,
                search_status="completed", molecules=[dict(smiles="CO", improved_both=joint)],
                summary=dict(distinct_new=1, nesso_scored_new=int(joint is not None),
                             joint_hits=int(joint is True), triple_hits=0))


def test_matched_actual_counts_and_unscored_inputs():
    control, treatment = readout("1", joint=None), readout("2", joint=True)
    result = compare(control, treatment)
    assert result["common_updates"] == 9000 and result["shared_novel_identities"] == 1
    assert result["control"]["comparable_to_founder"] == 0 and result["blend"]["comparable_to_founder"] == 1
    assert result["blend_minus_control_at_equal_updates"]["joint_hits"] == 1
    treatment["completed_updates"] = 4500
    result = compare(control, treatment)
    assert result["common_updates"] is None and result["blend_minus_control_at_equal_updates"] == {}


@pytest.mark.parametrize("field,value", [("geometry_seed", 1202), ("model_sha256", "d"*64),
                                         ("hard_recycles", 1), ("seed", 500)])
def test_different_hard_scoring_context_is_rejected(field, value):
    control, treatment = readout("1", joint=False), readout("2", joint=True)
    treatment[field] = value
    with pytest.raises(ValueError, match=field):
        compare(control, treatment)


def test_rejects_mismatched_discovery_budget_and_latent():
    keys = ("scoring_configs", "design", "search", "perturbation_seed", "graph_spec",
            "vae_checkpoint", "checkpoint_sha256", "confirmation_geometry_seeds")
    control = dict.fromkeys(keys)
    control.update(encoded_founder={"smiles": "CC", "latent": [[1.]]}, job_id="1", optimization_space="vae_latent",
                   budget=dict(fresh_attempts=108, founder_fresh_attempts=1, confirmation_fresh_attempts=8, updates=9000))
    blend = deepcopy(control)
    blend.update(job_id="2", nesso_blend={"seed": 12})
    check_search_pair(control, blend)
    blend["budget"]["fresh_attempts"] -= 1
    with pytest.raises(ValueError, match="budget"):
        check_search_pair(control, blend)
    blend["budget"] = control["budget"]
    blend["encoded_founder"]["latent"] = [[0.]]
    with pytest.raises(ValueError, match="encoded_founder"):
        check_search_pair(control, blend)


def test_standalone_rescore_writes_only_to_evaluation(tmp_path, monkeypatch):
    import combined_native
    from latentmol.design.protenix_nesso_readout import write_nesso_readout

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    class Worker:
        def __init__(self, seed, geometry, folder, assets, *, schema_founder):
            assert seed == 9312401 and schema_founder["smiles"] == "CC" and geometry is None
            self.identity = dict(seed=seed, model_sha256="a"*64, hard_recycles=5)
            self.audit = {"frozen": True}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        @staticmethod
        def fresh(smiles):
            return dict(status="scored", has_interface=True, affinity_pred_value=float(smiles == "CC"),
                        affinity_probability_binary=.6 if smiles == "CO" else .3)

    settings = dict(seed=9312401, assets_dir="/fixture")
    limits = dict(updates=9000, fresh_attempts=108, founder_fresh_attempts=1, confirmation_fresh_attempts=8)
    control, blend, out = (tmp_path / key for key in ("control", "blend", "eval"))
    for path in (control, blend):
        run = dict(job_id=path.name, encoded_founder=dict(smiles="CC"), design={}, search={}, perturbation_seed=1,
                   scoring_configs={"fresh": "c"*64}, graph_spec={}, vae_checkpoint={}, checkpoint_sha256="b"*64,
                   optimization_space="vae_latent", confirmation_geometry_seeds=[2203,3209], budget=limits,
                   parents=[dict(smiles="CC", fresh_loss=2.)])
        if path == blend:
            run["nesso_blend"] = settings
        write(path / "run.json", run)
        write(path / "result.json", dict(status="completed", completed_updates=9000, planned=limits,
              cycles=[dict(cycle=1, fresh_attempts=1, fresh_scored=1)], unique_fresh_attempts=1))
        for folder, smiles, loss, role in (("founder-score", "CC", 2., "initial"), ("cycle-01", "CO", 1., "edit")):
            write(path / folder / "fresh" / "summary.json", dict(status="completed", geometry_seed=1201,
                  candidates=[dict(smiles=smiles, arm="arm", step=0, status="scored", loss=loss, role=role)]))
    write_nesso_readout(blend, Worker.fresh, seed=settings["seed"], model_sha256="a"*64)
    write(blend / "nesso-schema-founder.json", dict(smiles="CC"))
    plan = tmp_path / "plan.json"
    write(plan, dict(nesso_blend=settings))
    monkeypatch.setattr(combined_native, "NessoWorker", Worker)
    monkeypatch.delenv("DBMOL_LAUNCH_READY", raising=False)
    monkeypatch.setattr(sys, "argv", ["compare", "--control", str(control), "--blend", str(blend),
                        "--output", str(out), "--plan", str(plan), "--plan-sha256", hashlib.sha256(plan.read_bytes()).hexdigest()])
    main()
    assert not (control / "nesso-readout.json").exists()
    result = json.loads((out / "comparison.json").read_bytes())
    assert result["control"]["triple_hits"] == 1 and result["shared_novel_identities"] == 1
