"""Continue an authenticated BRD4 campaign with explicit best-primal fallback."""

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--study", choices=("chain", "uniform", "noisy"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Continuation requires an allocated GPU")
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    from protenix_relaxed_gradients import (
        _case_specs,
        _contact_objective,
        _initial_graph,
    )
    from mosaic.small_molecule.gradient_checks import (
        array_digest,
        check_graph_gradients,
    )
    from mosaic.small_molecule.gradient_resume import (
        assert_identical_contexts,
        continue_case,
    )
    from mosaic.small_molecule.io import (
        file_sha256,
        load_context,
        load_graph,
        save_context,
        write_json,
    )
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.provenance import environment_identity, source_identity

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Requires the allocated GPU")
    if file_sha256(args.plan) != args.plan_sha256:
        raise ValueError("Continuation plan identity changed")
    plan = json.loads(args.plan.read_text())
    if plan["status"] != "ready":
        raise ValueError("Continuation readiness failed")
    item = plan["studies"][args.study]
    parent = Path(item["parent_directory"])
    if file_sha256(parent / "summary.json") != item["parent_summary_sha256"]:
        raise ValueError("Parent summary identity changed")
    previous = json.loads((parent / "summary.json").read_text())
    numerical_profile = {
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "jax_enable_x64": jax.config.jax_enable_x64,
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
    }
    if numerical_profile != previous["numerical_profile"]:
        raise ValueError("Continuation changed the actual numerical profile")
    config = item["config"]
    expected = json.loads(json.dumps(previous["config"]))
    expected["checks"]["acceptance_policy"] = "best_primal"
    if config != expected:
        raise ValueError("Continuation changed an unapproved setting")
    fixture_path = parent / "fixture.json"
    if file_sha256(fixture_path) != previous["fixture_sha256"]:
        raise ValueError("Fixture changed")
    fixture = json.loads(fixture_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        **previous,
        "status": "running",
        "cases": [],
        "job_id": os.environ["LSB_JOBID"],
        "config": config,
        "config_sha256": None,
        "continuation_plan_sha256": args.plan_sha256,
        "parent_job_id": item["parent_job_id"],
        "parent_summary_sha256": item["parent_summary_sha256"],
        "completion_scope": "explicit exploratory policy; inherited strict prefixes, recorded increases allowed",
    }
    report.pop("error", None)
    report.pop("source", None)
    report.update(
        numerical_profile=numerical_profile,
        prepared_context_reused=True,
        prepared_summary_path=str(parent / "summary.json"),
        prepared_summary_sha256=item["parent_summary_sha256"],
        prepared_configuration_policy="acceptance_policy_change_with_authenticated_prefix",
    )
    try:
        shutil.copy2(args.plan, args.output / "continuation_plan.json")
        (args.output / "fixture.json").write_bytes(fixture_path.read_bytes())
        write_json(args.output / "configuration.json", config)
        report["config_sha256"] = file_sha256(args.output / "configuration.json")
        model = FrozenProtenixTiny.load(cache_dir=args.cache_dir)
        context = load_context(parent / "context", previous["context_files"])
        pilot_dir = Path(plan["pilot_directory"])
        if file_sha256(pilot_dir / "summary.json") != plan["pilot_summary_sha256"]:
            raise ValueError("Pilot identity changed")
        pilot = json.loads((pilot_dir / "summary.json").read_text())
        assert_identical_contexts(
            context, load_context(pilot_dir / "context", pilot["context_files"])
        )
        before_model, before_context = array_digest(model), array_digest(context)
        if (
            model.identities != previous["model"]
            or before_model != previous["model_digest_before"]
            or before_context != previous["context_digest_before"]
        ):
            raise ValueError("Frozen model/context identity changed")
        report.update(
            model_digest_before=before_model,
            context_digest_before=before_context,
            context_files=save_context(args.output / "context", context),
            source=source_identity(Path(__file__).resolve().parents[1]),
            environment=environment_identity(),
        )
        hard = load_graph(parent / "hard_graph.npz", previous["hard_graph_sha256"])
        if context.n_atoms != hard.n_atoms:
            raise ValueError("Context/graph atom count mismatch")
        shutil.copy2(parent / "hard_graph.npz", args.output / "hard_graph.npz")
        report["hard_graph_sha256"] = previous["hard_graph_sha256"]
        objective, pocket = _contact_objective(config, context, fixture)
        report["pocket_token_indices"] = pocket
        loss = model.build_loss(context=context, loss=objective)
        key = jax.random.key(config["seed"])
        for index, (case, entry) in enumerate(
            zip(_case_specs(config), item["cases"], strict=True)
        ):
            if entry is None:
                result = check_graph_gradients(
                    loss,
                    _initial_graph(hard, case, key),
                    key=key,
                    output_directory=args.output / f"case-{index}",
                    **config["checks"],
                )
            else:
                result = continue_case(
                    loss,
                    entry=entry,
                    key=key,
                    output_directory=args.output / f"case-{index}",
                    checks=config["checks"],
                )
            report["cases"].append({**case, **result})
            write_json(args.output / "summary.json", report)
        report.update(
            model_digest_after=array_digest(model),
            context_digest_after=array_digest(context),
        )
        if (
            report["model_digest_after"] != before_model
            or report["context_digest_after"] != before_context
        ):
            raise ValueError("Frozen identity changed during continuation")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
