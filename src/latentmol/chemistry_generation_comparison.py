"""Saved, unfiltered generation observations for bounded chemistry experiments.

Use fixed CPU prior coordinates and sampling seeds, with no repair, filtering or
training-RNG consumption. Both valid and invalid sampled graphs are retained by
the existing large-corpus evaluator. These observations do not select training
states or substitute for native/retention/runtime gates.
"""
from pathlib import Path
import json

import torch

from .chemistry_early_health import preserve_observation_state
from .experiments import file_sha256, write_json
from .large_vae_evaluation import evaluate_codes


VERSION = "chemistry-fixed-generation-comparison-v1"
FAILED_VERSION = "chemistry-fixed-generation-failed-endpoint-v1"
DIAGNOSTIC_VERSION = "chemistry-fixed-generation-strength-diagnostic-v1"


def observe_strength_generation(plan, model, generators, output, *, completed_updates):
    """Unfiltered samples at an explicit nonpromotable coefficient-probe point.

    Reuse the identical latent/draw seeds and evaluator. The probe's100-update
    endpoint is not a completed2,500-update proof or a training checkpoint.
    Only this newly created observation's metadata is labelled before return;
    raw graph bytes and numerical observations are untouched.
    """
    from .chemistry_self_guidance_iteration import validate_self_guidance_iteration
    if (validate_self_guidance_iteration(plan) is None
            or type(completed_updates) is not int or completed_updates not in (0,32,100)
            or plan.get("self_strength_probe") != dict(version="valid-self-strength-probe-v1",
                updates=100, gradient_updates=[1,39,96,100])):
        raise ValueError("Require a declared original36 coefficient-probe observation")
    receipt = _observe(plan, model, generators, output, completed_updates=completed_updates)
    path = Path(receipt["path"])
    result = json.loads(path.read_text())
    result.update(version=DIAGNOSTIC_VERSION, self_guidance_iteration=plan["self_guidance_iteration"],
        diagnostic_only=True, promotable=False, epoch_complete=False,
        scope="Unfiltered fixed-seed100-update coefficient diagnostic; not a2,500-update proof or training")
    write_json(path, result)
    return dict(path=str(path),sha256=file_sha256(path))


def observe_generation(plan, model, generators, output, *, completed_updates):
    """Observe identical 1,024 latents/four categorical draws in each arm."""
    from .chemistry_short_continuation import validate_short_continuation
    if validate_short_continuation(plan) is None:
        raise ValueError("Generation comparison requires an explicit bounded campaign")
    if type(completed_updates) is not int or completed_updates not in (0, plan["guarded_restart"]["validation_updates"]):
        raise ValueError("Generation comparison must identify an exact proof endpoint")
    return _observe(plan, model, generators, output, completed_updates=completed_updates)


def observe_failed_generation(plan, model, generators, output, *, completed_updates, failure_gate):
    """Observe an actual failed point without turning it into a completed proof.

    The original failed gate is immutable. This separately versioned sidecar
    binds its exact bytes and actual update index, so both baseline and weak
    arms retain a fair start/failure comparison even when early checks stop the
    proof. There is no repair, filtering, checkpoint promotion or RNG advance.
    """
    from .chemistry_short_continuation import validate_short_continuation
    from .train import state_digest
    if validate_short_continuation(plan) is None:
        raise ValueError("Failed generation observation requires a declared bounded campaign")
    if (type(completed_updates) is not int
            or not 1 <= completed_updates <= plan["guarded_restart"]["validation_updates"]
            or type(failure_gate) is not dict or set(failure_gate) != {"path", "sha256"}):
        raise ValueError("Failed generation observation requires an actual applied proof endpoint")
    gate_path = Path(failure_gate["path"]).resolve(strict=True)
    root = Path(plan["campaign"]).resolve(strict=True)
    if (not gate_path.is_relative_to(root / "stability") or gate_path.name != "gate.json"
            or file_sha256(gate_path) != failure_gate["sha256"]):
        raise ValueError("Failed generation proof escaped its campaign or changed")
    proof = json.loads(gate_path.read_text())
    record = json.loads((gate_path.parent / "run.json").read_text())
    if (proof.get("status") != "failed" or proof.get("plan_sha256") != plan["_plan_sha256"]
            or proof.get("archive_sha256") != plan["archive_sha256"]
            or proof.get("manifest_sha256") != plan["manifest_sha256"]
            or proof.get("starting_checkpoint_sha256") != plan["starting_checkpoint"]["sha256"]
            or type(proof.get("completed_updates")) is not int or proof["completed_updates"] != completed_updates
            or type(proof.get("last_update")) is not int
            or proof.get("last_update") != plan["starting_checkpoint"]["updates"] + completed_updates
            or proof.get("epoch_complete") is not False
            or proof.get("validation_weights_discarded") is not True
            or Path(proof.get("run_dir", "")).resolve(strict=True) != gate_path.parent
            or type(proof.get("job_id")) is not str or not proof["job_id"].isdecimal()
            or int(proof["job_id"]) < 1 or record.get("environment", {}).get("job_id") != proof["job_id"]
            or record.get("source_archive_sha256") != plan["archive_sha256"]
            or record.get("source", {}).get("sha256") != plan["manifest_sha256"]):
        raise ValueError("Failed generation endpoint lost its scientific proof identity")
    model_sha = state_digest(model.state_dict())
    context = dict(path=str(gate_path), sha256=failure_gate["sha256"], run_id=gate_path.parent.name,
        job_id=proof.get("job_id"), primary_status="failed", last_update=proof["last_update"],
        endpoint_status=("failed_partial" if completed_updates < plan["guarded_restart"]["validation_updates"]
                         else "failed_full_budget"), observer_model_sha256=model_sha,
        epoch_complete=False, promotable=False)
    receipt = _observe(plan, model, generators, output, completed_updates=completed_updates, failure=context)
    if state_digest(model.state_dict()) != model_sha:
        raise RuntimeError("Failed generation observer modified the scientific model state")
    return receipt


def _observe(plan, model, generators, output, *, completed_updates, failure=None):
    """One shared fixed-seed/raw-output implementation for full and failed endpoints."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = plan["protocol"]["evaluation_seed"]
    protocol = dict(evaluation_seed=seed, evaluation_batch_size=32,
                    categorical_samples_per_latent=4)
    with preserve_observation_state(model, generators):
        model.eval()
        codes = torch.randn(1024, model.config.latent, generator=torch.Generator().manual_seed(seed))
        metrics = evaluate_codes(model, codes, output / "prior.jsonl", protocol)
    raw = output / "prior.jsonl.gz"
    result = dict(version=VERSION if failure is None else FAILED_VERSION, plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"],
        starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],
        completed_disposable_updates=completed_updates, prior_count=1024,
        categorical_samples_per_latent=4, evaluation_seed=seed, metrics=metrics,
        raw_graphs=dict(path=str(raw), sha256=file_sha256(raw)),
        scope="Matched unfiltered evaluation, no repair; not a valid-mass guarantee or a training checkpoint")
    if failure is not None:
        result.update(failure_proof=failure, epoch_complete=False, promotable=False)
    if "valid_reference_test" in plan:
        from .chemistry_valid_reference_policy import validate_valid_reference_test
        result.update(valid_reference_test=validate_valid_reference_test(plan),
            valid_prior_reference_guidance=dict(plan["protocol"]["valid_prior_reference_guidance"]),
            valid_reference_anchor=model._valid_reference_anchor.assert_unchanged())
    write_json(output / "metrics.json", result)
    return dict(path=str(output / "metrics.json"), sha256=file_sha256(output / "metrics.json"))
