"""Matched CPU speed/loss/gradient study; no optimizer or training promotion.

Execute only inside its immutable allocated HPC release. All modes see the
same saved logits at the production 1:3 ratio. Pools persist, every measured
pass starts with an empty logical target cache, and backend order rotates.
This study supplements, and cannot replace, native full-update GPU readiness.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.chemistry_continuation import (authenticated, digest, load_parent_config, nested_equal,
                                             read_plan, composition_reference)
from latentmol.data import GraphDataset, collate_graphs, validate_large_cache
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput
from latentmol.train import capture_rng_state, seed_everything
from paired_repair_loss_audit_helpers import loss_audit, compare_loss_audits, objective_terms


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest_value = hashlib.sha256(f"{value.dtype}|{tuple(value.shape)}|".encode())
    digest_value.update(value.numpy().tobytes())
    return digest_value.hexdigest()


def output_sha(decoded):
    return {field: tensor_sha(getattr(decoded, field)) for field in ("atoms", "charges", "bonds")}


def model_sha(model):
    return digest({name: tensor_sha(value) for name, value in model.state_dict().items()})


def semantic(receipt):
    """Exactly the production semantic receipt exclusions, nothing else."""
    result = {key: value for key, value in receipt.items()
              if key not in ("elapsed_seconds", "cache_hits", "search_checks")}
    result["attempts"] = [{key: value for key, value in row.items() if key != "cache_hit"}
                          for row in result["attempts"]]
    return result


def concat(real, prior):
    return DecoderOutput(*(torch.cat([getattr(real, field), getattr(prior, field)])
                           for field in ("atoms", "charges", "bonds")))


def fixed_fixture(spec):
    """Thirty-two probability-distinct O#C fixtures warm every worker.

    This deliberately invalid diagnostic is separate from real/prior cohorts.
    Its supported O=C witness must give nonzero atom, charge and bond gradients.
    Distinct probabilities prevent the exact teacher cache reducing32 searches
    to one and accidentally putting process startup in measured timings.
    """
    n, count = spec.max_nodes, 32
    atoms = torch.full((count, n, len(spec.atomic_numbers)), -12.)
    charges = torch.full((count, n, len(spec.formal_charges)), -12.)
    bonds = torch.full((count, n, n, len(spec.bond_names)), -12.)
    atoms[..., 0] = 0; charges[..., spec.formal_charges.index(0)] = 0; bonds[..., 0] = 0
    atoms[:, :2, 0] = -12
    atoms[:, 0, spec.atomic_numbers.index(8)] = 0
    atoms[:, 1, spec.atomic_numbers.index(6)] = 0
    for row in range(count):
        for i, j in ((0, 1), (1, 0)):
            bonds[row, i, j, 0] = -12
            bonds[row, i, j, spec.bond_names.index("SINGLE")] = -4
            bonds[row, i, j, spec.bond_names.index("DOUBLE")] = -.1 + row * .00001
            bonds[row, i, j, spec.bond_names.index("TRIPLE")] = 0
    return DecoderOutput(atoms, charges, bonds)


def materialize(model, data, seed, total):
    real_count = total // 4
    rows = np.random.default_rng(seed).choice(len(data), real_count, replace=False).tolist()
    hard = collate_graphs([data[row] for row in rows])
    posterior_rng = torch.Generator().manual_seed(seed + 100000)
    prior_rng = torch.Generator().manual_seed(seed + 200000)
    with torch.no_grad():
        mu, logvar = model.encode(hard.to_relaxed(model.spec))
        epsilon = torch.randn(mu.shape, generator=posterior_rng)
        real_z = mu + epsilon * (.5 * logvar).exp()
        prior_z = torch.randn(3 * real_count, model.config.latent, generator=prior_rng)
        real, prior = model.decode(real_z), model.decode(prior_z)
        occupancy = prior.probabilities().occupancy.detach().clone()
    return dict(rows=rows, hard=hard, mu=mu, logvar=logvar, epsilon=epsilon,
                real_z=real_z, prior_z=prior_z, real=real, prior=prior, occupancy=occupancy)


def audit_inputs(cohort, targets, model, config, protocol, composition):
    count = len(cohort["rows"])
    return dict(hard=cohort["hard"], real_targets=targets.subset(range(count)),
                prior_targets=targets.subset(range(count, count * 4)), spec=model.spec,
                config=config, protocol=protocol, mu=cohort["mu"], logvar=cohort["logvar"],
                reference_prior_occupancy=cohort["occupancy"], composition=composition,
                epoch=36, phase_start_epoch=35)


def parameter_audit(model, cohort, arguments):
    """Actual decoder VJP at fixed4+12 latent inputs; weights never change."""
    parameters = list(model.decoder.named_parameters())
    flags = [parameter.requires_grad for _, parameter in parameters]
    try:
        model.decoder.requires_grad_(True)
        real, prior = model.decode(cohort["real_z"]), model.decode(cohort["prior_z"])
        metrics, total = objective_terms(real, prior, **arguments)
        gradients = torch.autograd.grad(total, [parameter for _, parameter in parameters], allow_unused=True)
        values = {name: value for (name, _), value in zip(parameters, gradients)}
        attached = [value for value in gradients if value is not None]
        if not attached or any(not bool(value.isfinite().all()) for value in attached):
            raise ValueError("Decoder VJP is detached or nonfinite")
        norm = float(torch.stack([value.detach().double().square().sum() for value in attached]).sum().sqrt())
        if norm <= 0:
            raise ValueError("Decoder VJP is entirely zero")
        metrics["decoder_gradient_diagnostics"] = dict(attached_parameters=len(attached),
            nonzero_parameters=sum(bool(torch.count_nonzero(value)) for value in attached), norm=norm)
        return metrics, values
    finally:
        for (_, parameter), flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True); args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Paired molecular study requires an allocated DTU CPU job")
    authenticated(args.config, args.config_sha256)
    study = json.loads(Path(args.config).read_text())
    if (study["total_points"] != [16, 64, 128] or len(study["seeds"]) != 3 or study["repeats"] != 3
            or torch.get_num_threads() != 4 or int(os.environ.get("LSB_DJOB_NUMPROC", "0")) != 16):
        raise ValueError("Study differs from declared bounded1:3 CPU design")
    plan = read_plan(study["base_plan"], study["base_plan_sha256"])
    root = Path(study["run_root"])
    if (root / "study_result.json").exists():
        raise FileExistsError("Retain the existing study outcome")
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(status="running", job_id=os.environ["LSB_JOBID"], study_config_sha256=args.config_sha256,
                  base_plan_sha256=study["base_plan_sha256"], parent_sha256=plan["parent_checkpoint_sha256"],
                  scope="CPU teacher/loss/VJP supplement, not native full-update readiness or training",
                  cohorts=[], no_optimizer_or_weight_update=True)
    with ExperimentRun(root / "execution", "paired-repair-loss", study, Path.cwd()) as run, ExitStack() as resources:
        try:
            seed_everything(study["seed"])
            authenticated(plan["parent_checkpoint"], plan["parent_checkpoint_sha256"])
            authenticated(plan["config"], plan["config_sha256"])
            config = load_parent_config(Path(plan["config"]))
            cache = validate_large_cache(plan["data"])
            data = GraphDataset(plan["data"], "train", validated_cache=cache)
            if data.manifest["dataset_sha256"] != plan["dataset_sha256"] or data.manifest["split_sha256"] != plan["split_sha256"]:
                raise ValueError("Study data identity differs")
            model, _ = load_model(Path(plan["parent_checkpoint"]), "cpu", data.spec)
            model.eval().requires_grad_(False)
            before_weights = model_sha(model)
            composition = composition_reference(data)
            write_json(run.path / "composition.json", composition)
            modes = [("reference", "reference", 0), ("fast_serial", "optimized", 0),
                     ("workers4", "optimized", 4), ("workers12", "optimized", 12)]
            teachers = {name: resources.enter_context(AtomPreservingRepairTeacher(model.spec,
                ProbabilityRepairConfig(**plan["protocol"]["search"]), backend=backend, workers=workers))
                for name, backend, workers in modes}
            fixture, fixture_reference, fixture_records = fixed_fixture(model.spec), None, {}
            for name, _, _ in modes:
                tick = time.perf_counter()
                target = teachers[name].build_targets(fixture, candidate_ids=list(range(999000, 999032)), update=0)
                cold_seconds = time.perf_counter() - tick
                current = semantic(target.receipt)
                if fixture_reference is None: fixture_reference = current
                if current != fixture_reference or target.receipt["accepted_targets"] != 32:
                    raise ValueError("Controlled invalid fixture changed repair or failed to exercise it")
                leaves = DecoderOutput(*(getattr(fixture, field).detach().clone().requires_grad_()
                                         for field in ("atoms", "charges", "bonds")))
                loss = repair_loss(leaves, target, model.spec)
                grads = torch.autograd.grad(loss, (leaves.atoms, leaves.charges, leaves.bonds))
                if not all(bool(torch.count_nonzero(value)) for value in grads):
                    raise ValueError("Controlled fixture did not exercise all three output heads")
                fixture_records[name] = dict(cold_seconds=cold_seconds, semantic_sha256=digest(current),
                    loss=float(loss.detach()), gradients=[tensor_sha(value) for value in grads])
                teachers[name].clear_cache()
            if any((row["loss"], row["gradients"]) != (fixture_records["reference"]["loss"],
                    fixture_records["reference"]["gradients"]) for row in fixture_records.values()):
                raise ValueError("Controlled fixture changed NLL/logit gradients")
            report["controlled_fixture"] = fixture_records
            names = [row[0] for row in modes]
            parameter_reference, parameter_records = None, {}
            for seed in study["seeds"]:
                for total_points in study["total_points"]:
                    if time.monotonic() - started > study["wall_seconds"]:
                        raise TimeoutError("Bounded study budget exhausted; preserve partial evidence")
                    case = len(report["cohorts"])
                    cohort = materialize(model, data, seed, total_points)
                    decoded = concat(cohort["real"], cohort["prior"])
                    frozen_sha = output_sha(decoded)
                    ids = list(range(case * 128, case * 128 + total_points))
                    artifact = run.path / f"cohort-{case:02d}.pt"
                    torch.save({"rows": cohort["rows"], "real_z": cohort["real_z"], "prior_z": cohort["prior_z"],
                        "mu": cohort["mu"], "logvar": cohort["logvar"], "epsilon": cohort["epsilon"],
                        "real_logits": {field: getattr(cohort["real"], field) for field in ("atoms", "charges", "bonds")},
                        "prior_logits": {field: getattr(cohort["prior"], field) for field in ("atoms", "charges", "bonds")}}, artifact)
                    teachers["reference"].clear_cache()
                    canonical = teachers["reference"].build_targets(decoded, candidate_ids=ids, update=case)
                    expected_semantic = semantic(canonical.receipt)
                    write_json(run.path / f"cohort-{case:02d}-reference.json", canonical.receipt)
                    expected_audit = loss_audit(cohort["real"], cohort["prior"],
                        **audit_inputs(cohort, canonical, model, config, plan["protocol"], composition))
                    row = dict(seed=seed, total_points=total_points, real_points=len(cohort["rows"]),
                        prior_points=3 * len(cohort["rows"]), row_ids=cohort["rows"], input_sha256=frozen_sha,
                        inputs_file=str(artifact), inputs_file_sha256=file_sha256(artifact),
                        semantic_sha256=digest(expected_semantic), reference_loss=expected_audit.record(),
                        reference_statuses=canonical.receipt["status_counts"], repetitions=[], timing_summary={})
                    if case == 0:
                        parameter_metrics, parameter_reference = parameter_audit(model, cohort,
                            audit_inputs(cohort, canonical, model, config, plan["protocol"], composition))
                        parameter_records["reference"] = dict(metrics=parameter_metrics,
                            gradients={name: None if value is None else tensor_sha(value) for name, value in parameter_reference.items()})
                    for repeat in range(study["repeats"]):
                        offset = (case + repeat) % len(names)
                        order = names[offset:] + names[:offset]
                        for name in order:
                            if time.monotonic() - started > study["wall_seconds"]:
                                raise TimeoutError("Bounded paired-pass budget exhausted")
                            teachers[name].clear_cache()
                            rng = capture_rng_state({}, False)
                            tick = time.perf_counter()
                            target = teachers[name].build_targets(decoded, candidate_ids=ids, update=case)
                            seconds = time.perf_counter() - tick
                            if not nested_equal(rng, capture_rng_state({}, False)) or output_sha(decoded) != frozen_sha:
                                raise ValueError("Search mutated source logits or parent RNG")
                            if (semantic(target.receipt) != expected_semantic or target.receipt["search_checks"] != canonical.receipt["search_checks"]
                                    or target.receipt["cache_hits"] != canonical.receipt["cache_hits"]):
                                raise ValueError("Ordered search/target/budget/cache semantics differ")
                            actual_audit = loss_audit(cohort["real"], cohort["prior"],
                                **audit_inputs(cohort, target, model, config, plan["protocol"], composition))
                            parity = compare_loss_audits(expected_audit, actual_audit)
                            if not parity["passed"]:
                                write_json(run.path / f"failed-loss-parity-{case}-{repeat}-{name}.json", parity)
                                raise ValueError("Fixed-logit loss or gradient equivalence failed")
                            if case == 0 and repeat == 0 and name != "reference":
                                parameter_metrics, gradients = parameter_audit(model, cohort,
                                    audit_inputs(cohort, target, model, config, plan["protocol"], composition))
                                same = nested_equal(parameter_metrics, parameter_records["reference"]["metrics"])
                                same = same and nested_equal(gradients, parameter_reference)
                                parameter_records[name] = dict(passed=same, metrics=parameter_metrics,
                                    gradients={key: None if value is None else tensor_sha(value) for key, value in gradients.items()})
                                if not same: raise ValueError("Actual decoder parameter VJP changed")
                            observation = dict(repeat=repeat, order=order, backend=name, seconds=seconds,
                                search_checks=target.receipt["search_checks"], accepted_targets=target.receipt["accepted_targets"],
                                semantic_sha256=digest(semantic(target.receipt)), loss_parity=parity)
                            row["repetitions"].append(observation)
                            run.log(dict(stage="paired_pass", case=case, seed=seed, total_points=total_points,
                                         backend=name, repeat=repeat, seconds=seconds, exact_loss_and_gradients=True))
                    for name in names:
                        times = [value["seconds"] for value in row["repetitions"] if value["backend"] == name]
                        reference_times = {value["repeat"]: value["seconds"] for value in row["repetitions"] if value["backend"] == "reference"}
                        ratios = [reference_times[value["repeat"]] / value["seconds"] for value in row["repetitions"] if value["backend"] == name]
                        row["timing_summary"][name] = dict(median_seconds=statistics.median(times), min_seconds=min(times),
                            max_seconds=max(times), median_paired_speedup=statistics.median(ratios))
                    report["cohorts"].append(row)
                    write_json(root / "study_result.json", report)
                    write_json(run.path / f"cohort-{case:02d}-summary.json", row)
            if not any(row["reference_loss"]["metrics"]["repair_prior"] > 0 for row in report["cohorts"]):
                raise ValueError("Predeclared native cohorts did not exercise nonzero prior repair loss")
            if model_sha(model) != before_weights:
                raise ValueError("Study changed frozen model parameters/buffers")
            report.update(status="passed", decoder_parameter_gradient_check=parameter_records,
                before_model_sha256=before_weights, after_model_sha256=model_sha(model),
                elapsed_seconds=time.monotonic() - started, run_dir=str(run.path))
            write_json(root / "study_result.json", report)
            run.annotate(study_status="passed", model_unchanged=True)
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}",
                          elapsed_seconds=time.monotonic() - started, run_dir=str(run.path))
            write_json(root / "study_result.json", report)
            raise


if __name__ == "__main__":
    main()
