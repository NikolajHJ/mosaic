"""Bounded-search teacher experiment on a declared fixed prior training pool.

Search/RDKit never enters autograd. Only live probability of verified held
targets supplies the repair gradient. All arms execute the same maximum oracle
quota, including coefficient-zero controls, with failures retained. This is an
alternating finite-pool surrogate, not globally nearest-valid distance or an
unbiased expected stochastic repair objective. Native efficacy is untested
until the HPC pilot supplies actual held-out evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import torch

from latentmol.data import GraphDataset
from latentmol.decoder_finetune import FineTuneConfig, assess_advantage, fit_case, make_panel, subset_loader, tensor_sha
from latentmol.decoder_regularization import reference_anchors, validate_coefficient
from latentmol.evaluate import load_model, reconstruction_metrics
from latentmol.experiments import file_sha256, write_json
from latentmol.losses import LossConfig
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput
from latentmol.search_repair_guidance import SearchRepairTeacher
from latentmol.train import seed_everything, state_digest


def decode_panel(model, panel, batch_size=64):
    """One shared decoder; pool points have a fixed independent identity."""
    outputs = [model.decode(panel[start:start + batch_size].to(next(model.parameters()).device))
               for start in range(0, len(panel), batch_size)]
    return DecoderOutput(*(torch.cat([getattr(output, name) for output in outputs])
                           for name in ("atoms", "charges", "bonds")))


class SearchPanelGuidance:
    """Reusable training hook; epoch-refresh oracle and immutable targets.

    Common occupancy/aromatic anchors on this pool apply in every arm as well
    as the parent trainer's fresh-prior anchors. Pool occupancy can change at a
    later refresh; each raw→witness search keeps its own exact occupied slots.
    Unconditioned NLL includes all NO_ATOM choices and learns their retention.
    No target is applied to an unrelated fresh random latent point.
    """
    def __init__(self, panel, spec, repair_config, *, name, weight, batch_size=8, seed=71001):
        if batch_size < 1 or batch_size > len(panel):
            raise ValueError("Invalid search-pool batch size")
        self.name = name
        self.weight = validate_coefficient(weight)
        self.panel = panel.detach().cpu().clone()
        self.panel_sha = tensor_sha(self.panel)
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.teacher = SearchRepairTeacher(self.panel, spec, repair_config, generator=self.generator)
        self.refreshes = []
        self.contributing_points = self.scheduled_points = self.updates = self.weighted_eligible_updates = 0
        self.reference_panel = None

    def identity(self):
        return {"name": self.name, "repair_weight": self.weight, "panel_sha256": self.panel_sha,
                "pool_count": len(self.panel), "batch_size": self.batch_size,
                "repair_policy": self.teacher.repair_config.policy(), "refresh": "start_of_every_epoch",
                "scope": "Frozen per-refresh witnesses; no search/sampling pathwise gradient",
                "normalization": "batch mean including missing-target zeros; per graph N+K+K(K-1)/2",
                "common_pool_anchors": "original occupancy/aromatic KL, weight1 each, all arms"}

    def begin_epoch(self, model, reference, epoch, update, run_path):
        if tensor_sha(self.panel) != self.panel_sha:
            raise RuntimeError("Training latent pool changed")
        with torch.no_grad():
            if self.reference_panel is None:
                ref = decode_panel(reference, self.panel)
                self.reference_panel = DecoderOutput(*(getattr(ref, name).detach() for name in ("atoms", "charges", "bonds")))
            decoded = decode_panel(model, self.panel)
            receipt = self.teacher.refresh(decoded, update=update,
                directory=run_path / f"repair-refresh-epoch-{epoch:03d}")
        receipt_path = run_path / f"repair-refresh-epoch-{epoch:03d}" / f"search-refresh-{update:08d}.json"
        self.refreshes.append({**{key: value for key, value in receipt.items() if key != "attempts"},
                               "receipt_path": str(receipt_path), "receipt_file_sha256": file_sha256(receipt_path)})

    def loss(self, model, reference, update):
        ids = (torch.arange(self.batch_size) + update * self.batch_size) % len(self.panel)
        device = next(model.parameters()).device
        decoded = model.decode(self.panel[ids].to(device))
        ref = DecoderOutput(*(getattr(self.reference_panel, name)[ids.to(device)] for name in ("atoms", "charges", "bonds")))
        anchors = reference_anchors(decoded, ref, model.spec)
        objective = self.teacher.objective(decoded, latent_ids=ids)
        available = int(objective["available_mask"].sum())
        self.contributing_points += available
        self.scheduled_points += len(ids)
        self.updates += 1
        self.weighted_eligible_updates += int(self.weight > 0 and available > 0)
        loss = (self.weight * objective["total"] + anchors["occupancy_anchor"] + anchors["aromatic_anchor"])
        return loss, {"search_target_nll": objective["total"], "search_target_available_fraction": decoded.atoms.new_tensor(available / len(ids)),
                      "search_pool_occupancy_anchor": anchors["occupancy_anchor"],
                      "search_pool_aromatic_anchor": anchors["aromatic_anchor"]}

    def receipt(self):
        return {**self.identity(), "refreshes": self.refreshes, "scheduled_points": self.scheduled_points,
                "available_target_points": self.contributing_points, "updates": self.updates,
                "weighted_eligible_guidance_updates": self.weighted_eligible_updates,
                "actual_nonzero_repair_gradient_updates": "not inferred from eligibility",
                "search_generator_sha256": tensor_sha(self.generator.get_state())}


def probe_fixed_teacher_gradient(model, teacher, panel, directory):
    """FD the native FP32 scalar with labels/search pool held completely fixed.

    Distinguish a measurable slope from baseline/ULP noise. Gradient and seeded
    random unit directions are predeclared. Search absence/zero signal is
    uninformative, never called a passing native repair gradient. It does not
    gate scientific coverage or force a valid witness into the cohort.
    """
    directory.mkdir(parents=True, exist_ok=False)
    before = state_digest(model.state_dict())
    model.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        full = teacher.objective(decode_panel(model, panel))
    available = full["available_mask"].nonzero().flatten().cpu().tolist()
    if not available:
        result = {"status": "uninformative_no_native_witness", "gradient_verified": False}
    else:
        ids = [available[0]]
        z = panel[ids].to(device).detach().requires_grad_(True)
        def objective(value):
            return teacher.objective(model.decode(value), latent_ids=ids)["total"]
        value = objective(z)
        gradient = torch.autograd.grad(value, z)[0]
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite native search-target gradient")
        norm = float(gradient.norm())
        if norm == 0:
            result = {"status": "uninformative_zero_native_gradient", "gradient_verified": False, "latent_ids": ids}
        else:
            random = torch.randn(z.shape, generator=torch.Generator().manual_seed(71002)).to(device)
            directions = {"gradient_unit": gradient.detach() / gradient.norm(), "seeded_random_unit": random / random.norm()}
            rows = []
            with torch.no_grad():
                baseline = objective(z.detach())
                repeat = objective(z.detach())
                baseline_variation = abs(float(repeat - baseline))
                for name, direction in directions.items():
                    analytic = float((gradient * direction).sum())
                    for epsilon in (0.01, 0.003, 0.001):
                        plus, minus = objective(z.detach() + epsilon * direction), objective(z.detach() - epsilon * direction)
                        finite = float((plus.double() - minus.double()) / (2 * epsilon))
                        ulp_plus = abs(float(torch.nextafter(plus, plus.new_tensor(float('inf'))) - plus))
                        ulp_minus = abs(float(torch.nextafter(minus, minus.new_tensor(float('inf'))) - minus))
                        noise = (ulp_plus + ulp_minus + 2 * baseline_variation) / (2 * epsilon)
                        informative = abs(analytic) > 5 * noise
                        error = abs(finite - analytic)
                        passed = informative and error <= 0.03 * abs(analytic) + 2 * noise + 1e-7
                        rows.append({"direction": name, "epsilon": epsilon, "autograd": analytic, "finite_difference": finite,
                                     "scalar_roundoff_proxy": noise, "informative": informative, "passed": passed, "absolute_error": error})
            gradient_rows = [row for row in rows if row["direction"] == "gradient_unit"]
            verified = any(gradient_rows[i]["passed"] and gradient_rows[i + 1]["passed"] for i in range(2))
            direction_status = {}
            for name in directions:
                selected = [row for row in rows if row["direction"] == name]
                direction_status[name] = "verified_local_plateau" if any(selected[i]["passed"] and selected[i + 1]["passed"] for i in range(2)) else "inconclusive_no_measurable_agreement_plateau"
            verified = all(status == "verified_local_plateau" for status in direction_status.values())
            result = {"status": "verified" if verified else "inconclusive_or_uninformative",
                "gradient_verified": verified, "rows": rows, "latent_ids": ids, "direction_status": direction_status,
                "loss": float(value), "latent_gradient_norm": norm, "baseline_repeat_variation": baseline_variation,
                "dtype": str(value.dtype), "scope": "Native FP32 frozen VAE; same selected targets/IDs across all probes",
                "roundoff_scope": "Scalar ULP/repeat proxy, not a rigorous network arithmetic error bound; coarse curvature does not imply broken autograd"}
    if state_digest(model.state_dict()) != before:
        raise RuntimeError("Gradient probe changed pretrained weights")
    write_json(directory / "probe.json", result)
    return result


def read_protocol(path, expected_sha):
    if file_sha256(path) != expected_sha:
        raise ValueError("Search study config hash mismatch")
    protocol = json.loads(path.read_text())
    required = {"version", "initializer", "initializer_sha256", "data", "dataset_sha256", "split_sha256", "fine_tune",
                "seed", "arms", "train_limit", "validation_limit", "validation_prior_count", "final_prior_count",
                "final_repair_limit", "training_prior_count", "training_prior_batch", "repair", "case_budget_seconds",
                "controller_budget_seconds", "smoke_budget_seconds", "preceding_job"}
    if set(protocol) != required or protocol["version"] != 1:
        raise ValueError("Unknown/missing search study config fields")
    FineTuneConfig(**protocol["fine_tune"])
    ProbabilityRepairConfig(**protocol["repair"])
    for name in ("seed", "train_limit", "validation_limit", "validation_prior_count", "final_prior_count", "final_repair_limit",
                 "training_prior_count", "training_prior_batch", "case_budget_seconds", "controller_budget_seconds", "smoke_budget_seconds"):
        if type(protocol[name]) is not int or protocol[name] < (0 if name == "seed" else 1):
            raise ValueError("Invalid protocol integer: " + name)
    expected = [{"name": "control", "defect_weight": 0., "repair_weight": 0.},
                {"name": "defect", "defect_weight": 1., "repair_weight": 0.},
                {"name": "search-low", "defect_weight": 0., "repair_weight": .1},
                {"name": "search-high", "defect_weight": 0., "repair_weight": 1.}]
    if protocol["arms"] != expected or protocol["training_prior_batch"] > protocol["training_prior_count"]:
        raise ValueError("Protocol differs from prespecified matched pilot")
    if not isinstance(protocol["preceding_job"], str) or not protocol["preceding_job"].isdigit():
        raise ValueError("Preceding job must be a scheduler ID")
    return protocol


def informative_native_gradient(coverage, probe):
    """Coverage alone cannot pass the full comparison's derivative gate."""
    return (type(coverage.get("accepted_targets")) is int and coverage["accepted_targets"] > 0
            and probe.get("gradient_verified") is True
            and probe.get("direction_status") == {
                "gradient_unit": "verified_local_plateau",
                "seeded_random_unit": "verified_local_plateau"})


def run_study(protocol, output_root, source_root, *, device="cuda",
              evaluation_reserve_seconds=1800, require_full_data=False,
              require_native_gradient=False,
              scope="One-seed reduced-data pilot; fixed training prior pool; held-out generation; no synthesis/binding claim"):
    """Shared comparison engine behind separately validated pilot/full contracts.

    Defaults preserve source071's pilot behavior. The full-study controller
    requires every training graph, an informative actual GPU derivative gate
    and a larger evaluation reserve. Search checkpoints cannot resume their
    teacher/cache state; censored arms remain incomplete, never final results.
    """
    from latentmol.decoder_quality_evaluation import evaluate_random_panel, paired_compare
    initializer = Path(protocol["initializer"])
    if file_sha256(initializer) != protocol["initializer_sha256"]:
        raise ValueError("Search study initializer mismatch")
    if require_full_data and file_sha256(Path(protocol["data"]) / "dataset.npz") != protocol["dataset_sha256"]:
        raise ValueError("Full comparison protected dataset bytes changed")
    train, val, test = [GraphDataset(Path(protocol["data"]), split) for split in ("train", "val", "test")]
    if train.manifest["dataset_sha256"] != protocol["dataset_sha256"] or train.manifest["split_sha256"] != protocol["split_sha256"]:
        raise ValueError("Search study protected dataset/split mismatch")
    if protocol["train_limit"] > len(train) or protocol["validation_limit"] > len(val):
        raise ValueError("Do not silently shrink the prespecified pilot")
    if require_full_data and protocol["train_limit"] != len(train):
        raise ValueError("Full comparison must use every protected training molecule")
    if type(evaluation_reserve_seconds) is not int or evaluation_reserve_seconds <= 0:
        raise ValueError("Evaluation reserve must be a positive integer")
    if require_native_gradient and (device != "cuda" or not torch.cuda.is_available()):
        raise ValueError("Full comparison requires an actual CUDA gradient preflight")
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output_root / "protocol.json", protocol)
    config = FineTuneConfig(**protocol["fine_tune"])
    reference, checkpoint = load_model(initializer, device, expected_spec=train.spec)
    if checkpoint["dataset_sha256"] != protocol["dataset_sha256"] or checkpoint["train_config"]["mode"] != "vae":
        raise ValueError("Search study must start from the authenticated VAE")
    seed_everything(protocol["seed"])
    pool = make_panel(protocol["training_prior_count"], reference.config.latent, 71000)
    validation = make_panel(protocol["validation_prior_count"], reference.config.latent, 81000)
    final = make_panel(protocol["final_prior_count"], reference.config.latent, 91000)
    torch.save({"training": pool, "validation": validation, "final": final}, output_root / "latent-panels.pt")
    repair = ProbabilityRepairConfig(**protocol["repair"])
    # Native real-search coverage and directional probes are independent of the
    # pilot's effectiveness; absent witnesses remain a reportable outcome.
    probe_teacher = SearchRepairTeacher(pool, reference.spec, repair,
                                       generator=torch.Generator().manual_seed(71001))
    with torch.no_grad():
        probe_receipt = probe_teacher.refresh(decode_panel(reference, pool), update=0,
                                            directory=output_root / "native-search-coverage")
    probe = probe_fixed_teacher_gradient(reference, probe_teacher, pool, output_root / "native-gradient-probe")
    if require_native_gradient and not informative_native_gradient(probe_receipt, probe):
        write_json(output_root / "native_engineering_gate.json", {
            "status": "failed_or_uninformative", "native_gradient_probe": probe,
            "accepted_targets": probe_receipt["accepted_targets"],
            "scope": "No scientific arm launched: informative fixed-target CUDA gradient required"})
        raise RuntimeError("Full search comparison native gradient gate failed or was uninformative")
    # A small control/guided pair exercises the actual CUDA training hook.
    smoke_config = FineTuneConfig(**{**asdict(config), "epochs": 1})
    smoke = []
    common = dict(initializer=initializer, train_data=train, val_data=val, device=device, source_root=source_root)
    for name, weight in (("smoke-control", 0.), ("smoke-search", 1.)):
        hook = SearchPanelGuidance(pool[:16], train.spec, repair, name=name, weight=weight, batch_size=8)
        smoke.append(fit_case(**common, output_root=output_root / "smoke", config=smoke_config,
            seed=protocol["seed"], coefficient=0., train_limit=512, val_limit=64, validation_panel=validation[:32],
            budget_seconds=protocol["smoke_budget_seconds"], extra_guidance=hook))
    if smoke[0]["terminal_rng"] != smoke[1]["terminal_rng"] or smoke[0]["encoder_sha256"] != smoke[1]["encoder_sha256"]:
        raise RuntimeError("Native search/control smoke budgets differed")
    write_json(output_root / "native_engineering_gate.json", {"status": "passed", "smoke": smoke,
        "native_search_coverage": {key: value for key, value in probe_receipt.items() if key != "attempts"}, "native_gradient_probe": probe,
        "scope": "Finite updates/frozen encoder/budgets; zero native repair coverage is not a positive gradient result"})
    original_rec, _ = reconstruction_metrics(reference, subset_loader(test, len(test), 17, config.batch_size),
        LossConfig(**checkpoint["loss_config"]), device, chemistry_limit=len(test), training_mode="vae", graded=False,
        training_smiles=set(train.smiles))
    original_prior = evaluate_random_panel(reference, final, set(train.smiles), output_root / "original-prior",
        batch_size=config.batch_size, samples_per_latent=4, sample_seed=91001, repair_limit=protocol["final_repair_limit"])
    original_pool = evaluate_random_panel(reference, pool, set(train.smiles), output_root / "original-training-pool",
        batch_size=config.batch_size, samples_per_latent=4, sample_seed=71003, repair_limit=0)
    write_json(output_root / "original.json", {"reconstruction": original_rec, "prior": original_prior, "training_pool": original_pool})
    del reference
    cases = []
    control = None
    for arm in protocol["arms"]:
        remaining = protocol["controller_budget_seconds"] - (time.monotonic() - started)
        if remaining < protocol["case_budget_seconds"] + evaluation_reserve_seconds:
            cases.append({"arm": arm, "status": "not_attempted_time_reservation", "remaining_seconds": remaining})
            write_json(output_root / "cases.json", cases)
            continue
        fit, model, phase = None, None, "training"
        try:
            hook = SearchPanelGuidance(pool, train.spec, repair, name=arm["name"], weight=arm["repair_weight"],
                                       batch_size=protocol["training_prior_batch"])
            fit = fit_case(**common, output_root=output_root / "continuations", config=config,
                seed=protocol["seed"], coefficient=arm["defect_weight"], train_limit=protocol["train_limit"],
                val_limit=protocol["validation_limit"], validation_panel=validation,
                budget_seconds=protocol["case_budget_seconds"], extra_guidance=hook)
            phase = "endpoint_validation"
            model, endpoint = load_model(Path(fit["checkpoint"]), device, expected_spec=train.spec)
            if endpoint["epoch"] != config.epochs:
                raise ValueError("Do not evaluate an incomplete training endpoint as final")
            phase = "evaluation"
            reconstruction, _ = reconstruction_metrics(model, subset_loader(test, len(test), 17, config.batch_size),
                LossConfig(**checkpoint["loss_config"]), device, chemistry_limit=len(test), training_mode="vae", graded=False,
                training_smiles=set(train.smiles))
            evaluation_dir = Path(fit["run_dir"]) / "final-prior"
            prior = evaluate_random_panel(model, final, set(train.smiles), evaluation_dir,
                batch_size=config.batch_size, samples_per_latent=4, sample_seed=91001, repair_limit=protocol["final_repair_limit"])
            training_pool = evaluate_random_panel(model, pool, set(train.smiles), Path(fit["run_dir"]) / "training-pool",
                batch_size=config.batch_size, samples_per_latent=4, sample_seed=71003, repair_limit=0)
            item = {"arm": arm, "status": "completed", "fit": fit, "reconstruction": reconstruction, "prior": prior,
                "training_pool": training_pool,
                "evaluation_dir": str(evaluation_dir), "vs_original": paired_compare(output_root / "original-prior", evaluation_dir)}
            phase = "matching_and_comparison"
            if arm["name"] == "control":
                control = item
            elif control is not None:
                if fit["terminal_rng"] != control["fit"]["terminal_rng"]:
                    raise RuntimeError("Search pilot training RNG budgets differ")
                if fit["encoder_sha256"] != control["fit"]["encoder_sha256"]:
                    raise RuntimeError("Search comparison frozen encoders differ")
                left, right = control["fit"]["extra_guidance"], fit["extra_guidance"]
                if left["updates"] != right["updates"] or left["scheduled_points"] != right["scheduled_points"] or len(left["refreshes"]) != len(right["refreshes"]):
                    raise RuntimeError("Search pilot oracle/point schedules differ")
                item["vs_control"] = paired_compare(Path(control["evaluation_dir"]), evaluation_dir)
                item["advantage_assessment"] = assess_advantage(original_rec, control["reconstruction"], reconstruction,
                    item["vs_original"], item["vs_control"])
            write_json(Path(fit["run_dir"]) / "final_evaluation.json", item)
        except Exception as error:
            item = {"arm": arm, "status": "failed_or_censored", "error": repr(error),
                    "failure_phase": phase, "training_completed": fit is not None,
                    "run_dir": getattr(error, "experiment_run_dir", None) or (fit["run_dir"] if fit is not None else None)}
            if fit is not None:
                item["fit"] = fit
        finally:
            if model is not None:
                del model
        cases.append(item)
        write_json(output_root / "cases.json", cases)
    # Compare search guidance directly with the matched defect arm as well.
    defect = next((case for case in cases if case["status"] == "completed" and case["arm"]["name"] == "defect"), None)
    for case in cases:
        if defect is not None and case["status"] == "completed" and case["arm"]["repair_weight"] > 0:
            case["vs_defect"] = paired_compare(Path(defect["evaluation_dir"]), Path(case["evaluation_dir"]))
    complete = len(cases) == len(protocol["arms"]) and all(case["status"] == "completed" for case in cases)
    write_json(output_root / "summary.json", {"cases": cases, "complete": complete,
        "native_gradient_probe": probe, "original_reconstruction": original_rec,
        "scope": scope})
    if not complete:
        raise RuntimeError("Search pilot retained incomplete cases; full matched comparison unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Search pilot requires an allocated DTU job")
    run_study(read_protocol(args.config, args.sha256), args.output_root, Path.cwd())


if __name__ == "__main__":
    main()
