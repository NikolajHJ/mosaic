"""Bounded supervised mapping fit with train-only statistics and one audit.

Native diffusion schedules are supplied by the authenticated loaded model. The
only trained parameters are ConstrainedAtomBridge's small chemical head; VAE,
native encoder, normalization and every diffusion noise buffer remain frozen.
Feature acceptance never substitutes for denoiser, scorer or transport gates.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from latentmol.representation import HardGraphBatch
from latentmol.train import state_digest
from .bridge import (ATOMIC_NUMBER, BRIDGE_VERSION, PILOT_ELEMENTS, ConstrainedAtomBridge,
                     exact_native_probabilities)
from .paired_data import (SPLITS, check_deadline, deterministic_seed, load_dataset,
                          vae_identity, write_json)


TRAINING_VERSION = "diffsbdd-constrained-supervised-fit-v1"
AUDIT_TIMES = (0.0, 0.04, 0.10, 0.20)


def _bundle(vae, data, rows, device):
    """Recompute one exact natural decode; stale/fixed-mask targets cannot pass."""
    active = data["active"][rows].to(device)
    with torch.no_grad():
        decoded, features = vae.decode_with_features(data["z"][rows].to(device))
        hard = decoded.harden(vae.spec)
        stored = HardGraphBatch(*(data[key][rows].to(device=device, dtype=torch.long)
                                  for key in ("atoms", "charges", "bonds")))
        if any(not torch.equal(left, right) for left, right in
               ((hard.atoms, stored.atoms), (hard.charges, stored.charges), (hard.bonds, stored.bonds))):
            raise ValueError("Frozen VAE does not reproduce the actual stored natural decoded graph")
        if not torch.equal(hard.atoms != 0, active):
            raise ValueError("Natural decoder occupancy differs from paired-data mask")
        symbol_by_number = {number: index for index, number in enumerate(ATOMIC_NUMBER[x] for x in PILOT_ELEMENTS)}
        lookup = torch.full((len(vae.spec.atomic_numbers),), -1, device=device, dtype=torch.long)
        for index, number in enumerate(vae.spec.atomic_numbers):
            lookup[index] = symbol_by_number.get(number, -1)
        labels = lookup[hard.atoms]
        if bool((labels[active] < 0).any()):
            raise ValueError("Stored decoded graph has unsupported pilot elements")
        labels = labels.masked_fill(~active, 0)
    return decoded, features.atom_hidden.detach(), active, labels


def _common_noise(data, rows, level, reference, *, seed, purpose):
    values = []
    for index in rows:
        noise_seed = deterministic_seed(seed, purpose, int(data["seed"][index]), float(level))
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        values.append(torch.randn(reference.shape[1:], generator=generator, dtype=torch.float32))
    return torch.stack(values).to(reference)


def _schedule(schedule, level, reference):
    # Native alpha()/sigma() generally inflate [B,1] schedule values themselves;
    # NativeAtomBoundary also accepts a returned [B,1] or [B,1,1] tensor.
    times = reference.new_full((reference.shape[0], 1), float(level))
    alpha, sigma = schedule(times, reference)
    return alpha, sigma


def _per_example_mean(values, active):
    if values.ndim == active.ndim + 1:
        values = values.mean(-1)
    return (values * active).sum(-1) / active.sum(-1)


def _clean_target(labels, boundary, active):
    probabilities = F.one_hot(labels, len(PILOT_ELEMENTS)).float()
    return boundary.scatter(probabilities, PILOT_ELEMENTS) * active[..., None]


@torch.no_grad()
def training_statistics(vae, boundary, data, rows, noise_schedule, device, *,
                        batch_size=128, seed=17, deadline=None, times=AUDIT_TIMES):
    """Channel moments use train examples only and equal-example atom pooling."""
    if not rows or any(data["split"][index] != "train" for index in rows):
        raise ValueError("Embedding standardization must use training-only rows")
    sums, squares, count = None, None, 0
    for start in range(0, len(rows), batch_size):
        check_deadline(deadline)
        chosen = rows[start:start + batch_size]
        _, _, active, labels = _bundle(vae, data, chosen, device)
        target = _clean_target(labels, boundary, active)
        for level in times:
            noise = _common_noise(data, chosen, level, target, seed=seed, purpose="train_statistics")
            alpha, sigma = _schedule(noise_schedule, level, target)
            features = boundary.features(target, active, alpha=alpha, sigma=sigma, noise=noise)
            if not bool(features.isfinite().all()):
                raise FloatingPointError("Native training feature statistics are nonfinite")
            weight = active[..., None] / active.sum(-1)[:, None, None]
            mean = (features.double() * weight).sum((0, 1))
            second = (features.double().square() * weight).sum((0, 1))
            sums = mean if sums is None else sums + mean
            squares = second if squares is None else squares + second
            count += len(chosen)
    mean = sums / count
    variance = (squares / count - mean.square()).clamp_min(1e-4)
    return dict(mean=mean.float().cpu(), variance=variance.float().cpu(),
                variance_floor=1e-4, training_examples=len(rows), example_time_views=count,
                source="training split only, equal-example occupied-atom pooling")


def _loss(bridge, boundary, atom_hidden, labels, active, data, rows, noise_schedule,
          statistics, *, seed, purpose, times=AUDIT_TIMES, probabilities=None):
    logits = bridge(atom_hidden) if probabilities is None else None
    prediction = (boundary.scatter(logits.softmax(-1), bridge.elements) * active[..., None]
                  if probabilities is None else probabilities)
    target = _clean_target(labels, boundary, active)
    if logits is None:
        # Exact-codec loss is measured against the same six hard categories. The
        # full-native diagnostic may retain probability in other native classes.
        selected = torch.stack([prediction[..., boundary.native_elements.index(x)] for x in PILOT_ELEMENTS], -1)
        ce = -selected.gather(-1, labels[..., None]).squeeze(-1).clamp_min(1e-30).log()
    else:
        ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    ce_by_example = _per_example_mean(ce, active)
    variance = statistics["variance"].to(prediction)
    standardized, raw, per_time = [], [], {}
    for level in times:
        noise = _common_noise(data, rows, level, target, seed=seed, purpose=purpose)
        alpha, sigma = _schedule(noise_schedule, level, target)
        with torch.no_grad():
            expected = boundary.features(target, active, alpha=alpha, sigma=sigma, noise=noise)
        predicted = boundary.features(prediction, active, alpha=alpha, sigma=sigma, noise=noise)
        square = (predicted - expected).square()
        mse = _per_example_mean(square, active)
        standardized.append(_per_example_mean(square / variance, active))
        raw.append(mse)
        power = _per_example_mean(expected.square(), active)
        per_time[f"{level:g}"] = dict(mse=mse, target_power=power,
            per_example_nrmse=(mse / power.clamp_min(1e-12)).sqrt(),
            tiny_target=power < 1e-12, alpha=alpha, sigma=sigma)
    loss_by_example = ce_by_example + torch.stack(standardized).mean(0)
    metrics = dict(cross_entropy=ce_by_example, embedding_mse=torch.stack(raw).mean(0),
                   standardized_embedding_mse=torch.stack(standardized).mean(0), loss=loss_by_example)
    return loss_by_example.mean(), metrics, per_time, prediction


@torch.no_grad()
def evaluate(vae, boundary, bridge, data, rows, noise_schedule, statistics, device, *,
             batch_size=128, seed=17, deadline=None, mode="learned", times=AUDIT_TIMES):
    if not rows or mode not in ("learned", "exact_six_conditioned", "exact_native_full"):
        raise ValueError("Evaluation requires rows and an explicit mapping mode")
    bridge.eval()
    totals, by_time = Counter(), {}
    classes = {symbol: dict(count=0, correct=0) for symbol in PILOT_ELEMENTS}
    excluded_total, atom_total, example_count = 0.0, 0, 0
    for start in range(0, len(rows), batch_size):
        check_deadline(deadline)
        chosen = rows[start:start + batch_size]
        decoded, hidden, active, labels = _bundle(vae, data, chosen, device)
        conversion = None
        if mode != "learned":
            conversion = exact_native_probabilities(decoded, vae.spec, active, boundary.native_elements,
                allowed_elements=PILOT_ELEMENTS if mode == "exact_six_conditioned" else None)
            excluded_total += float(conversion.excluded_mass.sum())
        loss, metrics, per_time, probabilities = _loss(bridge, boundary, hidden, labels, active,
            data, chosen, noise_schedule, statistics, seed=seed, purpose="fixed_evaluation",
            times=times, probabilities=None if conversion is None else conversion.probabilities)
        if not bool(loss.isfinite()) or any(not bool(values.isfinite().all()) for values in metrics.values()):
            raise FloatingPointError("Nonfinite chemical feature evaluation")
        for key, values in metrics.items():
            totals[key] += float(values.sum())
        predicted_numbers = probabilities.argmax(-1)
        expected_numbers = boundary.scatter(F.one_hot(labels, len(PILOT_ELEMENTS)).float()).argmax(-1)
        for index, symbol in enumerate(PILOT_ELEMENTS):
            mask = active & (labels == index)
            classes[symbol]["count"] += int(mask.sum())
            classes[symbol]["correct"] += int((mask & (predicted_numbers == expected_numbers)).sum())
        for level, result in per_time.items():
            bucket = by_time.setdefault(level, dict(mse_sum=0.0, power_sum=0.0,
                per_example_nrmse=[], tiny_target_examples=0, alpha_min=math.inf,
                alpha_max=-math.inf, sigma_min=math.inf, sigma_max=-math.inf))
            bucket["mse_sum"] += float(result["mse"].sum())
            bucket["power_sum"] += float(result["target_power"].sum())
            bucket["per_example_nrmse"].extend(result["per_example_nrmse"].cpu().tolist())
            bucket["tiny_target_examples"] += int(result["tiny_target"].sum())
            for name in ("alpha", "sigma"):
                value = torch.as_tensor(result[name])
                bucket[name + "_min"] = min(bucket[name + "_min"], float(value.min()))
                bucket[name + "_max"] = max(bucket[name + "_max"], float(value.max()))
        atom_total += int(active.sum())
        example_count += len(chosen)
    for symbol, counts in classes.items():
        counts["accuracy"] = counts["correct"] / counts["count"] if counts["count"] else None
    for bucket in by_time.values():
        values = torch.tensor(bucket.pop("per_example_nrmse"), dtype=torch.float64)
        mse, power = bucket.pop("mse_sum") / example_count, bucket.pop("power_sum") / example_count
        bucket.update(embedding_mse=mse, target_power=power, embedding_rmse=math.sqrt(mse),
            embedding_nrmse=math.sqrt(mse / max(power, 1e-12)),
            p95_per_example_embedding_nrmse=float(torch.quantile(values, 0.95)))
    return dict(mode=mode, examples=example_count, occupied_atoms=atom_total,
        metrics={key: value / example_count for key, value in totals.items()},
        atom_agreement=sum(value["correct"] for value in classes.values()) / atom_total,
        per_element=classes, missing_elements=[key for key, value in classes.items() if value["count"] == 0],
        by_time=by_time, mean_excluded_ordinary_probability_mass=excluded_total / atom_total,
        tiny_target_denominator_floor=1e-12)


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def calibrate_fit_runtime(vae, boundary, bridge, data, rows_by_split, noise_schedule,
                          statistics, device, *, steps, batch_size, evaluate_every,
                          learning_rate, weight_decay, seed, deadline):
    """Disposable three-update train-only timing, with no actual-fit RNG use.

    The temporary head starts from the exact actual-fit initialization. Its
    optimizer/moments are discarded. Development timing never consults audit
    tensors or outputs; only prespecified audit row counts enter the forecast.
    """
    check_deadline(deadline)
    device = torch.device(device)
    before = {"head": state_digest(bridge.state_dict()), "vae": state_digest(vae.state_dict()),
              "native_boundary": state_digest(boundary.state_dict())}
    temporary = deepcopy(bridge).to(device).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(temporary.parameters(), lr=learning_rate, weight_decay=weight_decay)
    calibration_seed = deterministic_seed(seed, "disposable_runtime_calibration")
    generator = torch.Generator(device="cpu").manual_seed(calibration_seed)
    train_rows, measured, batch_receipts = rows_by_split["train"], [], []
    for index in range(3):
        check_deadline(deadline)
        # The first three complete wrapped training batches are fixed before
        # timing or outputs; no development/audit molecule influences them.
        rows = [train_rows[(index * batch_size + offset) % len(train_rows)]
                for offset in range(batch_size)]
        _synchronize(device)
        tick = time.monotonic()
        _, hidden, active, labels = _bundle(vae, data, rows, device)
        loss, _, _, _ = _loss(temporary, boundary, hidden, labels, active, data, rows,
            noise_schedule, statistics, seed=generator.initial_seed(),
            purpose=f"disposable_runtime_calibration_update_{index + 1}")
        if not bool(loss.isfinite()):
            raise FloatingPointError("Nonfinite disposable runtime-calibration training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not bool(p.grad.isfinite().all()) for p in temporary.parameters()):
            raise FloatingPointError("Nonfinite disposable runtime-calibration gradient")
        optimizer.step()
        _synchronize(device)
        measured.append(time.monotonic() - tick)
        batch_receipts.append(dict(index=index + 1, rows=rows, elapsed_seconds=measured[-1]))
    check_deadline(deadline)
    _synchronize(device)
    tick = time.monotonic()
    development = evaluate(vae, boundary, temporary, data, rows_by_split["dev"], noise_schedule,
        statistics, device, batch_size=batch_size, seed=calibration_seed, deadline=deadline)
    _synchronize(device)
    development_seconds = time.monotonic() - tick
    dev_batches = math.ceil(len(rows_by_split["dev"]) / batch_size)
    audit_batches = math.ceil(len(rows_by_split["audit"]) / batch_size)
    audit25_count = sum(int(data["active"][index].sum()) == 25 for index in rows_by_split["audit"])
    audit25_batches = math.ceil(audit25_count / batch_size)
    development_evaluations = math.ceil(steps / evaluate_every) + 1
    full_audit_equivalents = 3 * audit_batches / dev_batches
    audit25_equivalents = 3 * audit25_batches / dev_batches
    forecast = 1.5 * (steps * max(measured) + development_seconds *
        (development_evaluations + full_audit_equivalents + audit25_equivalents)) + 120.0
    after = {"head": state_digest(bridge.state_dict()), "vae": state_digest(vae.state_dict()),
             "native_boundary": state_digest(boundary.state_dict())}
    frozen = before == after
    del optimizer, temporary
    if not frozen:
        raise RuntimeError("Disposable calibration changed the actual head or frozen pretrained state")
    available = deadline - time.monotonic()
    finite = (all(math.isfinite(value) and value > 0 for value in measured)
              and math.isfinite(development_seconds) and development_seconds > 0)
    return dict(version="diffsbdd-disposable-fit-runtime-preflight-v1",
        passed=finite and forecast <= available, disposable_updates=3, actual_fit_updates=0,
        temporary_head_and_moments_discarded=True, frozen_and_actual_initial_state_unchanged=frozen,
        state_sha256_before=before, state_sha256_after=after, calibration_seed=calibration_seed,
        independent_training_row_stream=True, audit_outputs_consumed=False,
        row_selection="first three contiguous wrapped training-only batches", train_batches=batch_receipts,
        measured_slowest_training_update_seconds=max(measured),
        measured_complete_development_seconds=development_seconds,
        development_examples=development["examples"], scheduled_development_evaluations=development_evaluations,
        full_audit_development_equivalents=full_audit_equivalents,
        audit25_development_equivalents=audit25_equivalents,
        forecast_safety_factor=1.5, startup_and_checkpoint_margin_seconds=120.0,
        forecast_remaining_seconds=forecast, available_seconds=available,
        scope="measured bounded-operation engineering forecast; no audit/optimization or training-progress claim")


def train_bridge(vae, boundary, dataset_path, output, device, noise_schedule, *,
                 steps=10000, batch_size=128, learning_rate=1e-3, weight_decay=1e-4,
                 evaluate_every=250, seed=17, deadline=None, internal_seconds=3 * 3600 + 50 * 60,
                 hidden_width=96, width=128, min_atom_agreement=0.995, max_embedding_nrmse=0.05,
                 calibrate_runtime=True):
    """Fit a bounded head and audit the development-selected checkpoint once.

    The authenticated caller supplies ``noise_schedule(times[B,1], reference)``
    returning native alpha/sigma. time=0 is passed through the real schedule.
    Returned feature status cannot release any native or alternating-loop stage.
    Smaller budgets are supported for allocated synthetic contract fixtures;
    the launch protocol must enforce production values separately.
    """
    identity = vae_identity(vae)
    if (type(steps) is not int or not 1 <= steps <= 10000 or type(batch_size) is not int or batch_size < 1
            or type(evaluate_every) is not int or evaluate_every < 1
            or any(not math.isfinite(x) for x in (learning_rate, weight_decay, internal_seconds,
                                                 min_atom_agreement, max_embedding_nrmse))
            or learning_rate <= 0 or weight_decay < 0 or not 0 < internal_seconds <= 13800
            or not 0 <= min_atom_agreement <= 1 or max_embedding_nrmse <= 0
            or type(calibrate_runtime) is not bool):
        raise ValueError("Invalid bounded bridge-fitting budget or sealed feature thresholds")
    started = time.monotonic()
    deadline = min(started + internal_seconds, deadline) if deadline is not None else started + internal_seconds
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data, manifest, manifest_sha = load_dataset(dataset_path, vae)
    rows_by_split = {split: [index for index, value in enumerate(data["split"]) if value == split] for split in SPLITS}
    if any(not rows for rows in rows_by_split.values()):
        raise ValueError("All grouped training/development/audit splits must be nonempty")
    device = torch.device(device)
    vae.to(device)
    boundary.to(device).eval()
    if any(p.requires_grad for p in boundary.parameters()):
        raise ValueError("Native boundary parameters are trainable")
    before, boundary_before = state_digest(vae.state_dict()), state_digest(boundary.state_dict())
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        bridge = ConstrainedAtomBridge(hidden_width, width).to(device)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trained = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if (trained != {id(p) for p in bridge.parameters()}
            or trained & {id(p) for p in (*vae.parameters(), *boundary.parameters())}):
        raise RuntimeError("Mapping optimizer includes a pretrained parameter")
    plan = dict(version=TRAINING_VERSION, bridge_version=BRIDGE_VERSION, vae_checkpoint=identity,
        dataset_manifest_sha256=manifest_sha, head_configuration=bridge.configuration(),
        native_boundary=boundary.metadata(), native_boundary_state_sha256=boundary_before,
        seed=seed, steps=steps, batch_size=batch_size, learning_rate=learning_rate,
        weight_decay=weight_decay, evaluate_every=evaluate_every, internal_seconds=internal_seconds,
        time_levels=list(AUDIT_TIMES), selection="minimum development equal-example CE plus standardized MSE; earliest tie",
        trainable_scope="ConstrainedAtomBridge head only", desired_score_labels=False,
        feature_thresholds=dict(min_atom_agreement=min_atom_agreement, max_embedding_nrmse=max_embedding_nrmse),
        exact_controls=["exact_six_conditioned", "exact_native_full"], calibrate_runtime=calibrate_runtime,
        runtime_calibration_policy="three disposable train-only updates + complete dev timing; factor1.5 whole remaining fit and audits +120s")
    write_json(output / "training-plan.json", plan)
    result = dict(version=TRAINING_VERSION, status="running_not_passed", requested_steps=steps,
                  audit_consumed=False, native_denoiser_validation="not_run", optimization="not_run")
    completed, best_loss, best_step = 0, math.inf, None
    generator = torch.Generator(device="cpu").manual_seed(deterministic_seed(seed, "training_row_stream"))
    try:
        statistics = training_statistics(vae, boundary, data, rows_by_split["train"], noise_schedule,
            device, batch_size=batch_size, seed=seed, deadline=deadline)
        torch.save(statistics, output / "train-only-statistics.pt")
        if calibrate_runtime:
            initial_state = {key: value.detach().clone() for key, value in bridge.state_dict().items()}
            row_rng_before = generator.get_state().clone()
            preflight = calibrate_fit_runtime(vae, boundary, bridge, data, rows_by_split, noise_schedule,
                statistics, device, steps=steps, batch_size=batch_size, evaluate_every=evaluate_every,
                learning_rate=learning_rate, weight_decay=weight_decay, seed=seed, deadline=deadline)
            preflight["actual_training_row_rng_unchanged"] = torch.equal(row_rng_before, generator.get_state())
            if not preflight["actual_training_row_rng_unchanged"]:
                preflight["passed"] = False
            write_json(output / "runtime-preflight.json", preflight)
            result["runtime_preflight"] = preflight
            if not preflight["passed"]:
                result["status"] = "failed_runtime_forecast_before_fit"
                raise RuntimeError("Measured remaining mapping fit/development/audit forecast does not fit; actual fit not started")
            # Explicitly restore exact initial bytes and fresh empty moments.
            bridge.load_state_dict(initial_state, strict=True)
            optimizer = torch.optim.AdamW(bridge.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            result["runtime_preflight"] = {"status": "disabled_explicitly_for_synthetic_fixture", "passed": False}
        for step in range(1, steps + 1):
            check_deadline(deadline)
            bridge.train()
            indices = torch.randint(len(rows_by_split["train"]), (batch_size,), generator=generator).tolist()
            rows = [rows_by_split["train"][index] for index in indices]
            _, hidden, active, labels = _bundle(vae, data, rows, device)
            loss, _, _, _ = _loss(bridge, boundary, hidden, labels, active, data, rows, noise_schedule,
                statistics, seed=seed, purpose=f"fit_update_{step}")
            if not bool(loss.isfinite()):
                raise FloatingPointError("Nonfinite constrained-head training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not bool(p.grad.isfinite().all()) for p in bridge.parameters()):
                raise FloatingPointError("Nonfinite constrained-head gradient")
            optimizer.step()
            completed = step
            if step == 1 or step % evaluate_every == 0 or step == steps:
                development = evaluate(vae, boundary, bridge, data, rows_by_split["dev"], noise_schedule,
                    statistics, device, batch_size=batch_size, seed=seed, deadline=deadline)
                record = dict(step=step, training_loss=float(loss.detach()), development=development)
                with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                dev_loss = development["metrics"]["loss"]
                if dev_loss < best_loss:
                    best_loss, best_step = dev_loss, step
                    torch.save(dict(version=TRAINING_VERSION, head_configuration=bridge.configuration(),
                        bridge_state={key: value.detach().cpu() for key, value in bridge.state_dict().items()},
                        vae_checkpoint=identity, vae_state_sha256=before,
                        native_boundary=boundary.metadata(), native_boundary_state_sha256=boundary_before,
                        dataset_manifest_sha256=manifest_sha, statistics=statistics,
                        training=plan, step=step, development=development), output / "best-bridge.pt")
        saved = torch.load(output / "best-bridge.pt", map_location=device, weights_only=True)
        bridge.load_state_dict(saved["bridge_state"], strict=True)
        bridge.eval().requires_grad_(False)
        check_deadline(deadline)
        # Audit is intentionally touched only after all development selection.
        result["audit_consumed"] = True
        audits = {mode: evaluate(vae, boundary, bridge, data, rows_by_split["audit"], noise_schedule,
            statistics, device, batch_size=batch_size, seed=seed, deadline=deadline, mode=mode)
            for mode in ("learned", "exact_six_conditioned", "exact_native_full")}
        audit25_rows = [index for index in rows_by_split["audit"] if int(data["active"][index].sum()) == 25]
        audit25 = ({mode: evaluate(vae, boundary, bridge, data, audit25_rows, noise_schedule,
            statistics, device, batch_size=batch_size, seed=seed, deadline=deadline, mode=mode)
            for mode in ("learned", "exact_six_conditioned", "exact_native_full")}
            if audit25_rows else {"status": "no_25_atom_audit_views", "examples": 0})
        learned = audits["learned"]
        checks = dict(atom_agreement=learned["atom_agreement"] >= min_atom_agreement)
        checks.update({"embedding_nrmse_t=" + level: value["embedding_nrmse"] <= max_embedding_nrmse
                       for level, value in learned["by_time"].items()})
        result.update(status="passed_feature_audit_not_native_validated" if all(checks.values()) else "failed_feature_audit",
            checks=checks, audit=audits, best_step=best_step, best_development_loss=best_loss,
            audit_25_atom_subset=audit25,
            atom_count_coverage={split: dict(Counter(str(int(data["active"][index].sum()))
                for index in rows)) for split, rows in rows_by_split.items()},
            checkpoint_path=str((output / "best-bridge.pt").resolve()),
            requires_individual_hard_identity=True, requires_denoiser_and_gradient_gates=True)
    except Exception as error:
        if result["status"] == "running_not_passed":
            result["status"] = "failed_not_validated"
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        after, boundary_after = state_digest(vae.state_dict()), state_digest(boundary.state_dict())
        frozen = (before == after and boundary_before == boundary_after
                  and not vae.training and not boundary.atom_encoder.training
                  and not any(p.requires_grad for p in (*vae.parameters(), *boundary.parameters()))
                  and getattr(vae, "diffsbdd_checkpoint_identity", None) == identity)
        if not frozen:
            result["status"] = "failed_frozen_state_changed"
        result.update(completed_steps=completed, elapsed_seconds=time.monotonic() - started,
            frozen_vae_and_native_boundary_unchanged=frozen,
            vae_state_sha256_before=before, vae_state_sha256_after=after,
            native_boundary_state_sha256_before=boundary_before, native_boundary_state_sha256_after=boundary_after)
        write_json(output / "fit-result.json", result)
    if not frozen:
        raise RuntimeError("Frozen pretrained state changed during bridge fit")
    return result
