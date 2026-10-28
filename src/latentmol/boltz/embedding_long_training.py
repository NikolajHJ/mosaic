"""Disjoint frozen-epoch36 bridge continuation and development-only calibration.

Only the input bridge is fitted. Native projection targets remain the molecule
decoded by the original frozen VAE. All numerical work requires allocated HPC.
Calibration is an empirical feature-error boundary, never a chemistry or
gradient-faithfulness certificate. Original pilot files are only read.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.data import GraphDataset, collate_graphs, sha256_file
from latentmol.representation import HardGraphBatch
from latentmol.train import state_digest
from .embedding_bridge import BridgeTargets, HiddenEmbeddingBridge
from .embedding_training import (
    DATA_VERSION, TRAINING_VERSION, SPLITS, _check_deadline, _evaluate,
    _forward_rows, _load_dataset, _prior_split, _source_holdout_identities,
    _vae_identity, _write_json, conditioned_hard, partition_source_roots,
    slot_to_canonical_alignment,
)


VERSION = "epoch36-disjoint-warm-bridge-fit-v1"
DATA_EXTENSION = "epoch36-disjoint-bridge-data-v1"
CALIBRATION_VERSION = "epoch36-bridge-dev-q99-calibration-v1"
QUANTILE = 0.99
CAP_FLOOR = 1e-8


def _asset(asset):
    if (not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
            or not isinstance(asset["path"], str)
            or not isinstance(asset["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None):
        raise ValueError("Require an exact path/SHA256 asset")
    path = Path(asset["path"])
    if not path.is_file() or sha256_file(path) != asset["sha256"]:
        raise ValueError("Long-fit asset hash mismatch: " + str(path))
    return path


def _bindings(bindings):
    if not isinstance(bindings, dict) or not {"source_sha256", "plan_sha256"} <= set(bindings):
        raise ValueError("Long fit requires source and plan SHA256 bindings")
    for key in ("source_sha256", "plan_sha256"):
        if not isinstance(bindings[key], str) or re.fullmatch(r"[0-9a-f]{64}", bindings[key]) is None:
            raise ValueError("Malformed long-fit source/plan binding")
    return dict(bindings)


def _excluded(vae, prior_dataset, benchmark_starts):
    path = _asset(prior_dataset)
    old, manifest, digest = _load_dataset(path, vae)
    starts = torch.load(_asset(benchmark_starts), map_location="cpu", weights_only=True)
    if (set(starts) != {"cases", "graph_spec"} or starts["graph_spec"] != vae.spec.to_dict()
            or len(starts["cases"]) != 3):
        raise ValueError("Long fit requires the exact three benchmark starts/schema")
    groups, identities = set(old["group"]), set(old["identity"])
    for case in starts["cases"]:
        record = case["record"]
        matches = [i for i, (g, identity) in enumerate(zip(old["group"], old["identity"], strict=True))
                   if g == record["group"] and identity == record["identity"]
                   and old["split"][i] == "audit" and old["origin"][i] == "posterior_mean"]
        if (len(matches) != 1 or case["z"].dtype != old["z"].dtype
                or not torch.equal(case["z"], old["z"][matches[0]:matches[0] + 1])
                or case["active"].dtype != old["active"].dtype
                or not torch.equal(case["active"], old["active"][matches[0]])):
            raise ValueError("Benchmark start differs from authenticated old audit row")
    if (len({c["record"]["group"] for c in starts["cases"]}) != 3
            or len({c["record"]["identity"] for c in starts["cases"]}) != 3):
        raise ValueError("Benchmark groups/identities must remain distinct")
    roots = manifest.get("selected_source_roots")
    if not isinstance(roots, dict) or set(roots) != set(SPLITS):
        raise ValueError("Historical manifest lacks complete selected-source roots")
    root_indices = {index for rows in roots.values() for index in rows}
    if any(type(index) is not int or index < 0 for index in root_indices):
        raise ValueError("Malformed historical source-root indices")
    return old, manifest, dict(groups=groups, identities=identities, root_indices=root_indices), digest


def prepare_long_dataset(vae, data_path, output, seed, *, prior_dataset, benchmark_starts,
                         train_count=4096, dev_count=512, audit_count=512,
                         prior_attempts=1024, local_scale=0.05, deadline=None):
    """Use the original three-descendant/two-prior sampler on new source roots.

    All historical selected roots, accepted groups and decoded identities are
    excluded from every new partition. This protects new development/audit data
    from the bridge's warm-start history. Counts are attempts, not validity
    targets; rejected rows remain visible and are never replaced.
    """
    identity = _vae_identity(vae)
    if (type(prior_attempts) is not int or prior_attempts < 0
            or not math.isfinite(local_scale) or local_scale <= 0):
        raise ValueError("Require nonnegative prior attempts and positive local scale")
    old, old_manifest, excluded, prior_sha = _excluded(vae, prior_dataset, benchmark_starts)
    del old
    data = GraphDataset(data_path, "train")
    if (data.spec != vae.spec or data.manifest["dataset_sha256"] != old_manifest["source_dataset_sha256"]
            or data.manifest["split_sha256"] != old_manifest["source_split_sha256"]):
        raise ValueError("New and historical source dataset/schema bindings differ")
    if any(i >= len(data.smiles) for i in excluded["root_indices"]):
        raise ValueError("Historical source root is outside authenticated training split")
    counts = dict(train=train_count, dev=dev_count, audit=audit_count)
    eligible = [i for i, smile in enumerate(data.smiles)
                if i not in excluded["root_indices"] and smile not in excluded["identities"]]
    chosen = partition_source_roots([data.smiles[i] for i in eligible], seed, counts)
    roots = {split: [eligible[i] for i in rows] for split, rows in chosen.items()}
    owners = {data.smiles[i]: split for split, indices in roots.items() for i in indices}
    heldout = _source_holdout_identities(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = state_digest(vae.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(vae.parameters()).device
    samples, attempts, reasons = [], [], Counter()

    def retain(z, hard, split, group, origin):
        hard = hard.to("cpu")
        assessment = classify_repair_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], vae.spec)
        record = dict(attempt=len(attempts), split=split, group=group, origin=origin,
                      classification=assessment, accepted=False)
        canonical, reason, alignment = assessment["smiles"], None, None
        if group in excluded["groups"]:
            reason = "historical_bridge_group"
        elif not assessment["strict_teacher_valid"]:
            reason = "invalid:" + str(assessment["reason"])
        elif canonical in excluded["identities"]:
            reason = "historical_bridge_identity"
        elif canonical in heldout:
            reason = "source_heldout_identity"
        elif canonical in owners and owners[canonical] != split:
            reason = "cross_bridge_split_identity"
        else:
            try:
                alignment = slot_to_canonical_alignment(hard, vae.spec, canonical)
            except ValueError as error:
                reason = "identity_alignment"
                record["alignment_detail"] = str(error)
        if reason:
            reasons[reason] += 1
            record["rejection_reason"] = reason
        else:
            owners[canonical] = split
            samples.append(dict(z=z.detach().cpu().reshape(-1).float(),
                atoms=hard.atoms[0].to(torch.uint8), charges=hard.charges[0].to(torch.uint8),
                bonds=hard.bonds[0].to(torch.uint8), active=hard.atoms[0] != 0,
                slot_to_canonical=torch.tensor(alignment, dtype=torch.int16),
                split=split, group=group, origin=origin, identity=canonical))
            record.update(accepted=True, sample_index=len(samples) - 1)
        attempts.append(record)

    try:
        with torch.no_grad():
            for split in ("audit", "dev", "train"):
                for index in roots[split]:
                    _check_deadline(deadline)
                    graph = collate_graphs([data[index]]).to(device)
                    mu, logvar = vae.encode(graph.to_relaxed(vae.spec))
                    base = vae.decode(mu)
                    active = base.harden(vae.spec).atoms != 0
                    group = "source:" + hashlib.sha256(data.smiles[index].encode()).hexdigest()
                    if not bool(active.any()):
                        retain(mu, base.harden(vae.spec), split, group, "posterior_mean")
                        continue
                    posterior = mu + torch.randn(mu.shape, generator=generator).to(mu) * (0.5 * logvar).exp()
                    local = mu + local_scale * torch.randn(mu.shape, generator=generator).to(mu)
                    z = torch.cat((mu, posterior, local), 0)
                    hard = conditioned_hard(vae.decode(z), active.expand(3, -1), vae.spec)
                    for row, origin in enumerate(("posterior_mean", "posterior_sample", "local")):
                        retain(z[row], HardGraphBatch(hard.atoms[row:row + 1], hard.charges[row:row + 1],
                            hard.bonds[row:row + 1]), split, group, origin)
            for index in range(prior_attempts):
                _check_deadline(deadline)
                group = f"prior:{seed}:{index}"
                split = _prior_split(group, seed, counts)
                z = torch.randn((1, vae.config.latent), generator=generator).to(device)
                base = vae.decode(z)
                active = base.harden(vae.spec).atoms != 0
                if not bool(active.any()):
                    retain(z, base.harden(vae.spec), split, group, "prior")
                    continue
                local = z + local_scale * torch.randn(z.shape, generator=generator).to(z)
                points = torch.cat((z, local), 0)
                hard = conditioned_hard(vae.decode(points), active.expand(2, -1), vae.spec)
                for row, origin in enumerate(("prior", "prior_local")):
                    retain(points[row], HardGraphBatch(hard.atoms[row:row + 1], hard.charges[row:row + 1],
                        hard.bonds[row:row + 1]), split, group, origin)
        if state_digest(vae.state_dict()) != before or _vae_identity(vae) != identity:
            raise RuntimeError("Frozen VAE changed during disjoint paired preparation")
        if not samples:
            raise ValueError("No supported new paired rows; no replacement sampling allowed")
        payload = {name: torch.stack([r[name] for r in samples]) for name in
                   ("z", "atoms", "charges", "bonds", "active", "slot_to_canonical")}
        payload.update({name: [r[name] for r in samples] for name in ("split", "group", "origin", "identity")})
        torch.save(payload, output / "samples.pt")
        with (output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
            for record in attempts:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        manifest = dict(version=DATA_VERSION, status="prepared_not_trained", seed=seed,
            spec=vae.spec.to_dict(), spec_fingerprint=vae.spec.fingerprint(), vae_checkpoint=identity,
            vae_state_sha256_before=before, vae_state_sha256_after=before,
            source_dataset_sha256=data.manifest["dataset_sha256"],
            source_split_sha256=data.manifest["split_sha256"],
            source_manifest_sha256=sha256_file(data.cache_dir / "manifest.json"),
            selected_source_roots=roots, selected_source_root_counts=counts,
            source_heldout_identity_count=len(heldout), source_heldout_usage="identity exclusion only",
            prior_root_attempts=prior_attempts, local_scale=local_scale,
            graph_policy="root-natural-occupancy-conditioned-descendants-v1",
            root_order=["audit", "dev", "train", "hash-partitioned-prior"], attempts=len(attempts),
            accepted=len(samples), accepted_by_split={s: payload["split"].count(s) for s in SPLITS},
            unique_identities_by_split={s: len({r["identity"] for r in samples if r["split"] == s}) for s in SPLITS},
            rejected=dict(reasons), all_attempts_retained=True,
            samples_file="samples.pt", samples_sha256=sha256_file(output / "samples.pt"),
            attempts_file="attempts.jsonl", attempts_sha256=sha256_file(output / "attempts.jsonl"),
            long_data=dict(version=DATA_EXTENSION, prior_dataset=dict(prior_dataset),
                prior_manifest_sha256=prior_sha, benchmark_starts=dict(benchmark_starts),
                exclusion="all historical selected roots and accepted groups/decoded identities; every new split",
                excluded_source_roots=sorted(excluded["root_indices"]),
                excluded_groups=len(excluded["groups"]), excluded_identities=len(excluded["identities"]),
                old_train_replay_rows=0))
        _write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        _write_json(output / "preparation-failure.json", dict(status="failed_or_partial_not_prepared",
            error_type=type(error).__name__, error=str(error), completed_attempts=len(attempts),
            accepted=len(samples), attempts=attempts, frozen_vae_unchanged=state_digest(vae.state_dict()) == before))
        raise


def _load_long_dataset(path, vae):
    data, manifest, digest = _load_dataset(path, vae)
    extension = manifest.get("long_data", {})
    if extension.get("version") != DATA_EXTENSION or extension.get("old_train_replay_rows") != 0:
        raise ValueError("Long fit requires authenticated disjoint data metadata")
    _, old_manifest, exclusions, prior_sha = _excluded(
        vae, extension["prior_dataset"], extension["benchmark_starts"])
    if (prior_sha != extension["prior_manifest_sha256"]
            or manifest["source_dataset_sha256"] != old_manifest["source_dataset_sha256"]
            or manifest["source_split_sha256"] != old_manifest["source_split_sha256"]
            or sorted(exclusions["root_indices"]) != extension["excluded_source_roots"]
            or len(exclusions["groups"]) != extension["excluded_groups"]
            or len(exclusions["identities"]) != extension["excluded_identities"]
            or set(data["group"]) & exclusions["groups"]
            or set(data["identity"]) & exclusions["identities"]
            or any(set(rows) & exclusions["root_indices"] for rows in manifest["selected_source_roots"].values())):
        raise ValueError("Long-fit data leaks historical roots/groups/decoded identities")
    return data, manifest, digest


def _warm_bridge(vae, bank, asset, device, prior_manifest_sha256):
    saved = torch.load(_asset(asset), map_location="cpu", weights_only=True)
    if (saved.get("version") != TRAINING_VERSION or saved.get("vae_checkpoint") != _vae_identity(vae)
            or saved.get("vae_state_sha256") != state_digest(vae.state_dict())
            or saved.get("dataset_manifest_sha256") != prior_manifest_sha256
            or saved.get("bank_metadata") != bank.metadata()
            or set(saved.get("bank_state", {})) != set(bank.state_dict())):
        raise ValueError("Warm-start VAE/dataset/native-bank binding differs")
    for key, expected in bank.state_dict().items():
        actual = saved["bank_state"][key]
        if actual.dtype != expected.dtype or not torch.equal(actual, expected.detach().cpu()):
            raise ValueError("Warm-start native projection bytes differ")
    config = saved["bridge_configuration"]
    bridge = HiddenEmbeddingBridge(bank, **{key: config[key] for key in
        ("atom_input_dim", "pair_input_dim", "charge_input_dim", "hidden_dim")}).to(device)
    if bridge.configuration() != config:
        raise ValueError("Warm-start bridge configuration differs")
    bridge.load_state_dict(saved["bridge_state"], strict=True)
    return bridge


@torch.no_grad()
def per_example_errors(vae, bank, bridge, data, rows, device, batch_size, deadline=None):
    """Each metric uses exactly bank.loss's one-molecule group reductions."""
    if not rows:
        raise ValueError("Calibration/audit cohort cannot be empty")
    bridge.eval()
    errors = {d.key + "/mse": [] for d in bank.descriptors}
    for start in range(0, len(rows), batch_size):
        _check_deadline(deadline)
        chosen = rows[start:start + batch_size]
        _, targets, predictions = _forward_rows(vae, bank, bridge, data, chosen, device)
        for i in range(len(chosen)):
            one = BridgeTargets({k: v[i:i + 1] for k, v in targets.values.items()},
                targets.atom_mask[i:i + 1], targets.pair_mask[i:i + 1], targets.bond_present[i:i + 1])
            _, metrics = bank.loss({k: v[i:i + 1] for k, v in predictions.items()}, one)
            for key in errors:
                value = float(metrics[key])
                if not math.isfinite(value) or value < 0:
                    raise FloatingPointError("Nonfinite/negative calibration feature error")
                errors[key].append(value)
    return errors


def calibrate_caps(errors, *, split="dev", quantile=QUANTILE, floor=CAP_FLOOR):
    """Predeclared nearest-rank q99, using development examples only."""
    if split != "dev" or quantile != QUANTILE or floor != CAP_FLOOR:
        raise ValueError("Calibration is fixed dev-only q99 with floor1e-8")
    if not isinstance(errors, dict) or not errors or len({len(v) for v in errors.values()}) != 1:
        raise ValueError("Calibration needs nonempty equal-sized per-head errors")
    caps = {}
    for key, values in errors.items():
        if (not isinstance(key, str) or not key.endswith("/mse") or not values
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError("Malformed calibration feature errors")
        caps[key] = max(floor, sorted(values)[math.ceil(quantile * len(values)) - 1])
    return caps


def cap_coverage(errors, caps):
    if set(errors) != set(caps) or not caps:
        raise ValueError("Audit cap keys differ from calibrated projections")
    counts = {len(v) for v in errors.values()}
    if len(counts) != 1 or not next(iter(counts)):
        raise ValueError("Audit coverage requires a nonempty complete cohort")
    count = next(iter(counts))
    for key, values in errors.items():
        if not math.isfinite(caps[key]) or caps[key] < CAP_FLOOR or any(
                not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Malformed audit coverage errors/caps")
    return dict(examples=count,
        per_projection={k: dict(within=sum(v <= caps[k] for v in values), count=count,
            fraction=sum(v <= caps[k] for v in values) / count) for k, values in errors.items()},
        all_heads_within=sum(all(errors[k][i] <= caps[k] for k in caps) for i in range(count)),
        caps_adjusted=False, claim="empirical coverage; no native/chemistry/gradient guarantee")


def fit_long_bridge(vae, bank, dataset_path, warm_start, output, device, *, bindings,
                    steps=5000, batch_size=32, learning_rate=1e-3, seed=142036,
                    evaluate_every=100, deadline=None):
    """Warm-start bridge weights, reset AdamW, select only on new dev loss."""
    identity, bindings = _vae_identity(vae), _bindings(bindings)
    if (type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1
            or type(evaluate_every) is not int or evaluate_every < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid long bridge fitting budget")
    data, manifest, manifest_sha = _load_long_dataset(dataset_path, vae)
    rows = {s: [i for i, value in enumerate(data["split"]) if value == s] for s in SPLITS}
    if any(not value for value in rows.values()):
        raise ValueError("Every long-fit split requires accepted examples")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before, bank_before = state_digest(vae.state_dict()), state_digest(bank.state_dict())
    device = torch.device(device)
    vae.to(device)
    bank.to(device).eval().requires_grad_(False)
    # Initialization is overwritten by the authenticated warm-start; do not
    # advance the caller's process RNG even transiently.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        bridge = _warm_bridge(vae, bank, warm_start, device, manifest["long_data"]["prior_manifest_sha256"])
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=learning_rate, weight_decay=1e-4)
    ids = {id(p) for p in bridge.parameters()}
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != ids or ids & {id(p) for p in vae.parameters()}:
        raise RuntimeError("Long-fit optimizer escaped bridge-only scope")
    plan = dict(version=VERSION, bindings=bindings, dataset_manifest_sha256=manifest_sha,
        warm_start=dict(warm_start), steps=steps, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=1e-4, seed=seed, evaluate_every=evaluate_every,
        optimizer_history="new AdamW; bridge weights warm-started; old optimizer not resumed",
        desired_score_labels=False, trainable_scope="HiddenEmbeddingBridge only",
        selection="minimum new-dev batch-weighted group-balanced feature loss; earliest tie including step0",
        adoption="strict aggregate dev-loss improvement over warm start required; no native approval",
        calibration=dict(split="dev", quantile=QUANTILE, method="nearest_rank", floor=CAP_FLOOR,
                         metric="one-example NativeProjectionBank.loss per-projection /mse"))
    _write_json(output / "training-plan.json", plan)
    initial_dev = _evaluate(vae, bank, bridge, data, rows["dev"], device, batch_size, deadline)
    initial_loss = initial_dev["batch_weighted_metrics"]["loss"]
    best_loss, best_step, completed = initial_loss, 0, 0
    started = time.monotonic()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = dict(version=VERSION, status="running_not_passed", completed_steps=0,
                  audit_consumed=False, native_validation="not_run", optimization="not_run")

    def checkpoint(step, dev):
        torch.save(dict(version=TRAINING_VERSION, long_training_version=VERSION,
            bridge_configuration=bridge.configuration(),
            bridge_state={k: v.detach().cpu() for k, v in bridge.state_dict().items()},
            bank_metadata=bank.metadata(), bank_state={k: v.detach().cpu() for k, v in bank.state_dict().items()},
            vae_checkpoint=identity, vae_state_sha256=before, dataset_manifest_sha256=manifest_sha,
            training=plan, step=step, dev=dev), output / "best-bridge.pt")

    checkpoint(0, initial_dev)
    _write_json(output / "warm-start-dev.json", initial_dev)
    try:
        for step in range(1, steps + 1):
            _check_deadline(deadline)
            bridge.train()
            positions = torch.randint(len(rows["train"]), (batch_size,), generator=generator).tolist()
            chosen = [rows["train"][i] for i in positions]
            _, targets, predictions = _forward_rows(vae, bank, bridge, data, chosen, device)
            loss, _ = bank.loss(predictions, targets)
            if not bool(loss.isfinite()):
                raise FloatingPointError("Nonfinite long-fit supervised loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not bool(p.grad.isfinite().all()) for p in bridge.parameters()):
                raise FloatingPointError("Nonfinite long-fit bridge gradient")
            optimizer.step()
            completed = step
            if step == 1 or step % evaluate_every == 0 or step == steps:
                dev = _evaluate(vae, bank, bridge, data, rows["dev"], device, batch_size, deadline)
                with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(step=step, train_loss=float(loss.detach()), dev=dev),
                                            sort_keys=True, allow_nan=False) + "\n")
                value = dev["batch_weighted_metrics"]["loss"]
                if value < best_loss:
                    best_loss, best_step = value, step
                    checkpoint(step, dev)
        saved = torch.load(output / "best-bridge.pt", map_location=device, weights_only=True)
        bridge.load_state_dict(saved["bridge_state"], strict=True)
        bridge.eval().requires_grad_(False)
        bridge_sha = sha256_file(output / "best-bridge.pt")
        improved = best_step > 0 and best_loss < initial_loss
        result.update(dev_improved=improved, initial_dev_loss=initial_loss,
            best_dev_loss=best_loss, best_step=best_step, checkpoint_path=str(output / "best-bridge.pt"),
            checkpoint_sha256=bridge_sha)
        if not improved:
            result["status"] = "failed_dev_improvement_no_calibration_or_audit"
            return result
        errors = per_example_errors(vae, bank, bridge, data, rows["dev"], device, batch_size, deadline)
        caps = calibrate_caps(errors)
        calibration = dict(version=CALIBRATION_VERSION, status="calibrated_dev_only_not_native_validated",
            bindings=bindings, vae_checkpoint=identity, vae_state_sha256=before,
            bridge_sha256=bridge_sha, dataset_manifest_sha256=manifest_sha,
            bank_metadata=bank.metadata(), bank_state_sha256=bank_before, warm_start=dict(warm_start),
            training_plan_sha256=sha256_file(output / "training-plan.json"),
            split="dev", quantile=QUANTILE, quantile_method="nearest_rank", cap_floor=CAP_FLOOR,
            metric_policy="one-example NativeProjectionBank.loss per-projection /mse; pair NONE/present groups balanced",
            metric_names=sorted(caps), caps=caps, dev_errors=errors, dev_examples=len(rows["dev"]),
            dev_groups=len({data["group"][i] for i in rows["dev"]}),
            dev_identities=len({data["identity"][i] for i in rows["dev"]}),
            audit_used_for_calibration=False, desired_score_labels=False,
            claim="empirical dev boundary, not a calibrated probability or chemical/native gradient guarantee")
        _write_json(output / "calibration.json", calibration)
        # Caps and their exact byte receipt exist before the audit is touched.
        result["audit_consumed"] = True
        audit_errors = per_example_errors(vae, bank, bridge, data, rows["audit"], device, batch_size, deadline)
        audit = _evaluate(vae, bank, bridge, data, rows["audit"], device, batch_size, deadline)
        audit.update(cap_coverage=cap_coverage(audit_errors, caps), per_example_errors=audit_errors,
                     unsupported_categories_remain_unvalidated=True)
        _write_json(output / "audit.json", audit)
        result.update(status="completed_long_bridge_fit_not_native_validated", audit=audit,
            calibration_path=str(output / "calibration.json"),
            calibration_sha256=sha256_file(output / "calibration.json"),
            audit_sha256=sha256_file(output / "audit.json"), caps=caps)
        return result
    except BaseException as error:
        result.update(status="partial_runtime_not_passed" if isinstance(error, TimeoutError) else "failed_not_passed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        unchanged = (state_digest(vae.state_dict()) == before and state_digest(bank.state_dict()) == bank_before
                     and _vae_identity(vae) == identity and sha256_file(_asset(warm_start)) == warm_start["sha256"])
        result.update(completed_steps=completed, requested_steps=steps,
            elapsed_seconds=time.monotonic() - started, bindings=bindings,
            vae_checkpoint=identity, frozen_vae_and_bank_unchanged=unchanged,
            vae_state_sha256=before, bank_state_sha256=bank_before, dataset_manifest_sha256=manifest_sha,
            training_plan_sha256=sha256_file(output / "training-plan.json"),
            warm_start_unchanged=True, old_train_replay_rows=0)
        if not unchanged:
            result["status"] = "failed_frozen_identity"
        _write_json(output / "training-result.json", result)
        if not unchanged:
            raise RuntimeError("Long bridge fit changed frozen/reference assets")
