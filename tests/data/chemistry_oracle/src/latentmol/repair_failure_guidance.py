"""Opt-in experimental guidance for rows without a complete valid repair.

The source121 training objective remains unchanged. Preparation is detached:
it can run an additional bounded search and freezes identities/components.
The forward combines complete-witness NLL, edited-label partial NLL, and NEW
fixed-identity local bond admissibility / component-cut / nonempty risks.

The local term is negative log probability of remaining below a fixed native
valence bound, NOT the historical probability-weighted expected excess loss.
Its atom, charge, occupancy and rule choices have no direct gradient, avoiding
that term's direct incentive to choose a higher-valence element. Shared latent
or parameter updates can still change identities indirectly. Full aromatic,
Kekulization, radical and global validity constraints are not covered.
No loss here is an exact nearest-valid distance or a validity guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .atom_preserving_chemistry import RepairTargetBatch, _validated_decoded, repair_loss
from .chemistry_regularization import ALLOWED_VALENCES, RULE_VERSION
from .representation import DecoderOutput, HardGraphBatch
from .sampling import hard_graph_log_prob_terms


@dataclass(frozen=True)
class FailureGuidanceConfig:
    """Research coefficients; require allocated calibration before adoption."""
    version: str = "repair-failure-guidance-v1"
    extra_valid_weight: float = 1.0
    partial_weight: float = 1.0
    local_bond_weight: float = 0.1
    component_cut_weight: float = 0.1
    nonempty_weight: float = 0.1

    def __post_init__(self):
        if self.version != "repair-failure-guidance-v1":
            raise ValueError("Unknown experimental failure guidance version")
        for key, value in asdict(self).items():
            if key != "version" and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("Invalid failure guidance coefficient: " + key)
        if min(self.local_bond_weight, self.component_cut_weight, self.nonempty_weight) <= 0:
            raise ValueError("Failure risks must be explicitly enabled")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _components(atoms, bonds):
    remaining = {i for i, label in enumerate(atoms) if label}
    result = []
    while remaining:
        stack = [min(remaining)]
        remaining.remove(stack[0])
        group = []
        while stack:
            i = stack.pop()
            group.append(i)
            adjacent = sorted(j for j in remaining if bonds[i][j])
            remaining.difference_update(adjacent)
            stack.extend(adjacent)
        result.append(tuple(sorted(group)))
    return tuple(result)


@dataclass(frozen=True)
class FrozenFailureRow:
    candidate_id: int
    atoms: tuple
    charges: tuple
    bonds: tuple
    caps: tuple
    components: tuple
    witness: object = None


@dataclass(frozen=True)
class PreparedFailureGuidance:
    """Fixed reference for a loss/line search; explicitly refresh between steps."""
    base: RepairTargetBatch
    rows: tuple
    schema_sha256: str
    identity: str
    receipt: dict


def _row_record(row):
    return None if row is None else dict(candidate_id=row.candidate_id, atoms=row.atoms,
        charges=row.charges, bonds=row.bonds, caps=row.caps, components=row.components,
        witness_sha256=row.witness.identity if row.witness is not None else None)


def _prepared_identity(base, rows, spec, receipt):
    """Bind legacy targets AND receipts, not just newly introduced references."""
    if not isinstance(base, RepairTargetBatch):
        raise ValueError("Prepared guidance lost its original repair batch")
    target_records = []
    for target in base.targets:
        if target is not None:
            target.verify(spec)
        target_records.append(None if target is None else target.record())
    return _digest(dict(schema=spec.fingerprint(), candidate_ids=base.candidate_ids,
        targets=target_records, original_receipt=base.receipt,
        rows=[_row_record(row) for row in rows],
        coverage={k: v for k, v in receipt.items() if k != "identity"}))


@torch.no_grad()
def prepare_failure_guidance(decoded, base, spec, *, progress_search=None, progress_map=None):
    """Freeze failed rows; do not reinterpret errors as search exhaustion.

    ``progress_search`` is an optional ProgressSearchConfig. Complete new
    witnesses are independently rechecked. Partial targets remain explicitly
    invalid, atom-preserving diagnostic improvements. No best-invalid graph is
    substituted for a molecule scored by another branch.
    """
    from .probability_repair import probability_identity
    from .quality import graph_identity
    from .repair_progress import ProgressSearchConfig, search_progress_repair
    from .chemistry import encode_smiles, inspect_graph

    count = _validated_decoded(decoded, spec)
    if not isinstance(base, RepairTargetBatch) or len(base.targets) != count or len(base.candidate_ids) != count:
        raise ValueError("Original repair targets and decoder batch differ")
    if (base.receipt.get("schema_sha256") != spec.fingerprint()
            or base.receipt.get("attempted_candidates") != count
            or len(set(base.candidate_ids)) != count
            or any(type(value) is not int or value < 0 for value in base.candidate_ids)):
        raise ValueError("Original repair receipt lost its schema/count/candidate identity")
    attempts = base.receipt.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != count:
        raise ValueError("Require the original complete per-row coverage receipt")
    if progress_search is not None and not isinstance(progress_search, ProgressSearchConfig):
        raise ValueError("Extra search needs a declared progress policy")
    if progress_map is not None and (progress_search is None or not callable(progress_map)):
        raise ValueError("A progress mapper requires an explicit search policy")
    cpu = DecoderOutput(*(value.detach().cpu().clone() for value in (decoded.atoms, decoded.charges, decoded.bonds)))
    if cpu.atoms.dtype in (torch.float16, torch.bfloat16):
        cpu = DecoderOutput(*(value.float() for value in (cpu.atoms, cpu.charges, cpu.bonds)))
    hard = cpu.harden(spec)
    probabilities = (cpu.atoms.softmax(-1), cpu.charges.softmax(-1), cpu.symmetric_bond_logits.softmax(-1))
    # Validate EVERY source snapshot before scheduling additional native work.
    # Only NumPy arrays and explicit schema/config cross a process boundary.
    mapped = None
    if progress_map is not None:
        indices, tasks = [], []
        for index, (attempt, candidate_id) in enumerate(zip(attempts, base.candidate_ids)):
            raw = tuple(value[index] for value in (hard.atoms, hard.charges, hard.bonds))
            p = tuple(value[index] for value in probabilities)
            if (attempt.get("candidate_id") != candidate_id
                    or attempt.get("raw_sha256") != graph_identity(*raw, spec)
                    or attempt.get("probability_sha256") != probability_identity(p, spec)
                    or attempt.get("status") not in ("accepted", "skipped_already_valid", "censored", "unsupported")):
                raise ValueError("Parallel progress search requires exact original teacher snapshots")
            if attempt["status"] in ("censored", "unsupported") and bool(raw[0].any()):
                indices.append(index)
                tasks.append((tuple(value.numpy().copy() for value in raw),
                              tuple(value.numpy().copy() for value in p), spec, progress_search))
        results = progress_map(tasks)
        if type(results) is not list or len(results) != len(indices):
            raise ValueError("Progress mapper lost an ordered scheduled row")
        mapped = dict(zip(indices, results))
    rows, records = [], []
    for index, (attempt, old_target, candidate_id) in enumerate(zip(attempts, base.targets, base.candidate_ids)):
        if attempt.get("candidate_id") != candidate_id:
            raise ValueError("Failure row lost its original candidate identity")
        status = attempt.get("status")
        if status not in ("accepted", "skipped_already_valid", "censored", "unsupported"):
            raise ValueError("Repair errors cannot be hidden by fallback guidance")
        if (old_target is not None) != (status == "accepted"):
            raise ValueError("Original accepted-target status disagrees with target presence")
        raw = tuple(value[index] for value in (hard.atoms, hard.charges, hard.bonds))
        p = tuple(value[index] for value in probabilities)
        if (attempt.get("raw_sha256") != graph_identity(*raw, spec)
                or attempt.get("probability_sha256") != probability_identity(p, spec)):
            raise ValueError("Failure preparation requires the exact original teacher snapshot")
        if old_target is not None:
            old_target.verify(spec)
            if (old_target.candidate_id != candidate_id
                    or old_target.raw_sha256 != attempt["raw_sha256"]
                    or old_target.source_probability_sha256 != attempt["probability_sha256"]
                    or old_target.source_update != base.receipt.get("source_update")
                    or old_target.record() != attempt.get("target")):
                raise ValueError("Accepted target does not match its authenticated source attempt")
        if status in ("accepted", "skipped_already_valid"):
            rows.append(None)
            continue
        labels = tuple(tuple(int(x) for x in v.tolist()) for v in raw[:2])
        bonds = tuple(tuple(int(x) for x in v) for v in raw[2].tolist())
        caps = []
        for atom, charge in zip(*labels):
            allowed = ALLOWED_VALENCES.get(spec.atomic_numbers[atom], {}).get(spec.formal_charges[charge]) if atom else None
            caps.append(max(allowed) if allowed else None)
        witness, extra = None, None
        if progress_search is not None and any(labels[0]):
            extra = (mapped[index] if mapped is not None else
                     search_progress_repair(tuple(v.numpy() for v in raw), tuple(v.numpy() for v in p), spec, progress_search))
            if (type(extra) is not dict or extra.get("policy") != progress_search.record()
                    or type(extra.get("checks")) is not int
                    or not 0 <= extra["checks"] <= progress_search.max_edits * progress_search.checks_per_depth
                    or extra.get("source_probability_sha256") != _digest([v.tolist() for v in p])
                    or extra.get("status") not in ("valid", "partial", "no_progress", "already_valid", "unsupported")):
                raise ValueError("Progress result lost its exact probability/policy/budget provenance")
            witness = extra["witness"]
            if (witness is None) != (extra["status"] not in ("valid", "partial")):
                raise ValueError("Progress status and witness presence disagree")
            if witness is not None:
                witness.verify(spec)
                if witness.kind != extra["status"]:
                    raise ValueError("Progress witness kind and result status disagree")
                if witness.original != (labels[0], labels[1], bonds):
                    raise ValueError("Progress witness changed its source identity")
                if witness.kind == "valid":
                    inspected = inspect_graph(*witness.labels, spec)
                    if inspected.smiles is None or encode_smiles(inspected.smiles, spec)["smiles"] != inspected.smiles:
                        raise ValueError("Complete progress witness failed independent native validation")
        row = FrozenFailureRow(candidate_id, *labels, bonds, tuple(caps), _components(labels[0], bonds), witness)
        rows.append(row)
        records.append(dict(row=index, candidate_id=candidate_id, original_status=status,
            supported_local_atoms=sum(cap is not None for cap in caps),
            unsupported_local_atoms=sum(bool(a) and cap is None for a, cap in zip(labels[0], caps)),
            component_count=len(row.components), empty=not any(labels[0]),
            extra_search=None if extra is None else {k: v for k, v in extra.items() if k != "witness"},
            witness_kind=witness.kind if witness is not None else None,
            witness_sha256=witness.identity if witness is not None else None))
    receipt = dict(version="prepared-failure-guidance-v1", attempted_rows=count, failed_rows=len(records),
             rule_version=RULE_VERSION, search=None if progress_search is None else progress_search.record(),
             rows=records,
             fallback_rows=sum(row is not None and (row.witness is None or row.witness.kind != "valid") for row in rows),
             scope="Fixed references; fallback only without a complete target; all-attempt denominator",
             normalization={"valid": "mean over N+K+K*(K-1)/2 decisions, then all rows",
                            "partial": "mean over changed unordered bonds, then all rows",
                            "local": "sum supported local risks / all hard occupied atoms, then all rows",
                            "cuts": "mean over original component cuts, then all rows",
                            "nonempty": "one negative-log nonempty probability, then all rows"})
    identity = _prepared_identity(base, rows, spec, receipt)
    receipt["identity"] = identity
    return PreparedFailureGuidance(base, tuple(rows), spec.fingerprint(), identity, receipt)


def _safe_lse(terms):
    """Log-sum-exp with zero derivative for genuinely unreachable DP states."""
    stack = torch.stack(terms, -1)
    supported = torch.isfinite(stack).any(-1)
    safe = torch.where(supported[..., None], stack, torch.zeros_like(stack))
    value = torch.logsumexp(safe, -1)
    return torch.where(supported, value, torch.full_like(value, float("-inf")))


def _log_any_present(log_absent, log_present):
    """Disjoint first-present partition; stable even when absence rounds to 1."""
    if log_absent.numel() == 0:
        raise ValueError("An empty event set cannot provide a connection")
    prefix = torch.cat((log_absent.new_zeros(1), log_absent.cumsum(0)[:-1]))
    return torch.logsumexp(prefix + log_present, 0)


def _local_admissibility(log_bonds, rows, spec):
    """Exact log-space incident-order convolution for frozen hard identities.

    Aromatic counts as ONE: optimistic lower-bound scope, not aromaticity
    validation. Undercoordination is allowed because hydrogens are implicit.
    Only bond logits receive this gradient; no probability floor is applied.
    """
    index = {name: i for i, name in enumerate(spec.bond_names)}
    if set(index) != {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
        raise ValueError("Local risk requires the declared five bond categories")
    # Batch all failed molecules and focal atoms. Python loop count depends on
    # slot capacity and rule cap, not batch size, preventing a per-atom GPU loop.
    device = log_bonds.device
    active = torch.tensor([[bool(a) for a in row.atoms] for row in rows], device=device)
    supported = torch.tensor([[bool(a) and cap is not None for a, cap in zip(row.atoms, row.caps)]
                              for row in rows], device=device)
    caps = torch.tensor([[cap if cap is not None else 0 for cap in row.caps] for row in rows], device=device)
    maximum = max((cap for row in rows for cap in row.caps if cap is not None), default=0)
    distribution = [log_bonds.new_zeros(active.shape)] + [log_bonds.new_full(active.shape, float("-inf")) for _ in range(maximum)]
    # Frozen identities already tell us which columns are absent from EVERY
    # failed graph. Such columns contribute a constant order-zero event, so
    # skipping them removes identity DP transitions without changing the law.
    occupied_columns = sorted({j for row in rows for j, atom in enumerate(row.atoms) if atom})
    for j in occupied_columns:
        edge = log_bonds[:, :, j]
        orders = (edge[..., index["NONE"]], torch.logsumexp(edge[..., [index["SINGLE"], index["AROMATIC"]]], -1),
                  edge[..., index["DOUBLE"]], edge[..., index["TRIPLE"]])
        included = active & active[:, j, None]
        included = included & (torch.arange(spec.max_nodes, device=device) != j)[None]
        orders = tuple(torch.where(included, value, torch.full_like(value, 0.0 if order == 0 else float("-inf")))
                       for order, value in enumerate(orders))
        distribution = [_safe_lse([distribution[value - order] + orders[order]
                                   for order in range(min(3, value) + 1)]) for value in range(maximum + 1)]
    state = torch.stack(distribution, -1)
    allowed = torch.arange(maximum + 1, device=device) <= caps[..., None]
    log_mass = torch.logsumexp(torch.where(allowed, state, torch.full_like(state, float("-inf"))), -1)
    # Exact admissible mass is at most one; clamp only positive roundoff, not
    # small probability tails. Log-space computation retains confident gradients.
    penalty = torch.where(supported, -log_mass.clamp_max(0), torch.zeros_like(log_mass))
    return penalty.sum(-1) / active.sum(-1).clamp_min(1)


def _progress_nll(output, witness, spec):
    witness.verify(spec)
    atoms, charges, bonds = witness.labels
    device = output.atoms.device
    if witness.kind == "valid":
        graph = HardGraphBatch(*(torch.tensor(v, device=device, dtype=torch.long)[None]
                                 for v in (atoms, charges, bonds)))
        terms = hard_graph_log_prob_terms(output, graph, spec)
        if not bool(terms["supported"].all()):
            raise ValueError("Additional valid witness is outside the shared graph law")
        count = sum(bool(a) for a in atoms)
        return -(terms["atoms"] + terms["charges"] + terms["bonds"]).sum() / (spec.max_nodes + count + count * (count - 1) // 2)
    logp = F.log_softmax(output.symmetric_bond_logits[0], -1)
    edits = [-logp[i, j, bonds[i][j]] for i in range(spec.max_nodes) for j in range(i + 1, spec.max_nodes)
             if bonds[i][j] != witness.original[2][i][j]]
    if not edits:
        raise ValueError("A partial target needs actual changed bond labels")
    return torch.stack(edits).mean()


def failure_aware_repair_loss(decoded, prepared, spec, config=None):
    """Return separate differentiable components and immutable coverage metadata.

    Call explicitly as a new objective; this is not a drop-in change to a sealed
    source121 plan. Existing valid-target NLL and all-row normalization remain.
    Dense local risks apply on original failed rows even if extra search finds
    no witness. This provides additional chemical signals, not a promise of a
    nonzero gradient for every defect or of complete molecular validity.
    """
    config = FailureGuidanceConfig() if config is None else config
    if not isinstance(config, FailureGuidanceConfig):
        raise ValueError("Require an explicit failure guidance configuration")
    count = _validated_decoded(decoded, spec)
    if (not isinstance(prepared, PreparedFailureGuidance) or len(prepared.rows) != count
            or prepared.schema_sha256 != spec.fingerprint()
            or prepared.receipt.get("identity") != prepared.identity
            or prepared.identity != _prepared_identity(prepared.base, prepared.rows, spec, prepared.receipt)):
        raise ValueError("Prepared failure identities/schema changed")
    zero = sum(v.reshape(-1)[0] * 0 for v in (decoded.atoms, decoded.charges, decoded.bonds))
    values = {key: zero for key in ("extra_valid_nll", "partial_edited_nll", "local_bond_risk", "component_cut_risk", "nonempty_risk")}
    failed = [i for i, row in enumerate(prepared.rows)
              if row is not None and (row.witness is None or row.witness.kind != "valid")]
    if failed:
        logits = decoded.symmetric_bond_logits[failed]
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        all_log_bonds = F.log_softmax(logits, -1)
        if not bool(all_log_bonds.isfinite().all()):
            raise ValueError("Local bond log probabilities are nonfinite")
        values["local_bond_risk"] = _local_admissibility(all_log_bonds, [prepared.rows[i] for i in failed], spec).sum() / count
    for i, row in enumerate(prepared.rows):
        if row is None:
            continue
        active = [j for j, atom in enumerate(row.atoms) if atom]
        one = DecoderOutput(*(v[i:i + 1].float() if v.dtype in (torch.float16, torch.bfloat16) else v[i:i + 1]
                              for v in (decoded.atoms, decoded.charges, decoded.bonds)))
        if row.witness is not None and row.witness.kind == "valid":
            # A valid atom-edited witness may require a different valence cap.
            # Its full NLL replaces fallback guidance, avoiding contradictory
            # original-identity penalties. This is still one shared output.
            values["extra_valid_nll"] = values["extra_valid_nll"] + _progress_nll(one, row.witness, spec) / count
            continue
        log_bonds = F.log_softmax(one.symmetric_bond_logits[0], -1)
        if not bool(log_bonds.isfinite().all()):
            raise ValueError("Local bond log probabilities are nonfinite")
        if not active:
            log_atoms = F.log_softmax(one.atoms[0], -1)
            values["nonempty_risk"] = values["nonempty_risk"] - _log_any_present(
                log_atoms[:, 0], torch.logsumexp(log_atoms[:, 1:], -1)) / count
        if len(row.components) > 1:
            cuts = []
            none = spec.bond_names.index("NONE")
            occupied = [j for j in range(len(spec.bond_names)) if j != none]
            for component in row.components:
                outside = sorted(set(active) - set(component))
                edges = torch.stack([log_bonds[a, b] for a in component for b in outside])
                cuts.append(-_log_any_present(edges[:, none], torch.logsumexp(edges[:, occupied], -1)))
            values["component_cut_risk"] = values["component_cut_risk"] + torch.stack(cuts).mean() / count
        if row.witness is not None:
            key = "extra_valid_nll" if row.witness.kind == "valid" else "partial_edited_nll"
            values[key] = values[key] + _progress_nll(one, row.witness, spec) / count
    base = repair_loss(decoded, prepared.base, spec)
    total = (base + config.extra_valid_weight * values["extra_valid_nll"]
             + config.partial_weight * values["partial_edited_nll"]
             + config.local_bond_weight * values["local_bond_risk"]
             + config.component_cut_weight * values["component_cut_risk"]
             + config.nonempty_weight * values["nonempty_risk"])
    if not bool(total.isfinite()):
        raise ValueError("Failure-aware objective is nonfinite")
    return dict(total=total, valid_full_nll=base, **values, coverage=prepared.receipt,
                policy=asdict(config),
                limitations=("Local fixed-identity bound and component-cut surrogates; aromatic order-one lower bound; "
                             "unsupported identities, Kekulization and radical chemistry not fully covered; "
                             "no direct atom/charge/occupancy gradient from local bond risk; indirect drift remains possible"))
