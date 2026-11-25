"""Opt-in confidence guidance for existing strictly valid prior argmax graphs.

This is detached hard-label self-distillation, not a repair witness or an exact
valid-mass objective. Preparation authenticates the original complete classified
teacher snapshot; targets then stay fixed while live decoder logits receive a
categorical NLL. Invalid, repaired and unsupported rows contribute zero to the
ALL-prior-attempt mean. No classifier/search/argmax path is differentiated.

Only CPU hard labels and provenance survive preparation. Module-level imports
are standard-library-only so policy validation remains safe at the scheduler
boundary. Numerical fixtures must run on allocated DTU HPC before adoption.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json


VERSION = "valid-prior-self-nll-v1"


def validate_valid_prior_self_guidance(value):
    """Delegate coefficient/schema validation to the shared metadata authority."""
    from .chemistry_short_continuation import validate_valid_prior_self_guidance as validate
    return validate(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _graph_digest(graph):
    digest = hashlib.sha256()
    for name in ("atoms", "charges", "bonds"):
        value = getattr(graph, name)
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedValidSelfTargets:
    """One authenticated full CPU snapshot with optional ordered subset views.

    Keep the complete receipt even in a subset. The private full identity binds
    timing/cache observations against mutation, while the public scientific
    receipt excludes them so execution-parity comparisons remain meaningful.
    Public graph readers receive copies; no live target tensor is returned.
    """
    _base: object
    _graph: object
    _spec: object
    _candidate_ids: tuple[int, ...]
    _statuses: tuple[str, ...]
    _eligible: tuple[int, ...]
    _view: tuple[int, ...]
    _source_receipt_sha256: str
    _source_graph_sha256: str
    identity: str

    @property
    def candidate_ids(self):
        return tuple(self._candidate_ids[i] for i in self._view)

    @property
    def eligible_rows(self):
        accepted = set(self._eligible)
        return tuple(local for local, source in enumerate(self._view) if source in accepted)

    @property
    def graph(self):
        from .representation import HardGraphBatch
        _verify(self, self._spec)
        return HardGraphBatch(*(getattr(self._graph, name)[list(self._view)].clone()
                                for name in ("atoms", "charges", "bonds")))

    @property
    def receipt(self):
        _verify(self, self._spec)
        return _semantic_receipt(self)

    def subset(self, indices):
        """Subset an already prepared full snapshot, never reprepare sliced receipts."""
        _verify(self, self._spec)
        indices = tuple(indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < len(self._view) for i in indices)):
            raise ValueError("Self-target subset needs unique nonempty in-range indices")
        result = replace(self, _view=tuple(self._view[i] for i in indices), identity="")
        return replace(result, identity=_identity(result))


def _identity(prepared):
    return _digest(dict(version=VERSION, schema=prepared._spec.fingerprint(),
        receipt=prepared._source_receipt_sha256, graphs=prepared._source_graph_sha256,
        candidates=prepared._candidate_ids, statuses=prepared._statuses,
        eligible=prepared._eligible, view=prepared._view))


def _verify(prepared, spec):
    import torch
    from .atom_preserving_chemistry import RepairTargetBatch
    if not isinstance(prepared, PreparedValidSelfTargets) or prepared._spec.fingerprint() != spec.fingerprint():
        raise ValueError("Self-target preparation or shared schema changed")
    base = prepared._base
    count = len(prepared._candidate_ids)
    fields = (prepared._graph.atoms, prepared._graph.charges, prepared._graph.bonds)
    shapes = ((count, spec.max_nodes), (count, spec.max_nodes), (count, spec.max_nodes, spec.max_nodes))
    if (not isinstance(base, RepairTargetBatch) or base.candidate_ids != prepared._candidate_ids
            or len(base.targets) != count or _digest(base.receipt) != prepared._source_receipt_sha256
            or any(tuple(value.shape) != shape or value.dtype != torch.long or value.device.type != "cpu"
                   or value.requires_grad for value, shape in zip(fields, shapes))
            or _graph_digest(prepared._graph) != prepared._source_graph_sha256
            or prepared.identity != _identity(prepared)
            or not prepared._view or len(set(prepared._view)) != len(prepared._view)
            or any(type(i) is not int or not 0 <= i < count for i in prepared._view)
            or any((target is not None) != (status == "accepted")
                   for target, status in zip(base.targets, prepared._statuses))):
        raise ValueError("Detached self-target identity, receipt, labels or alignment changed")


def _semantic_receipt(prepared):
    view, eligible = prepared._view, prepared.eligible_rows
    base = prepared._base.receipt
    return dict(version=VERSION, schema_sha256=prepared._spec.fingerprint(), source_update=base["source_update"],
        candidate_ids=list(prepared.candidate_ids), eligible_candidate_ids=[prepared.candidate_ids[i] for i in eligible],
        scheduled_prior_attempts=len(view), strict_valid_self_targets=len(eligible),
        excluded_prior_attempts=len(view) - len(eligible),
        original_accepted_repair_targets=sum(prepared._statuses[i] == "accepted" for i in view),
        accepted_repair_witnesses_created=0,
        target_label_sha256=_graph_digest(prepared._graph),
        source_probability_sha256=[base["attempts"][i]["probability_sha256"] for i in view],
        normalization="N+K+K*(K-1)/2 per self-target; mean over ALL scheduled prior attempts",
        target_source="native_strict_classified_already_valid_argmax; no repair or target change",
        gradient_scope="live decoder logits to fixed detached labels; no selection/classifier/argmax gradient")


def prepare_valid_self_guidance(decoded_cpu, base, spec):
    """Authenticate a complete CPU prepass and freeze only native already-valid labels.

    This boundary does not run a classifier or repair search. It uses their
    existing aligned per-row receipts and independently checks the exact raw
    label/probability identities. Complete preparation must precede subsetting.
    """
    import torch
    from .atom_preserving_chemistry import (RepairTargetBatch, VERSION as REPAIR_VERSION,
        _validated_decoded, _validate_repair_loss_targets)
    from .chemistry_classification import VERSION as CLASSIFICATION_VERSION
    from .probability_repair import probability_identity
    from .quality import graph_identity
    from .representation import DecoderOutput, HardGraphBatch

    count = _validated_decoded(decoded_cpu, spec)
    if any(value.device.type != "cpu" for value in (decoded_cpu.atoms, decoded_cpu.charges, decoded_cpu.bonds)):
        raise ValueError("Prepare self-targets from the detached CPU teacher prepass")
    if (not isinstance(base, RepairTargetBatch) or len(base.targets) != count
            or base.receipt.get("version") != REPAIR_VERSION
            or base.receipt.get("schema_sha256") != spec.fingerprint()
            or base.receipt.get("attempted_candidates") != count
            or type(base.receipt.get("source_update")) is not int or base.receipt["source_update"] < 0):
        raise ValueError("Self-targets require the original complete classified teacher receipt")
    _validate_repair_loss_targets(base, spec, count)
    attempts = base.receipt.get("attempts")
    classification = base.receipt.get("chemical_classification", {})
    if (type(attempts) is not list or len(attempts) != count
            or classification.get("version") != CLASSIFICATION_VERSION
            or classification.get("attempts") != count):
        raise ValueError("Self-targets require complete per-attempt native classification")
    cpu = DecoderOutput(*(value.detach().clone().float() if value.dtype in (torch.float16, torch.bfloat16)
                          else value.detach().clone()
                          for value in (decoded_cpu.atoms, decoded_cpu.charges, decoded_cpu.bonds)))
    graph = cpu.harden(spec)
    probabilities = (cpu.atoms.softmax(-1), cpu.charges.softmax(-1), cpu.symmetric_bond_logits.softmax(-1))
    eligible, statuses = [], []
    for row, (candidate, attempt, target) in enumerate(zip(base.candidate_ids, attempts, base.targets)):
        status = attempt.get("status")
        record, search = attempt.get("chemical_classification"), attempt.get("search")
        raw = tuple(value[row] for value in (graph.atoms, graph.charges, graph.bonds))
        probability = tuple(value[row] for value in probabilities)
        if (attempt.get("candidate_id") != candidate
                or status not in ("accepted", "skipped_already_valid", "censored", "unsupported")
                or (target is not None) != (status == "accepted")
                or attempt.get("raw_sha256") != graph_identity(*raw, spec)
                or attempt.get("probability_sha256") != probability_identity(probability, spec)
                or type(record) is not dict or record.get("version") != CLASSIFICATION_VERSION
                or type(record.get("strict_teacher_valid")) is not bool or type(search) is not dict
                or search.get("raw_graph_id") != attempt["raw_sha256"]
                or search.get("probability_sha256") != attempt["probability_sha256"]
                or record["strict_teacher_valid"] != (search.get("raw_valid") is True)):
            raise ValueError("Self-target source classification/probabilities/labels lost provenance")
        already_valid = status == "skipped_already_valid"
        if already_valid != record["strict_teacher_valid"]:
            raise ValueError("Only strictly verified already-valid rows may receive self-guidance")
        if already_valid:
            if (target is not None or search.get("status") != "already_valid"
                    or record.get("failure_kind") != "none"
                    or type(record.get("smiles")) is not str or not record["smiles"]
                    or record.get("representation_smiles") != record["smiles"]
                    or any(record.get(key) is not True for key in ("graph_sanitization_valid", "raw_chemical_valid",
                                                                  "smiles_parse_valid", "schema_roundtrip_valid"))):
                raise ValueError("Already-valid self-target is not a native strict-valid graph")
            eligible.append(row)
        statuses.append(status)
    if (dict(Counter(statuses)) != base.receipt.get("status_counts")
            or base.receipt.get("accepted_targets") != statuses.count("accepted")
            or classification.get("counts", {}).get("strict_teacher_valid") != len(eligible)):
        raise ValueError("Self-target eligibility disagrees with original all-attempt counters")
    frozen = HardGraphBatch(*(value.detach().cpu().clone() for value in (graph.atoms, graph.charges, graph.bonds)))
    prepared = PreparedValidSelfTargets(base, frozen, spec, base.candidate_ids, tuple(statuses), tuple(eligible),
        tuple(range(count)), _digest(base.receipt), _graph_digest(frozen), "")
    return replace(prepared, identity=_identity(prepared))


def valid_prior_self_nll(decoded, prepared, spec, policy=None, *, target_weighting=None):
    """Full categorical self-NLL with all-prior denominator and fixed targets.

    The caller applies only the logical prior-microbatch fraction. This term is
    independent of the repair coefficient/ramp and creates no repair successes.
    Targets/eligibility remain CPU constants; only live decoder logits have a
    gradient. It sharpens selected modes, not categorical dependencies.
    An explicit target-weighting policy uses weighted target CE instead of
    exact graph NLL; the same target provenance/all-attempt mean still apply.
    """
    import torch
    from .atom_preserving_chemistry import _validated_decoded
    from .representation import DecoderOutput, HardGraphBatch
    from .sampling import hard_graph_log_prob_terms

    policy = validate_valid_prior_self_guidance({"version": VERSION, "weight": .01} if policy is None else policy)
    if policy is None:
        raise ValueError("Calling self-NLL requires an explicit self-guidance policy")
    if target_weighting is not None:
        from .chemistry_target_weighting import validate_chemistry_target_weighting, fixed_hard_target_ce
        target_weighting = validate_chemistry_target_weighting(target_weighting)
    count = _validated_decoded(decoded, spec)
    _verify(prepared, spec)
    if count != len(prepared._view):
        raise ValueError("Live decoder and fixed self-target views are misaligned")
    eligible = prepared.eligible_rows
    mask = torch.zeros(count, dtype=torch.bool)
    mask[list(eligible)] = True
    loss = sum(value.reshape(-1)[0] * 0 for value in (decoded.atoms, decoded.charges, decoded.bonds))
    if eligible:
        one = DecoderOutput(*(value[list(eligible)] for value in (decoded.atoms, decoded.charges, decoded.bonds)))
        if one.atoms.dtype in (torch.float16, torch.bfloat16):
            one = DecoderOutput(*(value.float() for value in (one.atoms, one.charges, one.bonds)))
        original_rows = [prepared._view[i] for i in eligible]
        target = HardGraphBatch(*(getattr(prepared._graph, name)[original_rows].to(one.atoms.device)
                                  for name in ("atoms", "charges", "bonds")))
        if target_weighting is None:
            terms = hard_graph_log_prob_terms(one, target, spec)
            if not bool(terms["supported"].all()):
                raise ValueError("Detached valid self-target left the shared graph law")
            occupied = (target.atoms != 0).sum(-1).to(one.atoms.dtype)
            decisions = spec.max_nodes + occupied + occupied * (occupied - 1) / 2
            loss = loss - ((terms["atoms"] + terms["charges"] + terms["bonds"]) / decisions).sum() / count
        else:
            loss = loss + fixed_hard_target_ce(one, target, spec, target_weighting).sum() / count
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Valid-prior self-NLL is nonfinite")
    result = dict(loss=loss, weighted_loss=loss * policy["weight"], valid_mask=mask,
                  eligible_rows=eligible, receipt=_semantic_receipt(prepared), policy=dict(policy))
    if target_weighting is not None:
        result["target_weighting"] = dict(target_weighting)
        result["receipt"].update(target_weighting=dict(target_weighting),
            normalization="N+K+E+0.1*NE per self-target; mean over ALL scheduled prior attempts",
            objective="Weighted categorical target CE; not exact graph NLL")
    return result
