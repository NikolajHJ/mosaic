"""Detached valid-target anchoring to an authenticated frozen epoch-36 decoder.

The reference and live decoder see the SAME fresh-prior latent points. Only
strictly valid reference argmax graphs become fixed categorical targets; their
eligibility does not disappear when the live graph becomes invalid. This is
distillation of reference identities, not a validity likelihood or repair. It
does not help points whose reference graph is invalid and may constrain changes
to otherwise valid identities. The separate epoch-35 occupancy anchor is not
replaced. No classifier, argmax, reference forward or target selection is in
autograd, and preparation never runs repair search.

Module-level imports are standard-library-only for metadata policy readers.
All numerical and chemistry fixtures belong on allocated DTU HPC.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import re


VERSION = "frozen-valid-prior-reference-v1"


def validate_valid_prior_reference_guidance(value):
    """Validate the closed alpha-0.1 candidate; None denotes historical absence.

    A caller that sees an explicit null protocol field must reject it rather
    than treating that field as an absent policy. This helper imports no model
    or chemistry runtime and returns a defensive copy.
    """
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"version", "weight"}
            or value.get("version") != VERSION
            or type(value.get("weight")) is not float or value["weight"] != .1):
        raise ValueError("Require explicit frozen-valid-prior-reference-v1 with weight 0.1")
    return dict(value)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _reference_identity(value):
    if (type(value) is not dict or set(value) != {"epoch", "checkpoint_sha256", "model_sha256"}
            or type(value.get("epoch")) is not int or value["epoch"] != 36
            or not all(_sha(value.get(key)) for key in ("checkpoint_sha256", "model_sha256"))):
        raise ValueError("Reference targets require the authenticated original epoch-36 identity")
    return deepcopy(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _graph_digest(graph):
    # Use the existing detached graph identity convention, including tensor
    # names/dtypes/shapes. The import remains lazy at numerical boundaries.
    from .chemistry_valid_self_guidance import _graph_digest as digest
    return digest(graph)


@dataclass(frozen=True)
class PreparedValidReferenceTargets:
    """An authenticated complete CPU reference snapshot and ordered subset view.

    Public accessors return copies. The private source receipt binds every
    ordered native classifier verdict, raw graph/probability identity, latent
    digest and reference checkpoint/model identity. A subset retains that full
    receipt, so it cannot silently reclassify a slice or change its denominator.
    """
    _graph: object
    _spec: object
    _candidate_ids: tuple[int, ...]
    _eligible: tuple[int, ...]
    _view: tuple[int, ...]
    _source_receipt: dict
    _source_receipt_sha256: str
    _source_graph_sha256: str
    identity: str

    @property
    def candidate_ids(self):
        return tuple(self._candidate_ids[i] for i in self._view)

    @property
    def eligible_rows(self):
        selected = set(self._eligible)
        return tuple(local for local, source in enumerate(self._view) if source in selected)

    @property
    def reference_identity(self):
        _verify(self, self._spec)
        return deepcopy(self._source_receipt["reference_identity"])

    @property
    def graph(self):
        from .representation import HardGraphBatch
        _verify(self, self._spec)
        return HardGraphBatch(*(getattr(self._graph, name)[list(self._view)].clone()
                                for name in ("atoms", "charges", "bonds")))

    @property
    def classification_records(self):
        """Return bounded-by-batch native records without exposing private state."""
        _verify(self, self._spec)
        return deepcopy([self._source_receipt["records"][i] for i in self._view])

    @property
    def receipt(self):
        _verify(self, self._spec)
        return _semantic_receipt(self)

    def subset(self, indices):
        _verify(self, self._spec)
        indices = tuple(indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < len(self._view) for i in indices)):
            raise ValueError("Reference-target subset needs unique nonempty in-range indices")
        result = replace(self, _view=tuple(self._view[i] for i in indices), identity="")
        return replace(result, identity=_identity(result))


def _identity(prepared):
    return _digest(dict(version=VERSION, schema=prepared._spec.fingerprint(),
        source=prepared._source_receipt_sha256, graphs=prepared._source_graph_sha256,
        candidates=prepared._candidate_ids, eligible=prepared._eligible, view=prepared._view))


def _verify(prepared, spec):
    import torch
    if (not isinstance(prepared, PreparedValidReferenceTargets)
            or prepared._spec.fingerprint() != spec.fingerprint()):
        raise ValueError("Reference-target preparation or shared schema changed")
    count = len(prepared._candidate_ids)
    fields = (prepared._graph.atoms, prepared._graph.charges, prepared._graph.bonds)
    shapes = ((count, spec.max_nodes), (count, spec.max_nodes), (count, spec.max_nodes, spec.max_nodes))
    if (prepared._source_receipt.get("version") != VERSION
            or prepared._source_receipt.get("schema_sha256") != spec.fingerprint()
            or prepared._source_receipt.get("candidate_ids") != list(prepared._candidate_ids)
            or _digest(prepared._source_receipt) != prepared._source_receipt_sha256
            or any(tuple(value.shape) != shape or value.dtype != torch.long or value.device.type != "cpu"
                   or value.requires_grad for value, shape in zip(fields, shapes))
            or _graph_digest(prepared._graph) != prepared._source_graph_sha256
            or prepared.identity != _identity(prepared)
            or not prepared._view or len(set(prepared._view)) != len(prepared._view)
            or any(type(i) is not int or not 0 <= i < count for i in prepared._view)):
        raise ValueError("Detached reference-target identity, receipt, labels or alignment changed")


def _semantic_receipt(prepared):
    from .chemistry_classification import classification_summary
    source, view = prepared._source_receipt, prepared._view
    records = [source["records"][i] for i in view]
    eligible = prepared.eligible_rows
    return dict(version=VERSION, schema_sha256=prepared._spec.fingerprint(),
        source_update=source["source_update"], candidate_ids=list(prepared.candidate_ids),
        eligible_candidate_ids=[prepared.candidate_ids[i] for i in eligible],
        scheduled_prior_attempts=len(view), strict_valid_reference_targets=len(eligible),
        excluded_prior_attempts=len(view)-len(eligible), accepted_repair_witnesses_created=0,
        reference_searches=0, reference_identity=deepcopy(source["reference_identity"]),
        latent_sha256=source["latent_sha256"], source_snapshot_sha256=prepared._source_receipt_sha256,
        target_label_sha256=prepared._source_graph_sha256,
        source_probability_sha256=[record["probability_sha256"] for record in records],
        raw_graph_sha256=[record["raw_sha256"] for record in records],
        reference_classification=classification_summary(records, max_examples_per_failure=0),
        normalization="N+K+K*(K-1)/2 per reference-target; mean over ALL scheduled prior attempts",
        target_source="frozen_original_epoch36_strict_valid_argmax_at_same_prior_z; classification only",
        gradient_scope="live decoder logits to fixed detached labels; no reference/classifier/argmax gradient")


def prepare_valid_reference_guidance(reference_decoded_cpu, spec, *, candidate_ids, update,
                                     reference_identity, latent_sha256, classifier=None):
    """Classify a detached frozen-reference snapshot once, without repair search.

    ``classifier``, when supplied, is a trusted native execution adapter taking
    ``(ordered_raw_numpy_graphs, spec)`` and returning native verdict dictionaries
    in the SAME order. It may use the already allocated classifier workers;
    changing labels, filtering/reordering rows or returning custom validity
    scores violates this contract. Input arrays are defensive read-only copies.
    Full preparation must happen before making microbatch subset views.
    """
    import torch
    from .atom_preserving_chemistry import _validated_decoded
    from .chemistry_classification import (VERSION as CLASSIFICATION_VERSION, STAGES,
        classify_repair_graph, classification_summary)
    from .probability_repair import probability_identity
    from .quality import graph_identity
    from .representation import DecoderOutput, HardGraphBatch

    count = _validated_decoded(reference_decoded_cpu, spec)
    if any(value.device.type != "cpu" for value in
           (reference_decoded_cpu.atoms, reference_decoded_cpu.charges, reference_decoded_cpu.bonds)):
        raise ValueError("Prepare reference targets from the detached CPU frozen-reference snapshot")
    ids = tuple(candidate_ids)
    if (len(ids) != count or len(set(ids)) != count
            or any(type(value) is not int or value < 0 for value in ids)
            or type(update) is not int or update < 0 or not _sha(latent_sha256)):
        raise ValueError("Reference targets require aligned unique point IDs, update and latent SHA256")
    reference = _reference_identity(reference_identity)
    cpu = DecoderOutput(*(value.detach().clone().float() if value.dtype in (torch.float16, torch.bfloat16)
                          else value.detach().clone() for value in
                          (reference_decoded_cpu.atoms, reference_decoded_cpu.charges, reference_decoded_cpu.bonds)))
    graph = cpu.harden(spec)
    probabilities = (cpu.atoms.softmax(-1), cpu.charges.softmax(-1), cpu.symmetric_bond_logits.softmax(-1))
    raw, graph_ids, probability_ids = [], [], []
    for row in range(count):
        fields = tuple(value[row].numpy().copy() for value in (graph.atoms, graph.charges, graph.bonds))
        for value in fields:
            value.setflags(write=False)
        raw.append(fields)
        graph_ids.append(graph_identity(*fields, spec))
        probability_ids.append(probability_identity(tuple(value[row] for value in probabilities), spec))
    verdicts = ([classify_repair_graph(*fields, spec) for fields in raw] if classifier is None
                else classifier(raw, spec))
    if type(verdicts) not in (list, tuple) or len(verdicts) != count:
        raise ValueError("Reference classifier must return every native verdict in input order")
    if any(graph_identity(*fields, spec) != identity for fields, identity in zip(raw, graph_ids)):
        raise ValueError("Reference classifier changed an original frozen graph")
    records, eligible = [], []
    for row, verdict in enumerate(verdicts):
        if (type(verdict) is not dict or verdict.get("version") != CLASSIFICATION_VERSION
                or type(verdict.get("strict_teacher_valid")) is not bool):
            raise ValueError("Reference targets require explicit native classification verdicts")
        verdict = deepcopy(verdict)
        if verdict["strict_teacher_valid"]:
            if (verdict.get("failure_kind") != "none" or verdict.get("reason") is not None
                    or any(verdict.get(stage) is not True for stage in STAGES)
                    or type(verdict.get("smiles")) is not str or not verdict["smiles"]
                    or verdict.get("representation_smiles") != verdict["smiles"]):
                raise ValueError("Reference target did not pass every native chemical/schema stage")
            eligible.append(row)
        records.append(dict(candidate_id=ids[row], raw_sha256=graph_ids[row],
                            probability_sha256=probability_ids[row], classification=verdict))
    # Reuse the native summary validator for excluded-row kinds, stage types and
    # reasons. This does not classify again and supplies no repair witness.
    classification_summary(records, max_examples_per_failure=0)
    source = dict(version=VERSION, schema_sha256=spec.fingerprint(), source_update=update,
        candidate_ids=list(ids), reference_identity=reference, latent_sha256=latent_sha256, records=records)
    frozen = HardGraphBatch(*(value.detach().cpu().clone() for value in (graph.atoms, graph.charges, graph.bonds)))
    prepared = PreparedValidReferenceTargets(frozen, spec, ids, tuple(eligible), tuple(range(count)),
        deepcopy(source), _digest(source), _graph_digest(frozen), "")
    return replace(prepared, identity=_identity(prepared))


def valid_prior_reference_nll(decoded, prepared, spec, policy):
    """Historical full-graph NLL of fixed valid reference labels, ALL-prior mean.

    Reference-invalid rows contribute explicit differentiable zero; live-invalid
    rows retain their frozen valid reference target. The coefficient is applied
    once here and is independent of the repair branch weights and epoch ramp.
    """
    import torch
    from .atom_preserving_chemistry import _validated_decoded
    from .representation import DecoderOutput, HardGraphBatch
    from .sampling import hard_graph_log_prob_terms

    policy = validate_valid_prior_reference_guidance(policy)
    if policy is None:
        raise ValueError("Calling reference NLL requires the explicit reference policy")
    count = _validated_decoded(decoded, spec)
    _verify(prepared, spec)
    if count != len(prepared._view):
        raise ValueError("Live decoder and fixed reference-target views are misaligned")
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
        terms = hard_graph_log_prob_terms(one, target, spec)
        if not bool(terms["supported"].all()):
            raise ValueError("Detached reference target left the shared graph law")
        occupied = (target.atoms != 0).sum(-1).to(one.atoms.dtype)
        decisions = spec.max_nodes + occupied + occupied * (occupied - 1) / 2
        # Deliberately match the old unweighted self-NLL operation order.
        loss = loss - ((terms["atoms"] + terms["charges"] + terms["bonds"]) / decisions).sum() / count
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Valid-prior reference NLL is nonfinite")
    return dict(loss=loss, weighted_loss=loss*policy["weight"], valid_mask=mask,
        eligible_rows=eligible, receipt=_semantic_receipt(prepared), policy=dict(policy),
        reference_identity=deepcopy(prepared._source_receipt["reference_identity"]))
