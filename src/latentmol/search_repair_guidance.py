"""Detached bounded-search witnesses as decoder cross-entropy references.

This is a heuristic alternating teacher, not a learned chemistry prior, an
exact gradient of validity probability, or globally nearest-valid projection.
Search/argmax/selection/RDKit checks occur outside autograd. Only the decoder's
unconditional categorical probability of an accepted fixed witness carries
gradients. Already-valid candidates receive no guidance, so the teacher does
not guarantee their preservation between refreshes.

Unlike fixed-occupancy latent DeFoG guidance, atom NLL here includes every
NO_ATOM decision. Occupancy therefore receives a gradient. Charges contribute
only at occupied atoms, and each occupied unordered bond pair (including NONE)
contributes once. Normalize by N + K + K(K-1)/2 per witness, then average over
the full prescribed minibatch, including zero for unavailable targets.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from latentmol.chemistry import inspect_graph
from latentmol.probability_repair import ProbabilityRepairConfig, measure_probability_edits, probability_identity, search_probability_repair
from latentmol.quality import graph_identity
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob_terms


VERSION = "bounded-search-witness-teacher-v1"


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _panel_hash(panel: Tensor) -> str:
    digest = hashlib.sha256(json.dumps({"shape": list(panel.shape), "dtype": str(panel.dtype)}, sort_keys=True).encode())
    digest.update(panel.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _graph_hash(graph: HardGraphBatch, spec: GraphSpec) -> str:
    return graph_identity(*(value[0] for value in (graph.atoms, graph.charges, graph.bonds)), spec)


def _validate_decoded(decoded: DecoderOutput, spec: GraphSpec, batch: int) -> None:
    if not isinstance(decoded, DecoderOutput):
        raise ValueError("Teacher requires DecoderOutput")
    if type(batch) is not int or batch < 1:
        raise ValueError("Teacher decoder batch must be nonempty")
    n = spec.max_nodes
    expected = ((batch, n, len(spec.atomic_numbers)), (batch, n, len(spec.formal_charges)),
                (batch, n, n, len(spec.bond_names)))
    values = (decoded.atoms, decoded.charges, decoded.bonds)
    if any(tuple(value.shape) != shape or not value.is_floating_point() for value, shape in zip(values, expected)):
        raise ValueError("Teacher decoder shapes/dtypes disagree with the panel/schema")
    if any(value.device != values[0].device or value.dtype != values[0].dtype or not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Teacher logits must be finite with a common dtype/device")


@dataclass(frozen=True)
class FrozenSearchRepairTarget:
    """Private detached labels plus source/target identities verified on reuse."""

    latent_id: int
    source_update: int
    graph: HardGraphBatch
    target_sha256: str
    raw_graph_sha256: str
    source_probability_sha256: str
    weighted_cost: float
    edit_count: int
    decisions: int
    schema_sha256: str

    def verify(self, spec: GraphSpec) -> None:
        if self.schema_sha256 != spec.fingerprint() or _graph_hash(self.graph, spec) != self.target_sha256:
            raise ValueError("Frozen search witness identity changed")
        if any(value.requires_grad or value.device.type != "cpu" for value in (self.graph.atoms, self.graph.charges, self.graph.bonds)):
            raise ValueError("Search witness must remain detached on CPU")
        count = int((self.graph.atoms != 0).sum())
        if type(self.decisions) is not int or self.decisions != spec.max_nodes + count + count * (count - 1) // 2:
            raise ValueError("Search witness decision-count normalization changed")

    def record(self) -> dict:
        return {"latent_id": self.latent_id, "source_update": self.source_update,
                "target_sha256": self.target_sha256, "raw_graph_sha256": self.raw_graph_sha256,
                "source_probability_sha256": self.source_probability_sha256,
                "weighted_cost": self.weighted_cost, "edit_count": self.edit_count,
                "decisions": self.decisions, "schema_sha256": self.schema_sha256,
                "labels": {name: value[0].tolist() for name, value in
                           (("atoms", self.graph.atoms), ("charges", self.graph.charges), ("bonds", self.graph.bonds))}}


def witness_nll(decoded: DecoderOutput, target: FrozenSearchRepairTarget, spec: GraphSpec) -> Tensor:
    """Stable full q(target) NLL; no conditioning on the target's occupied slots."""
    _validate_decoded(decoded, spec, 1)
    target.verify(spec)
    if decoded.atoms.dtype in (torch.float16, torch.bfloat16):
        # Log-softmax and the full decision sum need float32 range under AMP;
        # this cast remains differentiable and does not floor probabilities.
        decoded = DecoderOutput(*(value.float() for value in (decoded.atoms, decoded.charges, decoded.bonds)))
    graph = target.graph.to(decoded.atoms.device)
    terms = hard_graph_log_prob_terms(decoded, graph, spec, temperature=1.0, fixed_active=None)
    if not bool(terms["supported"].all()):
        raise ValueError("Frozen target is outside the unconditional graph law")
    result = -(terms["atoms"] + terms["charges"] + terms["bonds"]) / target.decisions
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Witness NLL is nonfinite; no epsilon flooring is applied")
    return result[0]


class SearchRepairTeacher:
    """A fixed latent-ID panel with refreshed detached, validated references.

    The caller must decode exactly panel coordinates in their declared order
    when refreshing, and supply the corresponding latent IDs for minibatches.
    No decoder or separate chemical identity is learned inside this teacher.
    Dedicated CPU RNG is recorded but never consumed by argmax-only refreshes.
    """

    def __init__(self, panel: Tensor, spec: GraphSpec, repair_config: ProbabilityRepairConfig, *,
                 generator: torch.Generator):
        if not isinstance(panel, Tensor) or panel.device.type != "cpu" or panel.ndim != 2 or min(panel.shape) < 1 or not panel.is_floating_point() or not bool(torch.isfinite(panel).all()):
            raise ValueError("Teacher panel must be a finite nonempty floating CPU matrix")
        if not isinstance(spec, GraphSpec) or len(spec.atomic_numbers) < 2:
            raise ValueError("Teacher requires a shared GraphSpec with real-atom categories")
        if not isinstance(repair_config, ProbabilityRepairConfig):
            raise ValueError("Teacher requires ProbabilityRepairConfig")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Teacher requires a dedicated CPU torch.Generator")
        self._panel = panel.detach().clone()
        self.spec, self.repair_config, self.generator = spec, repair_config, generator
        self._panel_sha256 = _panel_hash(self._panel)
        self._source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("search_repair_guidance.py", "probability_repair.py", "sampling.py")}
        self._targets: dict[int, FrozenSearchRepairTarget] = {}
        self._previous_occupancy: Tensor | None = None
        self._last_update: int | None = None
        self.last_receipt: dict | None = None
        self._witness_inspection_count = 0

    @property
    def panel(self) -> Tensor:
        return self._panel.clone()

    def record(self) -> dict:
        return {"version": VERSION, "panel_sha256": self._panel_sha256,
                "panel_shape": list(self._panel.shape), "schema_sha256": self.spec.fingerprint(),
                "source_module_sha256": dict(self._source_hashes),
                "repair_config": self.repair_config.policy(),
                "candidate_policy": "Current DecoderOutput.harden native unrestricted argmax; no stochastic draws",
                "target_policy": "Only invalid to independently verified valid weighted witness; already valid skipped",
                "selection_policy": "Best observed weighted witness from fixed bounded search; no global minimum claim",
                "occupancy_policy": "Current raw occupancy fixed only raw-to-witness and until refresh; next refresh may change slots/count",
                "occupancy_anchor_requirement": "Caller applies identical frozen-original occupancy anchors to controls and guidance arms",
                "normalization": "Per target N+K+K(K-1)/2; mean over full batch including unavailable zero",
                "preservation_guarantee_for_already_valid": False,
                "gradient_scope": "Current decoder logits to constant target NLL; no argmax/search/RDKit/selection gradients",
                "rng_draws_per_refresh": 0, "validity_guarantee": False}

    def _verify_panel(self) -> None:
        if _panel_hash(self._panel) != self._panel_sha256:
            raise ValueError("Fixed teacher latent panel was mutated")

    def _validated_target(self, raw: HardGraphBatch, probabilities, witness: dict,
                          latent_id: int, update: int, probability_sha256: str) -> FrozenSearchRepairTarget:
        if not isinstance(witness, dict) or not isinstance(witness.get("labels"), dict):
            raise ValueError("Search witness is missing full slot labels")
        fields = witness["labels"]
        if set(fields) != {"atoms", "charges", "bonds"}:
            raise ValueError("Search witness labels have an unexpected schema")
        # Boolean/floating labels must not be silently coerced into categories.
        flat = lambda value: sum((flat(item) for item in value), []) if isinstance(value, list) else [value]
        if any(type(label) is not int for field in fields.values() for label in flat(field)):
            raise ValueError("Search witness labels require exact integer categories")
        graph = HardGraphBatch(*(torch.tensor(fields[name], dtype=torch.long)[None] for name in ("atoms", "charges", "bonds")))
        n = self.spec.max_nodes
        shapes = ((1, n), (1, n), (1, n, n))
        limits = (len(self.spec.atomic_numbers), len(self.spec.formal_charges), len(self.spec.bond_names))
        if any(tuple(value.shape) != shape or bool(((value < 0) | (value >= limit)).any()) for value, shape, limit in
               zip((graph.atoms, graph.charges, graph.bonds), shapes, limits)):
            raise ValueError("Search witness labels are outside the full graph vocabulary")
        active = raw.atoms != 0
        if not torch.equal(graph.atoms != 0, active):
            raise ValueError("Search witness changed current raw occupied slots")
        pair = active[:, :, None] & active[:, None, :]
        if not torch.equal(graph.bonds, graph.bonds.transpose(1, 2)) or bool(graph.bonds.diagonal(dim1=1, dim2=2).any()) or bool((graph.bonds[~pair] != 0).any()) or bool((graph.charges[~active] != self.spec.formal_charges.index(0)).any()):
            raise ValueError("Search witness violates slot/padding/bond support")
        self._witness_inspection_count += 1
        inspected = inspect_graph(*(value[0] for value in (graph.atoms, graph.charges, graph.bonds)), self.spec)
        if inspected.smiles is None:
            raise ValueError(f"Search witness fails independent chemistry check: {inspected.failure_reason}")
        identity = _graph_hash(graph, self.spec)
        if witness.get("graph_id") != identity or witness.get("smiles") != inspected.smiles:
            raise ValueError("Search witness chemistry/label identity receipt disagrees")
        cost = measure_probability_edits(tuple(value[0] for value in (raw.atoms, raw.charges, raw.bonds)),
                                         tuple(value[0] for value in (graph.atoms, graph.charges, graph.bonds)),
                                         probabilities, self.spec, self.repair_config)
        if type(witness.get("edit_count")) is not int or witness["edit_count"] != cost["edit_count"] or cost["edit_count"] < 1:
            raise ValueError("Invalid-to-valid witness edit count is inconsistent")
        weighted = witness.get("weighted_cost")
        if isinstance(weighted, bool) or not isinstance(weighted, (int, float)) or not math.isfinite(weighted) or not math.isclose(weighted, cost["weighted_cost"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Search witness probability-weighted edit cost is inconsistent")
        count = int(active.sum())
        return FrozenSearchRepairTarget(latent_id, update, graph, identity, _graph_hash(raw, self.spec),
                                        probability_sha256, float(weighted), cost["edit_count"],
                                        n + count + count * (count - 1) // 2, self.spec.fingerprint())

    @torch.no_grad()
    def refresh(self, decoded: DecoderOutput, *, update: int, directory: Path | None = None) -> dict:
        """Search every declared argmax; failed refreshes never retain old targets."""
        if type(update) is not int or update < 0 or (self._last_update is not None and update <= self._last_update):
            raise ValueError("Teacher refresh update must increase monotonically")
        self._targets.clear()
        self._witness_inspection_count = 0
        self._verify_panel()
        _validate_decoded(decoded, self.spec, len(self._panel))
        cpu = DecoderOutput(*(value.detach().cpu() for value in (decoded.atoms, decoded.charges, decoded.bonds)))
        raw_graph = cpu.harden(self.spec)
        atoms, charges, bonds = raw_graph.atoms, raw_graph.charges, raw_graph.bonds
        active = atoms != 0
        probabilities = (cpu.atoms.softmax(-1), cpu.charges.softmax(-1), cpu.symmetric_bond_logits.softmax(-1))
        rng_before = self.generator.get_state().clone()
        attempts = []
        for latent_id in range(len(self._panel)):
            raw = HardGraphBatch(*(value[latent_id:latent_id+1].clone() for value in (atoms, charges, bonds)))
            observed = inspect_graph(*(value[0] for value in (raw.atoms, raw.charges, raw.bonds)), self.spec)
            attempt = {"latent_id": latent_id, "raw_graph_sha256": _graph_hash(raw, self.spec),
                       "raw_occupied_slots": torch.where(active[latent_id])[0].tolist(),
                       "raw_inspection": {"smiles": observed.smiles, "failure_reason": observed.failure_reason, "detail": observed.detail},
                       "raw_labels": {name: value[0].tolist() for name, value in
                                      (("atoms", raw.atoms), ("charges", raw.charges), ("bonds", raw.bonds))}}
            try:
                p = tuple(value[latent_id] for value in probabilities)
                search = search_probability_repair(tuple(value[0] for value in (raw.atoms, raw.charges, raw.bonds)), p, self.spec, self.repair_config)
                attempt["search"] = search
                if search.get("raw_graph_id") != attempt["raw_graph_sha256"] or search.get("probability_sha256") != probability_identity(p, self.spec) or search.get("policy") != self.repair_config.policy():
                    raise ValueError("Search candidate/probability/budget provenance disagrees")
                if observed.smiles is not None:
                    if search.get("status") != "already_valid" or not search.get("raw_valid"):
                        raise ValueError("Search disagrees with independent raw-valid inspection")
                    attempt["guidance_status"] = "skipped_already_valid"
                elif search.get("status") == "witness_found":
                    if search.get("raw_valid") or search.get("raw_graph_id") != attempt["raw_graph_sha256"]:
                        raise ValueError("Search raw candidate provenance disagrees")
                    target = self._validated_target(raw, p, search.get("weighted_witness"), latent_id, update, search["probability_sha256"])
                    self._targets[latent_id] = target
                    attempt.update(guidance_status="accepted", target=target.record())
                elif search.get("status") in ("censored", "unsupported"):
                    attempt["guidance_status"] = search["status"]
                else:
                    raise ValueError("Search returned an unexpected status")
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
                self._targets.pop(latent_id, None)
                attempt.update(guidance_status="rejected_or_error", error_type=type(error).__name__, error=str(error))
            attempts.append(attempt)
        if not torch.equal(rng_before, self.generator.get_state()):
            self._targets.clear()
            raise RuntimeError("Argmax search unexpectedly consumed the dedicated sampling RNG")
        receipt = {**self.record(), "update": update, "status": "refreshed",
                   "raw_occupancy_sha256": _panel_hash(active),
                   "raw_atom_counts": active.sum(-1).tolist(),
                   "changed_occupancy_latent_ids": torch.where((active != self._previous_occupancy).any(-1))[0].tolist()
                       if self._previous_occupancy is not None else None,
                   "atom_count_changes_from_previous_refresh": (active.sum(-1) - self._previous_occupancy.sum(-1)).tolist()
                       if self._previous_occupancy is not None else None,
                   "attempted_candidates": len(attempts), "accepted_targets": len(self._targets),
                   "guidance_status_counts": {status: sum(attempt["guidance_status"] == status for attempt in attempts)
                                              for status in sorted({attempt["guidance_status"] for attempt in attempts})},
                   "search_checks": sum(attempt.get("search", {}).get("checks", 0) for attempt in attempts),
                   "search_checks_scope": "Existing bounded search's candidate-diagnosis checks only; excludes its initial raw diagnosis and teacher inspections",
                   "explicit_raw_inspection_calls": len(attempts),
                   "explicit_witness_inspection_calls": self._witness_inspection_count,
                   "completed_search_calls": sum("search" in attempt for attempt in attempts),
                   "cost_accounting_scope": "Retained wrapper/search counters; not a count of every internal RDKit operation",
                   "search_capped_depths": sum(depth.get("capped", False) for attempt in attempts for depth in attempt.get("search", {}).get("checks_by_depth", [])),
                   "fixed_candidate_quota_vs_actual_cost": "Candidate quota equal; physical RDKit search checks may differ by outcome",
                   "cleared_old_targets": True, "rng_consumed": False, "attempts": attempts}
        receipt["receipt_sha256"] = _json_hash(receipt)
        if directory is not None:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"search-refresh-{update:08d}.json").open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self._last_update = update
        self._previous_occupancy = active.clone()
        self.last_receipt = json.loads(json.dumps(receipt, allow_nan=False))
        return json.loads(json.dumps(receipt, allow_nan=False))

    def objective(self, decoded: DecoderOutput, *, latent_ids: Sequence[int] | Tensor | None = None) -> dict:
        """Apply held references to the corresponding current decoder minibatch."""
        self._verify_panel()
        batch = decoded.atoms.shape[0]
        _validate_decoded(decoded, self.spec, batch)
        if latent_ids is None:
            if batch != len(self._panel):
                raise ValueError("Minibatches require explicit teacher latent IDs")
            indices = list(range(batch))
        elif isinstance(latent_ids, Tensor):
            if latent_ids.ndim != 1 or latent_ids.dtype != torch.long:
                raise ValueError("Teacher latent IDs must be long [batch]")
            indices = latent_ids.detach().cpu().tolist()
        else:
            indices = list(latent_ids)
        if len(indices) != batch or any(type(index) is not int or not 0 <= index < len(self._panel) for index in indices):
            raise ValueError("Teacher latent IDs disagree with the current batch/panel")
        values, available, targets = [], [], []
        for offset, latent_id in enumerate(indices):
            one = DecoderOutput(*(value[offset:offset+1] for value in (decoded.atoms, decoded.charges, decoded.bonds)))
            target = self._targets.get(latent_id)
            if target is None:
                # A single finite element per field avoids overflow in a large
                # low-precision sum while attaching exact zero gradients to all
                # decoder fields for an unavailable target.
                value = sum(field.reshape(-1)[0] * 0 for field in (one.atoms, one.charges, one.bonds))
                available.append(False)
                targets.append(None)
            else:
                value = witness_nll(one, target, self.spec)
                available.append(True)
                targets.append(target.target_sha256)
            values.append(value)
        per_graph = torch.stack(values)
        mask = torch.tensor(available, dtype=torch.bool, device=decoded.atoms.device)
        nll = per_graph.mean()
        return {"total": nll, "nll": nll, "per_graph": per_graph, "available_mask": mask,
                "attempted": batch, "eligible": int(mask.sum()),
                "eligible_only_nll": per_graph[mask].mean() if bool(mask.any()) else None,
                "status": "available" if bool(mask.any()) else "unavailable_no_valid_witness",
                "latent_ids": indices, "target_sha256": targets,
                "scope": {**self.record(), "source_refresh_update": self._last_update,
                          "current_target_fixed_during_objective": True, "chemistry_checks_in_objective": False}}
