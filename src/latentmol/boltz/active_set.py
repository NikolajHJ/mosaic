"""Piecewise fixed-context Boltz optimization with explicit initialization.

Within a step the ligand topology has a fixed set of decoder slots, and the
frozen Boltz trunk receives the same conditional soft atom/charge/bond features
as the established fixed-count adapter. Between steps, raw decoder NO_ATOM
probabilities select a new mask. A different count requires a separately native
prepared complex and pose; a mask of the same size only relabels graph slots.
The selector is discrete, so gradients describe only the current fixed context.
Encoded-chain pilots have HPC evidence in the ledger. The source056 unfiltered
prior-start extension is new and still requires HPC fixture and model validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re

import torch
import yaml

from latentmol.chemistry import inspect_graph
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from latentmol.graph_policy import graph_from_decoder
from latentmol.initialization import LatentStart, sample_prior
from latentmol.optimization import NonfiniteObjective, StepControl, backtracking_step
from latentmol.multimodel.config import Diagnostics, Objective, Optimizer, identifier, number
from latentmol.multimodel.engine import Counters, scalar_loss
from latentmol.multimodel.study import hard_digest, module_digest, tensor_digest, verified_file
from latentmol.representation import HardGraphBatch
from latentmol.sampling import sample_hard_graph
from .contracts import PreparedComplex
from .model import FrozenBoltzAffinity, load_affinity_model
from .provenance import context_digest
from .scoring import BoltzGraphBranch, boltz_branch_manifest
from .element_diagnostics import element_record


@dataclass(frozen=True)
class ActiveSetConfig:
    """One declared count bank, start, objective, and matched arm budget."""

    name: str
    mode: str
    decoder: dict
    affinity_checkpoint: dict
    context_manifest: dict | None
    contexts: dict[int, dict]
    anchor_count: int
    output_root: Path
    device: str
    temperature: float
    threshold: float
    recycling_steps: int
    objective: Objective
    optimizer: Optimizer
    diagnostics: Diagnostics
    allow_lossy_charge: bool
    context_identity: dict | None = None
    start: LatentStart = field(default_factory=LatentStart)
    arms: tuple[str, ...] = ("active_set", "fixed_initial_mask")
    element_diagnostics: bool = False
    # Absent by default and omitted from legacy records, preserving historical
    # configuration hashes. New runs must explicitly opt into this policy.
    fd_policy: str | None = None
    step_control: StepControl | None = None
    # Campaign controls are explicit opt-ins; old records retain their identity.
    initial_latent: dict | None = None
    gradient_audit_interval: int | None = None
    # New opt-in profile. Omitted from legacy config records and fingerprints.
    dbmol: object | None = None
    # Fixed-target chemistry guidance is a separate, opt-in loss on this graph.
    repair_guidance: object | None = None
    # Operational ceilings are opt-in and absent from historical identities.
    execution_budget: object | None = None

    @classmethod
    def load(cls, path: Path) -> "ActiveSetConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        allowed = {"version", "name", "mode", "decoder", "affinity_checkpoint",
                   "context_manifest", "contexts", "anchor_count", "output_root",
                   "device", "temperature", "threshold", "recycling_steps",
                   "objective", "optimizer", "diagnostics", "allow_lossy_charge", "start",
                   "arms", "element_diagnostics", "fd_policy", "step_control",
                   "initial_latent", "gradient_audit_interval", "dbmol", "repair_guidance", "execution_budget"}
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError("Active-set config must be a mapping with only supported keys")
        if type(raw.get("version")) is not int or raw["version"] != 1:
            raise ValueError("Unknown active-set config version")
        identifier(raw["name"])
        arms = raw.get("arms", ["active_set", "fixed_initial_mask"])
        if (not isinstance(arms, list) or not arms or len(set(arms)) != len(arms)
                or any(a not in ("active_set", "fixed_initial_mask") for a in arms)):
            raise ValueError("arms must list distinct supported optimization arms")
        if type(raw.get("element_diagnostics", False)) is not bool:
            raise ValueError("element_diagnostics must be boolean")
        if raw.get("fd_policy") not in (None, "local_v1", "branch_local_v2", "additive_branch_v3", "pointwise_native_ad_v4"):
            raise ValueError("fd_policy must be absent or an explicit supported derivative contract")
        step_control = (StepControl.from_mapping(raw["step_control"])
                        if raw.get("step_control") is not None else None)
        audit_interval = raw.get("gradient_audit_interval")
        if audit_interval is not None and (type(audit_interval) is not int or audit_interval < 1):
            raise ValueError("gradient_audit_interval must be a positive integer")
        if raw.get("mode") not in ("ae", "vae"):
            raise ValueError("Declare checkpoint mode as ae or vae")
        start_raw = raw.get("start", {})
        if not isinstance(start_raw, dict) or set(start_raw) - {"kind", "seed"}:
            raise ValueError("start requires only a supported kind and its optional seed")
        start = LatentStart(**start_raw)
        if start.kind == "standard_normal" and raw["mode"] != "vae":
            raise ValueError("Standard-normal prior study requires a VAE checkpoint")
        if ("contexts" in raw) == ("context_manifest" in raw):
            raise ValueError("Declare exactly one of contexts and context_manifest")
        if type(raw["anchor_count"]) is not int or raw["anchor_count"] < 1:
            raise ValueError("anchor_count must be a positive integer")
        number(raw.get("temperature", 1.0), "temperature", positive=True)
        number(raw.get("threshold", 0.5), "threshold", positive=True)
        if raw.get("threshold", 0.5) >= 1:
            raise ValueError("Selection threshold must be strictly between zero and one")
        if type(raw.get("recycling_steps", 0)) is not int or raw.get("recycling_steps", 0) < 0:
            raise ValueError("recycling_steps must be a nonnegative integer")
        if type(raw.get("allow_lossy_charge", True)) is not bool:
            raise ValueError("allow_lossy_charge must be a boolean")
        if not isinstance(raw.get("device", "cuda"), str) or not re.fullmatch(
                r"cuda(?::[0-9]+)?", raw.get("device", "cuda")):
            raise ValueError("Native active-set jobs require cuda or cuda:index")
        base = path.resolve().parent
        execution_budget = None
        if raw.get("execution_budget") is not None:
            from latentmol.execution_budget import ExecutionBudgetConfig
            execution_budget = ExecutionBudgetConfig.from_mapping(raw["execution_budget"])
        repair_guidance = None
        if raw.get("repair_guidance") is not None:
            from latentmol.repair_guidance.config import RepairGuidanceConfig
            repair_guidance = RepairGuidanceConfig.from_mapping(raw["repair_guidance"], base)
            if arms != ["fixed_initial_mask"] or raw.get("fd_policy") not in ("local_v1", "branch_local_v2", "additive_branch_v3", "pointwise_native_ad_v4") or step_control is None:
                raise ValueError("Repair guidance requires only fixed_initial_mask, local_v1 and bounded step_control")
        dbmol = None
        if raw.get("dbmol") is not None:
            from .dbmol_objective import DBMolObjectiveConfig
            dbmol = DBMolObjectiveConfig.from_mapping(raw["dbmol"], base)
        resolve = lambda value: str((base / value).resolve())
        def asset(value):
            if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
                raise ValueError("Each asset requires exactly path and SHA256")
            if (not isinstance(value["path"], str) or not value["path"]
                    or not isinstance(value["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
                raise ValueError("Each asset requires a nonempty path and a 64-character SHA256")
            return value
        asset(raw["decoder"])
        asset(raw["affinity_checkpoint"])
        initial_latent = None
        if raw.get("initial_latent") is not None:
            asset(raw["initial_latent"])
            initial_latent = {**raw["initial_latent"], "path": resolve(raw["initial_latent"]["path"])}
        manifest_record = None
        context_identity = None
        if "context_manifest" in raw:
            manifest_asset = asset(raw["context_manifest"])
            manifest_path = verified_file(resolve(manifest_asset["path"]),
                                          manifest_asset["sha256"])
            manifest_record = {"path": str(manifest_path),
                               "sha256": manifest_asset["sha256"]}
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("version") != 1:
                raise ValueError("Unknown count-context manifest version")
            entries = manifest["contexts"]
            context_base = manifest_path.parent
            context_identity = {key: manifest.get(key) for key in
                                ("target_sequence_sha256", "msa_sha256", "fixed_msa_row_indices")}
        else:
            entries = raw["contexts"]
            context_base = base
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Contexts must be a nonempty count-to-asset mapping")
        contexts = {}
        for key, value in entries.items():
            if not ((type(key) is int and key > 0) or
                    (isinstance(key, str) and re.fullmatch(r"[1-9][0-9]*", key))):
                raise ValueError("Context keys must be positive integer counts")
            if int(key) in contexts:
                raise ValueError("Duplicate context count")
            asset(value)
            contexts[int(key)] = {"path": str((context_base / value["path"]).resolve()),
                                  "sha256": value["sha256"]}
        if raw["anchor_count"] not in contexts:
            raise ValueError("Anchor count must have a native prepared context")
        objective = Objective(**raw.get("objective", {"output": "affinity_probability_binary",
                                                     "direction": "maximize"}))
        if objective.weight != 1.0:
            raise ValueError("Single Boltz objective uses weight 1; adjust scale explicitly")
        if objective.output not in ("affinity_probability_binary", "affinity_pred_value", "dbmol_total_loss"):
            raise ValueError("Unknown native Boltz affinity output")
        if (objective.output == "dbmol_total_loss") != (dbmol is not None):
            raise ValueError("dbmol_total_loss and explicit DBMol head profile must be selected together")
        if dbmol is not None and (objective.direction != "minimize" or objective.scale != 1 or objective.offset != 0):
            raise ValueError("DBMol total uses explicit internal weights, minimize, scale one and offset zero")
        if dbmol is not None and raw.get("fd_policy") not in ("local_v1", "branch_local_v2", "additive_branch_v3", "pointwise_native_ad_v4"):
            raise ValueError("DBMol requires the local_v1 per-component/total derivative gate")
        if raw.get("fd_policy") in ("branch_local_v2", "additive_branch_v3", "pointwise_native_ad_v4"):
            from .dbmol_objective import CONTACT_HEAD64_ARITHMETIC
            if (dbmol is None or dbmol.objective_arithmetic != CONTACT_HEAD64_ARITHMETIC
                    or raw.get("recycling_steps", 0) != 0 or arms != ["fixed_initial_mask"]
                    or step_control is None):
                raise ValueError(raw["fd_policy"] + " requires fixed-mask, bounded, zero-recycle contact-head DBMol")
        d = dict(raw.get("diagnostics", {}))
        if "fd_epsilons" in d:
            d["fd_epsilons"] = tuple(d["fd_epsilons"])
        return cls(name=raw["name"], mode=raw["mode"],
                   decoder={**raw["decoder"], "path": resolve(raw["decoder"]["path"])},
                   affinity_checkpoint={**raw["affinity_checkpoint"],
                                        "path": resolve(raw["affinity_checkpoint"]["path"])},
                   context_manifest=manifest_record,
                   contexts=contexts, anchor_count=raw["anchor_count"],
                   output_root=Path(resolve(raw["output_root"])),
                   device=raw.get("device", "cuda"), temperature=raw.get("temperature", 1.0),
                   threshold=raw.get("threshold", 0.5),
                   recycling_steps=raw.get("recycling_steps", 0), objective=objective,
                   optimizer=Optimizer(**raw.get("optimizer", {})),
                   diagnostics=Diagnostics(**d),
                   allow_lossy_charge=raw.get("allow_lossy_charge", True),
                   context_identity=context_identity, start=start, arms=tuple(arms),
                   element_diagnostics=raw.get("element_diagnostics", False),
                   fd_policy=raw.get("fd_policy"), step_control=step_control,
                   initial_latent=initial_latent, gradient_audit_interval=audit_interval, dbmol=dbmol,
                   repair_guidance=repair_guidance, execution_budget=execution_budget)

    def record(self) -> dict:
        from dataclasses import asdict
        result = asdict(self)
        result["output_root"] = str(self.output_root)
        if self.fd_policy is None:
            result.pop("fd_policy")
        if self.step_control is None:
            result.pop("step_control")
        if self.initial_latent is None:
            result.pop("initial_latent")
        if self.gradient_audit_interval is None:
            result.pop("gradient_audit_interval")
        if self.dbmol is None:
            result.pop("dbmol")
        else:
            result["dbmol"] = self.dbmol.record()
        if self.repair_guidance is None:
            result.pop("repair_guidance")
        else:
            result["repair_guidance"] = self.repair_guidance.record()
        if self.execution_budget is None:
            result.pop("execution_budget")
        return result


def load_initial_latent(asset: dict, latent_dimension: int, device: str) -> torch.Tensor:
    """Reuse exact saved z across policies; never re-encode, perturb or redraw it."""
    payload = torch.load(verified_file(asset["path"], asset["sha256"]),
                         map_location="cpu", weights_only=True)
    point = payload.get("z") if isinstance(payload, dict) else None
    if (not isinstance(point, torch.Tensor) or point.shape != (1, latent_dimension)
            or not point.is_floating_point() or not bool(point.isfinite().all())):
        raise ValueError("Saved initialization requires a finite floating [1, latent] z")
    return point.detach().to(device)


@dataclass
class Selection:
    """Detached between-step decision; not a differentiable occupancy feature."""

    active: torch.Tensor
    no_atom: torch.Tensor
    threshold: float

    @property
    def count(self) -> int:
        return int(self.active.sum())

    def record(self) -> dict:
        return {"active_slots": self.active.nonzero().flatten().tolist(),
                "count": self.count, "no_atom_probabilities": self.no_atom.tolist(),
                "minimum_threshold_margin": float((self.no_atom - self.threshold).abs().min()),
                "threshold": self.threshold}


@torch.no_grad()
def select_slots(decoded, *, temperature: float, threshold: float,
                 require_nonempty: bool = True) -> Selection:
    """Choose all slots with P(NO_ATOM) below threshold; ties are absent."""
    if decoded.atoms.ndim != 3 or decoded.atoms.shape[0] != 1:
        raise ValueError("Active-set Boltz accepts one decoded molecule")
    if not 0 < threshold < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid threshold or temperature")
    no_atom = (decoded.atoms.detach() / temperature).softmax(-1)[0, :, 0]
    if not bool(no_atom.isfinite().all()):
        raise ValueError("Nonfinite occupancy probabilities")
    active = no_atom < threshold
    if require_nonempty and not bool(active.any()):
        raise ValueError("Selector chose an empty ligand; no native context exists")
    return Selection(active, no_atom, threshold)


@torch.no_grad()
def audit_start(decoded, seed, spec, *, temperature, threshold):
    """Report the exact carbon-anchor eligibility before any native model call."""
    selection = select_slots(decoded, temperature=temperature, threshold=threshold,
                             require_nonempty=False)
    hard = decoded.harden(spec)
    fields = {key: bool(torch.equal(getattr(hard, key).cpu(), getattr(seed, key).cpu()))
              for key in ("atoms", "charges", "bonds")}
    same_mask = torch.equal(selection.active.cpu(), (seed.atoms[0] != 0).cpu())
    return selection, {"passed": bool(same_mask and all(fields.values())),
                       "selected_mask_matches_seed": bool(same_mask),
                       "ordinary_hard_fields_equal": fields,
                       "selection": selection.record(), "seed_sha256": hard_digest(seed),
                       "ordinary_hard_sha256": hard_digest(hard)}


@torch.no_grad()
def audit_random_start(decoded, spec, *, temperature, threshold, available_counts):
    """Describe prior chemistry without requiring validity or changing the draw."""
    selection = select_slots(decoded, temperature=temperature, threshold=threshold,
                             require_nonempty=False)
    chemistry = {"smiles": None, "failure": "empty_selection", "graph_sha256": None}
    if selection.count:
        graph = graph_from_decoder(decoded, spec, temperature=temperature,
                                   fixed_active=selection.active)
        hard = HardGraphBatch(graph.atoms.argmax(-1), graph.charges.argmax(-1),
                             graph.bonds.argmax(-1)).to("cpu")
        inspected = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
        chemistry = {"smiles": inspected.smiles, "failure": inspected.failure_reason,
                     "graph_sha256": hard_digest(hard),
                     "labels": {key: getattr(hard, key)[0].tolist()
                                for key in ("atoms", "charges", "bonds")}}
    return selection, {
        "passed": selection.count in available_counts and selection.count > 0,
        "selection": selection.record(), "available_counts": sorted(available_counts),
        "initial_hard_graph": chemistry,
        "scope": "Unfiltered prior draw; eligibility checks occupancy/context, not chemical validity",
        "carbon_template_pose_bias_remains": True,
    }


def bind_slots(template: PreparedComplex, active: torch.Tensor) -> PreparedComplex:
    """Relabel graph slots, preserving every native tensor and native atom row.

    Native rows are assigned in template slot order to selected slot order. This
    does not move atoms in the native pose or change their native chemistry.
    Validation proves the remapped carbon seed still reproduces native features.
    """
    if active.dtype != torch.bool or active.shape != (template.spec.max_nodes,):
        raise ValueError("Selected slot mask disagrees with GraphSpec")
    source = template.active.nonzero().flatten()
    target = active.nonzero().flatten().to(source.device)
    if source.numel() != target.numel():
        raise ValueError("A count change requires a different native template")
    atom_map = torch.full_like(template.slot_to_atom, -1)
    token_map = torch.full_like(template.slot_to_token, -1)
    atom_map[target], token_map[target] = template.slot_to_atom[source], template.slot_to_token[source]
    atoms = torch.zeros_like(template.seed.atoms)
    charges = torch.full_like(template.seed.charges, template.spec.formal_charges.index(0))
    bonds = torch.zeros_like(template.seed.bonds)
    atoms[:, target], charges[:, target] = template.seed.atoms[:, source], template.seed.charges[:, source]
    bonds[:, target[:, None], target[None, :]] = template.seed.bonds[:, source[:, None], source[None, :]]
    metadata = {**template.metadata, "bound_active_slots": target.tolist(),
                "template_context_sha256": context_digest(template)}
    if "slot_map" in template.metadata:
        relabel = dict(zip(source.tolist(), target.tolist()))
        metadata["template_slot_map"] = template.metadata["slot_map"]
        metadata["slot_map"] = [{**row, "slot": relabel[row["slot"]]}
                                for row in template.metadata["slot_map"]]
    bound = PreparedComplex(template.features, template.coordinates, atom_map, token_map,
                            HardGraphBatch(atoms, charges, bonds), template.spec,
                            metadata)
    bound.validate()
    return bound


class ContextBank:
    """Verified count-specific native features; only one GPU context is resident."""

    def __init__(self, config: ActiveSetConfig, spec):
        self.templates = {}
        target_identity = None
        for count, item in sorted(config.contexts.items()):
            path = verified_file(item["path"], item["sha256"])
            template = PreparedComplex.load(path)
            if template.spec != spec or int(template.active.sum()) != count:
                raise ValueError(f"Count {count}: prepared graph schema or count differs")
            if template.metadata.get("seed_smiles") != "C" * count:
                raise ValueError(f"Count {count}: expected an exact native linear carbon seed")
            identity = (template.metadata.get("target_sequence_sha256"),
                        template.metadata.get("msa_sha256"),
                        tuple(template.metadata.get("fixed_msa_row_indices", ())),
                        template.metadata.get("boltz_commit"))
            if not all(identity[:2]) or not identity[2]:
                raise ValueError(f"Count {count}: missing target/MSA preparation identity")
            if target_identity is not None and identity != target_identity:
                raise ValueError("Prepared counts do not share the exact target/MSA policy")
            target_identity = identity
            self.templates[count] = template
        self.target_identity = target_identity
        if config.context_identity is not None:
            actual = {"target_sequence_sha256": target_identity[0],
                      "msa_sha256": target_identity[1],
                      "fixed_msa_row_indices": list(target_identity[2])}
            if config.context_identity != actual:
                raise ValueError("Context catalog target/MSA header differs from its prepared snapshots")

    def bind(self, selection: Selection, device) -> PreparedComplex:
        if selection.count not in self.templates:
            raise ValueError(f"No native prepared pose for selected count {selection.count}")
        return bind_slots(self.templates[selection.count], selection.active.cpu()).to(device)


def make_branch(scorer, bank: ContextBank, selection: Selection, config: ActiveSetConfig,
                source_sha256: str) -> BoltzGraphBranch:
    """Switch context without reloading or updating Boltz weights."""
    prepared = bank.bind(selection, config.device)
    manifest = boltz_branch_manifest(prepared, adapter_source_sha256=source_sha256,
                                     recycling_steps=config.recycling_steps,
                                     dbmol_profile=getattr(scorer, "manifest_config", None))
    branch = BoltzGraphBranch(scorer, prepared, manifest,
                              allow_lossy_charge=config.allow_lossy_charge)
    branch.verify_context_unchanged()
    return branch


def effective_repair_weight(guidance, repair_weight=None):
    """Require scheduled lambda to be fixed explicitly outside all state probes."""
    if guidance is None:
        if repair_weight is not None:
            raise ValueError("A repair weight requires configured repair guidance")
        return None
    if repair_weight is None:
        if guidance.weight_schedule is not None:
            raise ValueError("Scheduled repair guidance requires an explicit fixed state weight")
        return guidance.weight
    if (isinstance(repair_weight, bool) or not isinstance(repair_weight, (int, float))
            or not math.isfinite(repair_weight) or repair_weight < 0):
        raise ValueError("Effective repair weight must be finite and nonnegative")
    return repair_weight


def repair_application(weight, target, *, gradient_measured=False):
    """Observation only: distinguish scheduled lambda from an included target.

    A configured positive coefficient contributes nothing when refresh produced
    no target. Availability records presence; ``used_target`` additionally
    requires a positive coefficient. It describes inclusion in the current
    objective, not evidence that a forward/oracle/backward has run: preliminary
    snapshots also carry this metadata. ``gradient_measured`` refers to the
    repair derivative, not the binding/combined backward; an absent or no-grad
    repair is not a measured zero gradient. Existing ``weight`` and
    ``effective_repair_weight`` retain their scheduled-coefficient meaning.
    """
    available = target is not None
    active = available and weight > 0
    result = {"scheduled_weight": weight, "target_available": available,
              "used_target": active, "applied_weight": weight if available else 0.,
              "guidance_active": active,
              "gradient_measured": active and bool(gradient_measured)}
    if not active:
        result["weighted_loss"] = 0.
    return result


def score_at(decoder, point, branch, config, counters, purpose, *, gradients,
             reject_nonfinite_objective=False, component_gradients=None, repair_target=None,
             repair_weight=None):
    """Differentiate only the conditional graph for this one selected context."""
    guidance = getattr(config, "repair_guidance", None)
    weight = effective_repair_weight(guidance, repair_weight)
    if repair_target is not None and (guidance is None or weight <= 0):
        raise ValueError("An accepted target requires a positive effective guidance weight")
    probe = point.detach().clone().requires_grad_(gradients)
    with torch.set_grad_enabled(gradients):
        decoded = decoder.decode(probe)
        graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                                   fixed_active=branch.prepared.active)
        output = counters.call("boltz", purpose, "forward", lambda: branch(graph), probe.device)
        if reject_nonfinite_objective:
            # Classification only, after the unchanged native forward. Malformed
            # heads still fail through scalar_loss; only a real scalar tensor
            # containing a nonfinite value is a recoverable line-search rejection.
            value = output.get(config.objective.output)
            if isinstance(value, torch.Tensor) and value.numel() == 1 and not bool(value.isfinite().all()):
                raise NonfiniteObjective("Native scalar trial objective is nonfinite")
        loss, row = scalar_loss(output, config.objective)
        binding_loss = loss
        if config.dbmol is not None:
            row["loss_profile"] = config.dbmol.profile
            row["components"] = {k: float(v.detach()) for k, v in output.items()}
            row["physical_trunks_per_forward"] = 2
            row["contact_selection"] = getattr(branch.scorer, "last_contact_selection", {})
        if component_gradients is not None:
            from .dbmol_objective import COMPONENT_OUTPUTS
            for key in COMPONENT_OUTPUTS:
                term = output[key]
                if not term.requires_grad:
                    raise ValueError(f"Detached DBMol component {key}")
                derivative, = counters.call("boltz", purpose + ":" + key, "backward", lambda:
                    torch.autograd.grad(term, probe, allow_unused=True, retain_graph=True), probe.device)
                if derivative is None or not bool(derivative.isfinite().all()):
                    raise ValueError(f"Missing/nonfinite DBMol component gradient {key}")
                component_gradients[key] = derivative.detach()
        if guidance is not None:
            row["binding_loss"] = float(binding_loss.detach())
            row["repair_guidance"] = {"status": "disabled_weight_zero" if weight == 0
                                       else "no_accepted_target", "weight": weight,
                                      **repair_application(weight, repair_target)}
        if repair_target is not None:
            from latentmol.repair_guidance.loss import fixed_target_nll
            repair_target.verify(branch.prepared.active)
            repair_loss, detail = fixed_target_nll(decoded, graph, repair_target, decoder.spec,
                temperature=config.temperature, normalization=guidance.normalization)
            loss = binding_loss + weight * repair_loss
            if not bool(loss.isfinite().all()):
                raise NonfiniteObjective("Nonfinite combined binding/repair objective")
            row["repair_guidance"] = {**detail, "status": "active", "weight": weight,
                                       **repair_application(weight, repair_target),
                                       "weighted_loss": float((weight * repair_loss).detach())}
            if gradients:
                # Diagnostic derivatives retain the same decoder computation.
                # Their scale/cosine is reported; no per-term normalization is applied.
                binding_gradient, = counters.call("boltz", purpose + ":binding", "backward", lambda:
                    torch.autograd.grad(binding_loss, probe, retain_graph=True, allow_unused=True), probe.device)
                repair_gradient, = counters.call("repair", purpose, "backward", lambda:
                    torch.autograd.grad(repair_loss, probe, retain_graph=True, allow_unused=True), probe.device)
                if any(g is None or not bool(g.isfinite().all()) for g in (binding_gradient, repair_gradient)):
                    raise ValueError("Missing/nonfinite binding or repair component gradient")
                a, b = binding_gradient.detach().double(), repair_gradient.detach().double()
                denominator = float(a.norm() * b.norm())
                row["repair_guidance"]["gradient_diagnostics"] = {
                    "binding_norm": float(a.norm()), "repair_norm": float(b.norm()),
                    "weighted_repair_norm": weight * float(b.norm()),
                    "binding_repair_cosine": float((a * b).sum()) / denominator if denominator > 0 else None}
                row["repair_guidance"]["gradient_measured"] = True
        gradient = None
        if gradients:
            if not loss.requires_grad:
                raise ValueError("Boltz objective detached from the molecular latent")
            # Optional diagnostics use the same backward pass, without changing
            # the scalar objective or adding detached conversions to the branch.
            inputs = (probe, decoded.atoms, graph.atoms) if config.element_diagnostics else (probe,)
            derivatives = counters.call("boltz", purpose, "backward", lambda:
                torch.autograd.grad(loss, inputs, allow_unused=True), probe.device)
            gradient = derivatives[0]
            if gradient is None or not bool(gradient.isfinite().all()):
                raise ValueError("Missing or nonfinite Boltz latent gradient")
            gradient = gradient.detach()
        if config.element_diagnostics:
            row["elements"] = element_record(decoded, graph, branch.prepared.active,
                                              decoder.spec, config.temperature,
                                              derivatives[1:] if gradients else None)
            if repair_target is not None:
                row["elements"]["gradient_scope_by_field"] = {
                    "decoded_atom_logits": "combined binding and weighted fixed-target NLL",
                    "shared_graph_atom_probabilities": "binding only; repair differentiates stable log-softmax of the same logits"}
        row.update(loss=float(loss.detach()),
                   gradient_norm=float(gradient.double().norm()) if gradient is not None else None,
                   context_sha256=branch.manifest.context_sha256)
        if config.fd_policy is not None or config.step_control is not None:
            # Metadata only: preserve the actual scalar precision before JSON
            # widening. Legacy score rows and scientific arithmetic are unchanged.
            row["loss_dtype"] = str(loss.dtype)
        return row, gradient


def validate_branch(decoder, point, branch, config, counters, *, report_path=None, repair_target=None,
                    repair_weight=None) -> dict:
    """Persist partial evidence even if native parity or a derivative gate fails."""
    report = {"passed": False, "status": "running", "native_seed_parity": False,
              "parity_heads": {}, "finite_differences": [],
              "context_sha256": branch.manifest.context_sha256,
              "active_slots": branch.prepared.active.nonzero().flatten().tolist(),
              "point_sha256": tensor_digest({"z": point}),
              "scope": "fixed selected mask and fixed native pose; no derivative through selector"}
    def persist():
        if report_path is not None:
            write_json(Path(report_path), report)
    persist()
    try:
        _validate_branch(decoder, point, branch, config, counters, report, persist, repair_target, repair_weight)
        report.update(passed=True, status="passed")
    except Exception as exc:
        report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        persist()
        raise
    persist()
    return report


def _validate_branch(decoder, point, branch, config, counters, report, persist, repair_target=None,
                     repair_weight=None):
    """Finite differences deliberately hold the selected mask and context fixed."""
    weight = effective_repair_weight(getattr(config, "repair_guidance", None), repair_weight)
    if getattr(config, "repair_guidance", None) is not None:
        report["repair_application"] = repair_application(weight, repair_target)
        persist()
    prepared = branch.prepared
    with torch.no_grad():
        native = counters.call("boltz", "seed_parity", "native_forward", lambda:
            branch.scorer.discrete_reference(prepared.features, prepared.coordinates), point.device)
        adapted = counters.call("boltz", "seed_parity", "forward", lambda:
            branch(prepared.seed.to_relaxed(prepared.spec)), point.device)
    if set(native) != set(adapted):
        raise ValueError("Native and adapted Boltz heads disagree")
    for key in native:
        report["parity_heads"][key] = {"passed": False}
        torch.testing.assert_close(adapted[key], native[key],
                                   rtol=config.diagnostics.parity_rtol,
                                   atol=config.diagnostics.parity_atol)
        if not bool(native[key].isfinite().all()) or not bool(adapted[key].isfinite().all()):
            raise ValueError(f"Nonfinite parity output for {key}")
        report["parity_heads"][key]["passed"] = True
        persist()
    report["native_seed_parity"] = True
    persist()
    components = {} if config.dbmol is not None else None
    if (getattr(config, "repair_guidance", None) is not None
            and config.repair_guidance.weight_schedule is not None):
        report["effective_repair_weight"] = weight
    measured, gradient = score_at(decoder, point, branch, config, counters,
                                  "derivative_analytic", gradients=True, component_gradients=components,
                                  repair_target=repair_target, repair_weight=weight)
    norm = float(gradient.double().norm())
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("No nonzero Boltz derivative at the selected context")
    report.update(gradient_norm=norm, measured=measured)
    if getattr(config, "repair_guidance", None) is not None:
        report["repair_application"] = repair_application(weight, repair_target,
            gradient_measured=measured["repair_guidance"]["gradient_measured"])
    persist()
    if config.fd_policy in ("branch_local_v2", "additive_branch_v3", "pointwise_native_ad_v4"):
        if config.fd_policy == "branch_local_v2":
            from .branch_local_fd import validate_branch_derivatives
        elif config.fd_policy == "additive_branch_v3":
            # Explicit opt-in only. Historical v2 outcomes remain untouched;
            # this candidate has its own native readiness and proof contract.
            from .additive_branch_fd import validate_branch_derivatives
        else:
            # Local AD consistency is a distinct, preselected contract. Native
            # finite-chord observations are retained as sensitivity diagnostics;
            # accepted optimization steps still need measured native descent.
            from .pointwise_ad import validate_branch_derivatives
        validate_branch_derivatives(decoder, point, branch, config, counters, report, persist,
            measured, gradient, components, target=repair_target, weight=weight)
        branch.verify_context_unchanged()
        return
    if config.fd_policy is not None:
        from .local_fd import POLICY, validate_local_fd
        if config.fd_policy != POLICY:
            raise ValueError("Unsupported local finite-difference policy")
        if repair_target is not None:
            from latentmol.repair_guidance.loss import target_score_at
            guidance = config.repair_guidance
            def repair_probe(candidate, gradients=False):
                return counters.call("repair", "finite_difference" if not gradients else "derivative_analytic",
                    "decoder_probe", lambda: target_score_at(decoder, candidate, prepared.active, repair_target,
                        temperature=config.temperature, normalization=guidance.normalization,
                        gradients=gradients), point.device)
            target_row, target_gradient = repair_probe(point, gradients=True)
            item = {"target_sha256": repair_target.identity, "gradient_norm": float(target_gradient.double().norm()),
                    "native_oracle_calls": 0, "target_held_fixed": True}
            report["repair_only_validation"] = item
            if item["gradient_norm"] <= 1e-12:
                item["status"] = "zero_at_current_point_uninformative"
            else:
                validate_local_fd(point, target_gradient, lambda p: repair_probe(p)[0],
                    baseline=target_row["loss"], report=item, persist=persist,
                    seed=config.diagnostics.seed, rtol=config.diagnostics.fd_rtol,
                    atol=config.diagnostics.fd_atol, minimum_passes=config.diagnostics.minimum_fd_passes)
                if not item["passed"]:
                    raise ValueError("Fixed repair-target finite-difference gate failed")
                item["status"] = "validated_nonzero"
            persist()
        if components is not None:
            report["component_gradients"] = {}
            for key, derivative in components.items():
                component_norm = float(derivative.double().norm())
                item = {"attached": True, "gradient_norm": component_norm,
                        "fixed_pose_and_mask": True}
                report["component_gradients"][key] = item
                persist()
                if component_norm <= 1e-12:
                    # Stationary/masked signal is not a detach or proven useful
                    # derivative. Preserve it; validate nonzero diagnostic points
                    # before making a claim about this component's gradients.
                    item["status"] = "zero_at_current_point_uninformative"
                    continue
                objective = Objective(output=key, direction="minimize")
                component_config = replace(config, objective=objective)
                def component_score(candidate, component_config=component_config):
                    return score_at(decoder, candidate, branch, component_config, counters,
                                    "component_finite_difference", gradients=False, repair_weight=weight)[0]
                validate_local_fd(point, derivative, component_score,
                    baseline=measured["components"][key], report=item, persist=persist,
                    seed=config.diagnostics.seed, rtol=config.diagnostics.fd_rtol,
                    atol=config.diagnostics.fd_atol,
                    minimum_passes=config.diagnostics.minimum_fd_passes)
                if not item["passed"]:
                    raise ValueError(f"DBMol component finite-difference gate failed: {key}")
                item["status"] = "validated_nonzero"
                persist()
        # Select all directions before evaluating any endpoints. Every endpoint
        # still uses the same conditional graph and same frozen native context;
        # the detached natural selector below is a warning only.
        local_report = {}
        report["local_fd_validation"] = local_report
        def local_score(candidate):
            row = score_at(decoder, candidate, branch, config, counters,
                           "finite_difference", gradients=False, repair_target=repair_target,
                           repair_weight=weight)[0]
            with torch.no_grad():
                natural = select_slots(decoder.decode(candidate), temperature=config.temperature,
                                       threshold=config.threshold, require_nonempty=False)
            return {**row, "natural_selector_crossed": not torch.equal(natural.active, prepared.active)}
        validate_local_fd(point, gradient, local_score, baseline=measured["loss"],
                          seed=config.diagnostics.seed, rtol=config.diagnostics.fd_rtol,
                          atol=config.diagnostics.fd_atol,
                          minimum_passes=config.diagnostics.minimum_fd_passes,
                          report=local_report, persist=persist)
        report["finite_differences"] = local_report["finite_differences"]
        persist()
        if not local_report["passed"]:
            raise ValueError("Conditional Boltz local_v1 finite-difference gate failed")
        branch.verify_context_unchanged()
        return
    generator = torch.Generator(device="cpu").manual_seed(config.diagnostics.seed)
    random = torch.randn(point.shape, generator=generator).to(point)
    directions = {"gradient": gradient / norm, "random": random / random.norm()}
    rows = report["finite_differences"]
    for label, direction in directions.items():
        analytic = float((gradient * direction).sum())
        for epsilon in config.diagnostics.fd_epsilons:
            plus = score_at(decoder, point + epsilon * direction, branch, config, counters,
                            "finite_difference", gradients=False)[0]["loss"]
            minus = score_at(decoder, point - epsilon * direction, branch, config, counters,
                             "finite_difference", gradients=False)[0]["loss"]
            numerical = (plus - minus) / (2 * epsilon)
            error = abs(analytic - numerical)
            passed = error <= config.diagnostics.fd_atol + config.diagnostics.fd_rtol * max(
                abs(analytic), abs(numerical))
            # A natural selector crossing is only a warning: the measured
            # derivative is intentionally conditional on the original context.
            with torch.no_grad():
                plus_mask = ((decoder.decode(point + epsilon * direction).atoms /
                              config.temperature).softmax(-1)[0, :, 0] < config.threshold)
                minus_mask = ((decoder.decode(point - epsilon * direction).atoms /
                               config.temperature).softmax(-1)[0, :, 0] < config.threshold)
            rows.append({"direction": label, "epsilon": epsilon, "analytic": analytic,
                         "central_difference": numerical, "absolute_error": error,
                         "passed": passed,
                         "natural_selector_crossed": bool(not torch.equal(plus_mask, prepared.active)
                                                        or not torch.equal(minus_mask, prepared.active))})
            persist()
    passed = all(len({row["epsilon"] for row in rows
                      if row["direction"] == label and row["passed"]})
                 >= config.diagnostics.minimum_fd_passes for label in directions)
    if not passed:
        raise ValueError("Conditional Boltz finite-difference gate failed")
    branch.verify_context_unchanged()


def record_candidates(path, arm, step, decoder, point, active, config):
    """Hard projection and unfiltered samples; conversion stays outside autograd."""
    with torch.no_grad():
        decoded = decoder.decode(point)
        graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                                   fixed_active=active)
        # Snapshot the already-created shared graph, without another model
        # call or RNG draw. Offline reports can then show full distributions.
        from latentmol.reporting.capture import capture_shared_graph
        capture_shared_graph(path.parent, step, graph, decoder.spec,
            active_slots=active.nonzero().flatten().tolist(), temperature=config.temperature)
        generator = torch.Generator(device=str(point.device)).manual_seed(
            config.diagnostics.seed + step)
        def harden_fixed():
            return HardGraphBatch(graph.atoms.argmax(-1), graph.charges.argmax(-1),
                                  graph.bonds.argmax(-1))
        draws = [("argmax", harden_fixed())]
        draws.extend(("categorical", sample_hard_graph(
            decoded, decoder.spec, generator=generator, temperature=config.temperature,
            fixed_active=active)) for _ in range(config.diagnostics.hard_samples))
        rows = []
        for index, (kind, hard) in enumerate(draws):
            hard = hard.to("cpu")
            inspected = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec)
            item = {"candidate_id": f"{arm}-s{step:04d}-d{index:03d}",
                    "arm": arm, "step": step, "draw": index, "kind": kind,
                    "graph_sha256": hard_digest(hard), "smiles": inspected.smiles,
                    "failure": inspected.failure_reason,
                    "heavy_atoms": int((hard.atoms != 0).sum()),
                    "graph_spec": decoder.spec.to_dict(),
                    "labels": {key: getattr(hard, key)[0].tolist()
                               for key in ("atoms", "charges", "bonds")}}
            rows.append(item)
            with path.open("a") as handle:
                handle.write(json.dumps(item, allow_nan=False) + "\n")
        return rows


def run_arm(name, decoder, start, initial: Selection, bank, scorer, config,
            source_sha256, run, counters):
    """Matched active-set or initial-mask control from identical molecular z."""
    if name not in ("active_set", "fixed_initial_mask"):
        raise ValueError("Unknown active-set comparison arm")
    arm_path = run.path / name
    arm_path.mkdir(exist_ok=False)
    point = start.detach().clone()
    selection = initial
    branch = None
    gates = {}
    transition_count = 0
    accepted_updates = 0
    attempted_updates = 0
    stalled = False
    full_budget_policy = config.step_control is not None and config.step_control.version == "best_trial_v2"
    improving_updates = forced_fallback_updates = no_op_updates = 0
    best_binding_states = {}
    guidance = getattr(config, "repair_guidance", None)
    scheduled = guidance is not None and guidance.weight_schedule is not None
    state_repair_weight = guidance.weight if guidance is not None else None
    previous_repair_weight = None
    last_accepted_repair_weight = None
    weight_jump_count = 0
    repair_target = None
    repair_receipt = None
    repair_summary = {"refreshes": 0, "attempts": 0, "accepted_targets": 0,
                      "guided_accepted_updates": 0}
    if guidance is not None:
        if name != "fixed_initial_mask" or config.fd_policy not in ("local_v1", "branch_local_v2", "additive_branch_v3", "pointwise_native_ad_v4") or config.step_control is None:
            raise ValueError("Repair guidance currently supports fixed-mask, gated, bounded optimization only")
        from latentmol.repair_guidance.provider import make_provider
        # The provider's old constant-weight enable gate must not disable a
        # ramp with an initially zero weight. All zero-state refreshes are
        # omitted below; this gate only admits a later positive phase.
        provider_config = (replace(guidance, weight=guidance.provider_activation_weight)
                           if scheduled else guidance)
        provider = make_provider(provider_config, source_sha256)
    def repair_term(target, purpose):
        if target is None:
            return 0.
        from latentmol.repair_guidance.loss import target_score_at
        return counters.call("repair", purpose, "decoder_probe", lambda: target_score_at(
            decoder, point, branch.prepared.active, target, temperature=config.temperature,
            normalization=guidance.normalization)[0]["loss"], point.device)
    def resolve_state_weight(step):
        """Resolve lambda once; state probes and trials never change it."""
        nonlocal state_repair_weight, previous_repair_weight, weight_jump_count
        state_repair_weight = guidance.weight_at(step, config.optimizer.steps)
        if scheduled:
            absolute = guidance.weight_schedule.version == "linear_updates_v1"
            if absolute:
                schedule = guidance.weight_schedule
                progress = min(1., max(0., (min(step, config.optimizer.steps - 1) - schedule.start_update) /
                    (schedule.end_update - schedule.start_update)))
                progress_policy = "clip((update_index-start_update)/(end_update-start_update),0,1); absolute indices"
            else:
                progress = (1.0 if config.optimizer.steps == 1 else
                            min(step, config.optimizer.steps - 1) / (config.optimizer.steps - 1))
                progress_policy = "min(update_index,N-1)/(N-1); singleton=1"
            receipt = {"step": step, "requested_updates": config.optimizer.steps,
                "progress": progress, "progress_policy": progress_policy,
                "effective_weight": state_repair_weight, "previous_weight": previous_repair_weight,
                "point_sha256": tensor_digest({"z": point}),
                "target_sha256": repair_target.identity if repair_target is not None else None,
                "is_optimization_improvement": False, "oracle_calls": 0,
                "endpoint_retains_last_update_weight": step == config.optimizer.steps,
                "scope": "Fixed across all analytic/FD/Armijo probes of this state"}
            if previous_repair_weight is not None and state_repair_weight != previous_repair_weight:
                nll = repair_term(repair_target, "schedule_objective_jump")
                receipt.update(repair_nll_held_target=nll,
                    weighted_objective_jump=(state_repair_weight - previous_repair_weight) * nll,
                    binding_unchanged=True, target_unchanged=True)
                weight_jump_count += 1
            else:
                receipt["weighted_objective_jump"] = 0.
            write_json(arm_path / f"repair-weight-{step:04d}.json", receipt)
            run.log({"stage": "repair_weight_schedule", "arm": name, **receipt})
    def refresh_repair(step):
        """Between-step oracle work; the result never changes inside a probe.

        A target refresh may jump the objective even at unchanged z. Binding is
        unchanged at this fixed context; record that jump separately from an
        accepted update. A failed refresh clears the previous target explicitly.
        """
        nonlocal repair_target, repair_receipt
        old_identity = repair_target.identity if repair_target is not None else None
        old_nll = repair_term(repair_target, "refresh_objective")
        repair_target, repair_receipt = provider.refresh(decoder, point, branch.prepared.active,
            temperature=config.temperature, step=step, directory=arm_path, counters=counters)
        if scheduled:
            repair_receipt = {**repair_receipt,
                "effective_objective_weight": state_repair_weight,
                "provider_activation_weight": guidance.provider_activation_weight,
                "provider_weight_scope": "Provider weight is an activation gate only; objective uses fixed runner state weight"}
        new_nll = repair_term(repair_target, "refresh_objective")
        jump = {"step": step, "point_sha256": tensor_digest({"z": point}),
                "context_sha256": branch.manifest.context_sha256,
                "old_target_sha256": old_identity,
                "new_target_sha256": repair_target.identity if repair_target is not None else None,
                "old_repair_loss": old_nll, "new_repair_loss": new_nll,
                "weighted_objective_jump": state_repair_weight * (new_nll - old_nll),
                "binding_unchanged": True, "is_optimization_improvement": False,
                "failed_refresh_clears_old_target": True, "receipt": repair_receipt}
        write_json(arm_path / f"repair-refresh-objective-{step:04d}.json", jump)
        if scheduled:
            jump["effective_weight"] = state_repair_weight
            jump["schedule_held_fixed"] = True
            write_json(arm_path / f"repair-refresh-objective-{step:04d}.json", jump)
        repair_summary["refreshes"] += 1
        repair_summary["attempts"] += len(repair_receipt.get("attempts", []))
        repair_summary["accepted_targets"] += int(repair_target is not None)
        run.log({"stage": "repair_target_refresh", "arm": name, **jump})
        # Keep the preliminary snapshot truthful even when the following FD
        # gate aborts before the ordinary scored-state snapshot is written.
        state["repair_application"] = repair_application(state_repair_weight, repair_target)
        torch.save(state, state_path)
    for step in range(config.optimizer.steps + 1):
        do_update = step < config.optimizer.steps
        if guidance is not None:
            resolve_state_weight(step)
        first_positive_phase = scheduled and state_repair_weight > 0 and (
            previous_repair_weight is None or previous_repair_weight == 0)
        refresh_due = (guidance is not None and do_update and (
            (not scheduled and step % guidance.refresh_interval == 0)
            or (scheduled and state_repair_weight > 0
                and (step % guidance.refresh_interval == 0 or first_positive_phase))))
        if scheduled:
            previous_repair_weight = state_repair_weight
        refreshed = False
        state_path = arm_path / f"state-{step:04d}.pt"
        state = {"z": point.detach().cpu(), "active": selection.active.detach().cpu(),
                 "no_atom": selection.no_atom.detach().cpu(), "context_sha256": None}
        if guidance is not None:
            state["repair_application"] = repair_application(state_repair_weight, repair_target)
        torch.save(state, state_path)
        if branch is None or not torch.equal(selection.active, branch.prepared.active):
            old_context = branch.manifest.context_sha256 if branch is not None else None
            old_count = int(branch.prepared.active.sum()) if branch is not None else None
            transition = {"step": step, "status": "attempting",
                          "old_context_sha256": old_context,
                          "new_context_sha256": None, "state_path": state_path.name,
                          "new_selection": selection.record(),
                          "old_score": None, "new_score": None,
                          "score_jump_changes_conditioned_graph_and_slot_assignment": old_context is not None,
                          "score_jump_may_change_pose_and_native_features":
                          old_count is not None and selection.count != old_count}
            transition_path = arm_path / f"transition-{step:04d}.json"
            gate_path = arm_path / f"gate-{step:04d}.json"
            write_json(transition_path, transition)
            try:
                if branch is not None:
                    transition["old_score"] = score_at(decoder, point, branch, config, counters,
                                                       "transition_old_context", gradients=False,
                                                       repair_weight=state_repair_weight)[0]
                    branch.verify_context_unchanged()
                    # Release the previous device context before allocating the
                    # next; only the frozen scorer is shared between branches.
                    branch = None
                branch = make_branch(scorer, bank, selection, config, source_sha256)
                transition["new_context_sha256"] = branch.manifest.context_sha256
                state["context_sha256"] = branch.manifest.context_sha256
                torch.save(state, state_path)
                if refresh_due:
                    refresh_repair(step)
                    refreshed = True
                validate_branch(decoder, point, branch, config, counters, report_path=gate_path,
                                repair_target=repair_target, repair_weight=state_repair_weight)
                transition["new_score"] = score_at(decoder, point, branch, config, counters,
                                                   "transition_new_context", gradients=False,
                                                   repair_target=repair_target, repair_weight=state_repair_weight)[0]
                transition["status"] = "completed"
            except Exception as exc:
                transition.update(status="failed", error=str(exc), error_type=type(exc).__name__)
                raise
            finally:
                if gate_path.exists():
                    gates[f"step-{step:04d}"] = json.loads(gate_path.read_text())
                    write_json(arm_path / "gates.json", gates)
                write_json(transition_path, transition)
                with (arm_path / "transitions.jsonl").open("a") as handle:
                    handle.write(json.dumps(transition, allow_nan=False) + "\n")
            transition_count += int(old_context is not None)
        if refresh_due and not refreshed:
            refresh_repair(step)
            refreshed = True
        # Recheck derivatives during a trajectory, including its budget endpoint.
        # Context-transition gates already cover this point and are not repeated.
        if ((refreshed or (config.gradient_audit_interval is not None
                and (step % config.gradient_audit_interval == 0 or step == config.optimizer.steps)))
                and f"step-{step:04d}" not in gates):
            gate_path = arm_path / f"gate-{step:04d}.json"
            try:
                validate_branch(decoder, point, branch, config, counters, report_path=gate_path,
                                repair_target=repair_target, repair_weight=state_repair_weight)
            finally:
                if gate_path.exists():
                    gates[f"step-{step:04d}"] = json.loads(gate_path.read_text())
                    write_json(arm_path / "gates.json", gates)
        score, gradient = score_at(decoder, point, branch, config, counters,
                                   "optimization" if do_update else "endpoint",
                                   gradients=do_update, repair_target=repair_target,
                                   repair_weight=state_repair_weight)
        with torch.no_grad():
            decoded = decoder.decode(point)
            natural = select_slots(decoded, temperature=config.temperature,
                                   threshold=config.threshold,
                                   require_nonempty=name == "active_set")
            candidates = record_candidates(arm_path / "candidates.jsonl", name, step,
                                           decoder, point, selection.active, config)
            state = {"z": point.detach().cpu(),
                     "active": selection.active.detach().cpu(),
                     "no_atom": natural.no_atom.detach().cpu(),
                     "context_sha256": branch.manifest.context_sha256}
            if guidance is not None:
                state["repair_target_sha256"] = repair_target.identity if repair_target is not None else None
                state["repair_refresh_step"] = repair_target.refresh_step if repair_target is not None else None
                state["repair_application"] = repair_application(state_repair_weight, repair_target,
                    gradient_measured=score["repair_guidance"]["gradient_measured"])
            if scheduled:
                state["effective_repair_weight"] = state_repair_weight
            torch.save(state, arm_path / f"state-{step:04d}.pt")
        row = {"step": step, "arm": name, "score": score,
               "selection": selection.record(), "natural_selection": natural.record(),
               "latent_displacement": float((point - start).norm()),
               "candidate_count": len(candidates),
               "valid_candidates": sum(item["smiles"] is not None for item in candidates),
               "argmax_smiles": candidates[0]["smiles"],
               "context_sha256": branch.manifest.context_sha256}
        if guidance is not None:
            row["repair_refresh"] = repair_receipt
        if scheduled:
            row["repair_weight_schedule"] = {"effective_weight": state_repair_weight,
                "requested_budget": config.optimizer.steps,
                "receipt_path": f"repair-weight-{step:04d}.json"}
        if config.gradient_audit_interval is not None:
            from copy import deepcopy
            row["oracle_cost_at_state"] = deepcopy(counters.values)
        if full_budget_policy:
            # Compare binding scores only within the same authenticated context.
            # Repair targets/coefficients change between updates, so ranking the
            # combined objective across those changes would be misleading.
            binding_loss = score.get("binding_loss", score["loss"])
            context_sha = branch.manifest.context_sha256
            active_slots = selection.record()["active_slots"]
            context_key = hashlib.sha256(json.dumps([context_sha, active_slots]).encode()).hexdigest()
            previous_best = best_binding_states.get(context_key)
            if previous_best is None or binding_loss < previous_best["loss"]:
                best_path = f"best-binding-{context_key}.pt"
                torch.save(state, arm_path / best_path)
                best_binding_states[context_key] = {"step": step, "loss": binding_loss,
                    "state_path": best_path, "context_sha256": context_sha,
                    "active_slots": active_slots, "context_mask_sha256": context_key,
                    "scope": "Saved scored states; binding-only signed/scaled loss within this fixed native context"}
        write_json(arm_path / f"metrics-{step:04d}.json", row)
        run.log({"stage": "active_set_step", **row})
        if not do_update:
            break
        if config.step_control is not None:
            step_report = {}
            report_path = arm_path / f"step-control-{step:04d}.json"
            def persist_step():
                write_json(report_path, step_report)
            def trial_score(candidate):
                observed = score_at(decoder, candidate, branch, config, counters,
                                    "step_control", gradients=False, reject_nonfinite_objective=True,
                                    repair_target=repair_target, repair_weight=state_repair_weight)[0]
                natural_trial = select_slots(decoder.decode(candidate), temperature=config.temperature,
                                             threshold=config.threshold, require_nonempty=False)
                return {**observed, "natural_selector_crossed": not torch.equal(
                    natural_trial.active, branch.prepared.active),
                    "natural_selection": natural_trial.record(),
                    "scope": "Trial retains original fixed mask/context/pose; natural selection is diagnostic only"}
            attempted_updates += 1
            controlled = backtracking_step(point, start, gradient,
                baseline_loss=score, learning_rate=config.optimizer.learning_rate,
                max_gradient_norm=config.optimizer.max_gradient_norm,
                trust_radius=config.optimizer.trust_radius, control=config.step_control,
                score=trial_score, report=step_report, persist=persist_step)
            row.update(update_status=controlled.report["status"], step_control_path=report_path.name,
                       step_control_oracle_calls=controlled.report["oracle_calls_attempted"])
            if full_budget_policy:
                row.update(selection_reason=controlled.report["acceptance_kind"],
                           strictly_improved=controlled.report["monotone_descent"],
                           applied_loss_improvement=controlled.report["actual_loss_improvement"],
                           no_op=controlled.report["no_op"])
            write_json(arm_path / f"metrics-{step:04d}.json", row)
            run.log({"stage": "active_set_step_control", "arm": name, "step": step,
                     "status": controlled.report["status"], "report_path": report_path.name,
                     "accepted": controlled.accepted,
                     "oracle_calls": controlled.report["oracle_calls_attempted"]})
            if not controlled.accepted:
                if full_budget_policy:
                    raise RuntimeError("Full-budget policy returned without a finite applied trial")
                # The current saved state is the endpoint; no successful update,
                # occupancy transition, or fictional exhausted-budget completion.
                stalled = True
                # A stall is a terminal observation, not a verified optimum.
                # Audit this actual endpoint when the campaign requests checks.
                if config.gradient_audit_interval is not None and f"step-{step:04d}" not in gates:
                    gate_path = arm_path / f"gate-{step:04d}.json"
                    try:
                        validate_branch(decoder, point, branch, config, counters, report_path=gate_path,
                                        repair_target=repair_target, repair_weight=state_repair_weight)
                    finally:
                        if gate_path.exists():
                            gates[f"step-{step:04d}"] = json.loads(gate_path.read_text())
                            write_json(arm_path / "gates.json", gates)
                break
            point = controlled.point
            accepted_updates += 1
            if full_budget_policy:
                improving_updates += int(controlled.report["monotone_descent"])
                forced_fallback_updates += int(controlled.report["acceptance_kind"] == "best_finite_fallback")
                no_op_updates += int(controlled.report["no_op"])
            if scheduled:
                last_accepted_repair_weight = state_repair_weight
            if repair_target is not None:
                repair_summary["guided_accepted_updates"] += 1
        else:
            # Preserve legacy proposal/projection arithmetic and metadata exactly.
            previous_point = point
            norm = float(gradient.double().norm())
            gradient = gradient * min(1.0, config.optimizer.max_gradient_norm / max(norm, 1e-12))
            proposal = point - config.optimizer.learning_rate * gradient
            delta = proposal - start
            point = (start + delta * min(1.0, config.optimizer.trust_radius /
                                       max(float(delta.double().norm()), 1e-12))).detach()
            if config.gradient_audit_interval is not None:
                # Opt-in campaign receipts expose the old update's safeguards;
                # none of these observations change its arithmetic or defaults.
                row["legacy_update"] = {
                    "gradient_scale": min(1.0, config.optimizer.max_gradient_norm / max(norm, 1e-12)),
                    "projection_scale": min(1.0, config.optimizer.trust_radius / max(float(delta.double().norm()), 1e-12)),
                    "actual_displacement_norm": float((point - previous_point).double().norm()),
                    "proposal_displacement_norm": float((proposal - previous_point).double().norm()),
                }
                write_json(arm_path / f"metrics-{step:04d}.json", row)
        if name == "active_set":
            with torch.no_grad():
                selection = select_slots(decoder.decode(point), temperature=config.temperature,
                                         threshold=config.threshold, require_nonempty=False)
            if selection.count == 0 or selection.count not in bank.templates:
                proposal_path = arm_path / f"rejected-state-{step:04d}.pt"
                torch.save({"z": point.detach().cpu(), "active": selection.active.cpu(),
                            "no_atom": selection.no_atom.cpu()}, proposal_path)
                rejected = {"step_after_update": step,
                            "reason": "empty_ligand" if selection.count == 0 else "native_count_context_unavailable",
                            "selection": selection.record(),
                            "available_counts": sorted(bank.templates),
                            "proposal_state_path": proposal_path.name}
                write_json(arm_path / f"rejected-transition-{step:04d}.json", rejected)
                run.log({"stage": "active_set_transition_rejected", "arm": name, **rejected})
                raise ValueError(f"No native prepared context for selected count {selection.count}")
        # Fixed control uses the identical start, update rule and step budget,
        # but retains the starting occupancy and native context throughout.
    branch.verify_context_unchanged()
    summary = {"arm": name, "steps": config.optimizer.steps,
               "context_switches": transition_count,
               "gated_contexts": len(gates), "endpoint": row,
               "fresh_native_pose_confirmation": "required_not_run"}
    if config.step_control is not None:
        summary.update(steps=accepted_updates, requested_steps=config.optimizer.steps,
                       accepted_updates=accepted_updates, attempted_updates=attempted_updates,
                       stalled=stalled, completed_requested_budget=not stalled,
                       optimization_status="stalled" if stalled else "completed_budget",
                       step_control=config.step_control.record(),
                       monotonicity_scope="Accepted trial losses decrease only within each fixed context; separately gated context switches may jump")
    if guidance is not None:
        summary["repair_guidance"] = {**repair_summary, "config": guidance.record(),
            "native_guidance_validation": "requires accepted native targets, nonzero target FD and guided accepted updates",
            "final_target_sha256": repair_target.identity if repair_target is not None else None}
        summary["monotonicity_scope"] = "Accepted trial loss decreases at fixed mask/context and fixed repair target; refresh jumps are separate"
        if scheduled:
            summary["repair_guidance"].update(weight_schedule=guidance.record()["weight_schedule"],
                start_weight=guidance.weight_at(0, config.optimizer.steps),
                requested_final_update_weight=guidance.weight_at(config.optimizer.steps - 1, config.optimizer.steps),
                final_observed_weight=state_repair_weight,
                last_accepted_update_weight=last_accepted_repair_weight,
                weight_jump_events=weight_jump_count,
                provider_activation_weight=guidance.provider_activation_weight,
                progress_policy=("Absolute update-index ramp; requested budget does not retime the ramp"
                    if guidance.weight_schedule.version == "linear_updates_v1" else
                    "min(update_index,N-1)/(N-1); singleton=1; requested budget fixed on early stall"),
                zero_phase_refresh_policy="No oracle at zero; refresh first positive state even off regular interval")
            summary["monotonicity_scope"] = "Accepted trial loss decreases at fixed mask/context, repair target and weight; schedule and target-refresh jumps are separate"
    if full_budget_policy:
        summary.update(applied_updates=accepted_updates, improving_updates=improving_updates,
            forced_fallback_updates=forced_fallback_updates, no_op_updates=no_op_updates,
            best_binding_states=best_binding_states,
            accepted_updates_scope="Compatibility count of applied selections, including uphill/equal/no-op fallbacks; not a descent count",
            monotonicity_scope="No monotone-loss guarantee: first strict improvement or best finite fallback per fixed state; context/target/weight changes remain separate")
    write_json(arm_path / "summary.json", summary)
    return summary


def run(config_path: Path) -> Path:
    """Execute only inside an explicit DTU GPU job from an immutable release."""
    import os
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require an HPC allocation and immutable source archive identity")
    config = ActiveSetConfig.load(config_path)
    source_sha256 = source_manifest(Path.cwd())["sha256"]
    record = {**config.record(), "config_path": str(config_path.resolve()),
              "config_file_sha256": file_sha256(config_path),
              "source_sha256": source_sha256,
              "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"],
              "scope": "Fixed-context proxy only; random-start extension pending HPC validation"}
    with ExperimentRun(config.output_root, config.name, record, Path.cwd(),
                       report_kind="molecule_optimization") as experiment:
        decoder_path = verified_file(config.decoder["path"], config.decoder["sha256"])
        affinity_path = verified_file(config.affinity_checkpoint["path"],
                                      config.affinity_checkpoint["sha256"])
        decoder, training = load_model(decoder_path, config.device)
        if training["train_config"]["mode"] != config.mode:
            raise ValueError("Declared AE/VAE mode differs from the verified checkpoint")
        decoder.eval().requires_grad_(False)
        bank = ContextBank(config, decoder.spec)
        experiment.annotate(decoder_dataset_sha256=training.get("dataset_sha256"),
                            decoder_mode=config.mode,
                            graph_spec_sha256=decoder.spec.fingerprint(),
                            target_sequence_sha256=bank.target_identity[0],
                            msa_sha256=bank.target_identity[1])
        budget_guard = None
        if config.execution_budget is not None:
            from latentmol.execution_budget import TrackedNativeBudget
            budget_guard = TrackedNativeBudget(config.execution_budget,
                lambda value: write_json(experiment.path / "execution-budget.json", value))
        counters = Counters(lambda value: write_json(experiment.path / "oracle_cost.json", value),
            before_call=budget_guard.before_call if budget_guard is not None else None)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        anchor = bank.templates[config.anchor_count]
        if config.dbmol is not None:
            config.dbmol.verify_reference(anchor, config.contexts[config.anchor_count]["sha256"])
        with torch.no_grad():
            if config.initial_latent is not None:
                start = load_initial_latent(config.initial_latent, decoder.config.latent, config.device)
            elif config.start.kind == "standard_normal":
                start = sample_prior(decoder.config.latent, config.start.seed, device=config.device)
            else:
                start, _ = decoder.encode(anchor.seed.to(config.device).to_relaxed(decoder.spec))
            start = start.detach()
            # Preserve the actual draw before decoding or checking occupancy.
            # An unusable prior sample remains this seed's result, never a redraw.
            start_record = {"z": start.cpu(), "anchor_count": config.anchor_count,
                            "initialization": {"kind": config.start.kind, "seed": config.start.seed},
                            "z_sha256": tensor_digest({"z": start})}
            if config.initial_latent is not None:
                start_record["initialization"].update(saved_z=config.initial_latent,
                    sampled_or_encoded_this_run=False)
            torch.save(start_record, experiment.path / "start.pt")
            if start.shape != (1, decoder.config.latent) or not bool(start.isfinite().all()):
                raise ValueError("Molecular latent start is malformed")
            decoded_start = decoder.decode(start)
            if config.start.kind == "encoded_seed":
                initial, start_audit = audit_start(decoded_start, anchor.seed, decoder.spec,
                                                   temperature=config.temperature, threshold=config.threshold)
            else:
                initial, start_audit = audit_random_start(
                    decoded_start, decoder.spec, temperature=config.temperature,
                    threshold=config.threshold, available_counts=bank.templates)
                start_audit.update(initialization=start_record["initialization"],
                                   z_sha256=start_record["z_sha256"])
            write_json(experiment.path / "start-audit.json", start_audit)
            torch.save({**start_record, "selection": initial.record()}, experiment.path / "start.pt")
            if not start_audit["passed"]:
                if config.start.kind == "encoded_seed":
                    raise ValueError("Encoded start must select and hard-decode to the exact native carbon-chain seed")
                raise ValueError(f"Random start selected count {initial.count} without an available native context")
        affinity_model = load_affinity_model(affinity_path).to(config.device)
        if config.dbmol is None:
            scorer = FrozenBoltzAffinity(affinity_model, config.recycling_steps)
        else:
            from .dbmol_model import FrozenBoltzDBMol, load_confidence_model
            structure_model = load_confidence_model(config.dbmol.confidence_checkpoint).to(config.device)
            scorer = FrozenBoltzDBMol(affinity_model, structure_model, config.dbmol, config.recycling_steps)
            experiment.annotate(loss_profile=config.dbmol.record(), physical_trunks_per_forward=2,
                gradient_scope="fixed pose and selected mask; two native frozen trunks/heads; no diffusion derivatives")
        original_decoder = module_digest(decoder)
        original_scorer = module_digest(scorer)
        experiment.annotate(initial_selection=initial.record(),
                            decoder_state_sha256=original_decoder,
                            scorer_state_sha256=original_scorer)
        summaries = {}
        for arm in config.arms:
            summaries[arm] = run_arm(arm, decoder, start, initial, bank, scorer,
                                     config, source_sha256, experiment, counters)
            if module_digest(decoder) != original_decoder or module_digest(scorer) != original_scorer:
                raise ValueError("Frozen decoder or Boltz model state changed")
        candidate_hashes = {arm: file_sha256(experiment.path / arm / "candidates.jsonl")
                            for arm in summaries}
        summary = {
            "arms": summaries, "oracle_cost": counters.values,
            "candidate_file_sha256": candidate_hashes,
            "frozen_state_unchanged": True,
            "claim_limit": "Proxy scores at count-specific fixed predicted poses; no validated binding, chemistry, or native fresh-pose gain."}
        if budget_guard is not None:
            summary["execution_budget"] = budget_guard.complete()
        write_json(experiment.path / "summary.json", summary)
        return experiment.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config), flush=True)


if __name__ == "__main__":
    main()
