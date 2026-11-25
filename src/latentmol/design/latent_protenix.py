"""Frozen-decoder latent design with a shared graph, structural repair and schedules.

Only ``z`` changes. The native scorer is injected, keeping this loop independent
of Protenix process ownership. Repair search is detached; its accepted witness
supplies the existing fixed-occupancy categorical NLL. A failed search or an
already-valid graph has no repair target. An explicit optional valence term
acts on the same probability graph independently of the repair target.

Each invocation writes a NEW segment directory. ``state.pt`` can resume the
same global trajectory in another directory, retaining the latent/noise stream
and absolute schedules. A shorter pilot alone does not shorten any schedule.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from latentmol.chemistry import inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.regularization import fixed_occupancy_valence_excess
from latentmol.guidance.repair_proposals import RepairProposalConfig, search_with_proposals
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.repair_guidance.loss import FrozenRepairTarget, fixed_target_nll
from latentmol.reporting.capture import capture_shared_graph
from latentmol.representation import DecoderOutput, HardGraphBatch, RelaxedGraph


VERSION = "latent-protenix-structural-v1"


@dataclass(frozen=True)
class LatentDesignConfig:
    """Declared scientific settings; segment wall/step budgets are CLI controls."""

    version: str = VERSION
    updates: int = 200
    schedule_horizon: int = 50000
    seed: int = 17
    noise_seed: int = 230017
    occupied_atoms: int = 25
    learning_rate: float = .3
    max_step_norm: float = .5
    chemistry_initial: float = 1.
    chemistry_final: float = 10.
    chemistry_hold: int = 20000
    chemistry_ramp_end: int = 30000
    temperature_initial: float = .7
    temperature_final: float = .4
    noise_initial: float = .4
    noise_zero_at: int = 30000
    snapshot_interval: int = 100
    checkpoint_interval: int = 100
    repair_search: dict = field(default_factory=lambda: dict(max_edits=3, checks_per_depth=32,
        beam_width=4, probability_weight=1., probability_floor=1e-12))
    structural_proposals: dict = field(default_factory=lambda: asdict(RepairProposalConfig()))
    valence_excess_weight: float = 0.
    # Opt-in linear ramp across the absolute update indices 0..schedule_horizon-1.
    max_step_norm_final: float | None = None

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("Unknown latent design configuration")
        for name in ("updates", "schedule_horizon", "occupied_atoms", "snapshot_interval", "checkpoint_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Require a positive integer: " + name)
        for name in ("seed", "noise_seed"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**63:
                raise ValueError("Invalid RNG seed")
        for name in ("learning_rate", "max_step_norm", "chemistry_initial", "chemistry_final",
                     "temperature_initial", "temperature_final"):
            v = getattr(self, name)
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError("Require finite positive " + name)
        v = self.valence_excess_weight
        if isinstance(v, bool) or not math.isfinite(v) or v < 0:
            raise ValueError("Require finite nonnegative valence_excess_weight")
        v = self.max_step_norm_final
        if v is not None and (isinstance(v, bool) or not math.isfinite(v) or v <= 0
                              or self.schedule_horizon < 2):
            raise ValueError("Step cap schedule requires a finite positive final cap and at least two steps")
        if not (0 <= self.noise_initial <= 1 and 0 <= self.chemistry_hold < self.chemistry_ramp_end
                <= self.schedule_horizon and 0 < self.noise_zero_at <= self.schedule_horizon
                and self.updates <= self.schedule_horizon):
            raise ValueError("Invalid absolute schedule bounds")
        ProbabilityRepairConfig(**self.repair_search)
        RepairProposalConfig(**self.structural_proposals)

    @classmethod
    def from_mapping(cls, raw):
        return cls(**raw.get("design", raw))

    @classmethod
    def for_restart(cls, settings):
        """Keep update noise opt-in for restart searches."""
        return cls(**({"noise_initial": 0.0} | settings))

    def identity(self):
        payload = asdict(self)
        if not self.valence_excess_weight:
            # Old checkpoints authenticated the config before this opt-in field.
            del payload["valence_excess_weight"]
        if self.max_step_norm_final is None:
            # Keep fixed-cap resume identities valid.
            del payload["max_step_norm_final"]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def temperature_end_step(self):
        return self.schedule_horizon


def schedules(config: LatentDesignConfig, step: int) -> dict:
    """Global state schedules, independent of the requested segment length."""
    if type(step) is not int or step < 0:
        raise ValueError("Schedule step must be a nonnegative integer")
    progress = min(1., max(0., (step-config.chemistry_hold) /
                            (config.chemistry_ramp_end-config.chemistry_hold)))
    return dict(chemistry_weight=config.chemistry_initial *
        (config.chemistry_final/config.chemistry_initial)**progress,
        temperature=config.temperature_initial +
        (config.temperature_final-config.temperature_initial)*min(step/config.temperature_end_step, 1.),
        noise_ratio=config.noise_initial*max(0., 1.-step/config.noise_zero_at))


def choose_active(decoded, spec, count):
    """Choose top occupancy slots ONCE; this first experiment fixes atom count."""
    if not 1 <= count <= spec.max_nodes or decoded.atoms.shape[0] != 1:
        raise ValueError("Incompatible fixed atom count or decoder batch")
    occupancy = decoded.atoms.detach().softmax(-1)[0, :, 1:].sum(-1)
    if not bool(torch.isfinite(occupancy).all()):
        raise ValueError("Nonfinite initial occupancy")
    active = torch.zeros(spec.max_nodes, dtype=torch.bool, device=decoded.atoms.device)
    # Stable sorting makes ties resolve by slot ID, with no RNG redraw.
    active[torch.argsort(occupancy, descending=True, stable=True)[:count]] = True
    return active


def hard_from_graph(graph):
    return HardGraphBatch(graph.atoms.argmax(-1), graph.charges.argmax(-1), graph.bonds.argmax(-1))


def structural_target(graph, spec, active, config, step):
    """Search exactly the shared fixed-mask/T graph, never a second softmax law."""
    hard = hard_from_graph(graph)
    raw = tuple(getattr(hard, k)[0].detach().cpu().numpy() for k in ("atoms", "charges", "bonds"))
    probabilities = tuple(getattr(graph, k)[0].detach().cpu().numpy() for k in ("atoms", "charges", "bonds"))
    report = search_with_proposals(raw, probabilities, spec,
        ProbabilityRepairConfig(**config.repair_search), RepairProposalConfig(**config.structural_proposals))
    witness = report.get("weighted_witness")
    target = None
    if witness is not None:
        labels = witness["labels"]
        repaired = HardGraphBatch(*(torch.tensor(labels[k], dtype=torch.long)[None]
                                  for k in ("atoms", "charges", "bonds")))
        target = FrozenRepairTarget.create(repaired, active.detach().cpu(),
            refresh_step=step, receipt_path="metrics.jsonl:step="+str(step))
    compact = dict(status=report["status"], checks=report["checks"],
        accepted_source=report.get("accepted_source"), macro_status=report.get("macro_status"),
        edit_count=None if witness is None else witness["edit_count"],
        weighted_cost=None if witness is None else witness["weighted_cost"],
        target_sha256=None if target is None else target.identity,
        distance_is_exact=False)
    return target, compact


def _finite_scalar(value, name):
    if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(value.isfinite().all()):
        raise ValueError(name + " must be one finite tensor scalar")
    return value.reshape(())


@dataclass(frozen=True)
class PreparedPoint:
    """One differentiable decoder evaluation and its fixed branch settings."""

    point: torch.Tensor
    active: torch.Tensor
    config: LatentDesignConfig
    step: int
    schedule: dict
    decoded: DecoderOutput
    graph: RelaxedGraph
    features: object | None = None


def prepare_point(decoder, point, active, config, step, *, logit_offset=None,
                  uses_decoder_hidden=False):
    """Build the single live graph shared by the scorer and chemistry losses."""
    point = point.detach().clone().requires_grad_(True)
    schedule = schedules(config, step)
    features = None
    if uses_decoder_hidden:
        decoded, features = decoder.decode_with_features(point)
    else:
        decoded = decoder.decode(point)
    if logit_offset is not None:
        if uses_decoder_hidden:
            raise ValueError("A logit offset requires the shared graph scorer")
        decoded = logit_offset(decoded, step)
    for name in ("atoms", "charges", "bonds"):
        if not bool(torch.isfinite(getattr(decoded, name)).all()):
            raise ValueError("Nonfinite decoder logits: " + name)
    graph = graph_from_decoder(decoded, decoder.spec, temperature=schedule["temperature"], fixed_active=active)
    return PreparedPoint(point, active, config, step, schedule, decoded, graph, features)


def finish_point(decoder, prepared, scored, *, target_bundle=None, guidance=None):
    """Finish one branch using a score attached to its prepared graph."""
    point, active, config, step = prepared.point, prepared.active, prepared.config, prepared.step
    schedule, decoded, graph = prepared.schedule, prepared.decoded, prepared.graph
    binding, aux = scored
    binding = _finite_scalar(binding, "Binding objective")
    if not binding.requires_grad:
        raise ValueError("Required binding path is detached")
    channel_norms = {}
    if step == 0 and getattr(decoder, "raw_coordinate_key", None) == "graph_logits":
        channels = ("atoms", "charges", "bonds")
        paths = tuple(getattr(graph, name) for name in channels)
        binding_gradient, *gradients = torch.autograd.grad(binding, (point, *paths), retain_graph=True,
                                                          allow_unused=True)
        for name, grad in zip(channels, gradients, strict=True):
            if grad is None or not bool(grad.isfinite().all()):
                raise ValueError(f"Missing or nonfinite direct graph binding gradient for {name}")
        channel_norms["direct_binding_channels"] = {k: float(g.norm()) for k, g in zip(channels, gradients)}
    else:
        binding_gradient, = torch.autograd.grad(binding, point, retain_graph=True, allow_unused=True)
    if binding_gradient is None or not bool(binding_gradient.isfinite().all()):
        raise ValueError("Missing/nonfinite binding latent gradient")
    extra = {}
    guidance_gradient = None
    target, repair = (structural_target(graph, decoder.spec, active, config, step)
                      if target_bundle is None else target_bundle)
    if guidance is not None:
        validity = ({"raw_valid": repair["status"] == "already_valid"}
                    if getattr(guidance, "requires_raw_valid", False) else {})
        guidance_gradient, metadata = guidance.evaluate(graph, active, point, binding_gradient, step, **validity)
        if (guidance_gradient.shape != point.shape or not bool(guidance_gradient.isfinite().all())):
            raise ValueError("Incompatible/nonfinite auxiliary guidance gradient")
        extra["synthesis"] = metadata
    if target is None:
        chemistry = sum(getattr(decoded, k).sum()*0 for k in ("atoms", "charges", "bonds"))
        detail = dict(normalized_nll=0., raw_nll=0., target_sha256=None,
                      normalization="mean_per_decision", target_available=False)
    else:
        chemistry, detail = fixed_target_nll(decoded, graph, target, decoder.spec,
            temperature=schedule["temperature"], normalization="mean_per_decision")
        detail["target_available"] = True
    valence_details = {}
    valence_norms = {}
    if config.valence_excess_weight:
        valence = _finite_scalar(fixed_occupancy_valence_excess(graph, decoder.spec, active), "Valence excess")
        valence_gradient, = torch.autograd.grad(valence, point, retain_graph=True, allow_unused=True)
        if valence_gradient is None or not bool(valence_gradient.isfinite().all()):
            raise ValueError("Missing/nonfinite valence excess latent path")
        valence_norm = float(valence_gradient.norm())
        valence_details = dict(valence_excess=valence.detach(), valence_excess_gradient=valence_gradient.detach())
        valence_norms = dict(valence_excess=valence_norm,
                             weighted_valence_excess=config.valence_excess_weight*valence_norm)
    chemistry = _finite_scalar(chemistry, "Chemistry objective")
    chemistry_gradient, = torch.autograd.grad(chemistry, point, allow_unused=True)
    if chemistry_gradient is None or not bool(chemistry_gradient.isfinite().all()):
        raise ValueError("Missing/nonfinite chemistry latent path")
    gradient = binding_gradient + schedule["chemistry_weight"]*chemistry_gradient
    if config.valence_excess_weight:
        gradient = gradient + config.valence_excess_weight*valence_gradient
    if guidance_gradient is not None:
        gradient = gradient + guidance_gradient
    if not bool(gradient.isfinite().all()):
        raise ValueError("Nonfinite combined latent gradient")
    binding_norm = float(binding_gradient.norm())
    chemistry_norm = float(chemistry_gradient.norm())
    cosine = (float(torch.sum(binding_gradient*chemistry_gradient)) /
              (binding_norm*chemistry_norm) if binding_norm and chemistry_norm else None)
    return dict(point=point.detach(), graph=graph, binding=binding.detach(), chemistry=chemistry.detach(),
        binding_gradient=binding_gradient.detach(), chemistry_gradient=chemistry_gradient.detach(),
        gradient=gradient.detach(), native_aux=aux, repair=repair,
        repair_detail=detail, schedule=schedule, target_bundle=(target, repair),
        gradient_norms=dict(binding=binding_norm, chemistry=chemistry_norm,
            weighted_chemistry=schedule["chemistry_weight"]*chemistry_norm,
            combined=float(gradient.norm()), cosine=cosine, **valence_norms, **channel_norms), **valence_details, **extra)


def evaluate_point(decoder, scorer, point, active, config, step, *, target_bundle=None,
                   logit_offset=None):
    """Both branches consume one graph; target/weight/T stay fixed in this call."""
    hidden_path = bool(getattr(scorer, "uses_decoder_hidden", False))
    prepared = prepare_point(decoder, point, active, config, step, logit_offset=logit_offset,
                             uses_decoder_hidden=hidden_path)
    scored = (scorer.score_hidden(prepared.features, prepared.graph, active, step=step)
              if hidden_path else scorer(prepared.graph))
    return finish_point(decoder, prepared, scored, target_bundle=target_bundle)


def latent_step(point, binding_gradient, gradient, config, step, generator):
    """Bound the final displacement, preserving the two branches' relative weights."""
    ratio = schedules(config, step)["noise_ratio"]
    deterministic = -config.learning_rate*gradient
    noise = torch.zeros_like(point)
    requested = ratio*config.learning_rate*float(binding_gradient.norm())
    if ratio:
        direction = torch.randn(point.shape, generator=generator, dtype=torch.float32)
        direction = direction.to(point)
        norm = direction.norm()
        if not bool(torch.isfinite(norm)) or not norm:
            raise ValueError("Degenerate noise direction")
        noise = direction*(requested/norm)
    displacement = deterministic+noise
    norm = float(displacement.norm())
    cap = config.max_step_norm
    if config.max_step_norm_final is not None:
        progress = min(1., step / (config.schedule_horizon - 1))
        cap = (1. - progress) * cap + progress * config.max_step_norm_final
    factor = min(1., cap/norm) if norm else 1.
    proposal = point+factor*displacement
    if not bool(proposal.isfinite().all()):
        raise ValueError("Nonfinite proposed latent")
    update = dict(noise_ratio=ratio, noise_norm=float(noise.norm()),
        requested_noise_norm=requested, deterministic_norm=float(deterministic.norm()),
        requested_displacement_norm=norm, displacement_norm=float((proposal-point).norm()),
        bound_factor=factor, accepted=True, policy="one_finite_bounded_sgd_update_no_descent_gate")
    if config.max_step_norm_final is not None:
        update["max_step_norm"] = cap
    return proposal.detach(), update


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def _model_identity(model):
    h = hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RawBlocks:
    """Compact all-state labels/z; full probabilities have a declared sparse cadence."""
    def __init__(self, directory, nodes, size, coordinate_key="latent"):
        self.directory, self.size = Path(directory), size
        self.coordinate_key = coordinate_key
        self.directory.mkdir()
        self.left, self.right = np.triu_indices(nodes, 1)
        self.rows = []

    def add(self, step, point, hard):
        self.rows.append((step, point.detach().cpu().numpy().reshape(-1),
            hard.atoms[0].cpu().numpy().astype(np.uint8), hard.charges[0].cpu().numpy().astype(np.uint8),
            hard.bonds[0].cpu().numpy()[self.left,self.right].astype(np.uint8)))
        if len(self.rows) >= self.size:self.flush()

    def flush(self):
        if not self.rows:return
        p=self.directory/f"states-{self.rows[0][0]:06d}-{self.rows[-1][0]:06d}.npz"
        with p.open("xb") as f:
            np.savez_compressed(f, steps=np.array([r[0] for r in self.rows]),
                **{self.coordinate_key: np.stack([r[1] for r in self.rows])}, atoms=np.stack([r[2] for r in self.rows]),
                charges=np.stack([r[3] for r in self.rows]), bonds_upper=np.stack([r[4] for r in self.rows]),
                upper_left=self.left, upper_right=self.right)
        self.rows.clear()


def run_design(decoder, scorer_factory, config, output, *, checkpoint_sha256,
               resume=None, max_updates=None, wall_seconds=None, native_metadata=None,
               start_state=None, step_update=None):
    """Run one bounded segment; source checkpoint/decoder weights are never written."""
    if resume is not None and step_update is not None:
        raise ValueError("A custom step policy is supported only for a new bounded pilot")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    arm=output/"arms"/"latent";arm.mkdir(parents=True)
    config_hash=config.identity();device=next(decoder.parameters()).device
    if step_update is not None:
        config_hash=hashlib.sha256((config_hash+":custom-step-pilot").encode()).hexdigest()
    decoder.eval().requires_grad_(False)
    if any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting parameter gradients")
    decoder_hash=_model_identity(decoder)
    generator=torch.Generator(device="cpu").manual_seed(config.noise_seed)
    start_description=dict(kind="standard_normal",seed=config.seed)
    if resume is not None and start_state is not None:
        raise ValueError("Choose either a new warm-start arm or an existing run resume")
    if resume is None:
        if start_state is None:
            start_generator=torch.Generator(device="cpu").manual_seed(config.seed)
            point=torch.randn((1,decoder.config.latent),generator=start_generator).to(device)
            step=0
            with torch.no_grad():active=choose_active(decoder.decode(point),decoder.spec,config.occupied_atoms)
        else:
            point=start_state["latent"].detach().clone().to(device)
            active=start_state["active"].detach().clone().to(device)
            step=start_state["step"]
            if (point.shape!=(1,decoder.config.latent) or not point.is_floating_point()
                    or not bool(point.isfinite().all()) or active.dtype!=torch.bool
                    or active.shape!=(decoder.spec.max_nodes,) or int(active.sum())!=config.occupied_atoms
                    or type(step) is not int or step<0):
                raise ValueError("Invalid explicit warm-start latent, slots or schedule step")
            start_description=dict(kind=start_state.get("kind", "saved_trajectory_latent"),schedule_step=step,
                                   provenance=start_state.get("provenance",{}))
        initial=point.clone()
        trajectory_statistics=dict(updates=0,valid_sources=0,repair_witnesses=0,
                                   already_valid_sources=0,censored_sources=0,bounded_steps=0)
    else:
        state=torch.load(resume,map_location="cpu",weights_only=True)
        if (state.get("version")!=VERSION or state.get("config_sha256")!=config_hash
                or state.get("checkpoint_sha256")!=checkpoint_sha256
                or state.get("decoder_sha256")!=decoder_hash):
            raise ValueError("Resume identity differs from config/decoder/checkpoint")
        point=state["latent"].to(device);initial=state["initial_latent"].to(device)
        active=state["active"].to(device);step=state["step"]
        start_description=state.get("start_description",start_description)
        trajectory_statistics=dict(state["trajectory_statistics"])
        generator.set_state(state["noise_rng"])
        if point.shape!=(1,decoder.config.latent) or active.shape!=(decoder.spec.max_nodes,) or int(active.sum())!=config.occupied_atoms:
            raise ValueError("Resume latent/active shapes differ")
    start_step=step
    end_step=min(config.updates,step+(max_updates if max_updates is not None else config.updates))
    if not 0<=step<end_step<=config.schedule_horizon:
        raise ValueError("Empty/invalid global segment budget")
    scorer=scorer_factory(decoder.spec,active)
    native_identity=getattr(scorer,"identity",native_metadata or {})
    try:
        if resume is not None and state.get("native_identity")!=native_identity:
            raise ValueError("Resume native scorer/context identity changed")
    except BaseException:
        close=getattr(scorer,"close",None)
        if close:close()
        raise
    report_config=dict(asdict(config),start=start_description,
                       optimizer=dict(steps=config.updates,learning_rate=config.learning_rate))
    metadata=dict(version=VERSION,run_id=output.name,name=getattr(scorer,"label","VAE + Protenix latent design"),status="running",
        config=report_config,config_sha256=config_hash,
        source=dict(sha256=os.environ.get("LATENTMOL_SOURCE_SHA256"),
                    campaign_sha256=os.environ.get("LATENTMOL_CAMPAIGN_SHA256")),
        checkpoint_sha256=checkpoint_sha256,decoder_sha256=decoder_hash,graph_spec=decoder.spec.to_dict(),
        active_slots=torch.nonzero(active).flatten().cpu().tolist(),start_step=start_step,
        requested_end_step=end_step,native=native_metadata or {},native_identity=native_identity,
        occupancy_policy="fixed_top_initial_occupancy_conditioned_count; no birth/death gradient",
        gradient_scope=getattr(scorer,"gradient_scope",
            "shared conditioned decoder probabilities -> frozen Protenix contact and detached repair-target CE -> latent only"),
        chemistry_normalization="fixed_active mean per decision:2*K+K*(K-1)/2; differs from unconditional VAE training",
        probabilities_every=config.snapshot_interval,raw_state_storage="raw-blocks/*.npz stores every latent/hard graph including invalid",
        claim_limit=getattr(scorer,"claim_limit","Exploratory contact-surrogate optimization, not affinity or experimental binding"))
    _json(output/"run.json",metadata)
    started=time.monotonic();diagnostics={};update_times=[];raw=RawBlocks(arm/"raw-blocks",decoder.spec.max_nodes,100)
    scorer_closed=False
    def checkpoint():
        temporary=output/"state.tmp"
        torch.save(dict(version=VERSION,config_sha256=config_hash,checkpoint_sha256=checkpoint_sha256,
            decoder_sha256=decoder_hash,step=step,latent=point.detach().cpu(),initial_latent=initial.detach().cpu(),
            active=active.detach().cpu(),noise_rng=generator.get_state(),native_identity=native_identity,
            start_description=start_description,
            trajectory_statistics=dict(trajectory_statistics)),temporary)
        temporary.replace(output/"state.pt")
    def record(evaluation, endpoint=False):
        graph=evaluation["graph"];hard=hard_from_graph(graph)
        inspected=inspect_graph(*(getattr(hard,k)[0].detach().cpu() for k in ("atoms","charges","bonds")),decoder.spec)
        snapshot=(step==start_step or step%config.snapshot_interval==0 or endpoint)
        repair=dict(evaluation["repair_detail"],applied_weight=evaluation["schedule"]["chemistry_weight"],
                    search=evaluation["repair"])
        metric=dict(step=step,score=dict(loss=float(evaluation["binding"])+evaluation["schedule"]["chemistry_weight"]*float(evaluation["chemistry"]),
            binding_loss=float(evaluation["binding"]),repair_guidance=repair,
            gradient_norm=evaluation["gradient_norms"]["combined"],gradient_norms=evaluation["gradient_norms"]),
            selection=dict(active_slots=metadata["active_slots"],count=config.occupied_atoms),
            latent_displacement=float((point-initial).norm()),latent_norm=float(point.norm()),
            temperature=evaluation["schedule"]["temperature"],noise_ratio=evaluation["schedule"]["noise_ratio"],
            raw_valid=inspected.smiles is not None,endpoint=endpoint,native_aux=evaluation["native_aux"])
        if config.valence_excess_weight:
            value = float(evaluation["valence_excess"])
            metric["score"]["valence_excess"] = dict(loss=value, applied_weight=config.valence_excess_weight)
            metric["score"]["loss"] += config.valence_excess_weight*value
        metrics.write(json.dumps(metric,allow_nan=False)+"\n");metrics.flush()
        candidate=dict(step=step,kind="argmax",draw=0,valid=inspected.smiles is not None,
            smiles=inspected.smiles,failure=inspected.failure_reason)
        if snapshot:
            candidate.update(labels={k:getattr(hard,k)[0].detach().cpu().tolist() for k in ("atoms","charges","bonds")},graph_spec=decoder.spec.to_dict())
            capture_shared_graph(arm,step,graph,decoder.spec,active_slots=metadata["active_slots"],temperature=evaluation["schedule"]["temperature"],
                scope=("Decoded probabilities for chemistry guidance and molecule observation; binding consumes learned hidden-state projections, not these probabilities."
                       if getattr(scorer,"uses_decoder_hidden",False) else None))
            target=evaluation["target_bundle"][0]
            if target is not None:
                _json(arm/f"repair-target-{step:04d}.json",dict(step=step,target_sha256=target.identity,
                    graph_spec=decoder.spec.to_dict(),active_slots=metadata["active_slots"],
                    temperature=evaluation["schedule"]["temperature"],search=evaluation["repair"],
                    scope="Detached target only; this graph was not substituted into the binding scorer",
                    labels={k:getattr(target.graph,k)[0].tolist() for k in ("atoms","charges","bonds")}))
        candidates.write(json.dumps(candidate,allow_nan=False)+"\n");candidates.flush()
        raw.add(step,point,hard)
        return metric
    try:
        checkpoint()
        with (arm/"metrics.jsonl").open("x") as metrics,(arm/"candidates.jsonl").open("x") as candidates,(arm/"updates.jsonl").open("x") as updates:
            while step<end_step:
                if wall_seconds is not None and time.monotonic()-started>wall_seconds-120:break
                tick=time.monotonic()
                evaluation=evaluate_point(decoder,scorer,point,active,config,step)
                if step==start_step:
                    # Same z/T/target, different chemistry weight: no extra native
                    # prediction or training of an unrecorded side trajectory.
                    for weight in (1.,10.):
                        g=evaluation["binding_gradient"]+weight*evaluation["chemistry_gradient"]
                        if config.valence_excess_weight:
                            g = g + config.valence_excess_weight*evaluation["valence_excess_gradient"]
                        diagnostics[str(weight)]=dict(finite=bool(g.isfinite().all()),
                            binding_gradient_norm=evaluation["gradient_norms"]["binding"],
                            chemistry_gradient_norm=evaluation["gradient_norms"]["chemistry"],
                            weighted_chemistry_gradient_norm=weight*evaluation["gradient_norms"]["chemistry"],
                            combined_gradient_norm=float(g.norm()),target_status=evaluation["repair"]["status"],
                            warning="zero repair guidance at this point" if not evaluation["gradient_norms"]["chemistry"] else None)
                    _json(output/"weight-probes.json",diagnostics)
                    if os.environ.get("DBMOL_LAUNCH_READY"):
                        _json(Path(os.environ["DBMOL_LAUNCH_READY"]),dict(status="models_and_inputs_ready",
                            scope="native_and_decoder_gradient_ready",nonce=os.environ.get("DBMOL_LAUNCH_NONCE"),
                            job_id=os.environ.get("LSB_JOBID")))
                metric=record(evaluation)
                if step_update is None:
                    next_point,update=latent_step(point,evaluation["binding_gradient"],evaluation["gradient"],config,step,generator)
                else:
                    next_point,update=step_update(decoder,scorer,evaluation,active,config,step)
                update.update(step=step,source_loss=metric["score"]["loss"])
                updates.write(json.dumps(update,allow_nan=False)+"\n");updates.flush()
                trajectory_statistics["updates"]+=1
                trajectory_statistics["valid_sources"]+=int(metric["raw_valid"])
                trajectory_statistics["repair_witnesses"]+=int(evaluation["repair"]["status"]=="witness_found")
                trajectory_statistics["already_valid_sources"]+=int(evaluation["repair"]["status"]=="already_valid")
                trajectory_statistics["censored_sources"]+=int(evaluation["repair"]["status"] in ("censored","unsupported"))
                trajectory_statistics["bounded_steps"]+=int(update["bound_factor"]<1)
                point=next_point;step+=1;update_times.append(time.monotonic()-tick)
                if step%config.checkpoint_interval==0:checkpoint();raw.flush()
                if step<=start_step+3 or step%100==0:
                    print(json.dumps(dict(step=step,binding_loss=metric["score"]["binding_loss"],
                        chemistry_loss=float(evaluation["chemistry"]),valid=metric["raw_valid"],seconds=update_times[-1])),flush=True)
            final=evaluate_point(decoder,scorer,point,active,config,step)
            final_metric=record(final,endpoint=True)
        raw.flush();checkpoint()
        if _model_identity(decoder)!=decoder_hash or any(p.grad is not None for p in decoder.parameters()):
            raise ValueError("Frozen decoder parameters/gradients changed")
        # Native close checks frozen weights/context. Complete it BEFORE a
        # success receipt exists, so a frozen-state failure cannot look passed.
        close=getattr(scorer,"close",None)
        if close:close()
        scorer_closed=True
        result=dict(status="completed" if step==config.updates else "segment_completed",version=VERSION,
            start_step=start_step,completed_step=step,updates_this_segment=step-start_step,
            config_sha256=config_hash,checkpoint_sha256=checkpoint_sha256,
            technical_checks=dict(finite_required_binding_path=True,finite_chemistry_path=True,
                frozen_decoder_unchanged=True,frozen_native_close_completed=True,
                same_shared_graph_and_temperature=not bool(getattr(scorer,"uses_decoder_hidden",False)),
                decoder_hidden_mapping_path=bool(getattr(scorer,"uses_decoder_hidden",False))),
            weight_probes=diagnostics,elapsed_seconds=time.monotonic()-started,
            mean_update_seconds=sum(update_times)/len(update_times) if update_times else None,
            warm_mean_update_seconds=sum(update_times[3:])/len(update_times[3:]) if len(update_times)>3 else None,
            state=str(output/"state.pt"),final=final_metric,trajectory_statistics=dict(trajectory_statistics))
        _json(output/"result.json",result);metadata.update(status=result["status"],completed_step=step);_json(output/"run.json",metadata)
        # Presentation consumes saved JSON only. A report failure has its own
        # receipt and cannot change the scientific outcome or rerun a model.
        from latentmol.reporting.optimization import default_report
        default_report(output)
        return result
    except BaseException as exc:
        raw.flush();checkpoint()
        _json(output/"failure.json",dict(status="failed",error=type(exc).__name__+": "+str(exc),step=step))
        metadata.update(status="failed",error=type(exc).__name__+": "+str(exc),completed_step=step)
        _json(output/"run.json",metadata)
        from latentmol.reporting.optimization import default_report
        default_report(output)
        raise
    finally:
        close=getattr(scorer,"close",None)
        if close and not scorer_closed:close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--checkpoint-sha256");p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--threads",type=int,default=4)
    p.add_argument("--mosaic-python",required=True);p.add_argument("--mosaic-config",type=Path,required=True)
    p.add_argument("--resume",type=Path);p.add_argument("--max-updates",type=int);p.add_argument("--wall-seconds",type=float)
    a=p.parse_args(argv);torch.set_num_threads(a.threads)
    config=LatentDesignConfig.from_mapping(json.loads(a.config.read_text()))
    identity=file_sha256(a.checkpoint)
    if a.checkpoint_sha256 and identity!=a.checkpoint_sha256:raise ValueError("Decoder checkpoint SHA differs")
    decoder,_=load_model(a.checkpoint,a.device)
    from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
    native_config=json.loads(a.mosaic_config.read_text())
    def factory(spec,active):
        return MosaicProtenixScorer(spec,active,
            command=[a.mosaic_python,"-m","latentmol.guidance.mosaic_protenix_worker"],
            config=native_config,stderr_path=a.output/"native.stderr.log",timeout=900,startup_timeout=900)
    return run_design(decoder,factory,config,a.output,checkpoint_sha256=identity,resume=a.resume,
        max_updates=a.max_updates,wall_seconds=a.wall_seconds,
        native_metadata=dict(config_path=str(a.mosaic_config),config_sha256=file_sha256(a.mosaic_config)))


if __name__=="__main__":
    main()
