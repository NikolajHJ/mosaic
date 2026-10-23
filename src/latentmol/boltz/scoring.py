"""Additive shared-graph facade over the existing frozen fixed-context Boltz path.

Existing pilots retain their direct adapter/scorer calls. This facade performs
the same calls, with explicit eligibility and provenance for branch composition.
It adds neither native featurization nor diffusion nor a new affinity predictor.
"""

import torch
from torch import nn

from latentmol.representation import RelaxedGraph
from latentmol.scoring import (BranchCapabilities, BranchManifest, InputSupport,
                               require_frozen_eval)
from .adapter import adapt_graph
from .contracts import AFFINITY_SHA256, BOLTZ_COMMIT, PreparedComplex
from .provenance import context_digest


def boltz_branch_manifest(prepared: PreparedComplex, *, adapter_source_sha256: str,
                          recycling_steps: int,
                          validation_report_sha256: str | None = None,
                          dbmol_profile: dict | None = None) -> BranchManifest:
    """Describe pinned assets; the existing loader verifies their actual bytes.

    Context identity is the tensor/context digest, not the prepared-file digest.
    Include the complete adapter/facade source in adapter_source_sha256, typically
    the immutable release hash. Validation evidence remains optional, not implied.
    """
    if recycling_steps < 0:
        raise ValueError("Recycling steps must be nonnegative")
    spec = prepared.spec
    name, checkpoint, settings = "boltz_fixed_context", AFFINITY_SHA256, (("recycling_steps", str(recycling_steps)),)
    context_policy = "Fixed pose, native auxiliary features, MSA and complete ligand slot map; no geometry or occupancy gradients."
    relaxation_policy = "Expected native element/charge/bond features; soft formal-charge distributions lose information when reduced to their mean; no molecular-weight correction; exact native seed-feature mapping only, not fresh-pose parity for changed molecules."
    if dbmol_profile is not None:
        from latentmol.multimodel.config import digest
        from .dbmol_objective import PROFILE
        if dbmol_profile.get("profile") != PROFILE:
            raise ValueError("Unsupported Boltz loss profile")
        name = "boltz_dbmol_fixed_context"
        # A combined checkpoint identity, not the SHA of either individual file.
        checkpoint = digest({"affinity_checkpoint_sha256": AFFINITY_SHA256,
                             "confidence_checkpoint_sha256": dbmol_profile["confidence_checkpoint"]["sha256"]})
        settings += (("loss_profile", PROFILE), ("dbmol_profile_sha256", digest(dbmol_profile)))
        context_policy += " Contact pocket uses frozen native protein residue identities; both checkpoints receive the same graph/pose."
        relaxation_policy += " Affinity and structure/confidence use their own frozen pretrained trunks; diffusion is bypassed, confidence inputs stay attached, and typed bonds use expected embeddings."
    return BranchManifest(
        name=name, graph_spec_sha256=spec.fingerprint(),
        model_source_version=BOLTZ_COMMIT, model_weights_sha256=checkpoint,
        adapter_version="fixed-graph-v1", adapter_source_sha256=adapter_source_sha256,
        context_sha256=context_digest(prepared),
        validation_report_sha256=validation_report_sha256,
        settings=settings,
        capabilities=BranchCapabilities(
            input_kind="relaxed_graph", occupancy="fixed", gradient_mode="partial",
            differentiable_channels=("atoms", "charges", "bonds"),
            supported_atomic_numbers=spec.atomic_numbers[1:],
            supported_formal_charges=spec.formal_charges,
            supported_bond_names=spec.bond_names,
            context_policy=context_policy,
            relaxation_policy=relaxation_policy))


class BoltzGraphBranch(nn.Module):
    """Expose named native heads while retaining gradients to the shared graph.

    Pass an already weight-frozen FrozenBoltzAffinity from the verified loader.
    The constructor sets eval mode but refuses trainable parameters. A compatible
    mock nn.Module can exercise the boundary without installing native Boltz.
    Manifest hashes are provenance assertions; this facade cannot authenticate
    arbitrary injected scorers. Production callers must construct the scorer via
    verified load_affinity_model and FrozenBoltzAffinity, then retain that audit.
    Soft charge distributions require ``allow_lossy_charge=True``; their mean is
    all that native Boltz receives. Composition independently requires its generic
    ``allow_lossy=True`` opt-in and retains this declared loss of information.
    PreparedComplex contains mutable tensors. Call verify_context_unchanged()
    before and after an optimization run; full context hashing is deliberately
    absent from the hot forward path.
    """

    def __init__(self, scorer: nn.Module, prepared: PreparedComplex,
                 manifest: BranchManifest, *, allow_lossy_charge: bool = False):
        super().__init__()
        prepared.validate()
        self.spec, self.prepared, self.manifest = prepared.spec, prepared, manifest
        self.allow_lossy_charge = allow_lossy_charge
        expected = boltz_branch_manifest(
            prepared, adapter_source_sha256=manifest.adapter_source_sha256,
            recycling_steps=scorer.recycling_steps,
            validation_report_sha256=manifest.validation_report_sha256,
            dbmol_profile=getattr(scorer, "manifest_config", None))
        if manifest != expected:
            raise ValueError("Boltz branch manifest differs from context, schema or scorer configuration")
        self._context_sha256 = expected.context_sha256
        if any(parameter.requires_grad for parameter in scorer.parameters()):
            raise ValueError("External scoring parameters must be frozen")
        self.scorer = scorer
        self.eval()
        require_frozen_eval(self)

    def verify_context_unchanged(self) -> str:
        """Check the captured tensor context and schema at explicit run boundaries."""
        if (self.manifest.context_sha256 != self._context_sha256
                or self.prepared.spec.fingerprint() != self.manifest.graph_spec_sha256
                or context_digest(self.prepared) != self._context_sha256):
            raise ValueError("Boltz prepared context changed after branch construction")
        if getattr(self.scorer, "manifest_config", None) is not None:
            current = boltz_branch_manifest(self.prepared,
                adapter_source_sha256=self.manifest.adapter_source_sha256,
                recycling_steps=self.scorer.recycling_steps,
                validation_report_sha256=self.manifest.validation_report_sha256,
                dbmol_profile=self.scorer.manifest_config)
            if current != self.manifest:
                raise ValueError("DBMol checkpoint, pocket or loss profile changed")
        return self._context_sha256

    def assess(self, graph: RelaxedGraph) -> InputSupport:
        """Fail closed instead of silently conditioning or repairing a graph."""
        spec, active = self.spec, self.prepared.active
        shapes = ((1, spec.max_nodes, len(spec.atomic_numbers)),
                  (1, spec.max_nodes, len(spec.formal_charges)),
                  (1, spec.max_nodes, spec.max_nodes, len(spec.bond_names)))
        values = (graph.atoms, graph.charges, graph.bonds)
        if any(tuple(value.shape) != shape for value, shape in zip(values, shapes)):
            return InputSupport("unsupported", ("Require one graph with the declared schema dimensions",))
        if any(value.device != active.device or not value.is_floating_point() for value in values):
            return InputSupport("unsupported", ("Graph probabilities and prepared context require one device and floating tensors",))
        if any(not value.isfinite().all() or (value < 0).any()
               or not torch.allclose(value.sum(-1), torch.ones_like(value[..., 0]), atol=1e-6, rtol=1e-5)
               for value in values):
            return InputSupport("unsupported", ("Require finite nonnegative normalized categorical probabilities",))
        if not torch.allclose(graph.occupancy[0], active.to(graph.atoms), atol=1e-6, rtol=0):
            return InputSupport("unsupported", ("Boltz requires the exact prepared occupancy; apply the shared graph policy first",))
        diagonal = torch.eye(spec.max_nodes, device=active.device, dtype=torch.bool)
        masked = ~(active[:, None] & active[None, :]) | diagonal
        if (not torch.allclose(graph.bonds, graph.bonds.transpose(1, 2), atol=1e-6, rtol=0)
                or bool(graph.bonds[0][masked][:, 1:].any())
                or not torch.all(graph.charges[0, ~active, spec.formal_charges.index(0)] == 1)):
            return InputSupport("unsupported", ("Require symmetric bonds, NONE on absent/diagonal pairs, and neutral padding",))
        active_charge = graph.charges[:, active]
        if not bool(((active_charge == 0) | (active_charge == 1)).all()):
            return InputSupport("lossy", ("Native Boltz receives only expected formal charge, not its categorical distribution",))
        return InputSupport("exact")

    def forward(self, graph: RelaxedGraph):
        require_frozen_eval(self)
        support = self.assess(graph)
        if support.status == "unsupported":
            raise ValueError("Unsupported Boltz input: " + "; ".join(support.reasons))
        if support.status == "lossy" and not self.allow_lossy_charge:
            raise ValueError("Lossy Boltz charge input requires allow_lossy_charge=True")
        features, bond_types = adapt_graph(graph, self.prepared)
        return self.scorer(features, bond_types, self.prepared.coordinates)
