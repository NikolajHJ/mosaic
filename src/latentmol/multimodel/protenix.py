"""Experimental Protenix-v1 distogram branch; no affinity or diffusion claim.

The upstream ``get_pairformer_output`` uses a training-dependent grad context
which disables trunk gradients in eval mode. This module spells out its frozen
eval equations instead: no global grad monkeypatch, no model.train(), no
diffusion or confidence-head bypass presented as an affinity score.

Source audit: bytedance/Protenix, ``protenix/model/protenix.py`` and
``protenix/model/modules/embedders.py``, 2026-10-01. Native checkpoint, source
and prepared-input hashes are mandatory; native execution is still untested.
"""

from __future__ import annotations

import json

import torch
from torch import Tensor, nn

from latentmol.representation import GraphSpec, RelaxedGraph
from latentmol.scoring import BranchCapabilities, BranchManifest, InputSupport, require_frozen_eval
from .external import NativeBundle, checked_import, freeze, tree_to, verify_native_assets


def differentiable_pairformer(model: nn.Module, features: dict, *, cycles: int,
                             chunk_size: int | None = None) -> Tensor:
    """Mirror the native no-inplace eval trunk, keeping every recycle attached.

    The forward operations match ``get_pairformer_output(..., mc_dropout=False,
    inplace_safe=False)``. Input gradients include all recycling iterations;
    this costs more memory than inference. Native parity/finite differences
    must pass before its derivative is used in a scientific experiment.
    """
    if model.train_confidence_only:
        raise ValueError("Confidence-only training configurations are unsupported")
    if cycles < 1:
        raise ValueError("Protenix cycles must be positive")
    s_inputs = model.input_embedder(features, inplace_safe=False, chunk_size=chunk_size)
    s_init = model.linear_no_bias_sinit(s_inputs)
    z_init = model.linear_no_bias_zinit1(s_init)[..., None, :] + model.linear_no_bias_zinit2(s_init)[..., None, :, :]
    z_init = z_init + model.relative_position_encoding(features["relp"])
    z_init = z_init + model.linear_no_bias_token_bond(features["token_bonds"].unsqueeze(-1))
    if "constraint_feature" in features:
        z_init = z_init + model.constraint_embedder(features["constraint_feature"])
    z, s = torch.zeros_like(z_init), torch.zeros_like(s_init)
    kwargs = dict(triangle_multiplicative=model.configs.triangle_multiplicative,
                  triangle_attention=model.configs.triangle_attention,
                  inplace_safe=False, chunk_size=chunk_size)
    for _ in range(cycles):
        z = z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
        if model.template_embedder.n_blocks > 0:
            z = z + model.template_embedder(features, z, **kwargs)
        z = model.msa_module(features, z, s_inputs, pair_mask=None, **kwargs)
        s = s_init + model.linear_no_bias_s(model.layernorm_s(s))
        s, z = model.pairformer_stack(s, z, pair_mask=None, **kwargs)
    return z


class ProtenixBranch(nn.Module):
    """Fixed-pocket distance consistency, deliberately distinct from affinity."""

    def __init__(self, model: nn.Module, bundle: NativeBundle, manifest: BranchManifest,
                 *, cycles: int, chunk_size: int | None = None):
        super().__init__()
        self.model, self.bundle, self.spec, self.manifest = freeze(model), bundle, bundle.spec, manifest
        self.cycles, self.chunk_size = cycles, chunk_size
        if cycles < 1 or (chunk_size is not None and chunk_size < 1):
            raise ValueError("Protenix cycles/chunk_size must be positive")
        if model.train_confidence_only:
            raise ValueError("Confidence-only Protenix configurations are unsupported")
        if not bundle.metadata.get("distance_target_source"):
            raise ValueError("Protenix bundle must identify the fixed reference distance_target_source")
        mask, bins = bundle.extra["distance_pair_mask"], bundle.extra["distance_target_bins"]
        t = bundle.ligand_token_mask.numel()
        if mask.shape != (t, t) or mask.dtype != torch.bool or not mask.any():
            raise ValueError("Provide a nonempty bool[T,T] distance_pair_mask")
        if bins.shape != (t, t) or bins.dtype != torch.long or (bins[mask] < 0).any():
            raise ValueError("Provide nonnegative long[T,T] native distance target bins")
        ligand = bundle.ligand_token_mask
        if (mask & ~(ligand[:, None] ^ ligand[None, :])).any():
            raise ValueError("Distance objective selects only ligand-to-context token pairs")
        # All nonligand context must be protein for this first adapter; native
        # DNA/RNA/covalent multimers need an explicit extension of the contract.
        if "protein_token_mask" not in bundle.extra:
            raise ValueError("Protenix bundle must contain prepared protein_token_mask")
        protein = bundle.extra["protein_token_mask"]
        if protein.shape != (t,) or protein.dtype != torch.bool or (protein & ligand).any() or (mask & ~(ligand[:, None] & protein[None, :] | protein[:, None] & ligand[None, :])).any():
            raise ValueError("Distance mask must select ligand-protein pairs only")
        expected_protein = torch.zeros_like(protein)
        native_protein_atoms = bundle.features["is_protein"].bool()
        atom_tokens = bundle.features["atom_to_token_idx"].long()
        if native_protein_atoms.shape != atom_tokens.shape:
            raise ValueError("Native is_protein must label atoms, not tokens")
        expected_protein[atom_tokens[native_protein_atoms]] = True
        if not torch.equal(expected_protein, protein):
            raise ValueError("protein_token_mask disagrees with native protein atom labels")
        self.register_buffer("pair_mask", mask)
        self.register_buffer("target_bins", bins)
        bundle.assert_seed_features(charge=True)
        self.losses = (
            "Native token_bonds retain only bond presence; bond order/aromaticity are not represented",
            "Reference conformer, atom names, tokenization, pocket and target distance bins remain fixed",
            "Canonical graph excludes stereochemistry; distogram consistency is not affinity or validity",
        )
        freeze(self)

    def assess(self, graph: RelaxedGraph) -> InputSupport:
        return self.bundle.assess(graph, self.losses)

    def _reduce(self, logits: Tensor) -> dict[str, Tensor]:
        """Average CE per selected pair and normalize by log(number of bins).

        The normalization makes uniform predictions equal one; the loss has no
        finite upper bound and is not a probability. Target bins specify an
        existing reference geometry rather than rewarding every short contact.
        """
        if logits.ndim != 3 or logits.shape[:2] != self.pair_mask.shape:
            raise ValueError("Expected unbatched native [T,T,K] distogram logits")
        if logits.shape[-1] < 2 or (self.target_bins[self.pair_mask] >= logits.shape[-1]).any():
            raise ValueError("Distance target bins exceed native head vocabulary")
        logp = logits[self.pair_mask].float().log_softmax(-1)
        target = self.target_bins[self.pair_mask]
        loss = -logp.gather(-1, target[:, None]).mean() / torch.log(logp.new_tensor(logp.shape[-1]))
        return {"distance_consistency_loss": loss.reshape(1),
                "distogram_entropy": (-(logp.exp() * logp).sum(-1).mean()).reshape(1)}

    def forward(self, graph: RelaxedGraph) -> dict[str, Tensor]:
        require_frozen_eval(self)
        support = self.assess(graph)
        if support.status == "unsupported":
            raise ValueError("; ".join(support.reasons))
        features = self.bundle.soft_features(graph, charge=True)
        z = differentiable_pairformer(self.model, features, cycles=self.cycles, chunk_size=self.chunk_size)
        return self._reduce(self.model.distogram_head(z))

    def native_reference(self) -> dict[str, Tensor]:
        """Native fixed seed trunk/head, without running unnecessary diffusion."""
        with torch.no_grad():
            features = tree_to(self.bundle.features, self.pair_mask.device)
            _, _, z = self.model.get_pairformer_output(features, N_cycle=self.cycles,
                                                       inplace_safe=False, chunk_size=self.chunk_size,
                                                       mc_dropout=False)
            return self._reduce(self.model.distogram_head(z))


def load_protenix(*, spec: GraphSpec, device: str, options: dict, source_sha256: str):
    """Load local v1 weights/config and prepared native features, never download."""
    from .runtime import BranchRuntime
    assets = verify_native_assets(options, "protenix")
    native = checked_import("protenix.model.protenix", "protenix", assets)
    try:
        from ml_collections import ConfigDict
    except ImportError as exc:
        raise ImportError("Protenix's native environment requires ml_collections") from exc
    with open(assets["model_config_path"], encoding="utf-8") as handle:
        config = json.load(handle)
    # v2 has different weight terms and is outside this adapter's audited scope.
    if "v1" not in config.get("model_name", "").lower() or "v2" in config["model_name"].lower():
        raise ValueError("This adapter supports explicitly identified Protenix-v1 configurations only")
    if config.get("train_confidence_only", False):
        raise ValueError("Protenix train_confidence_only must be false")
    # The bundled config must already select upstream differentiable torch
    # kernels; no hidden edits to external configuration or precision policy.
    if config.get("triangle_attention") != "torch" or config.get("triangle_multiplicative") != "torch":
        raise ValueError("Initial Protenix gradient experiments require torch triangle kernels")
    model = native.Protenix(ConfigDict(config))
    checkpoint = torch.load(assets["weights_path"], map_location="cpu", weights_only=True)
    if "model" not in checkpoint:
        raise ValueError("Expected native Protenix checkpoint['model'] state dict")
    state = checkpoint["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = freeze(model.to(device))
    bundle = NativeBundle.load(assets["bundle_path"], spec=spec, model="protenix_v1", device=device)
    cycles = int(options.get("cycles", config["model"]["N_cycle"]))
    chunk = options.get("chunk_size")
    manifest = BranchManifest(
        name="protenix_v1", graph_spec_sha256=spec.fingerprint(),
        model_source_version=assets["native_source_sha256"], model_weights_sha256=assets["weights_sha256"],
        adapter_version="protenix-distogram-experimental-v1", adapter_source_sha256=source_sha256,
        context_sha256=assets["bundle_sha256"],
        capabilities=BranchCapabilities(
            input_kind="relaxed_graph", occupancy="fixed", gradient_mode="partial",
            differentiable_channels=("atoms", "charges", "bonds"),
            supported_atomic_numbers=spec.atomic_numbers[1:], supported_formal_charges=spec.formal_charges,
            supported_bond_names=spec.bond_names,
            context_policy="Fixed reference conformer, names, tokenization, MSA, pocket and distance targets; no diffusion",
            relaxation_policy="Expected element and charge features plus bond presence; all trunk recycles attached; typed bonds lost"),
        settings=(("status", "experimental_native_unvalidated"), ("cycles", str(cycles)),
                  ("chunk_size", str(chunk)), ("model_config_sha256", assets["model_config_sha256"]),
                  ("objective", "structural_distance_consistency_not_affinity")))
    branch = ProtenixBranch(model, bundle, manifest, cycles=cycles, chunk_size=chunk)
    return BranchRuntime(branch=branch, seed=bundle.seed, active=bundle.active,
                         native_reference=branch.native_reference, assets=assets,
                         verify=bundle.assert_unchanged)
