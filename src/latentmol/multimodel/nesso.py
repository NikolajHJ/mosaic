"""Experimental Nesso-1 fixed-context affinity branch; native validation pending.

Source audit: recursionpharma/nesso, ``nesso/model/models/nesso1.py`` and
``nesso/model/modules/encoders.py``, 2026-10-01. The ordinary forward detaches
trunk inputs to its affinity heads and selects a pocket with hard thresholds.
Here the pretrained modules stay in eval mode, the pocket is prepared once,
and continuous chemistry reaches both affinity heads without those detaches.
This is a partial, fixed-context derivative, not differentiation of recropping.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from latentmol.representation import GraphSpec, RelaxedGraph
from latentmol.scoring import BranchCapabilities, BranchManifest, InputSupport, require_frozen_eval
from .external import NativeBundle, checked_import, freeze, tree_to, verify_native_assets

OUTPUTS = ("affinity_pred_value", "affinity_logits_binary", "affinity_probability_binary")


class NessoBranch(nn.Module):
    """One frozen native model plus a deterministic graph-to-feature adapter."""

    def __init__(self, model: nn.Module, bundle: NativeBundle, manifest: BranchManifest,
                 *, recycling_steps: int, bond_ids: Tensor):
        super().__init__()
        self.model, self.bundle, self.spec, self.manifest = freeze(model), bundle, bundle.spec, manifest
        self.recycling_steps = recycling_steps
        self.register_buffer("bond_ids", bond_ids)
        self.register_buffer("keep", bundle.extra["affinity_keep_indices"])
        self.charge_enabled = bool(model.input_embedder.atom_encoder.add_additional_atom_features)
        if recycling_steps < 0:
            raise ValueError("Nesso recycling_steps must be nonnegative")
        if not model.affinity_prediction:
            raise ValueError("Nesso checkpoint must contain both pretrained affinity heads")
        t = bundle.ligand_token_mask.numel()
        if self.keep.ndim != 1 or self.keep.dtype != torch.long or not self.keep.numel() or self.keep.unique().numel() != self.keep.numel() or (self.keep < 0).any() or (self.keep >= t).any():
            raise ValueError("Nesso requires distinct bounded affinity_keep_indices from the native seed forward")
        if not torch.isin(bundle.slot_to_token[bundle.active], self.keep).all():
            raise ValueError("Affinity crop must retain every canonical ligand token")
        if not (bundle.features["mol_type"][0, self.keep] == 0).any():
            raise ValueError("Affinity crop needs protein context")
        if bond_ids.shape != (len(self.spec.bond_names),):
            raise ValueError("Nesso typed-bond vocabulary does not match canonical graph")
        bundle.assert_seed_features(charge=self.charge_enabled)
        native_types = bundle.features["type_bonds"]
        ti = bundle.slot_to_token[bundle.active]
        seed_types = bond_ids[bundle.seed.bonds[0, bundle.active][:, bundle.active]]
        if not torch.equal(native_types[0, ti[:, None], ti[None, :]], seed_types):
            raise ValueError("Canonical seed does not reproduce native typed bonds")
        self.losses = ("Fixed reference conformer and atom-name features are retained as chemistry changes",)
        if self.charge_enabled:
            self.losses += ("Native chirality and hybridization stay fixed; canonical graph omits stereo",)
        else:
            self.losses += ("This Nesso checkpoint ignores formal charge; no charge gradient exists",)
        freeze(self)

    def assess(self, graph: RelaxedGraph) -> InputSupport:
        return self.bundle.assess(graph, self.losses)

    def _forward_features(self, graph: RelaxedGraph) -> dict[str, Tensor]:
        """Explicit native eval equations, fixed pocket, expected bond embeddings.

        Unlike the upstream inference wrapper, this method keeps the s/z and
        distogram paths attached. No native module is put into training mode.
        The 22-Angstrom first-cycle refinement crop is disabled in BOTH paths.
        """
        model, b = self.model, self.bundle
        f = b.soft_features(graph, charge=self.charge_enabled)
        s = model.input_embedder(f)
        z_init = model.z_init_1(s)[:, :, None] + model.z_init_2(s)[:, None, :]
        z_init = z_init + model.rel_pos(f) + model.token_bonds(f["token_bonds"].float())
        # Keep native protein/intermolecular entries; only ligand-ligand bond
        # categories are replaced by their exact expected embedding.
        typed = model.token_bonds_type(f["type_bonds"].long()).clone()
        ti = b.slot_to_token[b.active]
        values = graph.bonds[0, b.active][:, b.active] @ model.token_bonds_type.weight[self.bond_ids]
        typed[0, ti[:, None], ti[None, :]] = values
        z_init = z_init + typed
        z = torch.zeros_like(z_init)
        mask = f["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        for _ in range(self.recycling_steps + 1):
            z = z_init + model.z_recycle(model.z_norm(z))
            z = model.esm_module(z, s_inputs=s, s_esm=f["s_esm"], pair_mask=pair_mask, use_kernels=False)
            z = model.pairformer_module(z, pair_mask=pair_mask, use_kernels=False)
        logits = model.distogram_head(z + z.transpose(1, 2))
        keep = self.keep
        sf, zf = s[:, keep], z[:, keep][:, :, keep]
        pf = logits[:, keep][:, :, keep].softmax(-1)
        ff = dict(f)
        for key in ("mol_type", "token_pad_mask", "s_esm", "affinity_token_mask", "asym_id", "residue_index"):
            if key in ff:
                ff[key] = ff[key][:, keep].contiguous()
        # Upstream heads explicitly run fp32. No autocast policy is guessed.
        with torch.autocast(device_type=sf.device.type, enabled=False):
            one = model.affinity_module(s_inputs=sf.float(), z=zf.float(), pdistogram=pf.float(), feats=ff, use_kernels=False)
            two = model.affinity_module2(s_inputs=sf.float(), z=zf.float(), pdistogram=pf.float(), feats=ff, use_kernels=False)
        probability = (one["affinity_logits_binary"].sigmoid() + two["affinity_logits_binary"].sigmoid()) / 2
        return {"affinity_pred_value": (one["affinity_pred_value"] + two["affinity_pred_value"]) / 2,
                "affinity_probability_binary": probability,
                "affinity_logits_binary": torch.logit(probability.clamp(1e-6, 1 - 1e-6))}

    def forward(self, graph: RelaxedGraph) -> dict[str, Tensor]:
        require_frozen_eval(self)
        support = self.assess(graph)
        if support.status == "unsupported":
            raise ValueError("; ".join(support.reasons))
        return self._forward_features(graph)

    def native_reference(self) -> dict[str, Tensor]:
        """Ordinary seed forward with native discrete inputs, for parity only.

        Dynamic native pocket selection must agree with the saved fixed crop.
        This is not a re-featurization or fresh native rescore of changed graphs.
        """
        with torch.no_grad():
            out = self.model(tree_to(self.bundle.features, self.keep.device),
                             recycling_steps=self.recycling_steps, refine_protein_inference=False)
        if not torch.equal(out["affinity_keep_indices"], self.keep):
            raise ValueError("Prepared affinity crop differs from current native seed forward")
        return {key: out[key] for key in OUTPUTS}


def load_nesso(*, spec: GraphSpec, device: str, options: dict, source_sha256: str):
    """Load verified local native assets; real forward/backward remains unvalidated."""
    from .runtime import BranchRuntime
    assets = verify_native_assets(options, "nesso")
    native = checked_import("nesso.model.models.nesso1", "nesso", assets)
    const = checked_import("nesso.data.const", "nesso", assets)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Nesso's native environment requires safetensors") from exc
    config = json.loads(Path(assets["model_config_path"]).read_text(encoding="utf-8"))
    # The numerically matching native reference also disables custom kernels.
    config["use_kernels"] = False
    model = native.Nesso1(**config)
    model.load_state_dict(load_file(assets["weights_path"], device="cpu"), strict=True)
    model = freeze(model.to(device))
    bundle = NativeBundle.load(assets["bundle_path"], spec=spec, model="nesso1", device=device)
    recycling = int(options.get("recycling_steps", 0))
    charge = bool(model.input_embedder.atom_encoder.add_additional_atom_features)
    bond_ids = torch.tensor([0 if name == "NONE" else const.bond_type_ids[name] + 1
                             for name in spec.bond_names], device=device)
    manifest = BranchManifest(
        name="nesso1", graph_spec_sha256=spec.fingerprint(),
        model_source_version=assets["native_source_sha256"], model_weights_sha256=assets["weights_sha256"],
        adapter_version="nesso-fixed-context-experimental-v1", adapter_source_sha256=source_sha256,
        context_sha256=assets["bundle_sha256"],
        capabilities=BranchCapabilities(
            input_kind="relaxed_graph", occupancy="fixed", gradient_mode="partial",
            differentiable_channels=("atoms", "bonds", "charges") if charge else ("atoms", "bonds"),
            supported_atomic_numbers=spec.atomic_numbers[1:], supported_formal_charges=spec.formal_charges,
            supported_bond_names=spec.bond_names,
            context_policy="Fixed native conformer, atom names, tokenization, ESM and affinity pocket; no 22A refinement crop",
            relaxation_policy="Expected element/typed-bond embeddings; charge expectation only if consumed natively; full attached trunk within fixed context"),
        settings=(("status", "experimental_native_unvalidated"), ("recycling_steps", str(recycling)),
                  ("use_kernels", "false"), ("model_config_sha256", assets["model_config_sha256"])))
    branch = NessoBranch(model, bundle, manifest, recycling_steps=recycling, bond_ids=bond_ids)
    return BranchRuntime(branch=branch, seed=bundle.seed, active=bundle.active,
                         native_reference=branch.native_reference, assets=assets,
                         verify=bundle.assert_unchanged)
