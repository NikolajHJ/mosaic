"""Two frozen native Boltz trunks and the explicit fixed-pose DBMol objective.

This is a direct probability-graph boundary, independent of DBMol's VAE or
optimization code. The reference implementation was inspected, never imported.
Coordinates, occupancy, pocket, MSA and every nonchemical feature stay fixed.
Whole-path raw finite differences remain a separate mandatory allocated gate.
"""

from dataclasses import asdict
from functools import wraps
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import native

PROFILE = "mosaic-dbmol-fixed-jq1-v1"
PROTOCOL = "mosaic-boltz-dbmol-v1"
STRUCTURE_SHA256 = "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"
POCKET_SHA256 = "9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43"
ARITHMETIC = "contact_head_and_assembly_float64_v1"
CONFIDENCE_WEIGHTS = {
    "iptm": -1.0,
    "complex_iplddt": -0.3,
    "complex_ipde": 0.3,
    "complex_pde": 0.1,
    "confidence_score": -0.3,
    "ptm": -0.1,
}
COMPONENTS = (
    "dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss",
    "dbmol_total_loss",
)
METRICS = (
    "dbmol_total_loss", "dbmol_affinity_loss", "dbmol_contact_loss",
    "dbmol_confidence_loss", "dbmol_weighted_affinity_loss",
    "dbmol_weighted_contact_loss", "dbmol_weighted_confidence_loss",
    "affinity_pred_value", "affinity_probability_binary", "affinity_logits_binary",
    "dbmol_contact_pocket_loss", "dbmol_contact_anchor_loss", "complex_plddt",
    *CONFIDENCE_WEIGHTS,
)
OBJECTIVES = tuple(key for key in METRICS if key != "affinity_probability_binary")


def validate_graph(x, edges, context):
    """Require the exact occupied neutral FP32 25-slot categorical boundary."""
    if (
        x.dtype != torch.float32 or edges.dtype != torch.float32
        or x.shape != (25, 6) or edges.shape != (300, 5)
        or x.device != edges.device or x.device != context["coordinates"].device
        or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(edges).all())
    ):
        raise ValueError("DBMol requires finite same-device FP32 X[25,6]/E[300,5]")
    for value in (x, edges):
        if bool((value < 0).any()) or bool((value > 1).any()):
            raise ValueError("DBMol categorical probabilities leave the simplex")
        torch.testing.assert_close(
            value.sum(-1), torch.ones_like(value[:, 0]), atol=2e-6, rtol=0,
        )


def one_distogram(logits):
    if logits.ndim == 5 and logits.shape[3] == 1:
        return logits[:, :, :, 0]
    if logits.ndim == 4:
        return logits
    raise ValueError("Only the single native distogram is supported")


def native_bin_centers(logits):
    """Define the bin partition in native FP32 before contact-only widening."""
    if logits.shape[-1] != 64 or logits.dtype != torch.float32:
        raise ValueError("Native distogram must contain 64 FP32 bins")
    boundaries = torch.linspace(2.0, 22.0, 63, device=logits.device, dtype=logits.dtype)
    half = (boundaries[1] - boundaries[0]) / 2
    return torch.cat((boundaries[:1] - half,
                      (boundaries[:-1] + boundaries[1:]) / 2,
                      boundaries[-1:] + half))


def contact_loss(logits, pocket, ligand, *, bin_centers=None, diagnostics=None):
    """Frozen pocket coverage/anchor loss; native piecewise minima remain live."""
    logits = one_distogram(logits)
    if (logits.ndim != 4 or logits.shape[0] != 1
            or logits.shape[1] != logits.shape[2] or logits.shape[-1] != 64
            or pocket.dtype != torch.bool or ligand.dtype != torch.bool
            or pocket.ndim != 1 or pocket.shape != ligand.shape
            or pocket.numel() != logits.shape[1]
            or not bool(pocket.any()) or not bool(ligand.any())
            or bool((pocket & ligand).any()) or not bool(torch.isfinite(logits).all())):
        raise ValueError("Malformed contact logits or fixed token masks")
    if bin_centers is None:
        bin_centers = native_bin_centers(logits)
    if (bin_centers.shape != (64,) or bin_centers.device != logits.device
            or not bool(torch.isfinite(bin_centers).all())):
        raise ValueError("Malformed native distance representatives")
    near = bin_centers < 8.0
    if not bool(near.any()) or bool(near.all()):
        raise ValueError("Contact cutoff must partition native bins")
    selected = logits[0][pocket][:, ligand]
    log_mass = torch.logsumexp(selected.log_softmax(-1)[..., near], -1)
    pair_loss = -torch.logaddexp(log_mass, log_mass.new_tensor(math.log(1e-8)))
    per_residue, winners = pair_loss.min(-1)
    pocket_value, anchor = per_residue.mean(), pair_loss.min()
    if diagnostics is not None:
        detached = pair_loss.detach()
        pids, lids = pocket.nonzero().flatten().tolist(), ligand.nonzero().flatten().tolist()
        best = int(detached.argmin())
        diagnostics.update(
            pocket_token_indices=pids,
            best_ligand_token_per_pocket=[lids[i] for i in winners.tolist()],
            anchor_tokens=[pids[best // len(lids)], lids[best % len(lids)]],
        )
        if len(lids) > 1:
            two = detached.topk(2, dim=-1, largest=False).values
            diagnostics["smallest_best_vs_second_loss_margin"] = float((two[:, 1] - two[:, 0]).min())
    return {"dbmol_contact_loss": (pocket_value + anchor) / 2,
            "dbmol_contact_pocket_loss": pocket_value,
            "dbmol_contact_anchor_loss": anchor}


def assemble_components(outputs):
    """Recompose from live primitives, preserving grouping and FP64 arithmetic."""
    required = ("affinity_pred_value", "affinity_logits_binary",
                "dbmol_contact_loss", *CONFIDENCE_WEIGHTS)
    for key in required:
        value = outputs.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"Missing or nonscalar DBMol primitive: {key}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite DBMol primitive: {key}")
    values = {key: value.double().reshape(()) for key, value in outputs.items()}
    affinity = values["affinity_pred_value"] - values["affinity_logits_binary"]
    confidence = sum(weight * values[key] for key, weight in CONFIDENCE_WEIGHTS.items())
    result = dict(outputs)
    result.update(dbmol_affinity_loss=affinity, dbmol_confidence_loss=confidence,
                  dbmol_weighted_affinity_loss=affinity,
                  dbmol_weighted_contact_loss=2.5 * values["dbmol_contact_loss"],
                  dbmol_weighted_confidence_loss=2.5 * confidence)
    result["dbmol_total_loss"] = sum(result[key] for key in (
        "dbmol_weighted_affinity_loss", "dbmol_weighted_contact_loss",
        "dbmol_weighted_confidence_loss"))
    if not bool(torch.isfinite(result["dbmol_total_loss"]).all()):
        raise FloatingPointError("Nonfinite assembled DBMol loss")
    return result


def contact_distogram_float64(module, pair):
    """Widen BEFORE final native symmetrization and frozen affine projection."""
    linear = getattr(module, "distogram", None)
    if (not isinstance(linear, nn.Linear)
            or getattr(module, "num_distograms", None) != 1
            or getattr(module, "num_bins", None) != 64 or linear.out_features != 64
            or pair.ndim != 4 or pair.shape[1] != pair.shape[2]
            or pair.shape[-1] != linear.in_features or pair.dtype != torch.float32
            or module.training or any(p.requires_grad for p in module.parameters())
            or linear.weight.dtype != torch.float32
            or linear.bias is not None and linear.bias.dtype != torch.float32):
        raise ValueError("Require frozen eval native FP32 single 64-bin distogram")
    wide = pair.double()
    logits = F.linear(wide + wide.transpose(1, 2), linear.weight.double(),
                      None if linear.bias is None else linear.bias.double())
    return logits.reshape(*pair.shape[:3], 1, 64)


def fixed_pocket_masks(context):
    """Reuse Mosaic's authenticated JQ1 token IDs, never infer a new pocket."""
    pocket = context.get("pocket", {})
    audit = pocket.get("pocket_pose_audit", {})
    ids = audit.get("native_pocket_token_indices")
    if (pocket.get("pocket_definition_sha256") != POCKET_SHA256
            or pocket.get("diffusion_gradient") is not False
            or audit.get("status") != "passed" or not isinstance(ids, list) or not ids
            or any(type(index) is not int or index < 0 for index in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Require original authenticated JQ1 pocket token IDs")
    features = context["features"]
    pad = features["token_pad_mask"][0].bool()
    protein = (features["mol_type"][0] == 0) & pad
    if max(ids) >= pad.numel() or not bool(protein[ids].all()):
        raise ValueError("JQ1 token IDs must name occupied protein tokens")
    chosen = torch.zeros_like(pad)
    chosen[ids] = True
    ligand = features["affinity_token_mask"][0].bool() & pad
    if int(ligand.sum()) != 25 or bool((chosen & ligand).any()):
        raise ValueError("Require disjoint fixed JQ1 pocket and 25 ligand tokens")
    return chosen, ligand


def validate_frame_context(features, coordinates):
    """Reject unsupported layouts before native helpers can swallow exceptions."""
    pad, atom_pad, frames = (features[key] for key in ("token_pad_mask", "atom_pad_mask", "frames_idx"))
    if (pad.ndim != 2 or pad.shape[0] != 1
            or atom_pad.ndim != 2 or atom_pad.shape[0] != 1):
        raise ValueError("Confidence requires batch-one token/atom masks")
    if coordinates.shape not in ((1, atom_pad.shape[1], 3), (1, 1, atom_pad.shape[1], 3)):
        raise ValueError("Unsupported native confidence coordinate shape")
    if coordinates.requires_grad or not bool(torch.isfinite(coordinates).all()):
        raise ValueError("Confidence requires finite detached coordinates")
    coords = coordinates.reshape(1, -1, 3)
    if frames.shape == (*pad.shape, 3):
        rows = frames
    elif frames.shape == (1, 1, pad.shape[1], 3):
        rows = frames[:, 0]
    else:
        raise ValueError("Unsupported native confidence frame shape")
    if frames.dtype != torch.long:
        raise ValueError("Native frame indices must be int64")
    for mask in (pad[0], atom_pad[0]):
        if bool(((mask != 0) & (mask != 1)).any()):
            raise ValueError("Native padding masks must be binary")
        prefix = torch.arange(mask.numel(), device=mask.device) < mask.bool().sum()
        if not torch.equal(mask.bool(), prefix):
            raise ValueError("Confidence requires prefix padding on both axes")
    active = rows[pad.bool()]
    if bool(((active < 0) | (active >= coords.shape[1])).any()):
        raise ValueError("Native frame index outside coordinates")
    token_ids = features["asym_id"][0, pad[0].bool()]
    atom_ids = (features["atom_to_token"][0].float()
                @ features["asym_id"][0].float())[atom_pad[0].bool()].long()
    for ids in (token_ids, atom_ids):
        if torch.unique_consecutive(ids).tolist() != torch.unique(ids).tolist():
            raise ValueError("Confidence requires sorted contiguous chain axes")
    if torch.unique(token_ids).tolist() != torch.unique(atom_ids).tolist():
        raise ValueError("Confidence chain axes disagree")
    for chain in torch.unique(token_ids):
        tokens = (features["asym_id"][0] == chain) & pad[0].bool()
        types = features["mol_type"][0, tokens]
        if types.unique().numel() != 1:
            raise ValueError("Require one molecular type per native chain")
        if int(types[0]) == 3 and int(tokens.sum()) != int((atom_ids == chain).sum()):
            raise ValueError("Require one atom per occupied nonpolymer token")


def attached_trunk(model, features, types):
    """Native zero-recycle recurrence returning structural s and unmasked z."""
    if model.training or model.use_kernels:
        raise ValueError("Require frozen eval native PyTorch kernels")
    inputs = model.input_embedder(features)
    s_init = model.s_init(inputs)
    z_init = model.z_init_1(inputs)[:, :, None] + model.z_init_2(inputs)[:, None, :]
    z_init = z_init + model.rel_pos(features) + model.token_bonds(features["token_bonds"].float())
    if model.bond_type_feature:
        z_init = z_init + types @ model.token_bonds_type.weight
    z_init = z_init + model.contact_conditioning(features)
    s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
    mask = features["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s = s_init + model.s_recycle(model.s_norm(s))
    z = z_init + model.z_recycle(model.z_norm(z))
    if model.use_templates:
        z = z + model.template_module(z, features, pair_mask, use_kernels=False)
    z = z + model.msa_module(z, inputs, features, use_kernels=False)
    s, z = model.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)
    return inputs, s, z


def attached_confidence(module, inputs, s, z, features, coordinates, distogram, types):
    """Native confidence recurrence with attached soft typed-bond expectation."""
    inputs = module.s_inputs_norm(inputs)
    if not module.no_update_s:
        s = module.s_norm(s)
    if module.add_s_input_to_s:
        s = s + module.s_input_to_s(inputs)
    z = module.z_norm(z)
    if module.add_z_input_to_z:
        z = z + module.rel_pos(features)
        z = z + module.token_bonds(features["token_bonds"].float())
        if module.bond_type_feature:
            z = z + types @ module.token_bonds_type.weight
        z = z + module.contact_conditioning(features)
    z = z + module.s_to_z(inputs)[:, :, None, :] + module.s_to_z_transpose(inputs)[:, None, :, :]
    if module.add_s_to_z_prod:
        z = z + module.s_to_z_prod_out(module.s_to_z_prod_in1(inputs)[:, :, None, :]
                                     * module.s_to_z_prod_in2(inputs)[:, None, :, :])
    coordinates = coordinates.reshape(1, -1, 3).detach()
    representative = torch.bmm(features["token_to_rep_atom"].float(), coordinates)
    distances = torch.cdist(representative, representative)
    bins = (distances.unsqueeze(-1) > module.boundaries).sum(-1).long()
    z = z + module.dist_bin_pairwise_embed(bins)
    mask = features["token_pad_mask"]
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s, z = module.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)
    return module.confidence_heads(s=s, z=z, x_pred=coordinates, d=distances,
                                   feats=features, multiplicity=1,
                                   pred_distogram_logits=one_distogram(distogram))


def strict_confidence_scores(raw, features, coordinates):
    validate_frame_context(features, coordinates)
    from boltz.model.layers.confidence_utils import compute_frame_pred, compute_ptms
    coords = coordinates.reshape(1, -1, 3)
    _, frame_mask = compute_frame_pred(coords, features["frames_idx"], features, 1, inference=True)
    pad = features["token_pad_mask"].bool()
    different = features["asym_id"][:, :, None] != features["asym_id"][:, None, :]
    valid = frame_mask.reshape_as(pad)[:, :, None] * pad[:, :, None] * pad[:, None, :] * different
    if not bool(valid.any()):
        raise ValueError("No nondegenerate native confidence interface frames")
    ptm, iptm, *_ = compute_ptms(raw["pae_logits"], coords, features, 1)
    tm = iptm if not torch.allclose(iptm, torch.zeros_like(iptm)) else ptm
    result = {key: raw[key] for key in ("complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde")}
    result.update(ptm=ptm, iptm=iptm, confidence_score=(4 * raw["complex_plddt"] + tm) / 5)
    for key, value in result.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"Missing or nonscalar confidence primitive: {key}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite confidence primitive: {key}")
    return {key: value.reshape(()) for key, value in result.items()}


def install_structural_checkpoints(model):
    """Rematerialize native frozen eval blocks; no training-mode substitution."""
    from boltz.model.layers.pairformer import PairformerLayer, PairformerNoSeqLayer
    from boltz.model.modules.trunkv2 import MSALayer
    kinds = (PairformerLayer, PairformerNoSeqLayer, MSALayer)
    stacks = (model.pairformer_module, model.msa_module, model.confidence_module.pairformer_stack)
    counts = []
    for stack in stacks:
        counts.append(len(stack.layers))
        for block in stack.layers:
            if not isinstance(block, kinds):
                raise ValueError("Unexpected native structural checkpoint block")
            if hasattr(block, "_mosaic_dbmol_original_forward"):
                raise ValueError("Structural rematerialization already installed")
            original = block.forward

            @wraps(original)
            def rematerialized(*args, _block=block, _original=original, **kwargs):
                if _block.training:
                    raise ValueError("Native checkpoint parent must stay in eval")
                if not torch.is_grad_enabled():
                    return _original(*args, **kwargs)
                return checkpoint(_original, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)

            block._mosaic_dbmol_original_forward = original
            block.forward = rematerialized
    return {"use_reentrant": False, "preserve_rng_state": True, "stack_blocks": counts}


def load_structure_model(path):
    """Strict hash-authenticated structural checkpoint with its own trunk."""
    if native.file_digest(path) != STRUCTURE_SHA256:
        raise ValueError("Structural checkpoint SHA256 mismatch")
    native.verify_installation()
    from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2
    from boltz.model.models.boltz2 import Boltz2
    steering = BoltzSteeringParams()
    steering.fk_steering = steering.physical_guidance_update = steering.contact_guidance_update = False
    model = Boltz2.load_from_checkpoint(
        path, map_location="cpu", strict=True, ema=False, use_kernels=False,
        compile_pairformer=False, compile_structure=False, compile_confidence=False,
        compile_affinity=False, compile_msa=False, compile_templates=False,
        affinity_mw_correction=False,
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(subsample_msa=False, use_paired_feature=True)),
        steering_args=asdict(steering),
        predict_args={"recycling_steps": 0, "sampling_steps": 25, "diffusion_samples": 1},
    ).eval().requires_grad_(False)
    model.msa_module.subsample_msa = False
    if not model.confidence_prediction or not model.run_trunk_and_structure or not model.bond_type_feature:
        raise ValueError("Require native structural trunk/confidence/typed-bond head")
    install_structural_checkpoints(model)
    return model


class DBMolScorer(nn.Module):
    """Shared graph -> two frozen trunks -> separately audited loss components."""

    def __init__(self, affinity_model, structure_model, context):
        super().__init__()
        native.validate_context(context)
        if int(context["atom_indices"].numel()) != 25:
            raise ValueError("DBMol pilot is scoped to 25 occupied neutral slots")
        if any(value.requires_grad for value in context["features"].values()):
            raise ValueError("Prepared context features must be frozen")
        validate_frame_context(context["features"], context["coordinates"])
        self.pocket, self.ligand = fixed_pocket_masks(context)
        self.context = context
        self.affinity = affinity_model.eval().requires_grad_(False)
        self.structure = structure_model.eval().requires_grad_(False)
        self.last_contact_selection = {}
        self.eval()

    def _reduce(self, affinity, confidence, distogram, pair, features):
        self.last_contact_selection = {}
        contact_logits = contact_distogram_float64(self.structure.distogram_module, pair)
        if contact_logits.shape != distogram.shape:
            raise ValueError("Contact/native distogram layouts differ")
        contact = contact_loss(contact_logits, self.pocket, self.ligand,
                               bin_centers=native_bin_centers(distogram).double(),
                               diagnostics=self.last_contact_selection)
        return assemble_components({**affinity, **confidence, **contact})

    def forward(self, x, edges):
        validate_graph(x, edges, self.context)
        features, types = native.inject(x, edges, self.context)
        affinity_values = native.heads(self.affinity, features,
                                      native.attached_trunk(self.affinity, features, types),
                                      self.context["coordinates"])
        affinity = dict(zip(("affinity_pred_value", "affinity_probability_binary",
                             "affinity_logits_binary"), affinity_values.unbind()))
        inputs, s, pair = attached_trunk(self.structure, features, types)
        distogram = self.structure.distogram_module(pair)
        raw = attached_confidence(self.structure.confidence_module, inputs, s, pair,
                                  features, self.context["coordinates"], distogram, types)
        confidence = strict_confidence_scores(raw, features, self.context["coordinates"])
        result = self._reduce(affinity, confidence, distogram, pair, features)
        if torch.is_grad_enabled() and (x.requires_grad or edges.requires_grad):
            for key in (*COMPONENTS, "affinity_pred_value", "affinity_logits_binary", *CONFIDENCE_WEIGHTS):
                if not result[key].requires_grad:
                    raise ValueError(f"Detached DBMol objective path: {key}")
        return result

    @torch.no_grad()
    def native_reference(self):
        """Independent native integer trunks/heads at the unchanged seed/pose."""
        features, coords = self.context["features"], self.context["coordinates"]
        affinity_values = native.native_reference(self.affinity, self.context)
        affinity = dict(zip(("affinity_pred_value", "affinity_probability_binary",
                             "affinity_logits_binary"), affinity_values.unbind()))
        model = self.structure
        flags = ("skip_run_structure", "confidence_prediction", "affinity_prediction")
        previous = [getattr(model, flag) for flag in flags]
        try:
            model.skip_run_structure, model.confidence_prediction, model.affinity_prediction = True, False, False
            output = model(features, recycling_steps=0)
        finally:
            for flag, value in zip(flags, previous):
                setattr(model, flag, value)
        raw = model.confidence_module(
            s_inputs=model.input_embedder(features), s=output["s"], z=output["z"],
            x_pred=coords.reshape(1, -1, 3), feats=features,
            pred_distogram_logits=one_distogram(output["pdistogram"]),
            multiplicity=1, run_sequentially=False, use_kernels=False,
        )
        confidence = strict_confidence_scores(raw, features, coords)
        return self._reduce(affinity, confidence, output["pdistogram"], output["z"], features)
