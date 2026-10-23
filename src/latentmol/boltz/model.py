"""Frozen native Boltz-2 modules with an explicit partial input-gradient path.

Source: jwohlwend/boltz at b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc,
model/models/boltz2.py:forward and model/modules/affinity.py:AffinityModule.
The stock eval forward disables trunk autograd and detaches its affinity inputs.
Composition below keeps those paths and checkpoints selected block forward calls,
preserving native weights, block mathematics and parent-loop eval behavior.
"""

from dataclasses import asdict
from functools import wraps
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from latentmol.experiments import file_sha256
from .contracts import AFFINITY_SHA256


def checkpoint_eval_block(block: nn.Module) -> None:
    """Rematerialize one frozen native block while preserving its registered state.

    Native Boltz only enables its checkpoint flags in training mode. Replacing
    this block's Python forward callable leaves the native parent stack in eval
    mode, so its own chunk sizes, dropout behavior and arguments remain intact.
    No module is replaced and no parameter name changes. A separately captured
    bound method per block avoids late-bound loop closures during recomputation.

    Non-reentrant checkpointing supports ``torch.autograd.grad`` and frozen
    parameters. RNG state is preserved even though eval disables dropout. Calls
    under no_grad use the original forward directly (reference and FD passes).
    """
    if hasattr(block, "_latentmol_original_forward"):
        return
    original_forward = block.forward

    @wraps(original_forward)
    def forward(*args, **kwargs):
        if block.training:
            raise ValueError("Activation rematerialization requires the native block to remain in eval mode")
        if not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        return checkpoint(original_forward, *args, use_reentrant=False,
                          preserve_rng_state=True, **kwargs)

    block._latentmol_original_forward = original_forward
    block.forward = forward


def install_eval_checkpoints(model: nn.Module, *, include_confidence: bool = False) -> dict:
    """Checkpoint pinned native trunk/affinity and optional confidence blocks.

    A 64-block trunk retains large pair activations without this boundary. MSA
    blocks and both affinity pairformer stacks use the same policy. Individual
    block math and the native parent loops are unchanged; the cost is repeated
    forward work during backward. Embedders and diffusion stay as-is; the opt-in
    confidence stack uses this same policy without changing its native math.
    """
    from boltz.model.layers.pairformer import PairformerLayer, PairformerNoSeqLayer
    from boltz.model.modules.trunkv2 import MSALayer
    kinds = (PairformerLayer, PairformerNoSeqLayer, MSALayer)
    counts = {kind.__name__: 0 for kind in kinds}
    stacks = {"trunk_pairformer": model.pairformer_module, "trunk_msa": model.msa_module}
    heads = ([model.affinity_module1, model.affinity_module2] if model.affinity_ensemble
             else [model.affinity_module]) if model.affinity_prediction else []
    for index, head in enumerate(heads):
        stacks[f"affinity_pairformer_{index + 1}"] = head.pairformer_stack
    if include_confidence:
        stacks["confidence_pairformer"] = model.confidence_module.pairformer_stack
    stack_counts = {}
    for name, stack in stacks.items():
        stack_counts[name] = len(stack.layers)
        for block in stack.layers:
            if not isinstance(block, kinds):
                raise ValueError(f"Unexpected native block type in {name}: {type(block).__name__}")
            checkpoint_eval_block(block)
            counts[type(block).__name__] += 1
    if not counts["PairformerLayer"]:
        raise ValueError("Pinned native PairformerLayer blocks were not found")
    return {"enabled": True, "use_reentrant": False, "preserve_rng_state": True,
            "blocks": counts, "stacks": stack_counts, "native_eval_parent_loops_unchanged": True}


def affinity_runtime_overrides() -> dict:
    """Reproduce the pinned native CLI's supported checkpoint runtime settings.

    The official affinity checkpoint contains historical diffusion parameters
    (including ``mse_rotational_alignment``) unsupported by its current pinned
    AtomDiffusion constructor. The native CLI replaces entire runtime argument
    groups with these dataclasses; filtering arbitrary unknown keys would hide
    other compatibility failures. We use the same explicit replacement policy.

    MSA resampling is deliberately disabled because PreparedComplex already
    stores its fixed subset. ``predict_args`` reproduces native CLI defaults but
    is unused by this adapter, whose forward specifies its recycle count and
    never calls diffusion. Guidance is disabled exactly as in native affinity.
    """
    from boltz.main import (Boltz2DiffusionParams, BoltzSteeringParams,
                            MSAModuleArgs, PairformerArgsV2)
    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False
    return {
        "diffusion_process_args": asdict(Boltz2DiffusionParams()),
        "pairformer_args": asdict(PairformerArgsV2()),
        "msa_args": asdict(MSAModuleArgs(subsample_msa=False, use_paired_feature=True)),
        "steering_args": asdict(steering),
        "predict_args": {"recycling_steps": 5, "sampling_steps": 200,
                         "diffusion_samples": 5, "max_parallel_samples": 1,
                         "write_confidence_summary": False,
                         "write_full_pae": False, "write_full_pde": False},
    }


def load_affinity_model(path: Path):
    """Load the pinned official checkpoint strictly, without optional CUDA kernels.

    The official Lightning checkpoint is a trusted, hash-verified external asset.
    Boltz is imported only here, so AE/VAE environments do not require its deps.
    """
    if file_sha256(path) != AFFINITY_SHA256:
        raise ValueError("Affinity checkpoint does not match the pinned official asset")
    from .prepare import installed_boltz_provenance
    installed_boltz_provenance()
    from boltz.model.models.boltz2 import Boltz2
    model = Boltz2.load_from_checkpoint(
        path, map_location="cpu", strict=True, ema=False, use_kernels=False,
        compile_pairformer=False, compile_structure=False, compile_confidence=False,
        compile_affinity=False, compile_msa=False, compile_templates=False,
        affinity_mw_correction=False,
        **affinity_runtime_overrides(),
    )
    model.eval().requires_grad_(False)
    # PreparedComplex stores one explicit MSA subset. Native MSAModule would
    # otherwise redraw indices on every eval call, invalidating finite differences.
    model.msa_module.subsample_msa = False
    if not model.affinity_prediction or not model.run_trunk_and_structure:
        raise ValueError("Checkpoint must contain the native affinity trunk and heads")
    return model


def native_heads(model, features, z, coordinates, *, include_logits: bool = False):
    """Evaluate native affinity heads at a detached pose, preserving ensemble math.

    No molecular-weight correction is used, because a fixed reference weight
    becomes stale under element changes. Coordinates are deliberately detached;
    native affinity additionally bins distances discretely, preventing ordinary
    pose derivatives even if that detach were removed.
    """
    pad = features["token_pad_mask"][0]
    rec = (features["mol_type"][0] == 0) * pad
    lig = features["affinity_token_mask"][0].bool() * pad
    cross = lig[:, None]*rec[None, :] + rec[:, None]*lig[None, :] + lig[:, None]*lig[None, :]
    z = z * cross[None, :, :, None]
    inputs = model.input_embedder(features, affinity=True)
    modules = [model.affinity_module1, model.affinity_module2] if model.affinity_ensemble else [model.affinity_module]
    values, probabilities, logits = [], [], []
    for module in modules:
        out = module(s_inputs=inputs, z=z, x_pred=coordinates.detach(), feats=features,
                     multiplicity=1, use_kernels=False)
        values.append(out["affinity_pred_value"])
        probabilities.append(out["affinity_logits_binary"].sigmoid())
        if include_logits:
            logits.append(out["affinity_logits_binary"])
    result = {"affinity_pred_value": torch.stack(values).mean(0),
              "affinity_probability_binary": torch.stack(probabilities).mean(0)}
    if include_logits:
        # Mean raw native head logits is explicit; it is not logit(mean(sigmoid)).
        result["affinity_logits_binary"] = torch.stack(logits).mean(0)
    return result


def attached_trunk(model, features, bond_probabilities, recycling_steps):
    """Pinned native recurrence, with input derivatives and expected typed bonds.

    Both checkpoint-specific trunks use this same mapping. Return the unmasked
    pair representation: affinity masking must not erase structural-head inputs.
    The caller keeps all weights frozen and never invokes structure diffusion.
    """
    m = model
    if m.training or m.use_kernels:
        raise ValueError("Require eval mode and standard PyTorch kernels")
    inputs = m.input_embedder(features)
    s_init = m.s_init(inputs)
    z_init = m.z_init_1(inputs)[:, :, None] + m.z_init_2(inputs)[:, None, :]
    z_init = z_init + m.rel_pos(features) + m.token_bonds(features["token_bonds"].float())
    if m.bond_type_feature:
        z_init = z_init + bond_probabilities @ m.token_bonds_type.weight
    z_init = z_init + m.contact_conditioning(features)
    s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
    mask = features["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]
    for _ in range(recycling_steps + 1):
        s = s_init + m.s_recycle(m.s_norm(s))
        z = z_init + m.z_recycle(m.z_norm(z))
        if m.use_templates:
            z = z + m.template_module(z, features, pair_mask, use_kernels=False)
        z = z + m.msa_module(z, inputs, features, use_kernels=False)
        s, z = m.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=False)
    return inputs, s, z


class FrozenBoltzAffinity(nn.Module):
    """Feature-to-trunk-to-affinity derivative with occupancy and pose held fixed.

    All recycles participate in gradients here. The stock inference forward has
    the same numerical recurrence but no derivatives. A zero-recycle smoke is
    cheaper than the official five-recycle affinity prediction; record the value.
    """

    def __init__(self, model, recycling_steps: int = 0, *, include_logits: bool = False):
        super().__init__()
        if recycling_steps < 0:
            raise ValueError("Recycling steps must be nonnegative")
        self.model = model.eval().requires_grad_(False)
        self.recycling_steps = recycling_steps
        self.include_logits = include_logits
        self.activation_checkpoint_policy = install_eval_checkpoints(self.model)

    def forward(self, features, bond_probabilities, coordinates):
        """Use expected typed-bond embeddings instead of integer-index lookup."""
        _, _, z = attached_trunk(self.model, features, bond_probabilities, self.recycling_steps)
        return native_heads(self.model, features, z, coordinates, include_logits=self.include_logits)

    @torch.no_grad()
    def discrete_reference(self, features, coordinates):
        """Native discrete trunk plus native heads at the identical fixed pose.

        This uses Boltz2.forward for the trunk, with structure/confidence/affinity
        branches temporarily skipped, then its native heads. It is not a claim of
        equality to another full run whose diffusion samples a different pose.
        """
        m = self.model
        names = ("skip_run_structure", "confidence_prediction", "affinity_prediction")
        old = {name: getattr(m, name) for name in names}
        try:
            m.skip_run_structure, m.confidence_prediction, m.affinity_prediction = True, False, False
            native = m(features, recycling_steps=self.recycling_steps)
        finally:
            for name, value in old.items():
                setattr(m, name, value)
        return native_heads(m, features, native["z"], coordinates, include_logits=self.include_logits)
