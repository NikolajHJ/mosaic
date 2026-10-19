"""Frozen native Boltz integration, independent of JAX and molecular decoders.

The attached trunk follows the pinned upstream forward. Integer bond lookup is
extended by its probability-weighted pretrained embedding. Geometry, occupancy,
MSA and auxiliary features are fixed. This is a partial chemistry derivative,
not differentiation through pose prediction. Integration approach audited in
DBMol_exploratory; no VAE, latent state, learned bridge or DBMol objective used.
"""

import hashlib
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

BOLTZ_COMMIT = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"
AFFINITY_SHA256 = "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"
ATOMIC_NUMBERS = (6, 7, 8, 16, 9, 17)  # Mosaic C,N,O,S,F,Cl, not Protenix indices.
TYPED_BOND_IDS = (0, 2, 3, 4, 5)  # native OTHER=1 and COVALENT=6 are not trainable.


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(values):
    """Include names, shapes, dtypes and exact bytes; numerical work stays on HPC."""
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_installation():
    import json

    direct = json.loads(
        metadata.distribution("boltz").read_text("direct_url.json") or "{}"
    )
    if direct.get("vcs_info", {}).get("commit_id") != BOLTZ_COMMIT:
        raise ValueError("Boltz source differs from the audited pinned native commit")
    return direct


def load_model(path):
    """Strict cached affinity checkpoint; no downloads, training or random heads."""
    if file_digest(path) != AFFINITY_SHA256:
        raise ValueError("Native affinity checkpoint SHA mismatch")
    verify_installation()
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2

    steering = BoltzSteeringParams()
    steering.fk_steering = steering.physical_guidance_update = (
        steering.contact_guidance_update
    ) = False
    model = (
        Boltz2.load_from_checkpoint(
            path,
            map_location="cpu",
            strict=True,
            ema=False,
            use_kernels=False,
            compile_pairformer=False,
            compile_structure=False,
            compile_confidence=False,
            compile_affinity=False,
            compile_msa=False,
            compile_templates=False,
            affinity_mw_correction=False,
            diffusion_process_args=asdict(Boltz2DiffusionParams()),
            pairformer_args=asdict(PairformerArgsV2()),
            msa_args=asdict(
                MSAModuleArgs(subsample_msa=False, use_paired_feature=True)
            ),
            steering_args=asdict(steering),
            predict_args={
                "recycling_steps": 0,
                "sampling_steps": 25,
                "diffusion_samples": 1,
            },
        )
        .eval()
        .requires_grad_(False)
    )
    model.msa_module.subsample_msa = False
    if (
        not model.affinity_prediction
        or not model.run_trunk_and_structure
        or not model.bond_type_feature
    ):
        raise ValueError("Require trained affinity trunk, heads and typed-bond channel")
    # Keep native parent loops in eval; non-reentrant rematerialization supports
    # input gradients with entirely frozen parameters. Each closure owns its block.
    stacks = [model.msa_module, model.pairformer_module]
    heads = (
        [model.affinity_module1, model.affinity_module2]
        if model.affinity_ensemble
        else [model.affinity_module]
    )
    stacks.extend(head.pairformer_stack for head in heads)
    for stack in stacks:
        for block in stack.layers:
            original = block.forward

            def rematerialized(*args, _original=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(
                        _original,
                        *args,
                        use_reentrant=False,
                        preserve_rng_state=True,
                        **kwargs,
                    )
                return _original(*args, **kwargs)

            block.forward = rematerialized
    return model


def validate_context(context):
    """Check complete six-slot native correspondence and seed feature parity."""
    if context.get("version") != "mosaic-boltz-fixed-pose-v1":
        raise ValueError("Unknown prepared Boltz context")
    f, ai, ti = context["features"], context["atom_indices"], context["token_indices"]
    if (
        ai.shape != (6,)
        or ti.shape != (6,)
        or ai.unique().numel() != 6
        or ti.unique().numel() != 6
    ):
        raise ValueError("Require exactly six injective occupied ligand slots")
    if (
        context["coordinates"].requires_grad
        or not torch.isfinite(context["coordinates"]).all()
    ):
        raise ValueError("Require finite detached reference pose")
    ligand = f["affinity_token_mask"][0].bool() & f["token_pad_mask"][0].bool()
    if not torch.equal(ti.sort().values, ligand.nonzero().flatten()):
        raise ValueError("Slot map must cover the entire native affinity ligand")
    if not torch.all(f["atom_to_token"][0, ai, ti] == 1) or not torch.all(
        f["atom_to_token"][0, :, ti].sum(0) == 1
    ):
        raise ValueError("Require one atom per ligand token")
    if not torch.all(f["token_to_rep_atom"][0, ti, ai] == 1):
        raise ValueError("Representative atom map differs")
    if not torch.all(f["atom_pad_mask"][0, ai] == 1) or torch.any(
        f["ref_charge"][0, ai] != 0
    ):
        raise ValueError("Mosaic six occupied neutral slots required")
    mapped, types = inject(context["seed_X"], context["seed_E"], context)
    for name in ("ref_element", "ref_charge", "token_bonds"):
        torch.testing.assert_close(
            mapped[name], f[name].to(mapped[name]), atol=0, rtol=0
        )
    torch.testing.assert_close(
        types,
        nn.functional.one_hot(f["type_bonds"].long(), 7).to(types),
        atol=0,
        rtol=0,
    )


def inject(x, edges, context):
    """Linear extension of native atom/bond features, with shared unordered edges.

    No argmax or chemistry conversion occurs in the differentiable path. Formal
    charge remains fixed at zero, exactly as in Mosaic's current graph schema.
    """
    if x.shape != (6, 6) or edges.shape != (15, 5):
        raise ValueError("Boltz backend requires Mosaic six-slot graph shapes")
    f = dict(context["features"])
    ai, ti = context["atom_indices"], context["token_indices"]
    element_map = nn.functional.one_hot(
        torch.tensor(ATOMIC_NUMBERS, device=x.device), 128
    ).to(x)
    f["ref_element"] = f["ref_element"].to(x).index_copy(1, ai, (x @ element_map)[None])
    i, j = torch.triu_indices(6, 6, 1, device=x.device)
    dense = edges.new_zeros((6, 6, 5))
    dense[..., 0] = 1
    dense[i, j] = edges
    dense[j, i] = edges
    tokens = f["token_pad_mask"].shape[1]
    pairs = (ti[:, None] * tokens + ti[None, :]).flatten()
    connectivity = (1 - dense[..., 0]).reshape(1, -1, 1)
    f["token_bonds"] = (
        f["token_bonds"]
        .to(x)
        .reshape(1, tokens * tokens, 1)
        .index_copy(1, pairs, connectivity)
        .reshape(1, tokens, tokens, 1)
    )
    bond_map = nn.functional.one_hot(
        torch.tensor(TYPED_BOND_IDS, device=x.device), 7
    ).to(x)
    native = nn.functional.one_hot(f["type_bonds"].long(), 7).to(x)
    native = native.reshape(1, tokens * tokens, 7).index_copy(
        1, pairs, (dense @ bond_map).reshape(1, -1, 7)
    )
    return f, native.reshape(1, tokens, tokens, 7)


def attached_trunk(model, f, bond_types, recycling_steps=0):
    """Same native recurrence; frozen weights, explicit input derivatives."""
    inputs = model.input_embedder(f)
    s_init = model.s_init(inputs)
    z_init = model.z_init_1(inputs)[:, :, None] + model.z_init_2(inputs)[:, None, :]
    z_init = z_init + model.rel_pos(f) + model.token_bonds(f["token_bonds"].float())
    z_init = (
        z_init
        + bond_types @ model.token_bonds_type.weight
        + model.contact_conditioning(f)
    )
    s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
    mask = f["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]
    for _ in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))
        if model.use_templates:
            z = z + model.template_module(z, f, pair_mask, use_kernels=False)
        z = z + model.msa_module(z, inputs, f, use_kernels=False)
        s, z = model.pairformer_module(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=False
        )
    return z


def heads(model, f, z, coordinates):
    """Native two-member quantitative/probability ensemble; no MW correction.

    Only quantitative affinity is optimized. Binary probability/logits are
    recorded independently and are never substituted for the native ensemble.
    """
    pad = f["token_pad_mask"][0]
    rec = (f["mol_type"][0] == 0) * pad
    lig = f["affinity_token_mask"][0].bool() * pad
    cross = (
        lig[:, None] * rec[None, :]
        + rec[:, None] * lig[None, :]
        + lig[:, None] * lig[None, :]
    )
    inputs = model.input_embedder(f, affinity=True)
    modules = (
        [model.affinity_module1, model.affinity_module2]
        if model.affinity_ensemble
        else [model.affinity_module]
    )
    outputs = [
        head(
            s_inputs=inputs,
            z=z * cross[None, :, :, None],
            x_pred=coordinates.detach(),
            feats=f,
            multiplicity=1,
            use_kernels=False,
        )
        for head in modules
    ]
    return torch.stack(
        [
            torch.stack(
                [out["affinity_pred_value"].reshape(()) for out in outputs]
            ).mean(),
            torch.stack(
                [out["affinity_logits_binary"].reshape(()).sigmoid() for out in outputs]
            ).mean(),
            torch.stack(
                [out["affinity_logits_binary"].reshape(()) for out in outputs]
            ).mean(),
        ]
    )


def score(model, context, x, edges):
    f, types = inject(x, edges, context)
    return heads(model, f, attached_trunk(model, f, types), context["coordinates"])


@torch.no_grad()
def native_reference(model, context):
    """Independent integer native trunk, at the identical prepared seed/pose."""
    flags = ("skip_run_structure", "confidence_prediction", "affinity_prediction")
    previous = [getattr(model, flag) for flag in flags]
    try:
        (
            model.skip_run_structure,
            model.confidence_prediction,
            model.affinity_prediction,
        ) = True, False, False
        output = model(context["features"], recycling_steps=0)
    finally:
        for flag, value in zip(flags, previous):
            setattr(model, flag, value)
    return heads(model, context["features"], output["z"], context["coordinates"])
