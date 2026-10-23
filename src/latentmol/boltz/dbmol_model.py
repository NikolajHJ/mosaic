"""Frozen Boltz affinity and structural checkpoints on one canonical soft graph.

Each checkpoint keeps its own trained trunk; confidence weights are never
transplanted onto an affinity trunk. Contacts/confidence share the structural
trunk. Poses stay detached and structure diffusion is never called here.
Native and finite-difference HPC checks are mandatory for this new path.
"""

from pathlib import Path

import torch
from torch import nn

from latentmol.experiments import file_sha256
from latentmol.multimodel.config import digest
from .contracts import AFFINITY_SHA256
from .dbmol_objective import CONFIDENCE_WEIGHTS, CONTACT_HEAD64_ARITHMETIC, reduce_profile
from .contact_projection import contact_distogram_float64
from .model import (FrozenBoltzAffinity, affinity_runtime_overrides, attached_trunk,
                    install_eval_checkpoints)


def load_confidence_model(asset):
    """Strict native checkpoint loader; never initialize a missing head randomly."""
    from .prepare import installed_boltz_provenance
    path = Path(asset["path"])
    if file_sha256(path) != asset["sha256"]:
        raise ValueError("Confidence checkpoint SHA256 mismatch")
    installed_boltz_provenance()
    from boltz.model.models.boltz2 import Boltz2
    model = Boltz2.load_from_checkpoint(
        path, map_location="cpu", strict=True, ema=False, use_kernels=False,
        compile_pairformer=False, compile_structure=False, compile_confidence=False,
        compile_affinity=False, compile_msa=False, compile_templates=False,
        affinity_mw_correction=False, **affinity_runtime_overrides())
    if not model.confidence_prediction or not model.run_trunk_and_structure:
        raise ValueError("Checkpoint lacks its trained structural trunk/confidence head")
    model.eval().requires_grad_(False)
    model.msa_module.subsample_msa = False
    return model


def one_distogram(logits):
    if logits.ndim == 5 and logits.shape[3] == 1:
        return logits[:, :, :, 0]
    if logits.ndim == 4:
        return logits
    raise ValueError("Only the pinned single native distogram is supported")


def validate_frame_context(features, coordinates):
    """Exclude the native helper's caught ligand-frame assignment failure.

    compute_frame_pred assumes chains sorted/contiguous in both axes and one
    atom per nonpolymer token. Its internal assignment exception is swallowed;
    reject unsupported layouts before calling that helper instead. The pinned
    featurizer emits [batch, ensemble, token, 3] indices; its single-ensemble
    axis is native data, not an extra token axis. Keep the original tensor for
    native calls and use a view only for validation. Legacy [batch, token, 3]
    snapshots remain supported; multiple frame ensembles are unsupported here.
    """
    pad = features["token_pad_mask"]
    atom_pad = features["atom_pad_mask"]
    frames = features["frames_idx"]
    if pad.ndim != 2 or pad.shape[0] != 1 or atom_pad.ndim != 2 or atom_pad.shape[0] != 1:
        raise ValueError("Native confidence requires batch-one token/atom masks")
    if coordinates.shape not in ((1, atom_pad.shape[1], 3), (1, 1, atom_pad.shape[1], 3)):
        raise ValueError(f"Unsupported native confidence coordinate shape: {tuple(coordinates.shape)}")
    coords = coordinates.reshape(1, -1, 3)
    if not bool(coords.isfinite().all()):
        raise ValueError("Nonfinite native confidence coordinates")
    if frames.shape == (*pad.shape, 3):
        frame_rows = frames
    elif frames.shape == (pad.shape[0], 1, pad.shape[1], 3):
        frame_rows = frames[:, 0]
    else:
        raise ValueError(f"Unsupported native confidence frame shape: {tuple(frames.shape)}")
    if frames.dtype != torch.long:
        raise ValueError(f"Unsupported native confidence frame dtype: {frames.dtype}")
    for mask in (pad[0].bool(), atom_pad[0].bool()):
        prefix = torch.arange(mask.numel(), device=mask.device) < mask.sum()
        if not torch.equal(mask, prefix):
            raise ValueError("Native confidence requires prefix padding on both axes")
    active_frames = frame_rows[pad.bool()]
    if bool(((active_frames < 0) | (active_frames >= coords.shape[1])).any()):
        raise ValueError("Native confidence frame index out of bounds")
    token_ids = features["asym_id"][0, pad[0].bool()]
    atom_ids = (features["atom_to_token"][0].float()
                @ features["asym_id"][0].float())[atom_pad[0].bool()].long()
    for ids in (token_ids, atom_ids):
        if torch.unique_consecutive(ids).tolist() != torch.unique(ids).tolist():
            raise ValueError("Native confidence requires sorted contiguous chain axes")
    if torch.unique(token_ids).tolist() != torch.unique(atom_ids).tolist():
        raise ValueError("Native confidence chain axes disagree")
    for chain in torch.unique(token_ids):
        tokens = (features["asym_id"][0] == chain) & pad[0].bool()
        types = features["mol_type"][0, tokens]
        if types.unique().numel() != 1:
            raise ValueError("Native confidence requires one molecular type per chain")
        if int(types[0]) == 3 and int(tokens.sum()) != int((atom_ids == chain).sum()):
            raise ValueError("Native confidence requires one atom per nonpolymer token")


def attached_confidence(module, inputs, s, z, features, coordinates, distogram, bonds):
    """Mirror ConfidenceModule.forward for batch/multiplicity one, without detach.

    Only the optional typed-bond lookup changes: its integer embedding becomes
    the same probability-weighted embedding used by the shared-graph trunk.
    Native geometric binning/masks remain fixed at the reference coordinates.
    Hard-input parity must certify this copy against the original native call.
    """
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
            from .embedding_injection import bond_type_embedding
            z = z + bond_type_embedding(module, bonds)
        z = z + module.contact_conditioning(features)
    z = z + module.s_to_z(inputs)[:, :, None, :] + module.s_to_z_transpose(inputs)[:, None, :, :]
    if module.add_s_to_z_prod:
        z = z + module.s_to_z_prod_out(
            module.s_to_z_prod_in1(inputs)[:, :, None, :]
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
    """Recompute native PTM aggregation outside its silent exception fallback.

    A malformed frame/context must fail in no_grad parity passes as well as in
    autograd calls. Zero signal from legitimate masking is retained explicitly;
    component diagnostics distinguish that from a detached path.
    """
    validate_frame_context(features, coordinates)
    from boltz.model.layers.confidence_utils import compute_frame_pred, compute_ptms
    _, frame_mask = compute_frame_pred(coordinates.reshape(1, -1, 3),
                                       features["frames_idx"], features, 1, inference=True)
    pad = features["token_pad_mask"].bool()
    different = features["asym_id"][:, :, None] != features["asym_id"][:, None, :]
    valid_pairs = frame_mask.reshape_as(pad)[:, :, None] * pad[:, :, None] * pad[:, None, :] * different
    if not bool(valid_pairs.any()):
        raise ValueError("No nondegenerate native interface frames for DBMol confidence")
    ptm, iptm, *_ = compute_ptms(raw["pae_logits"], coordinates.reshape(1, -1, 3), features, 1)
    # Same native prediction-step aggregate, including its declared fallback.
    tm = iptm if not torch.allclose(iptm, torch.zeros_like(iptm)) else ptm
    scores = {k: raw[k] for k in ("complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde")}
    scores.update(ptm=ptm, iptm=iptm, confidence_score=(4 * raw["complex_plddt"] + tm) / 5)
    for key, value in scores.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"Invalid native confidence aggregate: {key}")
        if not bool(value.isfinite().all()):
            from latentmol.optimization import NonfiniteObjective
            raise NonfiniteObjective(f"Nonfinite native confidence aggregate: {key}")
    return scores


class FrozenBoltzDBMol(nn.Module):
    """Profile-specific scalar outputs, consumed by the unchanged latent optimizer.

    Two checkpoint-specific trunks are physically executed per forward. This is
    more expensive than the old affinity-only scorer. All branches receive the
    exact same adapted chemistry/features and fixed pose; there is one molecule.
    """

    def __init__(self, affinity_model, confidence_model, config, recycling_steps=0):
        super().__init__()
        self.recycling_steps = recycling_steps
        self.dbmol_config = config
        self.last_contact_selection = {}
        self.affinity = FrozenBoltzAffinity(affinity_model, recycling_steps, include_logits=True)
        self.structure = confidence_model.eval().requires_grad_(False)
        self.activation_checkpoint_policy = {
            "affinity": self.affinity.activation_checkpoint_policy,
            "structure": install_eval_checkpoints(self.structure, include_confidence=True)}
        self.eval()

    @property
    def manifest_config(self):
        return self.dbmol_config.record()

    @property
    def combined_checkpoint_sha256(self):
        return digest({"affinity_checkpoint_sha256": AFFINITY_SHA256,
                       "confidence_checkpoint_sha256": self.dbmol_config.confidence_checkpoint["sha256"]})

    def _reduce(self, affinity, structure, distogram, features, *, pair=None):
        self.last_contact_selection = {}
        contact = None
        if self.dbmol_config.objective_arithmetic == CONTACT_HEAD64_ARITHMETIC:
            if pair is None:
                raise ValueError("Contact-head arithmetic requires the same structural pair state")
            contact = contact_distogram_float64(self.structure.distogram_module, pair)
        return reduce_profile(affinity, structure, distogram, features, self.dbmol_config,
                              diagnostics=self.last_contact_selection, contact_distogram=contact)

    def forward(self, features, bonds, coordinates):
        validate_frame_context(features, coordinates)
        affinity = self.affinity(features, bonds, coordinates)
        inputs, s, z = attached_trunk(self.structure, features, bonds, self.recycling_steps)
        distogram = self.structure.distogram_module(z)
        confidence = attached_confidence(self.structure.confidence_module, inputs, s, z,
                                         features, coordinates, distogram, bonds)
        scores = strict_confidence_scores(confidence, features, coordinates)
        result = self._reduce(affinity, scores, distogram, features, pair=z)
        from .embedding_injection import any_injected_requires_grad
        if torch.is_grad_enabled() and (features["ref_element"].requires_grad or bonds.requires_grad
                                        or any_injected_requires_grad()):
            for key in ("affinity_pred_value", "affinity_logits_binary", "dbmol_contact_loss", *CONFIDENCE_WEIGHTS):
                if not result[key].requires_grad:
                    raise ValueError(f"Detached DBMol term: {key}")
        return result

    @torch.no_grad()
    def discrete_reference(self, features, coordinates):
        """Native integer trunk/head math at identical pose; no diffusion call."""
        validate_frame_context(features, coordinates)
        affinity = self.affinity.discrete_reference(features, coordinates)
        m = self.structure
        names = ("skip_run_structure", "confidence_prediction", "affinity_prediction")
        old = {name: getattr(m, name) for name in names}
        try:
            m.skip_run_structure, m.confidence_prediction, m.affinity_prediction = True, False, False
            native = m(features, recycling_steps=self.recycling_steps)
        finally:
            for name, value in old.items():
                setattr(m, name, value)
        raw = m.confidence_module(s_inputs=m.input_embedder(features), s=native["s"], z=native["z"],
                                 x_pred=coordinates.reshape(1, -1, 3), feats=features,
                                 pred_distogram_logits=one_distogram(native["pdistogram"]),
                                 multiplicity=1, run_sequentially=False, use_kernels=False)
        return self._reduce(affinity, strict_confidence_scores(raw, features, coordinates),
                            native["pdistogram"], features, pair=native["z"])
