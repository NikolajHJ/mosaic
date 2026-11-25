"""Explicit DBMol loss reductions; no model, sampling, repair or chemistry calls.

Paper: arXiv:2607.19237v1, equations 2--6 and Appendix C.1. The paper does
not specify epsilon, ensemble-logit aggregation or cutoff bin rounding. Those
choices are declared here rather than presented as exact upstream replication.
HPC numerical/native validation is required before this profile is scientific
evidence. Historical binary-only configurations do not opt into this module.
"""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re

import torch

from latentmol.multimodel.config import number


PROFILE = "dbmol_fixed_pose_v1"
NATIVE_ARITHMETIC = "native_v1"
CONTACT64_ARITHMETIC = "contact_and_assembly_float64_v1"
CONTACT_HEAD64_ARITHMETIC = "contact_head_and_assembly_float64_v1"
CONFIDENCE_WEIGHTS = {"iptm": 1.0, "complex_iplddt": 0.3,
                      "complex_ipde": 0.3, "complex_pde": 0.1,
                      "confidence_score": 0.3, "ptm": 0.1}
CONFIDENCE_SIGNS = {"iptm": -1.0, "complex_iplddt": -1.0,
                    "complex_ipde": 1.0, "complex_pde": 1.0,
                    "confidence_score": -1.0, "ptm": -1.0}
COMPONENT_OUTPUTS = ("dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss")


@dataclass(frozen=True)
class DBMolObjectiveConfig:
    """One sealed checkpoint and pocket definition, shared across count contexts.

    Residue indices and asym IDs are native Boltz tensor indices, not PDB author
    numbering. Pocket generation is a separate HPC preprocessing action. Never
    update this mask using the optimized ligand or rank pockets by outcomes.
    """

    confidence_checkpoint: dict
    pocket_residues: tuple[dict, ...]
    profile: str = PROFILE
    pocket_origin: str = "explicit_native_residues"
    pocket_reference: dict | None = None
    affinity_regression_weight: float = 1.0
    affinity_logit_weight: float = 1.0
    contact_weight: float = 2.5
    confidence_weight: float = 2.5
    ipsae_weight: float = 0.0
    confidence_weights: dict = field(default_factory=lambda: dict(CONFIDENCE_WEIGHTS))
    distance_cutoff: float = 8.0
    epsilon: float = 1e-8
    binder_logit_policy: str = "mean_native_head_logits"
    bin_policy: str = "native_64_midpoint_lt_cutoff"
    # Explicit changed arithmetic, never a silent reinterpretation of old runs.
    # CONTACT64 widens the reduction/assembly only. CONTACT_HEAD64 additionally
    # widens the contact-only affine distogram projection, before rounding.
    # Confidence retains native logits; trunks and repair NLL stay native.
    objective_arithmetic: str = NATIVE_ARITHMETIC

    @classmethod
    def from_mapping(cls, raw, base: Path):
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("DBMol profile contains unsupported keys")
        data = dict(raw)
        asset = data.get("confidence_checkpoint")
        if (not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
                or not isinstance(asset["path"], str) or not asset["path"]
                or not isinstance(asset["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])):
            raise ValueError("DBMol requires an explicit confidence checkpoint path/SHA256")
        data["confidence_checkpoint"] = {**asset, "path": str((base / asset["path"]).resolve())}
        residues = data.get("pocket_residues")
        if not isinstance(residues, (list, tuple)) or not residues:
            raise ValueError("DBMol requires a nonempty frozen pocket")
        keys = []
        for residue in residues:
            if (not isinstance(residue, dict) or set(residue) != {"asym_id", "residue_index"}
                    or any(type(residue[k]) is not int or residue[k] < 0 for k in residue)):
                raise ValueError("Pocket residues require native nonnegative integer asym_id/residue_index")
            keys.append((residue["asym_id"], residue["residue_index"]))
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate pocket residue")
        data["pocket_residues"] = tuple({"asym_id": a, "residue_index": r} for a, r in sorted(keys))
        obj = cls(**data)
        if obj.objective_arithmetic not in (NATIVE_ARITHMETIC, CONTACT64_ARITHMETIC, CONTACT_HEAD64_ARITHMETIC):
            raise ValueError("Unsupported DBMol objective arithmetic")
        if obj.profile != PROFILE or obj.binder_logit_policy != "mean_native_head_logits" or obj.bin_policy != "native_64_midpoint_lt_cutoff":
            raise ValueError("Unsupported DBMol profile/ensemble/bin policy")
        if obj.pocket_origin not in ("explicit_native_residues", "initial_pose_heavy_atom_8A"):
            raise ValueError("Declare the frozen pocket's origin")
        if obj.pocket_origin == "initial_pose_heavy_atom_8A":
            ref = obj.pocket_reference
            if (not isinstance(ref, dict) or set(ref) != {"prepared_sha256", "context_sha256"}
                    or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in ref.values())):
                raise ValueError("Initial-pose pocket requires authenticated prepared/context hashes")
        elif obj.pocket_reference is not None:
            raise ValueError("Explicit pocket does not use inferred-pose provenance")
        for name in ("affinity_regression_weight", "affinity_logit_weight", "contact_weight", "confidence_weight", "ipsae_weight"):
            number(getattr(obj, name), name)
        number(obj.distance_cutoff, "distance_cutoff", positive=True)
        number(obj.epsilon, "epsilon", positive=True)
        if obj.epsilon >= 1 or not 2 < obj.distance_cutoff < 22:
            raise ValueError("Require epsilon below one and cutoff inside pinned native distance range")
        if not isinstance(obj.confidence_weights, dict) or set(obj.confidence_weights) != set(CONFIDENCE_WEIGHTS):
            raise ValueError("Declare exactly the six supported native confidence weights")
        for key, value in obj.confidence_weights.items():
            number(value, key)
        if not obj.affinity_regression_weight and not obj.affinity_logit_weight and not obj.contact_weight and not obj.confidence_weight:
            raise ValueError("At least one DBMol loss weight must be positive")
        return obj

    def record(self):
        result = asdict(self)
        result["pocket_residues"] = list(self.pocket_residues)
        if self.ipsae_weight == 0:
            result.pop("ipsae_weight")
        if self.objective_arithmetic == NATIVE_ARITHMETIC:
            # Old absent-key resolved configs and identity hashes stay exact.
            result.pop("objective_arithmetic")
        return result

    def verify_reference(self, prepared, prepared_sha256):
        """Authenticate the fixed reference, never a subsequently selected pose."""
        if self.pocket_origin == "initial_pose_heavy_atom_8A":
            from .provenance import context_digest
            expected = {"prepared_sha256": prepared_sha256,
                        "context_sha256": context_digest(prepared)}
            if self.pocket_reference != expected:
                raise ValueError("DBMol pocket reference differs from the frozen anchor")


def pocket_token_mask(features, config):
    """Resolve the same biological residues after ligand count/slot changes."""
    pad = features["token_pad_mask"][0].bool()
    protein = (features["mol_type"][0] == 0) & pad
    mask = torch.zeros_like(protein)
    for residue in config.pocket_residues:
        chosen = protein & (features["asym_id"][0] == residue["asym_id"])
        chosen = chosen & (features["residue_index"][0] == residue["residue_index"])
        if int(chosen.sum()) != 1:
            raise ValueError("Frozen pocket residue missing or ambiguous in native context")
        mask |= chosen
    ligand = features["affinity_token_mask"][0].bool() & pad
    if not mask.any() or not ligand.any() or (mask & ligand).any():
        raise ValueError("Require disjoint nonempty pocket and ligand token masks")
    return mask, ligand


def native_bin_centers(logits):
    """Construct the pinned representatives in the ORIGINAL native head dtype.

    Widening first can change rounded bin membership near a supplied cutoff.
    Build these before the precision candidate casts its logits to FP64.
    """
    if logits.shape[-1] != 64:
        raise ValueError("Pinned native Boltz distogram requires 64 bins")
    boundaries = torch.linspace(2.0, 22.0, 63, device=logits.device, dtype=logits.dtype)
    half = (boundaries[1] - boundaries[0]) / 2
    return torch.cat((boundaries[:1] - half,
                      (boundaries[:-1] + boundaries[1:]) / 2,
                      boundaries[-1:] + half))


def contact_loss(logits, pocket, ligand, *, cutoff=8.0, epsilon=1e-8, bin_centers=None,
                 diagnostics=None):
    """Equation 6: equal pocket-average and strongest-pair anchor losses.

    Use the native protein-to-ligand orientation, one distogram and batch one.
    Minima are piecewise differentiable; they are not molecule argmax sampling.
    Log-space evaluation keeps low-contact predictions numerically informative.
    """
    if logits.ndim == 5 and logits.shape[3] == 1:
        logits = logits[:, :, :, 0]
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != logits.shape[2]:
        raise ValueError("Require one square native distogram")
    if pocket.dtype != torch.bool or ligand.dtype != torch.bool or pocket.shape != ligand.shape or pocket.numel() != logits.shape[1]:
        raise ValueError("Malformed contact token masks")
    if not pocket.any() or not ligand.any() or (pocket & ligand).any():
        raise ValueError("Contact masks must be nonempty and disjoint")
    if bin_centers is None:
        bin_centers = native_bin_centers(logits)
    if bin_centers.shape != (logits.shape[-1],) or not bool(torch.isfinite(bin_centers).all()):
        raise ValueError("Explicit bin representatives must cover every distogram category")
    near = bin_centers.to(logits.device) < cutoff
    if not near.any() or bool(near.all()):
        raise ValueError("Contact cutoff must partition the native distance bins")
    selected = logits[0][pocket][:, ligand]
    log_mass = torch.logsumexp(selected.log_softmax(-1)[..., near], -1)
    pair_loss = -torch.logaddexp(log_mass, log_mass.new_tensor(math.log(epsilon)))
    best_per_residue, winners = pair_loss.min(-1)
    pocket_loss, anchor_loss = best_per_residue.mean(), pair_loss.min()
    if diagnostics is not None:
        # Detached bookkeeping only. The original minima above drive gradients;
        # this witness reveals winner changes/near ties in finite differences.
        detached = pair_loss.detach()
        pocket_indices = pocket.nonzero().flatten().tolist()
        ligand_indices = ligand.nonzero().flatten().tolist()
        anchor = int(detached.argmin())
        diagnostics.update(best_ligand_token_per_pocket=[ligand_indices[i] for i in winners.tolist()],
                           pocket_token_indices=pocket_indices,
                           anchor_tokens=[pocket_indices[anchor // len(ligand_indices)],
                                          ligand_indices[anchor % len(ligand_indices)]])
        if len(ligand_indices) > 1:
            two = detached.topk(2, dim=-1, largest=False).values
            diagnostics["smallest_best_vs_second_loss_margin"] = float((two[:, 1] - two[:, 0]).min())
    return {"dbmol_contact_loss": (pocket_loss + anchor_loss) / 2,
            "dbmol_contact_pocket_loss": pocket_loss,
            "dbmol_contact_anchor_loss": anchor_loss}


def reduce_profile(affinity, structure, distogram, features, config, *, diagnostics=None,
                   contact_distogram=None):
    """Shared production/diagnostic entry point; the native tensors stay frozen.

    The reduction-only mode uses the same logits. The head mode requires a
    separately widened projection of the SAME final pair state; native logits
    still define the bin partition and feed confidence. Neither mode changes
    minima, cutoff, epsilon or frozen weights. Fresh gates are mandatory.
    """
    pocket, ligand = pocket_token_mask(features, config)
    centers = None
    arithmetic = getattr(config, "objective_arithmetic", NATIVE_ARITHMETIC)
    if arithmetic == CONTACT_HEAD64_ARITHMETIC:
        if (not isinstance(contact_distogram, torch.Tensor)
                or contact_distogram.dtype != torch.float64
                or contact_distogram.shape != distogram.shape
                or contact_distogram.device != distogram.device):
            raise ValueError("Contact-head arithmetic requires explicit FP64 logits from the shared pair state")
        centers = native_bin_centers(distogram).double()
        distogram = contact_distogram
    elif contact_distogram is not None:
        raise ValueError("Separate contact logits require the declared contact-head arithmetic")
    elif arithmetic == CONTACT64_ARITHMETIC:
        centers = native_bin_centers(distogram).double()
        distogram = distogram.double()
    contact = contact_loss(distogram, pocket, ligand, cutoff=config.distance_cutoff,
                           epsilon=config.epsilon, bin_centers=centers,
                           diagnostics=diagnostics)
    return reduce_objective({**affinity, **structure, **contact}, config)


def reduce_objective(outputs, config):
    """Apply declared signs/weights once; retain native primitives for audit.

    FP64 assembly cannot restore precision already lost inside native heads.
    Repair weighting is outside this module and is deliberately unchanged.
    """
    required = {"affinity_pred_value", "affinity_logits_binary", "dbmol_contact_loss", *CONFIDENCE_WEIGHTS}
    ipsae_weight = getattr(config, "ipsae_weight", 0.0)
    if ipsae_weight:
        required.add("ligand_ipsae")
    for key in required:
        value = outputs.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"Missing, nonfinite or nonscalar DBMol term: {key}")
        if not bool(value.isfinite().all()):
            from latentmol.optimization import NonfiniteObjective
            raise NonfiniteObjective(f"Nonfinite DBMol term: {key}")
    values = ({k: value.double() if isinstance(value, torch.Tensor) else value
               for k, value in outputs.items()}
              if getattr(config, "objective_arithmetic", NATIVE_ARITHMETIC)
              in (CONTACT64_ARITHMETIC, CONTACT_HEAD64_ARITHMETIC) else outputs)
    affinity = (config.affinity_regression_weight * values["affinity_pred_value"]
                - config.affinity_logit_weight * values["affinity_logits_binary"])
    confidence = sum(config.confidence_weights[k] * CONFIDENCE_SIGNS[k] * values[k]
                     for k in CONFIDENCE_WEIGHTS)
    if ipsae_weight:
        confidence = confidence - ipsae_weight * values["ligand_ipsae"]
    result = dict(outputs)
    result.update(dbmol_affinity_loss=affinity, dbmol_confidence_loss=confidence,
                  dbmol_weighted_affinity_loss=affinity,
                  dbmol_weighted_contact_loss=config.contact_weight * values["dbmol_contact_loss"],
                  dbmol_weighted_confidence_loss=config.confidence_weight * confidence)
    result["dbmol_total_loss"] = sum(result[k] for k in (
        "dbmol_weighted_affinity_loss", "dbmol_weighted_contact_loss", "dbmol_weighted_confidence_loss"))
    if not bool(result["dbmol_total_loss"].isfinite().all()):
        from latentmol.optimization import NonfiniteObjective
        raise NonfiniteObjective("Nonfinite weighted DBMol total")
    return result
