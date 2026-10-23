"""Supervised decoder-hidden to frozen native Boltz chemical embeddings.

Targets are exact *chemical contributions* at the earliest linear projections,
not contextual Boltz states, scores, geometry, atom names or absence embeddings.
The native element slice follows ref_pos[3], ref_charge[1], ref_element[128].
Every checkpoint owns separate targets. The bank clones tensors, never retains
Boltz or VAE modules, and the trainable bridge does not own the target bank.

This experimental smooth interpolation is not a hard-identity guarantee. Native
identity/forward audits and independently rebuilt hard-candidate scores remain
required before claiming useful molecular optimization.
"""

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import re

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from latentmol.representation import GraphSpec, HardGraphBatch
from .contracts import NATIVE_BOND_IDS


BRIDGE_VERSION = "epoch36-native-embedding-bridge-v1"
_KINDS = frozenset(("atom_element", "atom_charge", "connectivity", "type"))


@dataclass(frozen=True)
class ProjectionDescriptor:
    """Identity of one checkpoint-specific native chemical linear contribution."""

    key: str
    kind: str
    rows: int
    width: int
    native_path: str

    def record(self) -> dict:
        return dict(key=self.key, kind=self.kind, rows=self.rows,
                    width=self.width, native_path=self.native_path)


@dataclass(frozen=True)
class BridgeTargets:
    """Aligned hard-graph targets; masked rows are excluded, never native padding."""

    values: dict[str, Tensor]
    atom_mask: Tensor
    pair_mask: Tensor
    bond_present: Tensor


def _validate_hard(hard: HardGraphBatch, spec: GraphSpec) -> tuple[Tensor, Tensor]:
    if hard.atoms.ndim != 2 or hard.atoms.shape[1] != spec.max_nodes:
        raise ValueError("Hard graph needs [batch, GraphSpec.max_nodes] atom labels")
    shape = hard.atoms.shape
    for value, expected, count in (
            (hard.atoms, shape, len(spec.atomic_numbers)),
            (hard.charges, shape, len(spec.formal_charges)),
            (hard.bonds, (*shape, shape[1]), len(spec.bond_names))):
        if value.shape != expected or value.dtype != torch.long:
            raise ValueError("Hard graph labels have malformed shape or dtype")
        if value.device != hard.atoms.device or bool(((value < 0) | (value >= count)).any()):
            raise ValueError("Hard graph labels need shared device and bounded categories")
    if not torch.equal(hard.bonds, hard.bonds.transpose(1, 2)):
        raise ValueError("Hard graph bonds must be symmetric")
    active = hard.atoms != 0
    pairs = active[:, :, None] & active[:, None, :]
    diagonal = torch.eye(shape[1], device=active.device, dtype=torch.bool)[None]
    if bool((hard.bonds.masked_select(~pairs | diagonal) != 0).any()):
        raise ValueError("Hard graph requires NONE on diagonal and absent pairs")
    neutral = spec.formal_charges.index(0)
    if bool((hard.charges.masked_select(~active) != neutral).any()):
        raise ValueError("Absent slots must have neutral categorical charge")
    if any(number < 0 or number >= 128 for number in spec.atomic_numbers):
        raise ValueError("Boltz element vocabulary is indexed by atomic number below 128")
    if any(name not in NATIVE_BOND_IDS for name in spec.bond_names):
        raise ValueError("Graph bond vocabulary is unsupported by native Boltz")
    return active, pairs


class NativeProjectionBank(nn.Module):
    """Frozen, serializable prototypes copied from authenticated native models.

    Construction does not authenticate a model's file identity. The caller must
    strict-load authenticated checkpoints and retain source/config/file receipts.
    Optional SHA bindings are metadata assertions, not a checkpoint audit.
    """

    def __init__(self, projections: Mapping[str, Tensor],
                 descriptors: list[ProjectionDescriptor], *,
                 checkpoint_sha256: Mapping[str, str] | None = None):
        super().__init__()
        if not descriptors or len({d.key for d in descriptors}) != len(descriptors):
            raise ValueError("Projection keys must be unique and nonempty")
        if set(projections) != {d.key for d in descriptors}:
            raise ValueError("Projection tensors and descriptors disagree")
        self.descriptors = tuple(descriptors)
        self.checkpoint_sha256 = dict(checkpoint_sha256 or {})
        for name, sha in self.checkpoint_sha256.items():
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError(f"Malformed checkpoint SHA256: {name}")
        for index, descriptor in enumerate(self.descriptors):
            value = projections[descriptor.key]
            expected_rows = {"atom_element": 128, "atom_charge": 1,
                             "connectivity": 2, "type": 7}
            if (descriptor.kind not in _KINDS or descriptor.rows != expected_rows[descriptor.kind]
                    or descriptor.width < 1 or value.shape != (descriptor.rows, descriptor.width)
                    or not value.is_floating_point() or not bool(value.isfinite().all())):
                raise ValueError(f"Malformed native projection: {descriptor.key}")
            self.register_buffer(f"projection_{index:03d}", value.detach().clone())

    @property
    def dimensions(self) -> dict[str, int]:
        return {descriptor.key: descriptor.width for descriptor in self.descriptors}

    def projection(self, key: str) -> Tensor:
        for index, descriptor in enumerate(self.descriptors):
            if descriptor.key == key:
                return getattr(self, f"projection_{index:03d}")
        raise KeyError(key)

    def metadata(self) -> dict:
        return {"version": BRIDGE_VERSION,
                "raw_atom_layout": ["ref_pos:3", "ref_charge:1", "ref_element:128"],
                "checkpoint_sha256": dict(self.checkpoint_sha256),
                "projections": [descriptor.record() for descriptor in self.descriptors]}

    @classmethod
    def from_metadata(cls, metadata: dict, state_dict: Mapping[str, Tensor]):
        if (metadata.get("version") != BRIDGE_VERSION or metadata.get("raw_atom_layout")
                != ["ref_pos:3", "ref_charge:1", "ref_element:128"]):
            raise ValueError("Unsupported native embedding-bank metadata")
        descriptors = [ProjectionDescriptor(**record) for record in metadata["projections"]]
        expected = {f"projection_{i:03d}" for i in range(len(descriptors))}
        if set(state_dict) != expected:
            raise ValueError("Native embedding-bank state keys disagree with metadata")
        bank = cls({d.key: state_dict[f"projection_{i:03d}"]
                    for i, d in enumerate(descriptors)}, descriptors,
                   checkpoint_sha256=metadata.get("checkpoint_sha256", {}))
        bank.load_state_dict(state_dict, strict=True)
        return bank

    @classmethod
    def from_models(cls, models: Mapping[str, nn.Module], *,
                    checkpoint_sha256: Mapping[str, str] | None = None,
                    confidence_models: Iterable[str] | None = None):
        """Extract input atom projections and trunk/optional confidence bonds.

        Only the pinned early-feature layout is supported. Backbone/residue
        extensions are rejected explicitly; no width or trained head is guessed.
        By default only the DBMol ``structure`` role executes confidence. A
        checkpoint's stored confidence flag alone does not establish reachability
        in FrozenBoltzAffinity. Custom call graphs may pass explicit role names.
        Disabled confidence Z-input branches contribute no unreachable targets.
        """
        projections, descriptors = {}, []

        def add(key, kind, value, path):
            value = value.detach().clone()
            projections[key] = value
            descriptors.append(ProjectionDescriptor(key, kind, value.shape[0], value.shape[1], path))

        def bonds(name, module, path, prefix):
            linear = module.token_bonds
            if (not isinstance(linear, nn.Linear) or linear.in_features != 1
                    or linear.bias is not None):
                raise ValueError(f"Unsupported native connectivity projection: {path}")
            # Include the zero prototype: no bond is not a missing training row.
            add(f"{name}.{prefix}_connectivity", "connectivity",
                torch.cat((torch.zeros_like(linear.weight.T), linear.weight.T), 0),
                f"{path}.token_bonds")
            if module.bond_type_feature:
                embedding = module.token_bonds_type
                if (not isinstance(embedding, nn.Embedding) or embedding.num_embeddings != 7
                        or embedding.embedding_dim != linear.out_features):
                    raise ValueError(f"Unsupported native typed-bond projection: {path}")
                add(f"{name}.{prefix}_type", "type", embedding.weight,
                    f"{path}.token_bonds_type")

        if not models:
            raise ValueError("At least one native model is required")
        if checkpoint_sha256 is not None and set(checkpoint_sha256) != set(models):
            raise ValueError("Checkpoint bindings must cover exactly the supplied models")
        if isinstance(confidence_models, str):
            raise ValueError("Confidence model roles must be an iterable of names")
        confidence_roles = ({name for name, model in models.items()
                             if name == "structure" and getattr(model, "confidence_prediction", False)}
                            if confidence_models is None else set(confidence_models))
        if not confidence_roles <= set(models):
            raise ValueError("Confidence model roles must be present in the supplied models")
        for name, model in models.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError("Native model names must be unambiguous identifiers")
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError("Native target model must be frozen and in eval mode")
            encoder = model.input_embedder.atom_encoder
            if (getattr(encoder, "use_atom_backbone_feat", False)
                    or getattr(encoder, "use_residue_feats_atoms", False)):
                raise ValueError("Extended native atom feature layouts are unsupported")
            linear = encoder.embed_atom_features
            expected = 132 + (0 if encoder.use_no_atom_char else 256)
            if not isinstance(linear, nn.Linear) or linear.in_features != expected:
                raise ValueError("Native atom projection width disagrees with pinned feature layout")
            add(f"{name}.atom_element", "atom_element", linear.weight[:, 4:132].T,
                f"{name}.input_embedder.atom_encoder.embed_atom_features.element")
            add(f"{name}.atom_charge", "atom_charge", linear.weight[:, 3:4].T,
                f"{name}.input_embedder.atom_encoder.embed_atom_features.charge")
            bonds(name, model, name, "trunk")
            if name in confidence_roles:
                if not getattr(model, "confidence_prediction", False):
                    raise ValueError("Requested confidence role lacks its native trained module")
                confidence = model.confidence_module
                if confidence.add_z_input_to_z:
                    bonds(name, confidence, f"{name}.confidence_module", "confidence")
        return cls(projections, descriptors, checkpoint_sha256=checkpoint_sha256)

    @torch.no_grad()
    def targets(self, hard: HardGraphBatch, spec: GraphSpec) -> BridgeTargets:
        """Generate exact frozen early features for explicitly aligned hard labels.

        Chemical sanitization/support and identity preprocessing are the caller's
        responsibility; this function validates tensor schema, never repairs it.
        Diagonal active pairs retain native NONE typed embeddings. Absent slots
        are zero *excluded* targets, not claims of native NO_ATOM semantics.
        """
        active, pairs = _validate_hard(hard, spec)
        numbers = hard.atoms.new_tensor(spec.atomic_numbers)[hard.atoms]
        charges = hard.atoms.new_tensor(spec.formal_charges)[hard.charges]
        native_ids = hard.atoms.new_tensor([NATIVE_BOND_IDS[n] for n in spec.bond_names])[hard.bonds]
        present = hard.bonds != 0
        values = {}
        for descriptor in self.descriptors:
            projection = self.projection(descriptor.key)
            if projection.device != hard.atoms.device:
                raise ValueError("Target bank and hard graphs must share a device")
            if descriptor.kind == "atom_element":
                value, mask = projection[numbers], active
            elif descriptor.kind == "atom_charge":
                value, mask = charges[..., None].to(projection) * projection[0], active
            elif descriptor.kind == "connectivity":
                value, mask = projection[present.long()], pairs
            else:
                value, mask = projection[native_ids], pairs
            values[descriptor.key] = value.masked_fill(~mask[..., None], 0)
        return BridgeTargets(values, active, pairs, present & pairs)

    def loss(self, predictions: Mapping[str, Tensor], targets: BridgeTargets):
        """Return summed per-projection MSE and explicit group/count diagnostics.

        Each atom target kind has its own mean. Each pair target averages the
        populated NONE and present-bond group means equally. Output dimensions
        are mean-reduced, so wide native layers and plentiful NONE pairs do not
        silently dominate. Empty groups contribute zero with count zero.
        """
        if set(predictions) != set(self.dimensions) or set(targets.values) != set(predictions):
            raise ValueError("Prediction and target projection keys disagree")
        metrics, losses = {}, []
        for descriptor in self.descriptors:
            key = descriptor.key
            prediction, target = predictions[key], targets.values[key]
            if prediction.shape != target.shape:
                raise ValueError(f"Prediction/target shapes disagree: {key}")
            if target.requires_grad:
                raise ValueError("Native embedding targets must remain detached")
            row_error = (prediction - target).square().mean(-1)
            if descriptor.kind.startswith("atom_"):
                groups = {"atom": targets.atom_mask}
            else:
                groups = {"none": targets.pair_mask & ~targets.bond_present,
                          "present": targets.pair_mask & targets.bond_present}
            group_losses = []
            for name, mask in groups.items():
                count = mask.sum()
                value = row_error.masked_select(mask).sum() / count.clamp_min(1)
                metrics[f"{key}/{name}_count"] = count.detach()
                metrics[f"{key}/{name}_mse"] = value.detach()
                group_losses.append(value)
            populated = sum((mask.any().to(row_error) for mask in groups.values()))
            value = sum(group_losses) / populated.clamp_min(1)
            metrics[f"{key}/mse"] = value.detach()
            losses.append(value)
        total = torch.stack(losses).sum()
        metrics["loss"] = total.detach()
        return total, metrics

    @torch.no_grad()
    def identity_metrics(self, predictions: Mapping[str, Tensor],
                         hard: HardGraphBatch, spec: GraphSpec) -> dict[str, Tensor]:
        """Nearest native chemical prototypes; not a global identity guarantee."""
        active, pairs = _validate_hard(hard, spec)
        if set(predictions) != set(self.dimensions):
            raise ValueError("Identity probe projection keys disagree")
        metrics = {}
        for descriptor in self.descriptors:
            projection = self.projection(descriptor.key)
            if descriptor.kind == "atom_element":
                prototypes = projection[hard.atoms.new_tensor(spec.atomic_numbers)[1:]]
                labels, mask = hard.atoms - 1, active
            elif descriptor.kind == "atom_charge":
                prototypes = projection[0] * projection.new_tensor(spec.formal_charges)[:, None]
                labels, mask = hard.charges, active
            elif descriptor.kind == "connectivity":
                prototypes, labels, mask = projection, (hard.bonds != 0).long(), pairs
            else:
                prototypes = projection[hard.atoms.new_tensor([NATIVE_BOND_IDS[n] for n in spec.bond_names])]
                labels, mask = hard.bonds, pairs
            value = predictions[descriptor.key]
            if value.shape != (*mask.shape, descriptor.width):
                raise ValueError(f"Identity probe shape disagreement: {descriptor.key}")
            # cdist avoids retaining a [batch,N,N,categories,width] allocation.
            predicted = torch.cdist(value.reshape(-1, descriptor.width), prototypes).argmin(-1).reshape_as(labels)
            correct = ((predicted == labels) & mask).sum()
            count = mask.sum()
            metrics[f"{descriptor.key}/identity_accuracy"] = correct.to(value) / count.clamp_min(1)
            metrics[f"{descriptor.key}/identity_count"] = count
        return metrics


class HiddenEmbeddingBridge(nn.Module):
    """Small shared atom/charge/pair MLPs fitted only to native early features.

    Per-projection heads use one shared trunk per chemical input group. Symmetry
    is imposed on actual final bond-head input *before* evaluating the pair MLP.
    The bank is deliberately not retained: state contains only bridge parameters.
    All parameters must be frozen before latent optimization.
    """

    def __init__(self, bank: NativeProjectionBank, *, atom_input_dim: int = 96,
                 pair_input_dim: int = 336, charge_input_dim: int = 192,
                 hidden_dim: int = 128):
        super().__init__()
        if min(atom_input_dim, pair_input_dim, charge_input_dim, hidden_dim) < 1:
            raise ValueError("Bridge dimensions must be positive")
        self.atom_input_dim, self.pair_input_dim = atom_input_dim, pair_input_dim
        self.charge_input_dim, self.hidden_dim = charge_input_dim, hidden_dim
        self.descriptors = tuple(bank.descriptors)
        self.atom_trunk = nn.Sequential(nn.LayerNorm(atom_input_dim),
                                       nn.Linear(atom_input_dim, hidden_dim), nn.SiLU())
        self.charge_trunk = nn.Sequential(nn.LayerNorm(charge_input_dim),
                                         nn.Linear(charge_input_dim, hidden_dim), nn.SiLU())
        self.pair_trunk = nn.Sequential(nn.LayerNorm(pair_input_dim),
                                       nn.Linear(pair_input_dim, hidden_dim), nn.SiLU())
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, d.width) for d in self.descriptors)

    def configuration(self) -> dict:
        return {"version": BRIDGE_VERSION, "atom_input_dim": self.atom_input_dim,
                "pair_input_dim": self.pair_input_dim, "charge_input_dim": self.charge_input_dim,
                "hidden_dim": self.hidden_dim,
                "dimensions": {d.key: d.width for d in self.descriptors}}

    def forward(self, atom_hidden: Tensor, bond_input: Tensor, active: Tensor,
                charge_input: Tensor | None = None) -> dict[str, Tensor]:
        if atom_hidden.ndim != 3 or atom_hidden.shape[-1] != self.atom_input_dim:
            raise ValueError("Malformed final decoder atom hidden states")
        batch, nodes = atom_hidden.shape[:2]
        if bond_input.shape != (batch, nodes, nodes, self.pair_input_dim):
            raise ValueError("Malformed final decoder bond-head input")
        if active.shape == (nodes,):
            active = active[None].expand(batch, -1)
        if active.shape != (batch, nodes) or active.dtype != torch.bool:
            raise ValueError("Bridge requires an explicit boolean occupancy mask")
        if (active.device != atom_hidden.device or bond_input.device != atom_hidden.device
                or bond_input.dtype != atom_hidden.dtype):
            raise ValueError("Bridge hidden states and occupancy must share device/dtype")
        needs_charge = any(d.kind == "atom_charge" for d in self.descriptors)
        if needs_charge and (charge_input is None
                             or charge_input.shape != (batch, nodes, self.charge_input_dim)
                             or charge_input.device != atom_hidden.device
                             or charge_input.dtype != atom_hidden.dtype):
            raise ValueError("Charge projection requires aligned final charge-head input")
        atom = self.atom_trunk(atom_hidden)
        charge = self.charge_trunk(charge_input) if needs_charge else None
        pair = self.pair_trunk((bond_input + bond_input.transpose(1, 2)) * 0.5)
        pair_mask = active[:, :, None] & active[:, None, :]
        values = {}
        for descriptor, head in zip(self.descriptors, self.heads, strict=True):
            if descriptor.kind == "atom_element":
                hidden, mask = atom, active
            elif descriptor.kind == "atom_charge":
                hidden, mask = charge, active
            else:
                hidden, mask = pair, pair_mask
            values[descriptor.key] = head(hidden).masked_fill(~mask[..., None], 0)
        return values
