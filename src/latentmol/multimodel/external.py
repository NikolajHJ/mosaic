"""Prepared native-input boundary for optional frozen external models.

Native feature preparation is deliberately outside the differentiable path.
Export one *ordinary hard molecule* using the upstream featurizer, preserve its
atom correspondence, and package the result with :func:`export_native_bundle`.
These adapters are experimental and have not been validated on native weights.
Loading never downloads checkpoints or invents a replacement scoring model.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from latentmol.representation import GraphSpec, HardGraphBatch, RelaxedGraph
from latentmol.scoring import InputSupport


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_source_sha256(root: str | Path, package: str) -> str:
    """Hash relative names and bytes of every Python file in a native package.

    ``root`` is the checkout/install directory containing ``nesso`` or
    ``protenix``. The compact JSON list of [relative POSIX path, file SHA256]
    pairs is hashed, in sorted path order. Uncommitted edits are included.
    Dependencies and compiled kernels must additionally be recorded by the run.
    """
    root = Path(root).resolve()
    paths = sorted((root / package).rglob("*.py"))
    if not paths:
        raise ValueError(f"No native Python source under {root / package}")
    records = [[p.relative_to(root).as_posix(), file_sha256(p)] for p in paths]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def verify_native_assets(options: dict, package: str) -> dict:
    """Fail before loading any model when required provenance is absent/mismatched."""
    assets: dict[str, Any] = {"validation_status": "experimental_native_unvalidated"}
    for stem in ("bundle", "weights", "model_config"):
        path_key, hash_key = f"{stem}_path", f"{stem}_sha256"
        if not options.get(path_key) or not options.get(hash_key):
            raise ValueError(f"{package} requires {path_key} and {hash_key}")
        path = Path(options[path_key]).resolve()
        observed = file_sha256(path)
        if observed != options[hash_key]:
            raise ValueError(f"{package} {stem} SHA256 mismatch")
        assets[path_key], assets[hash_key] = str(path), observed
    if not options.get("native_source_root") or not options.get("native_source_sha256"):
        raise ValueError(f"{package} requires native_source_root and native_source_sha256")
    root = Path(options["native_source_root"]).resolve()
    observed = native_source_sha256(root, package)
    if observed != options["native_source_sha256"]:
        raise ValueError(f"{package} native source SHA256 mismatch")
    assets.update(native_source_root=str(root), native_source_sha256=observed)
    return assets


def checked_import(module: str, package: str, assets: dict):
    """An installed optional dependency must resolve inside the audited checkout."""
    try:
        loaded = importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Install {package} and its pinned dependencies in the HPC model environment; "
            "this loader does not install or download them."
        ) from exc
    expected = Path(assets["native_source_root"]).resolve() / package
    if not getattr(loaded, "__file__", None) or not Path(loaded.__file__).resolve().is_relative_to(expected):
        raise ValueError(f"Imported {module} is outside the audited source directory {expected}")
    return loaded


def tree_to(value: Any, device: str | torch.device) -> Any:
    """Clone a tensor/primitive tree; arbitrary serialized native objects are rejected."""
    if isinstance(value, Tensor):
        if value.is_floating_point() and not value.isfinite().all():
            raise ValueError("Prepared features must be finite")
        return value.detach().clone().to(device)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree_to(v, device) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported native bundle value: {type(value).__name__}")


def tensor_tree_sha256(value: Any) -> str:
    """Content identity for fixed native features, including nested cache tensors.

    The byte view also supports bfloat16, which NumPy cannot represent directly.
    Device location is intentionally excluded; dtype, shape and bytes are not.
    """
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, Tensor):
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item):
                digest.update(json.dumps(key).encode())
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}[".encode())
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
            digest.update(b";")

    visit(value)
    return digest.hexdigest()


def export_native_bundle(path: str | Path, *, model: str, spec: GraphSpec,
                         features: dict, seed: HardGraphBatch, slot_to_atom: Tensor,
                         slot_to_token: Tensor, ligand_atom_mask: Tensor,
                         ligand_token_mask: Tensor, element_index: Tensor,
                         metadata: dict, extra: dict | None = None) -> str:
    """Write tensor-only native features from an upstream hard-input preparation.

    ``element_index`` explicitly maps GraphSpec atomic categories to native
    one-hot columns (index zero must be -1: NO_ATOM has no native row).
    Supply masks over *all* native atoms/tokens belonging to the ligand. Native
    bond types are supplied by Nesso's official featurizer, not inferred here.
    ``extra`` is ``affinity_keep_indices`` for Nesso, or fixed
    ``distance_pair_mask``/``distance_target_bins`` for Protenix. A native
    preparation must compute those first; this function never runs a model.

    Metadata must identify the upstream preparation command, native source,
    seed molecule and feature losses. This exporter establishes shape/identity
    consistency, not a scientific claim of native feature parity.
    """
    payload = dict(schema_version=1, model=model, spec=spec.to_dict(), features=features,
                   seed=dict(atoms=seed.atoms, charges=seed.charges, bonds=seed.bonds),
                   slot_to_atom=slot_to_atom, slot_to_token=slot_to_token,
                   ligand_atom_mask=ligand_atom_mask, ligand_token_mask=ligand_token_mask,
                   element_index=element_index, metadata=metadata, extra=extra or {})
    payload = tree_to(payload, "cpu")
    NativeBundle(payload, spec=spec, model=model, device="cpu")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Use a unique prepared bundle path: {path}")
    torch.save(payload, path)
    return file_sha256(path)


class NativeBundle:
    """Validated fixed-occupancy correspondence and immutable native context."""

    def __init__(self, payload: dict, *, spec: GraphSpec, model: str, device: str):
        if model not in ("nesso1", "protenix_v1"):
            raise ValueError("Unknown native bundle model")
        if payload.get("schema_version") != 1 or payload.get("model") != model:
            raise ValueError("Wrong native bundle schema/model")
        if GraphSpec.from_dict(payload["spec"]).fingerprint() != spec.fingerprint():
            raise ValueError("Native bundle canonical graph schema differs")
        value = tree_to(payload, device)
        self.spec, self.model = spec, model
        self.features, self.metadata, self.extra = value["features"], value["metadata"], value["extra"]
        self.seed = HardGraphBatch(**value["seed"])
        self.slot_to_atom, self.slot_to_token = value["slot_to_atom"], value["slot_to_token"]
        self.element_index = value["element_index"]
        self.ligand_atom_mask, self.ligand_token_mask = value["ligand_atom_mask"], value["ligand_token_mask"]
        self.batched = model == "nesso1"
        self.validate()
        self._original_digest = self.context_digest()

    def context_digest(self) -> str:
        """Include every tensor that can affect scores or graph correspondence."""
        return tensor_tree_sha256(dict(features=self.features, metadata=self.metadata,
                                       extra=self.extra, slot_to_atom=self.slot_to_atom,
                                       slot_to_token=self.slot_to_token, element_index=self.element_index,
                                       ligand_atom_mask=self.ligand_atom_mask,
                                       ligand_token_mask=self.ligand_token_mask,
                                       seed=dict(atoms=self.seed.atoms, charges=self.seed.charges,
                                                 bonds=self.seed.bonds), spec=self.spec.to_dict()))

    def assert_unchanged(self) -> None:
        """Check before/after a gate or optimization, outside timed model calls."""
        if self.context_digest() != self._original_digest:
            raise ValueError("Prepared native context was mutated after loading")

    @property
    def active(self) -> Tensor:
        return self.slot_to_atom >= 0

    def validate(self) -> None:
        n = self.spec.max_nodes
        if not isinstance(self.metadata, dict) or not self.metadata.get("preparation_command"):
            raise ValueError("Native bundle metadata requires preparation_command")
        if not self.metadata.get("native_source_version") or not self.metadata.get("seed_identity"):
            raise ValueError("Native bundle metadata requires native_source_version and seed_identity")
        for labels, shape, size in ((self.seed.atoms, (1, n), len(self.spec.atomic_numbers)),
                                    (self.seed.charges, (1, n), len(self.spec.formal_charges)),
                                    (self.seed.bonds, (1, n, n), len(self.spec.bond_names))):
            if labels.shape != shape or labels.dtype != torch.long or (labels < 0).any() or (labels >= size).any():
                raise ValueError("Invalid canonical hard seed")
        if not torch.equal(self.seed.bonds, self.seed.bonds.transpose(1, 2)) or self.seed.bonds.diagonal(dim1=1, dim2=2).any():
            raise ValueError("Seed must have symmetric bonds and no self-bonds")
        for mapping in (self.slot_to_atom, self.slot_to_token):
            if mapping.shape != (n,) or mapping.dtype != torch.long or (mapping < -1).any():
                raise ValueError("Slot maps must be long[N] with only -1 as absent sentinel")
        if not self.active.any() or not torch.equal(self.active, self.slot_to_token >= 0) or not torch.equal(self.active, self.seed.atoms[0] != 0):
            raise ValueError("Nonempty fixed occupancy must agree across maps and seed")
        inactive_pair = ~(self.active[:, None] & self.active[None, :])
        if self.seed.bonds[0][inactive_pair].any():
            raise ValueError("Seed bonds to absent slots are unsupported")
        if (self.seed.charges[0, ~self.active] != self.spec.formal_charges.index(0)).any():
            raise ValueError("Absent seed slots must carry neutral charge")
        for mapping, mask in ((self.slot_to_atom, self.ligand_atom_mask), (self.slot_to_token, self.ligand_token_mask)):
            if mask.ndim != 1 or mask.dtype != torch.bool:
                raise ValueError("Native ligand masks must be bool vectors")
            ids = mapping[self.active]
            if ids.unique().numel() != ids.numel() or not torch.equal(ids.sort().values, mask.nonzero().flatten()):
                raise ValueError("Slot maps must cover each native ligand atom/token exactly once")
        indices = self.element_index
        if indices.shape != (len(self.spec.atomic_numbers),) or indices.dtype != torch.long or indices[0] != -1:
            raise ValueError("Explicit native element_index must have one entry per canonical atom type")
        if (indices[1:] < 0).any() or (indices[1:] >= 128).any() or indices[1:].unique().numel() != indices.numel() - 1:
            raise ValueError("Native element indices must be bounded and distinct")
        # Audited upstream vocabularies: Nesso featurizer one-hots atomic number;
        # Protenix constants.get_all_elems starts at hydrogen (column zero).
        offset = 0 if self.batched else 1
        expected_indices = indices.new_tensor([-1, *(z - offset for z in self.spec.atomic_numbers[1:])])
        if not torch.equal(indices, expected_indices):
            raise ValueError("Element map differs from this adapter's audited native vocabulary")
        f = self.features
        a, t = self.ligand_atom_mask.numel(), self.ligand_token_mask.numel()
        expected_element = (1, a, 128) if self.batched else (a, 128)
        if f["ref_element"].shape != expected_element:
            raise ValueError(f"Expected native ref_element shape {expected_element}")
        expected_bonds = (1, t, t, 1) if self.batched else (t, t)
        if f["token_bonds"].shape != expected_bonds:
            raise ValueError(f"Expected native token_bonds shape {expected_bonds}")
        native_bonds = f["token_bonds"][0, ..., 0] if self.batched else f["token_bonds"]
        cross_boundary = self.ligand_token_mask[:, None] ^ self.ligand_token_mask[None, :]
        if native_bonds[cross_boundary].any():
            raise ValueError("Covalent ligand-context attachments are unsupported by this canonical graph adapter")
        ai, ti = self.slot_to_atom[self.active], self.slot_to_token[self.active]
        if self.batched:
            native_ligand = (f["mol_type"][0] == 3) & f["token_pad_mask"][0].bool()
            affinity_ligand = f["affinity_token_mask"][0].bool() & f["token_pad_mask"][0].bool()
            if not torch.equal(native_ligand, self.ligand_token_mask) or not torch.equal(affinity_ligand, self.ligand_token_mask):
                raise ValueError("Prepared graph must cover the complete single native affinity ligand")
            atom_to_token = f["atom_to_token"][0, ai]
            expected = F.one_hot(ti, t).to(atom_to_token)
            if atom_to_token.shape != expected.shape or not torch.equal(atom_to_token, expected):
                raise ValueError("Nesso atom/token correspondence differs from canonical slot mapping")
            native_atom_ligand = (f["atom_to_token"][0].float() @ native_ligand.float()) > 0
            if not torch.equal(native_atom_ligand, self.ligand_atom_mask):
                raise ValueError("Nesso ligand atom coverage is incomplete")
        else:
            if f["is_ligand"].shape != (a,) or not torch.equal(f["is_ligand"].bool(), self.ligand_atom_mask):
                raise ValueError("Prepared graph must cover the complete native ligand atom set")
            if not torch.equal(f["atom_to_token_idx"][ai].long(), ti):
                raise ValueError("Protenix atom/token correspondence differs from canonical slot mapping")

    @classmethod
    def load(cls, path: str | Path, *, spec: GraphSpec, model: str, device: str):
        return cls(torch.load(path, map_location="cpu", weights_only=True), spec=spec, model=model, device=device)

    def assess(self, graph: RelaxedGraph, losses: tuple[str, ...]) -> InputSupport:
        n = self.spec.max_nodes
        fields = ((graph.atoms, (1, n, len(self.spec.atomic_numbers))),
                  (graph.charges, (1, n, len(self.spec.formal_charges))),
                  (graph.bonds, (1, n, n, len(self.spec.bond_names))))
        for field, shape in fields:
            if field.shape != shape or not field.is_floating_point():
                return InputSupport("unsupported", ("Only one floating graph with the complete prepared schema is supported",))
            if field.device != self.slot_to_atom.device or field.device != self.features["ref_element"].device:
                return InputSupport("unsupported", ("All graph channels and the native bundle must share one device",))
            if not field.isfinite().all() or (field < -1e-7).any() or (field > 1 + 1e-7).any():
                return InputSupport("unsupported", ("Relaxed categorical probabilities must be finite and in [0,1]",))
            if not torch.allclose(field.sum(-1), torch.ones_like(field[..., 0]), atol=1e-5, rtol=0):
                return InputSupport("unsupported", ("Relaxed categorical probabilities must sum to one",))
        if not torch.allclose(graph.bonds, graph.bonds.transpose(1, 2), atol=1e-6, rtol=0):
            return InputSupport("unsupported", ("Canonical bonds must be symmetric",))
        if not torch.allclose(graph.occupancy[0], self.active.to(graph.atoms), atol=1e-6, rtol=0):
            return InputSupport("unsupported", ("Native branch requires fixed occupied slots",))
        inactive_pair = ~(self.active[:, None] & self.active[None, :])
        excluded = inactive_pair | torch.eye(n, dtype=torch.bool, device=self.active.device)
        if not torch.allclose(graph.bonds[0, ..., 0][excluded], torch.ones_like(graph.bonds[0, ..., 0][excluded]), atol=1e-6, rtol=0):
            return InputSupport("unsupported", ("Padding and diagonal bonds must be NONE",))
        neutral = self.spec.formal_charges.index(0)
        if not torch.allclose(graph.charges[0, ~self.active, neutral], torch.ones_like(graph.charges[0, ~self.active, neutral]), atol=1e-6, rtol=0):
            return InputSupport("unsupported", ("Absent-slot charges must be neutral",))
        return InputSupport("lossy", losses) if losses else InputSupport("exact")

    def soft_features(self, graph: RelaxedGraph, *, charge: bool) -> dict:
        """Replace chemistry tensors with expectations; never harden or call RDKit."""
        features = dict(self.features)
        active, ai, ti = self.active, self.slot_to_atom[self.active], self.slot_to_token[self.active]
        table = F.one_hot(self.element_index[1:], 128).to(graph.atoms)
        element = graph.atoms[0, active, 1:] @ table
        ref_element = features["ref_element"].float().clone()
        if self.batched:
            ref_element[0, ai] = element
        else:
            ref_element[ai] = element
        features["ref_element"] = ref_element
        if charge:
            ref_charge = features["ref_charge"].float().clone()
            values = graph.charges[0, active] @ graph.charges.new_tensor(self.spec.formal_charges)
            if self.batched:
                ref_charge[0, ai] = values
            else:
                ref_charge[ai] = values
            features["ref_charge"] = ref_charge
        bonds = features["token_bonds"].float().clone()
        present = 1 - graph.bonds[0, active][:, active, 0]
        if self.batched:
            bonds[0, ti[:, None], ti[None, :], 0] = present
        else:
            bonds[ti[:, None], ti[None, :]] = present
        features["token_bonds"] = bonds
        return features

    def assert_seed_features(self, *, charge: bool) -> None:
        """Independent native prepared tensors must match adapter-produced hard features."""
        actual = self.soft_features(self.seed.to_relaxed(self.spec), charge=charge)
        for key in (("ref_element", "token_bonds", "ref_charge") if charge else ("ref_element", "token_bonds")):
            if actual[key].shape != self.features[key].shape or not torch.allclose(actual[key], self.features[key].to(actual[key]), atol=1e-6, rtol=0):
                raise ValueError(f"Canonical seed does not reproduce native {key}; inspect atom order and encoding")


def freeze(module: nn.Module) -> nn.Module:
    """Freeze model parameters while retaining input autograd."""
    return module.requires_grad_(False).eval()
