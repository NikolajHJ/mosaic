"""Thin, deterministic access to the pinned native conditional DiffSBDD core.

All denoising/schedule/centering equations execute in upstream methods.  This
module owns only state, explicit Gaussian draws and coordinate-frame bookkeeping.
It does not load Lightning or implement a replacement diffusion network.
Numerical qualification belongs to allocated HPC, including reference replay.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Iterator

import torch
from torch import Tensor


UPSTREAM_COMMIT = "5d0d38d16c8932a0339fd2ce3f67ade98bbdff27"

_SOURCE_ANCHORS = {
    "equivariant_diffusion/dynamics.py": "16b008598de7c61c0b5575e3af02f9b1a9e6697559864df1591314e4b4ec6b9f",
    "equivariant_diffusion/conditional_model.py": "260bb941e05a3beaa0f1aef7aebba86aa2474d5f5db75637ec1498e3ad0e47b4",
    "equivariant_diffusion/en_diffusion.py": "841f95e8d47fd1bc27f50b76f605bf6d0369308c68c7a65b199e51b00b30d8ef",
    "constants.py": "dba13f16d831d48d1b9bb1b427207d0603a48c8687273d68ee7b599fe57afec9",
    "LICENSE": "f3dd092bcfdc8408c5326ecf3ab3119c8da604e8f7730a5e9ab6a3c41a02ead0",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_native_source(source_root, expected_manifest_sha256=None) -> dict:
    """Verify the raw git-archive manifest, or the complete pinned Git checkout.

    Archive manifests are produced outside this numerical loader from raw Git
    blobs and sealed by the controller. Every listed member is checked, including
    license/non-Python assets, and extra Python files are refused before imports.
    """
    root = Path(source_root).resolve(strict=True)
    manifest_path = root / ".diffsbdd-source.json"
    normalized_eol = False
    if manifest_path.is_file():
        manifest_sha = _file_sha256(manifest_path)
        if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
            raise RuntimeError("DiffSBDD source manifest SHA256 mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("commit") != UPSTREAM_COMMIT or not isinstance(manifest.get("members"), dict):
            raise RuntimeError("Unpinned/malformed DiffSBDD source archive manifest")
        members = manifest["members"]
        basis = "raw_git_archive_manifest"
    elif (root / ".git").exists():
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args])

        if git("rev-parse", "HEAD").decode().strip() != UPSTREAM_COMMIT:
            raise RuntimeError("DiffSBDD Git checkout is not the pinned commit")
        names = git("ls-tree", "-r", "--name-only", UPSTREAM_COMMIT).decode().splitlines()
        members = {name: hashlib.sha256(git("show", f"{UPSTREAM_COMMIT}:{name}")).hexdigest()
                   for name in names if name.endswith(".py") or name in _SOURCE_ANCHORS}
        manifest_sha = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()
        normalized_eol = True
        basis = "pinned_git_blobs_with_CRLF_normalized_runtime_comparison"
    else:
        raise RuntimeError("Native source requires its sealed archive manifest or pinned Git checkout")
    for relative, expected in members.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Unsafe source manifest member: {relative}")
        raw = path.read_bytes()
        checked = raw.replace(b"\r\n", b"\n") if normalized_eol else raw
        if hashlib.sha256(checked).hexdigest() != expected:
            raise RuntimeError(f"Native source member hash mismatch: {relative}")
    for relative, pinned_sha in _SOURCE_ANCHORS.items():
        if members.get(relative) != pinned_sha:
            raise RuntimeError(f"Source manifest differs from reviewed pinned blob: {relative}")
    extras = [str(path.relative_to(root)).replace("\\", "/")
              for path in root.rglob("*.py")
              if ".git" not in path.parts and
              str(path.relative_to(root)).replace("\\", "/") not in members]
    if extras:
        raise RuntimeError(f"Unmanifested Python source in native root: {extras[:10]}")
    return {"commit": UPSTREAM_COMMIT, "source_root": str(root),
            "manifest_sha256": manifest_sha, "basis": basis,
            "checked_members": len(members), "members": members,
            "license": "MIT", "license_sha256": _SOURCE_ANCHORS["LICENSE"]}


def load_native_core(source_root, checkpoint_path, expected_sha256,
                     device="cpu", *, expected_source_manifest_sha256=None):
    """Strict pure-PyTorch extraction of the authenticated native DDPM core.

    Reproduces the constructor in pinned ``LigandPocketDDPM.__init__`` without
    importing Lightning, metrics, docking, wandb or its old BioPython helpers.
    Returns ``(frozen_core, dataset_info, receipt)``. All saved ``ddpm.*`` state,
    including gamma/schedule and buffer tensors, must load strictly. This source's
    metric objects are ordinary Python classes and have no state_dict buffers;
    any non-DDPM state key is therefore an error, rather than silently discarded.
    """
    from copy import deepcopy
    from importlib import import_module

    source_receipt = authenticate_native_source(
        source_root, expected_source_manifest_sha256)
    root = Path(source_root).resolve(strict=True)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("An authenticated checkpoint SHA256 is mandatory")
    actual_sha = _file_sha256(checkpoint)
    if actual_sha != expected_sha256:
        raise RuntimeError("DiffSBDD checkpoint SHA256 mismatch")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested native CUDA device is unavailable")
    # Authenticate before unpickling the public Lightning checkpoint (contains
    # argparse.Namespace). weights_only=False is explicit, never a fallback.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_data, dict):
        raise RuntimeError("Native checkpoint must contain a mapping")
    saved_hp = checkpoint_data.get("hyper_parameters")
    if saved_hp is None or "state_dict" not in checkpoint_data:
        raise RuntimeError("Native Lightning checkpoint lacks metadata/state_dict")

    def mapping(value):
        if isinstance(value, dict):
            return value
        if hasattr(value, "__dict__"):
            return vars(value)
        raise RuntimeError("Expected saved native hyperparameter namespace")

    hp = mapping(saved_hp)
    if hp.get("mode") != "pocket_conditioning" or hp.get("dataset") != "crossdock":
        raise RuntimeError("Only the declared CrossDocked ConditionalDDPM is admitted")
    if hp.get("pocket_representation", "CA") != "full-atom" or hp.get("virtual_nodes", False):
        raise RuntimeError("Require full-atom pocket without virtual ligand nodes")
    egnn, diffusion = mapping(hp["egnn_params"]), mapping(hp["diffusion_params"])
    for module_name, module in list(sys.modules.items()):
        if module_name in {"utils", "constants", "equivariant_diffusion"} or module_name.startswith("equivariant_diffusion."):
            module_path = getattr(module, "__file__", None)
            namespace_paths = list(getattr(module, "__path__", ()))
            trusted_namespace = module_name == "equivariant_diffusion" and namespace_paths and all(
                Path(path).resolve().is_relative_to(root) for path in namespace_paths)
            if not trusted_namespace and (module_path is None or not Path(module_path).resolve().is_relative_to(root)):
                raise RuntimeError(f"Conflicting untrusted native import: {module_name}")
    sys.path.insert(0, str(root))
    try:
        dynamics_module = import_module("equivariant_diffusion.dynamics")
        conditional_module = import_module("equivariant_diffusion.conditional_model")
        constants_module = import_module("constants")
    finally:
        sys.path.remove(str(root))
    dataset_info = deepcopy(constants_module.dataset_params[hp["dataset"]])
    atom_count = len(dataset_info["atom_decoder"])
    # Required fields come from saved metadata; optional fields reproduce only
    # the exact upstream __dict__.get semantics shown in the reviewed constructor.
    dynamics_kwargs = {name: egnn[name] for name in (
        "joint_nf", "hidden_nf", "n_layers", "attention", "tanh", "norm_constant",
        "inv_sublayers", "sin_embedding", "normalization_factor", "aggregation_method",
        "reflection_equivariant")}
    dynamics_kwargs.update({name: egnn.get(name) for name in (
        "edge_cutoff_ligand", "edge_cutoff_pocket", "edge_cutoff_interaction",
        "edge_embedding_dim")})
    dynamics_kwargs.update(atom_nf=atom_count, residue_nf=atom_count,
                           n_dims=3, device=str(target_device),
                           act_fn=torch.nn.SiLU(), update_pocket_coords=False)
    dynamics = dynamics_module.EGNNDynamics(**dynamics_kwargs)
    core_kwargs = dict(
        dynamics=dynamics, atom_nf=atom_count, residue_nf=atom_count, n_dims=3,
        timesteps=diffusion["diffusion_steps"],
        noise_schedule=diffusion["diffusion_noise_schedule"],
        noise_precision=diffusion["diffusion_noise_precision"],
        loss_type=diffusion["diffusion_loss_type"],
        norm_values=diffusion["normalize_factors"],
        size_histogram=hp["node_histogram"], virtual_node_idx=None)
    core = conditional_module.ConditionalDDPM(**core_kwargs)
    state = checkpoint_data["state_dict"]
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Empty/invalid native state_dict")
    noncore = sorted(key for key in state if not key.startswith("ddpm."))
    if noncore:
        raise RuntimeError(f"Unexpected noncore checkpoint state (not discarded): {noncore}")
    core_state = {key[len("ddpm."):]: value for key, value in state.items()}
    core.load_state_dict(core_state, strict=True)
    # Native load_state_dict can cast a stored tensor to the constructor dtype;
    # report both and verify equality after exactly that native cast.
    for key, loaded in core.state_dict().items():
        if not torch.equal(loaded.cpu(), core_state[key].cpu().to(loaded.dtype)):
            raise RuntimeError(f"Native checkpoint tensor failed exact restore: {key}")
    core.to(target_device).eval().requires_grad_(False)
    constructor = {k: v for k, v in dynamics_kwargs.items() if k != "act_fn"}
    histogram = hp["node_histogram"]
    histogram_list = histogram.tolist() if hasattr(histogram, "tolist") else histogram
    histogram_sha = hashlib.sha256(json.dumps(histogram_list, separators=(",", ":")).encode()).hexdigest()
    receipt = {
        "schema": "diffsbdd-native-core-load-v1", "source": source_receipt,
        "checkpoint_sha256": actual_sha, "checkpoint_path": str(checkpoint),
        "mode": hp["mode"], "dataset": hp["dataset"],
        "pocket_representation": hp["pocket_representation"],
        "atom_decoder": list(dataset_info["atom_decoder"]),
        "dynamics_constructor": constructor,
        "diffusion_constructor": {k: v for k, v in core_kwargs.items()
                                  if k not in {"dynamics", "size_histogram"}},
        "node_histogram_sha256": histogram_sha,
        "normalization_biases": list(core.norm_biases),
        "strict_state_keys": sorted(core_state), "state_tensor_count": len(core_state),
        "state_dtypes": {k: str(v.dtype) for k, v in core_state.items()},
        "noncore_state_keys_discarded": [],
        "noncore_policy": "none: pinned metric classes own no state_dict buffers",
        "lightning_imported_for_load": False,
        "native_checkpoint_runtime_replay": "not_yet_run",
    }
    return core, dataset_info, receipt


def _batch_mean(x: Tensor, mask: Tensor) -> Tensor:
    # Use the very same operation/package as ConditionalDDPM.sample_given_pocket.
    from torch_scatter import scatter_mean

    return scatter_mean(x, mask, dim=0)


def _copy_pocket(pocket: dict) -> dict:
    return {key: value.detach().clone() if isinstance(value, Tensor) else value
            for key, value in pocket.items()}


@dataclass(frozen=True)
class NativeState:
    """Normalized noisy x/h and the jointly translated pocket at integer level.

    ``frame_offset`` is per-example, in *normalized* coordinate units:
    receptor-frame xyz = (internal xyz + frame_offset[mask]) * coordinate_scale.
    NativeState has no hidden EGNN embedding and no diffused bond tensor.
    """

    ligand: Tensor
    pocket: Tensor
    ligand_mask: Tensor
    pocket_mask: Tensor
    level: int
    frame_offset: Tensor

    @property
    def batch_size(self) -> int:
        return self.frame_offset.shape[0]

    def clone(self) -> "NativeState":
        return NativeState(*(x.detach().clone() for x in (
            self.ligand, self.pocket, self.ligand_mask, self.pocket_mask)),
            self.level, self.frame_offset.detach().clone())


@dataclass(frozen=True)
class CleanEstimate:
    normalized: Tensor
    positions: Tensor
    features: Tensor
    predicted_noise: Tensor
    level: int


@dataclass(frozen=True)
class FinalSample:
    """Raw native point cloud; no builder, largest-fragment filter or UFF."""

    positions: Tensor
    one_hot: Tensor
    pocket_positions: Tensor
    pocket_features: Tensor
    ligand_mask: Tensor
    pocket_mask: Tensor
    internal_ligand: Tensor
    internal_pocket: Tensor


class GaussianStream:
    """Independent, step-indexed device Gaussian draws shared across arms.

    Every key gets a private generator. Rejection/diagnostics therefore cannot
    advance the next diffusion draw. These draws are also injected into the
    unmodified high-level sampler for reference replay; parity is not presumed.
    """

    def __init__(self, seed: int, namespace: str = "diffsbdd-sampling-v1"):
        self.seed = int(seed)
        self.namespace = namespace
        self.receipts: list[dict] = []
        self.raw_draws: list[tuple[str, Tensor]] = []

    def fork(self) -> "GaussianStream":
        return GaussianStream(self.seed, self.namespace)

    def draw(self, key: str, size, device, dtype=None) -> Tensor:
        dtype = torch.get_default_dtype() if dtype is None else dtype
        seed_bytes = hashlib.sha256(
            f"{self.namespace}:{self.seed}:{key}".encode("utf-8")).digest()
        draw_seed = int.from_bytes(seed_bytes[:8], "little") & ((1 << 63) - 1)
        generator = torch.Generator(device=device).manual_seed(draw_seed)
        value = torch.randn(tuple(size), generator=generator, device=device,
                            dtype=dtype)
        self.receipts.append({"key": key, "seed": draw_seed,
                              "shape": list(size), "device": str(device),
                              "dtype": str(dtype)})
        self.raw_draws.append((key, value.detach().cpu().clone()))
        return value


@contextmanager
def _replace_gaussian(core, provider: Callable) -> Iterator[None]:
    """Patch only the core instance, restore even after an upstream exception.

    This wrapper is single-process/single-sampler by contract. The native method
    resolves ``self.sample_gaussian`` at runtime and retains its own arithmetic.
    """
    had_own = "sample_gaussian" in core.__dict__
    previous = core.__dict__.get("sample_gaussian")
    core.sample_gaussian = provider
    try:
        yield
    finally:
        if had_own:
            core.sample_gaussian = previous
        else:
            delattr(core, "sample_gaussian")


class NativeStepper:
    """Expose exact native transitions on a frozen ConditionalDDPM instance."""

    def __init__(self, core):
        if int(core.n_dims) != 3 or int(core.atom_nf) < 1 or int(core.T) < 1:
            raise ValueError("Invalid native conditional core metadata")
        if getattr(core.dynamics, "update_pocket_coords", False):
            raise ValueError("Only a fixed-coordinate conditional pocket is admitted")
        if getattr(core, "vnode_idx", None) is not None:
            raise ValueError("Virtual-node model is outside the fixed-count protocol")
        if core.training or any(p.requires_grad for p in core.parameters()):
            raise ValueError("DiffSBDD must be eval() and frozen before wrapping")
        self.core = core
        self.timesteps = int(core.T)
        self.coordinate_scale = float(core.norm_values[0])
        if self.coordinate_scale <= 0:
            raise ValueError("Coordinate normalization scale must be positive")

    def _check(self, state: NativeState) -> None:
        if not 0 <= state.level <= self.timesteps:
            raise ValueError("Diffusion level outside the loaded native schedule")
        if state.ligand.shape != (len(state.ligand_mask), 3 + self.core.atom_nf):
            raise ValueError("Ligand state shape does not match native vocabulary")
        if state.pocket.shape[0] != len(state.pocket_mask):
            raise ValueError("Pocket state and batch mask disagree")
        for mask in (state.ligand_mask, state.pocket_mask):
            if mask.dtype != torch.long:
                raise ValueError("Native batch masks must have dtype long")
            if set(mask.detach().cpu().tolist()) != set(range(state.batch_size)):
                raise ValueError("Every batch entry needs ligand and pocket nodes")
        if state.frame_offset.shape != (state.batch_size, 3):
            raise ValueError("Missing per-example global frame")
        if not all(torch.isfinite(x).all().item() for x in
                   (state.ligand, state.pocket, state.frame_offset)):
            raise ValueError("Nonfinite native state")

    @contextmanager
    def _draw(self, stream: GaussianStream, key: str) -> Iterator[None]:
        calls = []

        def provider(size, device):
            if calls:
                raise RuntimeError(f"Native transition requested extra noise: {key}")
            calls.append(key)
            return stream.draw(key, size, device)

        with _replace_gaussian(self.core, provider):
            yield
        if len(calls) != 1:
            raise RuntimeError(f"Native transition did not consume its one draw: {key}")

    def _offset_after(self, old_pocket: Tensor, new_pocket: Tensor,
                      mask: Tensor, offset: Tensor) -> Tensor:
        # A native recentering translates every pocket node by the same vector.
        # First-node deltas avoid averaging-roundoff in the saved global frame.
        delta = old_pocket[:, :3] - new_pocket[:, :3]
        shifts = torch.stack([delta[(mask == i).nonzero()[0, 0]]
                              for i in range(len(offset))])
        if not torch.allclose(delta, shifts[mask], atol=2e-5, rtol=2e-5):
            raise RuntimeError("Native method changed pocket geometry, not translation")
        return offset + shifts

    @torch.no_grad()
    def initialize(self, pocket: dict, num_nodes_lig: Tensor,
                   stream: GaussianStream) -> NativeState:
        """Mirror only upstream initialization; normal sampling remains native."""
        from importlib import import_module

        core = self.core
        pocket_copy = _copy_pocket(pocket)
        _, normalized = core.normalize(pocket=pocket_copy)
        original = torch.cat((normalized["x"], normalized["one_hot"]), dim=1)
        batch_size = len(normalized["size"])
        if len(num_nodes_lig) != batch_size or (num_nodes_lig <= 0).any():
            raise ValueError("Positive fixed ligand count required for every pocket")
        # Call upstream helper rather than reproduce its mask ordering.
        utils = import_module(core.__class__.__module__).utils
        ligand_mask = utils.num_nodes_to_batch_mask(
            batch_size, num_nodes_lig, normalized["x"].device)
        mean_x = _batch_mean(normalized["x"], normalized["mask"])
        mean_h = torch.zeros((batch_size, core.atom_nf),
                             device=normalized["x"].device)
        mean = torch.cat((mean_x, mean_h), dim=1)[ligand_mask]
        sigma = torch.ones_like(normalized["size"]).unsqueeze(1)
        with self._draw(stream, "initial"):
            ligand, shifted_pocket = core.sample_normal_zero_com(
                mean, original, sigma, ligand_mask, normalized["mask"])
        offset = self._offset_after(original, shifted_pocket, normalized["mask"],
                                   mean_x.new_zeros((batch_size, 3)))
        core.assert_mean_zero_with_mask(ligand[:, :3], ligand_mask)
        state = NativeState(ligand, shifted_pocket, ligand_mask,
                            normalized["mask"], self.timesteps, offset)
        self._check(state)
        return state

    @torch.no_grad()
    def clean_estimate(self, state: NativeState) -> CleanEstimate:
        self._check(state)
        t = torch.full((state.batch_size, 1), state.level,
                       device=state.ligand.device) / self.timesteps
        gamma = self.core.gamma(t)
        eps, _ = self.core.dynamics(state.ligand, state.pocket, t,
                                    state.ligand_mask, state.pocket_mask)
        clean = self.core.xh_given_zt_and_epsilon(
            state.ligand, eps, gamma, state.ligand_mask)
        xyz, features = self.core.unnormalize(clean[:, :3], clean[:, 3:])
        xyz = xyz + self.coordinate_scale * state.frame_offset[state.ligand_mask]
        return CleanEstimate(clean, xyz, features, eps, state.level)

    @torch.no_grad()
    def reverse_step(self, state: NativeState,
                     stream: GaussianStream) -> NativeState:
        self._check(state)
        if state.level == 0:
            raise ValueError("Level zero has a separate native terminal draw")
        s = torch.full((state.batch_size, 1), state.level - 1,
                       device=state.ligand.device) / self.timesteps
        t = torch.full((state.batch_size, 1), state.level,
                       device=state.ligand.device) / self.timesteps
        with self._draw(stream, f"reverse:{state.level - 1}"):
            ligand, pocket = self.core.sample_p_zs_given_zt(
                s, t, state.ligand.detach(), state.pocket.detach(),
                state.ligand_mask, state.pocket_mask)
        offset = self._offset_after(state.pocket, pocket, state.pocket_mask,
                                   state.frame_offset)
        out = NativeState(ligand, pocket, state.ligand_mask, state.pocket_mask,
                          state.level - 1, offset)
        self._check(out)
        return out

    @torch.no_grad()
    def clean_input(self, positions: Tensor, features: Tensor,
                    state: NativeState) -> Tensor:
        """Convert a receptor-frame clean proposal to this state's native frame."""
        xyz = positions / self.coordinate_scale - state.frame_offset[state.ligand_mask]
        h = (features.float() - self.core.norm_biases[1]) / self.core.norm_values[1]
        result = torch.cat((xyz, h), dim=1).to(state.ligand)
        if result.shape != state.ligand.shape or not torch.isfinite(result).all():
            raise ValueError("Invalid clean proposal tensor")
        return result

    @torch.no_grad()
    def correct(self, state: NativeState, baseline: Tensor,
                proposed: Tensor) -> NativeState:
        """Preserve the current residual: Y += alpha(t) * (Fnew - Fbase).

        No-op returns the very same state, with no extra centering roundoff or
        Gaussian draw. Both F tensors must already be normalized and in-frame.
        """
        self._check(state)
        if baseline.shape != state.ligand.shape or proposed.shape != baseline.shape:
            raise ValueError("Clean state correction shape mismatch")
        if not torch.isfinite(baseline).all() or not torch.isfinite(proposed).all():
            raise ValueError("Nonfinite clean correction")
        if torch.equal(baseline, proposed):
            return state
        t = torch.full((state.batch_size, 1), state.level,
                       device=state.ligand.device) / self.timesteps
        alpha = self.core.alpha(self.core.gamma(t), state.ligand)
        ligand = state.ligand + alpha[state.ligand_mask] * (proposed - baseline)
        pocket = state.pocket.clone()
        ligand[:, :3], pocket[:, :3] = self.core.remove_mean_batch(
            ligand[:, :3], pocket[:, :3], state.ligand_mask, state.pocket_mask)
        offset = self._offset_after(state.pocket, pocket, state.pocket_mask,
                                   state.frame_offset)
        out = NativeState(ligand, pocket, state.ligand_mask, state.pocket_mask,
                          state.level, offset)
        self._check(out)
        return out

    @torch.no_grad()
    def finalize(self, state: NativeState, stream: GaussianStream) -> FinalSample:
        self._check(state)
        if state.level != 0:
            raise ValueError("Terminal p(x,h|z0) requires level zero")
        with self._draw(stream, "terminal"):
            x, h, px, ph = self.core.sample_p_xh_given_z0(
                state.ligand, state.pocket, state.ligand_mask,
                state.pocket_mask, state.batch_size)
        # Preserve the high-level native return_frames=1 drift correction.
        from torch_scatter import scatter_add

        self.core.assert_mean_zero_with_mask(x, state.ligand_mask)
        if scatter_add(x, state.ligand_mask, dim=0).abs().max().item() > 5e-2:
            x, px = self.core.remove_mean_batch(
                x, px, state.ligand_mask, state.pocket_mask)
        # Terminal sampling can itself recenter. Recover that shift against the
        # pocket in raw units, rather than assuming the level-zero frame persists.
        normalized_px = px / self.coordinate_scale
        offset = self._offset_after(state.pocket, normalized_px,
                                   state.pocket_mask, state.frame_offset)
        return FinalSample(
            x + self.coordinate_scale * offset[state.ligand_mask], h,
            px + self.coordinate_scale * offset[state.pocket_mask], ph,
            state.ligand_mask, state.pocket_mask,
            torch.cat((x, h), dim=1), torch.cat((px, ph), dim=1))

    @torch.no_grad()
    def sample_reference(self, pocket: dict, num_nodes_lig: Tensor,
                         stream: GaussianStream):
        """Unmodified upstream sampler driven by the explicit same step draws."""
        keys = iter(["initial"] + [f"reverse:{i}" for i in
                    reversed(range(self.timesteps))] + ["terminal"])
        consumed = []

        def provider(size, device):
            try:
                key = next(keys)
            except StopIteration as exc:
                raise RuntimeError("Native reference requested extra noise") from exc
            consumed.append(key)
            return stream.draw(key, size, device)

        with _replace_gaussian(self.core, provider):
            result = self.core.sample_given_pocket(
                _copy_pocket(pocket), num_nodes_lig, return_frames=1)
        if len(consumed) != self.timesteps + 2:
            raise RuntimeError("Native reference Gaussian draw count changed")
        return result

    @torch.no_grad()
    def replay(self, pocket: dict, num_nodes_lig: Tensor,
               stream: GaussianStream) -> FinalSample:
        state = self.initialize(pocket, num_nodes_lig, stream)
        while state.level:
            state = self.reverse_step(state, stream)
        return self.finalize(state, stream)


def assert_native_replay(stepper: NativeStepper, pocket: dict,
                         num_nodes_lig: Tensor, seed: int) -> dict:
    """Mandatory same-runtime exact control, before any guidance or fitting."""
    stream = GaussianStream(seed)
    ligand, pocket_out, lm, pm = stepper.sample_reference(
        pocket, num_nodes_lig, stream)
    replay_stream = stream.fork()
    replay = stepper.replay(pocket, num_nodes_lig, replay_stream)
    equal = {
        "ligand": torch.equal(ligand, replay.internal_ligand),
        "pocket": torch.equal(pocket_out, replay.internal_pocket),
        "ligand_mask": torch.equal(lm, replay.ligand_mask),
        "pocket_mask": torch.equal(pm, replay.pocket_mask),
        "draw_count": len(stream.receipts) == stepper.timesteps + 2,
        "draw_key_shape_dtype_device_seed_receipts": stream.receipts == replay_stream.receipts,
    }
    if not all(equal.values()):
        raise RuntimeError(f"Native same-runtime replay failed: {equal}")
    return {"passed": True, "checks": equal, "draws": stream.receipts,
            "upstream_commit": UPSTREAM_COMMIT}
