"""Pinned official PyTorch Protenix-Tiny trunk for the occupancy gate study.

Only ``protenix_tiny_default_v0.5.0`` on official source commit
510faf9f5cc8fbbbb8d9055934d3fa7e6877c76d is supported. The attached callable
is derived from ``Protenix.get_pairformer_output`` by removing precisely its
audited eval-mode grad-disabling context. All other native statements and the
full tuple return are preserved. No diffusion, confidence head, training mode
or learned bridge is used. The old manual path is forensic-only.

The official MSA module remains in eval mode. Tiny has exactly one MSA block,
which is marked ``is_last_block`` and therefore never calls the eval MSAStack
in-place update path. MSA sampling remains native; callers must restore the
same Python/NumPy/Torch/CUDA RNG state for paired parity/FD evaluations.
"""

from __future__ import annotations

import ast
import __future__
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import textwrap
import types
from typing import Any

import torch
from torch import Tensor, nn

from latentmol.multimodel.external import (
    file_sha256, native_source_sha256, tensor_tree_sha256, tree_to,
)


SOURCE_COMMIT = "510faf9f5cc8fbbbb8d9055934d3fa7e6877c76d"
SOURCE_SHA256 = "4e5dea1855e439f6d6682262cb820285dfa6a15476a9868cbd1ab6a225fa14b3"
WEIGHTS_SHA256 = "7ad252e023d61f94572f51ab60c2a58f3a12205271898fe581fa38d20de9566b"
MODEL_NAME = "protenix_tiny_default_v0.5.0"
# The package digest does not cover the separate upstream configs package.
CONFIG_SOURCE_SHA256 = {
    "configs_base.py": "13c36fb5624bb804aee78cd95d04b08832dcabc7de8378886dd99a67251d66b9",
    "configs_data.py": "4699fc39387203bb861437dfa2dec89cc73e375539d37b2140c2b82fa6796e69",
    "configs_inference.py": "0a1b8f79f6b770af74d3e27f3b60b21f05848aa3477bc33ca9b16abcd7755bab",
    "configs_model_type.py": "d7f59f1b2d83b3657555893be085502b04c38b21a069e8ebb4b05229a8f12552",
}


def _config_sources(root: Path) -> dict[str, str]:
    actual = {name: file_sha256(root / "configs" / name)
              for name in CONFIG_SOURCE_SHA256}
    if actual != CONFIG_SOURCE_SHA256:
        raise ValueError("Pinned Protenix configuration source bytes differ")
    return actual


def _check_imports(root: Path) -> None:
    """Reject preloaded or transitively imported copies outside the audited tree."""
    for name, module in tuple(sys.modules.items()):
        if name not in ("protenix", "configs") and not name.startswith(("protenix.", "configs.")):
            continue
        location = getattr(module, "__file__", None)
        if location is not None:
            if not Path(location).resolve().is_relative_to(root / name.split(".")[0]):
                raise ValueError(f"Imported {name} is outside pinned native source: {location}")
        else:
            locations = tuple(getattr(module, "__path__", ()))
            if not locations or any(not Path(p).resolve().is_relative_to(root) for p in locations):
                raise ValueError(f"Cannot authenticate native namespace {name}")


def _deep_update(target: dict, update: dict) -> dict:
    """Same recursive model-specific override as official runner/inference.py."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _tiny_configs(root: Path):
    base = importlib.import_module("configs.configs_base").configs
    data = importlib.import_module("configs.configs_data").data_configs
    inference = importlib.import_module("configs.configs_inference").inference_configs
    variants = importlib.import_module("configs.configs_model_type").model_configs
    parse = importlib.import_module("protenix.config.config").parse_configs
    _check_imports(root)
    merged = copy.deepcopy({**base, "data": data, **inference})
    _deep_update(merged, variants[MODEL_NAME])
    # Declared gate execution settings, common to native and attached forwards.
    # Model architecture is the upstream Tiny variant; no checkpoint key is dropped.
    args = (
        f"--model_name {MODEL_NAME} --load_strict true "
        "--triangle_multiplicative torch --triangle_attention torch "
        "--enable_tf32 false --enable_efficient_fusion false "
        "--enable_diffusion_shared_vars_cache false --dtype fp32 "
        "--blocks_per_ckpt None --train_confidence_only false"
    )
    configs = parse(merged, arg_str=args, fill_required_with_null=True)
    if (configs.model.pairformer.n_blocks != 8 or configs.model.msa_module.n_blocks != 1
            or configs.model.template_embedder.n_blocks != 0
            or configs.model_name != MODEL_NAME or not configs.load_strict):
        raise ValueError("Resolved configuration is not the audited Tiny0.5 architecture")
    return configs


def _attached_pairformer(model: nn.Module, features: dict[str, Any]) -> Tensor:
    """Preserved attempt003 manual equations, used only for forensic comparisons."""
    s_inputs = model.input_embedder(features, inplace_safe=False, chunk_size=None)
    z_constraint = (model.constraint_embedder(features["constraint_feature"])
                    if "constraint_feature" in features else None)
    s_init = model.linear_no_bias_sinit(s_inputs)
    z_init = (model.linear_no_bias_zinit1(s_init)[..., None, :]
              + model.linear_no_bias_zinit2(s_init)[..., None, :, :])
    z_init = z_init + model.relative_position_encoding(features["relp"])
    z_init = z_init + model.linear_no_bias_token_bond(features["token_bonds"].unsqueeze(-1))
    if z_constraint is not None:
        z_init = z_init + z_constraint
    z, s = torch.zeros_like(z_init), torch.zeros_like(s_init)
    # This is precisely one full recurrence, not a detached partial-recycle path.
    z = z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
    kwargs = dict(triangle_multiplicative="torch", triangle_attention="torch",
                  inplace_safe=False, chunk_size=None)
    if model.template_embedder.n_blocks > 0:
        z = z + model.template_embedder(features, z, **kwargs)
    z = model.msa_module(features, z, s_inputs, pair_mask=None, **kwargs)
    s = s_init + model.linear_no_bias_s(model.layernorm_s(s))
    _, z = model.pairformer_stack(s, z, pair_mask=None, **kwargs)
    return z


def _make_attached_pairformer(native_method):
    """Remove only the audited native eval gradient guard from a bound method.

    Keep every equation, branch, call, tensor lifetime and tuple return from
    the authenticated native source. The untouched native method remains the
    independent forward reference. Never enable training mode for gradients.
    """
    if not isinstance(native_method, types.MethodType):
        raise ValueError("Require a bound native method")
    original = native_method.__func__
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != "get_pairformer_output"
            or tree.body[0].decorator_list):
        raise ValueError("Require the exact undecorated native pairformer method")
    guards = [node for node in ast.walk(tree) if isinstance(node, (ast.With, ast.AsyncWith))]
    expected = ast.parse("torch.set_grad_enabled(self.training and "
        "(not self.train_confidence_only) and cycle_no == (N_cycle - 1))", mode="eval").body
    if (len(guards) != 1 or not isinstance(guards[0], ast.With)
            or len(guards[0].items) != 1 or guards[0].items[0].optional_vars is not None
            or ast.dump(guards[0].items[0].context_expr) != ast.dump(expected)):
        raise ValueError("Native gradient guard is absent, duplicated or changed")
    guard = guards[0]

    class RemoveAuditedGuard(ast.NodeTransformer):
        def visit_With(self, node):
            if node is not guard:
                raise ValueError("Unaudited native context manager")
            return node.body

    rewritten = ast.fix_missing_locations(RemoveAuditedGuard().visit(tree))
    namespace = dict(original.__globals__)
    future_mask = 0
    for name in __future__.all_feature_names:
        future_mask |= getattr(__future__, name).compiler_flag
    code = compile(rewritten, inspect.getsourcefile(original) or "<native-pairformer>", "exec",
        flags=original.__code__.co_flags & future_mask, dont_inherit=True)
    exec(code, namespace)
    function = namespace[original.__name__]
    function._protenix_ast_metadata = dict(version="native-eval-grad-context-v1",
        removed_grad_contexts=1, native_method_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        attached_ast_sha256=hashlib.sha256(ast.dump(rewritten).encode()).hexdigest(),
        native_source_modified=False, training_enabled=False)
    return types.MethodType(function, native_method.__self__)


def _register_dormant_checkpoint_esm(model: nn.Module, state: dict[str, Tensor]) -> dict:
    """Retain one legacy zero projection without enabling native ESM execution.

    The authenticated default-Tiny checkpoint contains this legacy parameter,
    whereas the pinned native constructor registers it only when ESM is enabled.
    Require exactly that discrepancy, then register the official native layer so
    the complete, unmodified checkpoint can still load with ``strict=True``.
    Never call this a generic missing/unexpected-key tolerance.
    """
    key = "input_embedder.linear_esm.weight"
    embedder = model.input_embedder
    if (model.configs.esm.enable is not False
            or embedder.esm_configs.get("enable") is not False
            or hasattr(embedder, "linear_esm")):
        raise ValueError("Dormant compatibility requires disabled ESM and no existing projection")
    if (embedder.c_token != 384 or model.configs.esm.embedding_dim != 2560
            or embedder.esm_configs.get("embedding_dim") != 2560):
        raise ValueError("Dormant compatibility requires the exact default-Tiny dimensions")
    expected = model.state_dict()
    if set(state) - set(expected) != {key} or set(expected) - set(state):
        raise ValueError("Checkpoint discrepancy is not exactly the one dormant ESM parameter")
    weight = state[key]
    if (not isinstance(weight, Tensor) or weight.layout != torch.strided
            or weight.device.type != "cpu" or weight.dtype != torch.float32
            or tuple(weight.shape) != (449, 2560)):
        raise ValueError("Dormant ESM parameter shape/device/dtype differs")
    if not bool(torch.isfinite(weight).all()) or bool(torch.count_nonzero(weight)):
        raise ValueError("Dormant ESM parameter must be finite and exactly zero")
    metadata = dict(version="tiny-default-dormant-esm-v1", parameter_name=key,
        shape=list(weight.shape), dtype=str(weight.dtype), all_zero=True,
        esm_enabled=False, strict_load=True,
        tensor_sha256=tensor_tree_sha256({key: weight}))
    factory = importlib.import_module("protenix.model.modules.primitives").LinearNoBias
    embedder.linear_esm = factory(2560, 449)
    return metadata


class FrozenTiny:
    """Expose independent native logits and an attached, frozen trunk derivative.

    State snapshots hash every named parameter and buffer, including buffers
    excluded from state_dict, and all resolved config values. Feature hashes
    surround each forward. These are integrity checks, not a native parity pass.
    """

    def __init__(self, model: nn.Module, source_root: Path, weights_path: Path,
                 checkpoint_compatibility: dict):
        self.model = model.eval().requires_grad_(False)
        self._attached_native_method = _make_attached_pairformer(model.get_pairformer_output)
        self.source_root, self.weights_path = source_root, weights_path
        self.provenance = dict(model_name=MODEL_NAME, source_commit=SOURCE_COMMIT,
            source_sha256=SOURCE_SHA256, weights_sha256=WEIGHTS_SHA256,
            config_sources=dict(CONFIG_SOURCE_SHA256), cycles=1, dtype="float32",
            triangle_attention="torch", triangle_multiplicative="torch",
            layernorm="torch", diffusion=False, confidence=False,
            checkpoint_compatibility=copy.deepcopy(checkpoint_compatibility),
            attached_execution=copy.deepcopy(self._attached_native_method._protenix_ast_metadata),
            status="native_validation_pending")
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise ValueError("Every native module must remain in eval mode")
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ValueError("Native parameters must remain frozen without accumulated gradients")
        if self.model.train_confidence_only:
            raise ValueError("Confidence-only configuration is unsupported")
        if (self.model.configs.esm.enable is not False
                or self.model.input_embedder.esm_configs.get("enable") is not False
                or not hasattr(self.model.input_embedder, "linear_esm")
                or bool(torch.count_nonzero(self.model.input_embedder.linear_esm.weight))):
            raise ValueError("Legacy checkpoint projection must remain zero and ESM disabled")
        if os.environ.get("LAYERNORM_TYPE") != "torch":
            raise ValueError("Require declared LAYERNORM_TYPE=torch")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError("FP32 parity gates require TF32 disabled")
        if (self.model.configs.triangle_attention != "torch"
                or self.model.configs.triangle_multiplicative != "torch"
                or self.model.configs.enable_efficient_fusion):
            raise ValueError("Only audited unfused torch kernels are supported")
        msa_blocks = self.model.msa_module.blocks
        if len(msa_blocks) != 1 or not msa_blocks[0].is_last_block:
            raise ValueError("Tiny must bypass the eval MSAStack in-place path")

    def snapshot_state(self) -> dict:
        self._assert_frozen()
        _check_imports(self.source_root)
        return dict(
            parameters=tensor_tree_sha256(dict(self.model.named_parameters())),
            buffers=tensor_tree_sha256(dict(self.model.named_buffers())),
            config=json.dumps(self.model.configs.to_dict(), sort_keys=True, separators=(",", ":")),
            module_modes=tuple((name, module.training) for name, module in self.model.named_modules()),
            source_sha256=native_source_sha256(self.source_root, "protenix"),
            config_sources=_config_sources(self.source_root),
            weights_sha256=file_sha256(self.weights_path),
        )

    def assert_unchanged(self, snapshot: dict) -> None:
        if self.snapshot_state() != snapshot:
            raise ValueError("Frozen Tiny weights, buffers, config, source or execution mode changed")

    def prepare_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run the official fixed-geometry feature-cache preparation on a copy.

        Ordinary ``Protenix.forward`` at the pinned source calls
        ``generate_relp`` followed by ``update_input_feature_dict`` before
        entering the trunk. The latter derives d_lm/v_lm/pad_info from the
        supplied native reference geometry and space identities. No pose or
        prediction is generated. Prepare once before injecting any soft q;
        both reference and attached forwards then consume the same context.
        """
        self._assert_frozen()
        if not isinstance(features, dict):
            raise TypeError("Native feature preparation requires a tensor/primitive dictionary")

        def has_grad(value):
            if isinstance(value, Tensor):
                return value.requires_grad
            if isinstance(value, dict):
                return any(has_grad(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return any(has_grad(item) for item in value)
            return False

        if has_grad(features):
            raise ValueError("Prepare hard native context before adding differentiable occupancy")
        before_features = tensor_tree_sha256(features)
        before_state = self.snapshot_state()
        copied = tree_to(features, next(self.model.parameters()).device)
        native_module = importlib.import_module("protenix.model.protenix")
        _check_imports(self.source_root)
        with torch.no_grad():
            copied = self.model.relative_position_encoding.generate_relp(copied)
            copied = native_module.update_input_feature_dict(copied)
        if not isinstance(copied, dict) or any(
                key not in copied for key in ("relp", "d_lm", "v_lm", "pad_info")):
            raise ValueError("Official native feature preparation did not produce required caches")
        if tensor_tree_sha256(features) != before_features:
            raise ValueError("Native preparation mutated the raw input bundle")
        self.assert_unchanged(before_state)
        return copied

    @staticmethod
    def _cycles(cycles: int) -> None:
        if isinstance(cycles, bool) or cycles != 1:
            raise ValueError("This pinned native occupancy phase supports exactly one recycle")

    def native_logits(self, features: dict[str, Any], cycles: int = 1) -> Tensor:
        """Run the actual official eval trunk/head as an independent reference."""
        self._cycles(cycles)
        self._assert_frozen()
        before = tensor_tree_sha256(features)
        with torch.no_grad():
            _, _, z = self.model.get_pairformer_output(features, N_cycle=1,
                inplace_safe=False, chunk_size=None, mc_dropout=False)
            logits = self.model.distogram_head(z)
        if tensor_tree_sha256(features) != before:
            raise ValueError("Official native forward mutated prepared feature context")
        self._assert_frozen()
        return logits

    def attached_logits(self, features: dict[str, Any], cycles: int = 1) -> Tensor:
        """Preserve the caller's input graph and autograd mode through the trunk."""
        self._cycles(cycles)
        self._assert_frozen()
        before = tensor_tree_sha256(features)
        _, _, z = self._attached_native_method(features, N_cycle=1,
            inplace_safe=False, chunk_size=None, mc_dropout=False)
        logits = self.model.distogram_head(z)
        if tensor_tree_sha256(features) != before:
            raise ValueError("Attached native equations mutated prepared feature context")
        self._assert_frozen()
        return logits

    def legacy_logits(self, features: dict[str, Any], cycles: int = 1) -> Tensor:
        """Evaluate the preserved failed manual path for declared diagnostics only."""
        self._cycles(cycles)
        self._assert_frozen()
        before = tensor_tree_sha256(features)
        logits = self.model.distogram_head(_attached_pairformer(self.model, features))
        if tensor_tree_sha256(features) != before:
            raise ValueError("Legacy diagnostic mutated prepared feature context")
        self._assert_frozen()
        return logits


def load_frozen_tiny(source_root: str | Path, source_sha256: str,
                     weights_path: str | Path, weights_sha256: str,
                     device: str = "cuda", seed: int = 17) -> FrozenTiny:
    """Authenticate cached official assets, resolve Tiny config and strictly load.

    This function performs no download or installation. Call only inside an
    allocated HPC job; local review of this module is static/AST-only.
    """
    root, weights = Path(source_root).resolve(), Path(weights_path).resolve()
    if source_sha256 != SOURCE_SHA256 or native_source_sha256(root, "protenix") != source_sha256:
        raise ValueError("Official pinned Protenix package source SHA256 mismatch")
    if weights_sha256 != WEIGHTS_SHA256 or file_sha256(weights) != weights_sha256:
        raise ValueError("Official Protenix-Tiny0.5 weights SHA256 mismatch")
    _config_sources(root)
    # Set before any upstream import; the default otherwise compiles CUDA LN.
    os.environ["LAYERNORM_TYPE"] = "torch"
    sys.path.insert(0, str(root))
    _check_imports(root)
    configs = _tiny_configs(root)
    native_module = importlib.import_module("protenix.model.protenix")
    _check_imports(root)
    layers = importlib.import_module("protenix.model.triangular.layers")
    if layers.fastln_is_installed:
        raise ValueError("Native LayerNorm was imported with an unaudited fused implementation")
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = native_module.Protenix(configs)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Require the official Tiny checkpoint['model'] state dictionary")
    state = checkpoint["model"]
    if not state or any(not isinstance(k, str) or not isinstance(v, Tensor) for k, v in state.items()):
        raise ValueError("Native checkpoint model state must contain named tensors")
    prefixed = [name.startswith("module.") for name in state]
    if any(prefixed):
        if not all(prefixed):
            raise ValueError("Mixed DDP and ordinary checkpoint names are unsupported")
        state = {name[len("module."):]: value for name, value in state.items()}
    compatibility = _register_dormant_checkpoint_esm(model, state)
    _check_imports(root)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return FrozenTiny(model, root, weights, compatibility)
