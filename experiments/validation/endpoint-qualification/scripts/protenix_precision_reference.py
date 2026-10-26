"""Separate, authenticated precision reference; never production arithmetic.

The only attention change is removal of the installed JAX 0.11.1 core's FP32
softmax narrowing when its logits are FP64. The original public attention API,
masks, grouping and padding remain native. The hook exists in memory inside an
allocated, fresh x64 reference process and is restored on every exit. No package,
weight, context, saved molecule, optimizer setting or scientific result is edited.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
import textwrap


VERSION = "protenix-separate-fp64-reference-v2"
JAX_VERSION = "0.11.1"
FUNCTIONS_SHA256 = "12e8ef763d56f4cc92189101b0b01e1a458042f851274a5d3f4ad3042430536b"
CORE_AST_SHA256 = "24bcb2fa8e3ad101b0896972da1e9704c52ce702e82719323366ee45d1a894ae"
CORE_NAME = "_dot_product_attention_core"
NATIVE_SHA256 = "37f337839034afd2a814c1ed6b76867b67918f73dd56cdd365e67b8fed882a6d"
CHARGE_AST_SHA256 = "acec16e3de87879f2ea772deb29ce9157125f5f81632846c478da394925419e0"
NATIVE_CASTS = {
    "OuterProductMean.__call__": {
        "ast_sha256": "7d8f95ca77f6e672cbb0b72cd1fc1aa61d4858188eb5063c73883d6042123283",
        "receiver": "mask", "dtype_receiver": "m", "function": "__call__"},
    "average_over_atoms": {
        "ast_sha256": "b9b5b7a543ea33be8a8385297ed085350a5aa50a4bf1beabca603c833bbddcc5",
        "receiver": "atom_mask", "dtype_receiver": "x_atom", "function": "average_over_atoms"},
}
_ACTIVE = False
_NATIVE_ACTIVE = False


def _require(value, message):
    if not value:
        raise ValueError(message)


def _allocated():
    _require(str(os.environ.get("LSB_JOBID", "")).isdigit(),
             "Precision reference numerical operations require allocated DTU HPC")


def _ast_digest(tree):
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def clone_attention_core(source):
    """Return a code object with exactly one authenticated expression changed.

    This AST/byte operation itself needs no numerical imports. A changed upstream
    function, second cast, body edit or signature is rejected before compilation.
    Comments/whitespace are irrelevant; every executable AST node is pinned.
    """
    tree = ast.parse(source)
    _require(_ast_digest(tree) == CORE_AST_SHA256, "Unauthenticated native attention core AST")
    original = deepcopy(tree)
    matches = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "padded_logits"
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "padded_logits" and node.value.func.attr == "astype"
                and len(node.value.args) == 1 and not node.value.keywords
                and ast.dump(node.value.args[0], include_attributes=False)
                == ast.dump(ast.parse("np.float32", mode="eval").body, include_attributes=False)):
            matches.append(node)
    _require(len(matches) == 1, "Exactly one native padded-logit narrowing is required")
    matches[0].value.args[0] = ast.copy_location(
        ast.Attribute(value=ast.Name(id="logits", ctx=ast.Load()), attr="dtype", ctx=ast.Load()),
        matches[0].value.args[0])
    # Reversibility proves that the one expression is the complete AST change.
    restored = deepcopy(tree)
    found = [node for node in ast.walk(restored) if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == "padded_logits" and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "astype"]
    _require(len(found) == 1, "Unexpected modified attention AST")
    found[0].value.args[0] = ast.parse("np.float32", mode="eval").body
    _require(ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False),
             "Attention reference changed more than the declared narrowing")
    ast.fix_missing_locations(tree)
    return compile(tree, "<authenticated-protenix-fp64-attention-core>", "exec"), {
        "version": VERSION, "original_core_ast_sha256": CORE_AST_SHA256,
        "reference_core_ast_sha256": _ast_digest(tree), "changed_expressions": 1,
        "change": "padded_logits.astype(np.float32) -> padded_logits.astype(logits.dtype)",
        "production_execution_changed": False,
    }


def authenticate_runtime():
    """Authenticate the actual installed numerical consumer, before any hook."""
    _allocated()
    import jax
    import jax._src.nn.functions as functions
    _require(jax.__version__ == JAX_VERSION, "Unauthenticated JAX runtime version")
    path = Path(functions.__file__).resolve()
    _require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == FUNCTIONS_SHA256,
             "Unauthenticated installed JAX attention source bytes")
    function = getattr(functions, CORE_NAME)
    _require(function.__globals__ is functions.__dict__, "Unexpected native attention function globals")
    source = inspect.getsource(function)
    _, audit = clone_attention_core(source)
    return {**audit, "jax_version": jax.__version__, "functions_file": str(path),
            "functions_sha256": FUNCTIONS_SHA256}


def clone_native_mask_function(source, name):
    """Clone one exact native function, changing its one fixed mask cast only."""
    _require(name in NATIVE_CASTS, "Undeclared native precision function")
    declared = NATIVE_CASTS[name]
    tree = ast.parse(textwrap.dedent(source))
    _require(_ast_digest(tree) == declared["ast_sha256"], "Unauthenticated native mask AST: " + name)
    original = deepcopy(tree)
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "astype"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == declared["receiver"]
             and len(node.args) == 1 and not node.keywords
             and ast.dump(node.args[0], include_attributes=False)
             == ast.dump(ast.parse("jnp.float32", mode="eval").body, include_attributes=False)]
    _require(len(found) == 1, "Exactly one declared native mask cast is required")
    found[0].args[0] = ast.copy_location(ast.parse(declared["dtype_receiver"] + ".dtype", mode="eval").body,
                                       found[0].args[0])
    restored = deepcopy(tree)
    reverse = [node for node in ast.walk(restored) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "astype"
               and isinstance(node.func.value, ast.Name) and node.func.value.id == declared["receiver"]]
    _require(len(reverse) == 1, "Unexpected native mask clone")
    reverse[0].args[0] = ast.parse("jnp.float32", mode="eval").body
    _require(ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False),
             "Native reference changed more than its declared mask cast")
    ast.fix_missing_locations(tree)
    return compile(tree, "<authenticated-protenix-reference-mask>", "exec"), {
        "name": name, "original_ast_sha256": declared["ast_sha256"],
        "reference_ast_sha256": _ast_digest(tree), "changed_expressions": 1,
        "change": declared["receiver"] + ".astype(jnp.float32) -> "
                  + declared["receiver"] + ".astype(" + declared["dtype_receiver"] + ".dtype)",
        "optimization_execution_changed": False}


def clone_native_charge_function(source):
    """Cast the one integer arcsinh input to its reference weight dtype.

    Int32 arcsinh otherwise narrows to float32 in an x64 process. The cloned
    reference changes this operand only; production and integer metadata stay
    exact. Reversal proves that no other executable AST node changed.
    """
    tree = ast.parse(textwrap.dedent(source))
    _require(_ast_digest(tree) == CHARGE_AST_SHA256, "Unauthenticated atom encoder charge AST")
    original = deepcopy(tree)
    expected = ast.parse('input_feature_dict["ref_charge"]', mode="eval").body
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
               and node.func.value.id == "jnp" and node.func.attr == "arcsinh"
               and len(node.args) == 1 and not node.keywords
               and ast.dump(node.args[0]) == ast.dump(expected)]
    _require(len(matches) == 1, "Exactly one native integer-charge arcsinh operand required")
    matches[0].args[0] = ast.Call(
        func=ast.Attribute(value=deepcopy(expected), attr="astype", ctx=ast.Load()),
        args=[ast.parse("self.linear_no_bias_ref_charge.weight.dtype", mode="eval").body], keywords=[])
    restored = deepcopy(tree)
    calls = [node for node in ast.walk(restored) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "arcsinh"]
    _require(len(calls) == 1, "Unexpected reference charge clone")
    calls[0].args[0] = deepcopy(expected)
    _require(ast.dump(restored) == ast.dump(original), "Reference charge clone changed another executable node")
    ast.fix_missing_locations(tree)
    return compile(tree, "<authenticated-protenix-reference-charge>", "exec"), {
        "name": "AtomAttentionEncoder.__call__", "original_ast_sha256": CHARGE_AST_SHA256,
        "reference_ast_sha256": _ast_digest(tree), "changed_expressions": 1,
        "change": "arcsinh(ref_charge) -> arcsinh(ref_charge.astype(reference_charge_weight.dtype))",
        "reference_source": ast.unparse(tree),
        "integer_metadata_changed": False, "optimization_execution_changed": False}


def authenticate_native_runtime():
    _allocated()
    import protenix.protenij as native
    path = Path(native.__file__).resolve()
    _require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == NATIVE_SHA256,
             "Unauthenticated installed Protenij source bytes")
    rows = []
    for name, function in (("OuterProductMean.__call__", native.OuterProductMean.__call__),
                           ("average_over_atoms", native.average_over_atoms)):
        _require(function.__globals__ is native.__dict__, "Unexpected native mask function globals")
        _, audit = clone_native_mask_function(inspect.getsource(function), name)
        rows.append(audit)
    _, charge = clone_native_charge_function(inspect.getsource(native.AtomAttentionEncoder.__call__))
    return {"native_file": str(path), "native_sha256": NATIVE_SHA256,
            "mask_functions": rows, "charge_function": charge}


def assert_reference_process():
    """The caller must use a fresh process, separately from production FP32."""
    _allocated()
    import jax
    _require(bool(jax.config.jax_enable_x64), "Separate x64-enabled reference process required")
    return {"jax_enable_x64": True, "production_fp32_process_must_be_separate": True,
            "optimization_execution_changed": False}


@contextmanager
def attention_precision_context():
    """Temporarily install the narrow core clone in the reference process only.

    The installed source file is never written. Cache clearing prevents an old
    trace from hiding the hook or a reference trace from surviving restoration.
    Nested/concurrent contexts in this process fail closed.
    """
    global _ACTIVE
    assert_reference_process()
    _require(not _ACTIVE, "Nested/concurrent precision reference hooks are forbidden")
    audit = authenticate_runtime()
    import jax
    import jax._src.nn.functions as functions
    original = getattr(functions, CORE_NAME)
    namespace = dict(original.__globals__)
    code, _ = clone_attention_core(inspect.getsource(original))
    exec(code, namespace)
    replacement = namespace[CORE_NAME]
    jax.clear_caches()
    _ACTIVE = True
    setattr(functions, CORE_NAME, replacement)
    try:
        yield audit
    finally:
        changed = getattr(functions, CORE_NAME) is not replacement
        setattr(functions, CORE_NAME, original)
        _ACTIVE = False
        jax.clear_caches()
        _require(not changed, "Precision reference hook was unexpectedly replaced during its scope")
        _require(hashlib.sha256(Path(functions.__file__).read_bytes()).hexdigest() == FUNCTIONS_SHA256,
                 "Installed attention source changed during reference evaluation")


@contextmanager
def native_precision_context():
    """Compose four fixed precision expressions in the separate FP64 reference.

    Authenticate/configure original assets with bootstrap_native_assets before
    entering, and enter before loading/forward-tracing FrozenProtenixTiny. The
    geometry hook imports average_over_atoms at call time and sees this clone.
    Diffusion/confidence code and every installed package byte remain untouched.
    """
    global _NATIVE_ACTIVE
    assert_reference_process()
    _require(not _NATIVE_ACTIVE, "Nested/concurrent native precision hooks are forbidden")
    audit = authenticate_native_runtime()
    import jax
    import protenix.protenij as native
    original_mean = native.average_over_atoms
    original_outer = native.OuterProductMean.__call__
    original_encoder = native.AtomAttentionEncoder.__call__
    replacements = []
    for name, function in (("average_over_atoms", original_mean),
                           ("OuterProductMean.__call__", original_outer)):
        namespace = dict(function.__globals__)
        code, _ = clone_native_mask_function(inspect.getsource(function), name)
        exec(code, namespace)
        replacement = namespace[NATIVE_CASTS[name]["function"]]
        replacement.__qualname__ = function.__qualname__
        replacements.append(replacement)
    namespace = dict(original_encoder.__globals__)
    code, _ = clone_native_charge_function(inspect.getsource(original_encoder))
    exec(code, namespace)
    charge_replacement = namespace["__call__"]
    charge_replacement.__qualname__ = original_encoder.__qualname__
    jax.clear_caches()
    _NATIVE_ACTIVE = True
    native.average_over_atoms = replacements[0]
    native.OuterProductMean.__call__ = replacements[1]
    native.AtomAttentionEncoder.__call__ = charge_replacement
    try:
        with attention_precision_context() as attention:
            yield {**audit, "attention": attention, "changed_expressions": 4,
                   "optimization_execution_changed": False}
    finally:
        changed = (native.average_over_atoms is not replacements[0]
                   or native.OuterProductMean.__call__ is not replacements[1]
                   or native.AtomAttentionEncoder.__call__ is not charge_replacement)
        native.average_over_atoms = original_mean
        native.OuterProductMean.__call__ = original_outer
        native.AtomAttentionEncoder.__call__ = original_encoder
        _NATIVE_ACTIVE = False
        jax.clear_caches()
        _require(not changed, "Native precision hooks changed during their scope")
        _require(hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == NATIVE_SHA256,
                 "Installed native source changed during reference evaluation")


def _array_record(value):
    import numpy as np
    array = np.asarray(value)
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def promote_tree(tree, *, label="reference"):
    """Losslessly promote floating leaves; retain integer/static leaves exactly.

    Returns a new pytree and explicit per-leaf original/promoted/castback hashes.
    There is no serialization or mutation. The caller separately authenticates
    the original frozen model/context digest and retains it unchanged.
    """
    assert_reference_process()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    pairs, structure = jax.tree_util.tree_flatten_with_path(tree)
    promoted, rows = [], []
    for path, value in pairs:
        name = jax.tree_util.keystr(path)
        if eqx.is_array(value):
            original = np.asarray(value)
            before = _array_record(original)
            _require(original.dtype.kind != "c", "Complex reference leaves are not supported")
            if original.dtype.kind == "f":
                _require(np.isfinite(original).all(), "Nonfinite original frozen floating leaf")
                converted = jnp.asarray(value).astype(jnp.float64)
                _require(str(converted.dtype) == "float64", "Reference floating leaf was not promoted")
                castback = np.asarray(converted).astype(original.dtype)
                _require(_array_record(castback) == before, "Reference promotion was not lossless")
                rows.append({"path": name, "kind": "floating", "original": before,
                             "promoted": _array_record(converted), "castback": _array_record(castback),
                             "lossless_castback": True})
            else:
                converted = value
                rows.append({"path": name, "kind": "unchanged_array", "original": before,
                             "promoted": _array_record(converted), "unchanged": True})
            _require(_array_record(value) == before, "Original frozen array mutated during promotion")
        else:
            converted = value
            rows.append({"path": name, "kind": "unchanged_python_leaf",
                         "type": type(value).__module__ + "." + type(value).__qualname__,
                         "repr": repr(value), "unchanged": True})
        promoted.append(converted)
    result = jax.tree_util.tree_unflatten(structure, promoted)
    floating = [row for row in rows if row["kind"] == "floating"]
    _require(floating, "Reference pytree contains no floating arrays")
    return result, {"version": VERSION, "label": str(label), "leaves": rows,
                    "floating_leaves": len(floating), "all_castbacks_exact": True,
                    "original_tree_mutated": False, "optimization_execution_changed": False}


def floating_dtype_audit(tree, *, expected="float64"):
    """Fail on unexpected floating output/gradient/model leaves, not integer IDs."""
    _allocated()
    import jax
    import equinox as eqx
    import numpy as np
    _require(expected in ("float32", "float64"), "Explicit FP32/FP64 dtype expected")
    rows = []
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if eqx.is_array(value) and np.asarray(value).dtype.kind in "fc":
            row = {"path": jax.tree_util.keystr(path), **_array_record(value)}
            _require(row["dtype"] == expected, "Unexpected floating precision at " + row["path"])
            _require(np.isfinite(np.asarray(value)).all(), "Nonfinite reference result")
            rows.append(row)
    _require(rows, "No floating arrays found for precision audit")
    return {"expected": expected, "floating_leaves": len(rows), "leaves": rows, "passed": True}
