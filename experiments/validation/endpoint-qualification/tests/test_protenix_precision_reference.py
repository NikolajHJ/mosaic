"""Allocated-only precision-reference controls, never a native model pass."""
from copy import deepcopy
import ast
import importlib.util
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/protenix_precision_reference.py"
SPEC = importlib.util.spec_from_file_location("protenix_precision_reference_fixture", PATH)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


@pytest.fixture(autouse=True)
def allocated_only():
    assert str(os.environ.get("LSB_JOBID", "")).isdigit(), "Reference controls require allocated DTU HPC"


@pytest.fixture
def x64():
    import jax
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    jax.clear_caches()
    try:
        yield jax
    finally:
        jax.clear_caches()
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def fp32():
    """Use actual production-style defaults, including untyped native zeros."""
    import jax
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    jax.clear_caches()
    try:
        yield jax
    finally:
        jax.clear_caches()
        jax.config.update("jax_enable_x64", previous)


def core_source():
    import jax._src.nn.functions as functions
    return inspect.getsource(getattr(functions, r.CORE_NAME))


def test_module_has_no_eager_numerical_imports():
    tree = ast.parse(PATH.read_text())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [name.name for node in imports if isinstance(node, ast.Import) for name in node.names]
    assert not {"jax", "numpy", "equinox", "mosaic"}.intersection(names)


def test_installed_runtime_is_authenticated():
    audit = r.authenticate_runtime()
    assert audit["jax_version"] == "0.11.1"
    assert audit["functions_sha256"] == r.FUNCTIONS_SHA256
    assert audit["changed_expressions"] == 1


def test_exact_ast_clone_changes_one_expression():
    source = core_source()
    code, audit = r.clone_attention_core(source)
    assert code is not None and audit["original_core_ast_sha256"] == r.CORE_AST_SHA256
    assert audit["reference_core_ast_sha256"] != audit["original_core_ast_sha256"]
    assert audit["changed_expressions"] == 1 and audit["production_execution_changed"] is False


def test_comments_and_whitespace_do_not_change_pinned_executable_ast():
    _, a = r.clone_attention_core(core_source())
    _, b = r.clone_attention_core("# declaration-only comment\n\n" + core_source())
    assert a == b


@pytest.mark.parametrize("before,after", [
    ("logits *= jnp.array(scale, dtype=logits.dtype)", "logits *= jnp.array(scale * 2, dtype=logits.dtype)"),
    ("padded_logits.astype(np.float32)", "padded_logits.astype(np.float64)"),
    ("axis=-1", "axis=0"),
    ("return encoded", "return encoded * 0"),
    ("scale, q_seqlen", "scale_changed, q_seqlen"),
])
def test_changed_native_core_ast_is_rejected(before, after):
    source = core_source()
    assert before in source
    with pytest.raises(ValueError, match="Unauthenticated"):
        r.clone_attention_core(source.replace(before, after, 1))


def test_second_cast_and_any_extra_body_statement_are_rejected():
    with pytest.raises(ValueError, match="Unauthenticated"):
        r.clone_attention_core(core_source() + "\nunused = 1\n")


def test_numerical_entry_requires_allocation(monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(ValueError, match="allocated"):
        r.authenticate_runtime()
    with pytest.raises(ValueError, match="allocated"):
        r.assert_reference_process()
    with pytest.raises(ValueError, match="allocated"):
        r.promote_tree({})


def test_fp32_process_cannot_install_reference_hook(x64):
    x64.config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="x64-enabled"):
        with r.attention_precision_context():
            raise AssertionError("must not enter")


def test_changed_installed_source_identity_fails_before_hook(x64, monkeypatch):
    import jax._src.nn.functions as functions
    original = getattr(functions, r.CORE_NAME)
    monkeypatch.setattr(r, "FUNCTIONS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source bytes"):
        with r.attention_precision_context():
            raise AssertionError("must not enter")
    assert getattr(functions, r.CORE_NAME) is original


def test_hook_is_restored_on_normal_and_exceptional_exit(x64):
    import jax._src.nn.functions as functions
    original = getattr(functions, r.CORE_NAME)
    with r.attention_precision_context():
        assert getattr(functions, r.CORE_NAME) is not original
    assert getattr(functions, r.CORE_NAME) is original
    with pytest.raises(RuntimeError, match="deliberate fixture"):
        with r.attention_precision_context():
            raise RuntimeError("deliberate fixture")
    assert getattr(functions, r.CORE_NAME) is original


def test_nested_hook_fails_closed_and_outer_hook_still_restores(x64):
    import jax._src.nn.functions as functions
    original = getattr(functions, r.CORE_NAME)
    with r.attention_precision_context():
        with pytest.raises(ValueError):
            with r.attention_precision_context():
                raise AssertionError("must not enter")
    assert getattr(functions, r.CORE_NAME) is original


def attention_inputs(dtype, case="ordinary"):
    import jax.numpy as jnp
    batch, query_count, key_count, query_heads, key_heads, width = 2, 3, 4, 4, 4, 3
    if case == "grouped":
        key_heads = 2
    if case == "multiquery":
        key_heads = 1
    q = jnp.sin(jnp.arange(batch * query_count * query_heads * width, dtype=dtype)).reshape(
        batch, query_count, query_heads, width) / 3
    k = jnp.cos(jnp.arange(batch * key_count * key_heads * width, dtype=dtype)).reshape(
        batch, key_count, key_heads, width) / 4
    v = jnp.sin(jnp.arange(batch * key_count * key_heads * width, dtype=dtype) / 5).reshape(
        batch, key_count, key_heads, width)
    options = {"implementation": "xla", "scale": 1.0}
    if case in ("bias", "masked", "all_masked", "causal", "grouped", "multiquery"):
        options["bias"] = jnp.cos(jnp.arange(batch * query_heads * query_count * key_count, dtype=dtype)).reshape(
            batch, query_heads, query_count, key_count) / 7
    if case in ("masked", "all_masked"):
        mask = jnp.ones((batch, 1, query_count, key_count), dtype=bool).at[:, :, :, 1].set(False)
        if case == "all_masked":
            mask = mask.at[:, :, 1, :].set(False)
        options["mask"] = mask
    if case == "causal":
        options["is_causal"] = True
    if case == "padding":
        options.update(query_seq_lengths=jnp.asarray([2, 3], dtype=jnp.int32),
                       key_value_seq_lengths=jnp.asarray([3, 2], dtype=jnp.int32))
    if case == "window":
        options["local_window_size"] = (1, 2)
    if case == "unbatched":
        q, k, v = q[0], k[0], v[0]
    return q, k, v, options


@pytest.mark.parametrize("case", ["ordinary", "bias", "masked", "all_masked", "causal", "padding",
                                  "window", "grouped", "multiquery", "unbatched"])
@pytest.mark.parametrize("residual", [False, True])
def test_fp32_public_api_parity_including_masks_padding_and_grouping(x64, case, residual):
    import jax.numpy as jnp
    import numpy as np
    q, k, v, options = attention_inputs(jnp.float32, case)
    baseline = x64.nn.dot_product_attention(q, k, v, return_residual=residual, **options)
    baseline = x64.tree.map(lambda a: np.asarray(a).copy(), baseline)
    with r.attention_precision_context():
        observed = x64.nn.dot_product_attention(q, k, v, return_residual=residual, **options)
        x64.tree.map(lambda a, b: np.testing.assert_array_equal(a, np.asarray(b)), baseline, observed)


@pytest.mark.parametrize("case", ["ordinary", "bias", "masked", "all_masked", "causal", "padding",
                                  "window", "grouped", "multiquery", "unbatched"])
@pytest.mark.parametrize("residual", [False, True])
def test_standalone_attention_clone_matches_actual_fp32_runtime(fp32, case, residual):
    """The x64 hook cannot itself establish parity of x64-disabled defaults."""
    import jax.numpy as jnp
    import jax._src.nn.functions as functions
    import numpy as np
    original = getattr(functions, r.CORE_NAME)
    code, _ = r.clone_attention_core(inspect.getsource(original))
    namespace = dict(original.__globals__)
    exec(code, namespace)
    clone = namespace[r.CORE_NAME]
    q, k, v, options = attention_inputs(jnp.float32, case)
    baseline = fp32.nn.dot_product_attention(q, k, v, return_residual=residual, **options)
    baseline = fp32.tree.map(lambda value: np.asarray(value).copy(), baseline)
    fp32.clear_caches()
    # This is a pure allocated fixture of the authenticated standalone clone,
    # not installation of the separately gated FP64 reference context.
    setattr(functions, r.CORE_NAME, clone)
    try:
        observed = fp32.nn.dot_product_attention(q, k, v, return_residual=residual, **options)
        def compare(before, after):
            assert before.dtype == np.float32 and np.asarray(after).dtype == np.float32
            np.testing.assert_array_equal(before, np.asarray(after))
        fp32.tree.map(compare, baseline, observed)
    finally:
        setattr(functions, r.CORE_NAME, original)
        fp32.clear_caches()
    assert getattr(functions, r.CORE_NAME) is original


@pytest.mark.parametrize("field", [0, 1, 2])
@pytest.mark.parametrize("case", ["ordinary", "masked", "grouped"])
def test_fp64_attention_independent_central_differences_match_gradient(x64, field, case):
    import jax.numpy as jnp
    import numpy as np
    q, k, v, options = attention_inputs(jnp.float64, case)
    values = (q, k, v)
    index = (0, 1, 0, 1)
    with r.attention_precision_context():
        def objective(*arrays):
            output = x64.nn.dot_product_attention(*arrays, **options)
            return jnp.sum(jnp.sin(output))
        analytic = x64.grad(objective, argnums=field)(*values)[index]
        for epsilon in (0.0001, 0.00003):
            plus, minus = list(values), list(values)
            plus[field] = plus[field].at[index].add(epsilon)
            minus[field] = minus[field].at[index].add(-epsilon)
            finite_difference = (float(objective(*plus)) - float(objective(*minus))) / (2 * epsilon)
            np.testing.assert_allclose(finite_difference, np.asarray(analytic), rtol=2e-5, atol=2e-8)
        audit = r.floating_dtype_audit({"scalar": objective(*values),
                                       "gradient": x64.grad(objective, argnums=field)(*values)})
        assert audit["expected"] == "float64" and audit["passed"]


def test_fp64_core_avoids_the_original_softmax_downcast(x64):
    import jax.numpy as jnp
    q, k, v, options = attention_inputs(jnp.float64)
    with r.attention_precision_context():
        traced = x64.make_jaxpr(lambda a, b, c: x64.nn.dot_product_attention(a, b, c, **options))(q, k, v)
        assert "new_dtype=float32" not in str(traced)


def test_float_promotion_is_lossless_and_does_not_modify_original_arrays(x64):
    import jax.numpy as jnp
    import numpy as np
    original = {"weights": jnp.asarray([.1, -.3, 1], dtype=jnp.float32),
                "features": np.asarray([[.25, .75]], dtype=np.float32),
                "indices": jnp.asarray([0, 2], dtype=jnp.int32),
                "mask": np.asarray([True, False]), "static": (17, "same", .3)}
    before = {name: np.asarray(original[name]).copy() for name in ("weights", "features", "indices", "mask")}
    promoted, audit = r.promote_tree(original, label="fixture")
    assert audit["all_castbacks_exact"] is True and audit["floating_leaves"] == 2
    assert audit["optimization_execution_changed"] is False
    assert str(promoted["weights"].dtype) == str(promoted["features"].dtype) == "float64"
    assert promoted["indices"] is original["indices"] and promoted["mask"] is original["mask"]
    assert promoted["static"] == original["static"]
    for name, value in before.items():
        np.testing.assert_array_equal(value, np.asarray(original[name]))
    for row in audit["leaves"]:
        if row["kind"] == "floating":
            assert row["original"] == row["castback"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_frozen_float_promotion_is_rejected(x64, value):
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="Nonfinite"):
        r.promote_tree({"weight": jnp.asarray([value], dtype=jnp.float32)})


def test_complex_frozen_leaf_is_rejected(x64):
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="Complex"):
        r.promote_tree({"weight": jnp.asarray([1 + 2j], dtype=jnp.complex64)})


def test_dtype_audit_rejects_mixed_float32_reference(x64):
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="Unexpected floating precision"):
        r.floating_dtype_audit({"one": jnp.asarray([1], dtype=jnp.float64),
                                "narrowed": jnp.asarray([1], dtype=jnp.float32)})


def test_dtype_audit_rejects_missing_and_nonfinite_floating_results(x64):
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="No floating"):
        r.floating_dtype_audit({"indices": jnp.asarray([1], dtype=jnp.int32)})
    with pytest.raises(ValueError, match="Nonfinite"):
        r.floating_dtype_audit({"value": jnp.asarray([jnp.nan], dtype=jnp.float64)})


@lru_cache(maxsize=1)
def native_functions():
    from mosaic.small_molecule.features import bootstrap_native_assets
    bootstrap_native_assets('/work3/s234818/mosaic/cache/mosaic/protenix', include_model=False)
    import protenix.protenij as native
    return native, {"average_over_atoms": native.average_over_atoms,
                    "OuterProductMean.__call__": native.OuterProductMean.__call__,
                    "AtomAttentionEncoder.__call__": native.AtomAttentionEncoder.__call__}


@pytest.mark.parametrize("name", ["average_over_atoms", "OuterProductMean.__call__"])
def test_exact_native_mask_clone_changes_only_one_declared_cast(name):
    native, functions = native_functions()
    code, audit = r.clone_native_mask_function(inspect.getsource(functions[name]), name)
    assert code is not None and audit["changed_expressions"] == 1
    assert audit["original_ast_sha256"] == r.NATIVE_CASTS[name]["ast_sha256"]
    assert audit["reference_ast_sha256"] != audit["original_ast_sha256"]


@pytest.mark.parametrize("name", ["average_over_atoms", "OuterProductMean.__call__"])
def test_changed_native_mask_function_is_rejected(name):
    native, functions = native_functions()
    source = inspect.getsource(functions[name])
    assert "jnp.float32" in source
    with pytest.raises(ValueError, match="Unauthenticated native mask AST"):
        r.clone_native_mask_function(source.replace("jnp.float32", "jnp.float64", 1), name)


def test_native_source_and_hook_restoration_are_authenticated(x64):
    native, originals = native_functions()
    from mosaic.small_molecule import native_hook
    geometry_original = native_hook.GeometryMaskedAtomInput.__call__
    audit = r.authenticate_native_runtime()
    assert audit["native_sha256"] == r.NATIVE_SHA256
    assert len(audit["mask_functions"]) == 2
    with pytest.raises(RuntimeError, match="fixture interruption"):
        with r.native_precision_context() as active:
            assert active["changed_expressions"] == 5
            assert native.average_over_atoms is not originals["average_over_atoms"]
            assert native.OuterProductMean.__call__ is not originals["OuterProductMean.__call__"]
            assert native.AtomAttentionEncoder.__call__ is not originals["AtomAttentionEncoder.__call__"]
            assert native_hook.GeometryMaskedAtomInput.__call__ is not geometry_original
            raise RuntimeError("fixture interruption")
    assert native.average_over_atoms is originals["average_over_atoms"]
    assert native.OuterProductMean.__call__ is originals["OuterProductMean.__call__"]
    assert native.AtomAttentionEncoder.__call__ is originals["AtomAttentionEncoder.__call__"]
    assert native_hook.GeometryMaskedAtomInput.__call__ is geometry_original


def test_geometry_charge_clone_authenticates_actual_wrapper():
    from mosaic.small_molecule.native_hook import GeometryMaskedAtomInput
    _, audit = r.clone_geometry_charge_function(inspect.getsource(GeometryMaskedAtomInput.__call__))
    assert audit["original_ast_sha256"] == r.GEOMETRY_CHARGE_AST_SHA256
    assert audit["changed_expressions"] == 1
    assert audit["integer_metadata_changed"] is False
    assert audit["optimization_execution_changed"] is False


@pytest.mark.parametrize("before,after", [
    ('jnp.arcsinh', 'jnp.sinh'),
    ('features["ref_charge"]', 'features["other_charge"]'),
    ('position * geometry', 'position + geometry'),
])
def test_geometry_charge_clone_rejects_any_other_wrapper_change(before, after):
    from mosaic.small_molecule.native_hook import GeometryMaskedAtomInput
    source = inspect.getsource(GeometryMaskedAtomInput.__call__)
    assert before in source
    with pytest.raises(ValueError, match="Unauthenticated geometry-masked encoder AST"):
        r.clone_geometry_charge_function(source.replace(before, after, 1))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("batched", [False, True])
def test_actual_geometry_charge_prefix_parity_dtype_and_derivative(x64, dtype, batched):
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule import native_hook
    original = native_hook.GeometryMaskedAtomInput.__call__
    code, audit = r.clone_geometry_charge_function(inspect.getsource(original))
    namespace = dict(original.__globals__); exec(code, namespace)
    def prefix(source):
        tree = ast.parse(__import__('textwrap').dedent(source)); fn = tree.body[0]
        end = next(i for i, node in enumerate(fn.body) if isinstance(node, ast.AugAssign)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == "linear_no_bias_ref_charge")
        fn.body = fn.body[:end + 1] + [ast.Return(ast.Name("c_l", ast.Load()))]
        ast.fix_missing_locations(tree)
        values = dict(original.__globals__); exec(compile(tree, '<actual-geometry-charge-prefix>', 'exec'), values)
        return values['__call__']
    class Linear:
        def __init__(self, weight): self.weight = weight
        def __call__(self, value): return value @ self.weight.T
    kind = getattr(jnp, dtype); charge = jnp.asarray([-2, -1, 0, 1, 2], dtype=jnp.int32)
    if batched: charge = jnp.stack([charge, charge])
    features = {'ref_charge': charge, 'ref_pos': jnp.zeros((*charge.shape, 3), dtype=kind),
                'ref_mask': jnp.ones(charge.shape, dtype=kind),
                'ref_space_uid': jnp.zeros(charge.shape, dtype=jnp.int32),
                'atom_to_token_idx': jnp.zeros(charge.shape, dtype=jnp.int32),
                native_hook.GEOMETRY_FEATURE: jnp.ones(charge.shape, dtype=kind)}
    encoder = SimpleNamespace(has_coords=False, input_feature_keys_ordered=[],
        linear_no_bias_ref_pos=Linear(jnp.zeros((2, 3), dtype=kind)),
        linear_no_bias_ref_charge=Linear(jnp.ones((2, 1), dtype=kind)))
    layer = SimpleNamespace(native=encoder)
    cloned = prefix(audit['reference_source']); result = cloned(layer, features)
    expected = jnp.arcsinh(charge.astype(kind))[..., None] * jnp.ones((2,), dtype=kind)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(features['ref_charge']), np.asarray(charge))
    assert str(result.dtype) == dtype
    if dtype == 'float32':
        np.testing.assert_array_equal(np.asarray(prefix(inspect.getsource(original))(layer, features)), np.asarray(result))
    else:
        graph = x64.make_jaxpr(lambda q: cloned(layer, dict(features, ref_charge=q)))(charge)
        assert __import__('protenix_precision_native').audit_jaxpr(graph)['floating_operations_float64']
        def score(position): return jnp.sum(cloned(layer, dict(features, ref_pos=position)))
        assert str(x64.grad(score)(features['ref_pos']).dtype) == 'float64'


def test_geometry_runtime_source_pin_is_mandatory(x64, monkeypatch):
    monkeypatch.setattr(r, 'GEOMETRY_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='Unauthenticated geometry-masked encoder source bytes'):
        r.authenticate_geometry_runtime()


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_native_atom_aggregation_x64_context_hook_parity_and_fp64_gradient(x64, masked, batched):
    import jax.numpy as jnp
    import numpy as np
    native, originals = native_functions()
    indices = jnp.asarray([0, 0, 1, 1, 2], dtype=jnp.int32)
    mask = jnp.asarray([True, True, True, False, False]) if masked else None
    values = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 7
    if batched:
        values = jnp.stack([values, values + 1])
    baseline = np.asarray(originals["average_over_atoms"](values, indices, 3, atom_mask=mask)).copy()
    with r.native_precision_context():
        np.testing.assert_array_equal(baseline, np.asarray(native.average_over_atoms(values, indices, 3, atom_mask=mask)))
        values64 = values.astype(jnp.float64)
        result = native.average_over_atoms(values64, indices, 3, atom_mask=mask)
        assert str(result.dtype) == "float64"
        def score(x):
            return jnp.sum(jnp.sin(native.average_over_atoms(x, indices, 3, atom_mask=mask)))
        position = (0, 0, 1) if batched else (0, 1)
        derivative = x64.grad(score)(values64)[position]
        for epsilon in (.0001, .00003):
            observed = (float(score(values64.at[position].add(epsilon)))
                        - float(score(values64.at[position].add(-epsilon)))) / (2 * epsilon)
            np.testing.assert_allclose(observed, np.asarray(derivative), atol=2e-9, rtol=2e-6)
        assert "new_dtype=float32" not in str(x64.make_jaxpr(score)(values64))


@pytest.mark.parametrize("masked", [False, True])
def test_native_outer_product_x64_context_hook_parity_and_fp64_mask_arithmetic(x64, masked):
    import jax.numpy as jnp
    import numpy as np
    native, originals = native_functions()
    layer = SimpleNamespace(layer_norm=lambda x: x, linear_1=lambda x: x,
                            linear_2=lambda x: x, linear_out=lambda x: x, eps=.001)
    values = jnp.sin(jnp.arange(24, dtype=jnp.float32)).reshape(3, 4, 2)
    mask = jnp.asarray([True, True, False, True]) if masked else None
    baseline = np.asarray(originals["OuterProductMean.__call__"](layer, values, token_mask=mask)).copy()
    with r.native_precision_context():
        np.testing.assert_array_equal(baseline, np.asarray(native.OuterProductMean.__call__(layer, values, token_mask=mask)))
        def score(x):
            return jnp.sum(jnp.sin(native.OuterProductMean.__call__(layer, x, token_mask=mask)))
        values64 = values.astype(jnp.float64)
        assert "new_dtype=float32" not in str(x64.make_jaxpr(score)(values64))
        derivative = x64.grad(score)(values64)[0, 1, 0]
        for epsilon in (.0001, .00003):
            observed = (float(score(values64.at[0, 1, 0].add(epsilon)))
                        - float(score(values64.at[0, 1, 0].add(-epsilon)))) / (2 * epsilon)
            np.testing.assert_allclose(observed, np.asarray(derivative), atol=2e-9, rtol=2e-6)


def standalone_native_clone(name):
    _, originals = native_functions()
    original = originals[name]
    code, _ = r.clone_native_mask_function(inspect.getsource(original), name)
    namespace = dict(original.__globals__)
    exec(code, namespace)
    return original, namespace[r.NATIVE_CASTS[name]["function"]]


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_standalone_atom_clone_matches_actual_fp32_runtime(fp32, masked, batched):
    import jax.numpy as jnp
    import numpy as np
    original, clone = standalone_native_clone("average_over_atoms")
    indices = jnp.asarray([0, 0, 1, 1, 2], dtype=jnp.int32)
    mask = jnp.asarray([True, True, True, False, False]) if masked else None
    values = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 7
    if batched:
        values = jnp.stack([values, values + 1])
    baseline = np.asarray(original(values, indices, 3, atom_mask=mask)).copy()
    observed = np.asarray(clone(values, indices, 3, atom_mask=mask))
    assert baseline.dtype == observed.dtype == np.float32
    np.testing.assert_array_equal(baseline, observed)


@pytest.mark.parametrize("masked", [False, True])
def test_standalone_outer_clone_matches_actual_fp32_runtime(fp32, masked):
    import jax.numpy as jnp
    import numpy as np
    original, clone = standalone_native_clone("OuterProductMean.__call__")
    layer = SimpleNamespace(layer_norm=lambda x: x, linear_1=lambda x: x,
                            linear_2=lambda x: x, linear_out=lambda x: x, eps=.001)
    values = jnp.sin(jnp.arange(24, dtype=jnp.float32)).reshape(3, 4, 2)
    mask = jnp.asarray([True, True, False, True]) if masked else None
    baseline = np.asarray(original(layer, values, token_mask=mask)).copy()
    observed = np.asarray(clone(layer, values, token_mask=mask))
    assert baseline.dtype == observed.dtype == np.float32
    np.testing.assert_array_equal(baseline, observed)


def test_charge_clone_authenticates_the_complete_actual_encoder():
    _, functions = native_functions()
    code, audit = r.clone_native_charge_function(inspect.getsource(functions['AtomAttentionEncoder.__call__']))
    assert code is not None and audit['changed_expressions'] == 1
    assert audit['original_ast_sha256'] == r.CHARGE_AST_SHA256
    assert audit['integer_metadata_changed'] is False and audit['optimization_execution_changed'] is False


@pytest.mark.parametrize('before,after', [
    ('jnp.arcsinh', 'jnp.sinh'),
    ('["ref_charge"]', '["other_charge"]'),
    ('batch_shape =', 'changed_batch_shape ='),
])
def test_charge_clone_rejects_changed_native_encoder(before, after):
    _, functions = native_functions()
    source = inspect.getsource(functions['AtomAttentionEncoder.__call__'])
    assert before in source and source.replace(before, after, 1) != source
    with pytest.raises(ValueError, match='Unauthenticated atom encoder charge AST'):
        r.clone_native_charge_function(source.replace(before, after, 1))


@pytest.mark.parametrize('dtype', ['float32','float64'])
@pytest.mark.parametrize('batched', [False,True])
def test_actual_charge_prefix_preserves_fp32_and_keeps_fp64(x64, dtype, batched):
    import jax.numpy as jnp
    import numpy as np
    native, functions = native_functions()
    original = functions['AtomAttentionEncoder.__call__']
    code, clone_audit = r.clone_native_charge_function(inspect.getsource(original))
    namespace = dict(original.__globals__); exec(code, namespace)
    # Execute the exact authenticated native prefix through charge embedding;
    # stop before the attention trunk, which remains covered by the full native
    # GPU gate. Both clone and original use identical prefix extraction.
    def prefix(function, source):
        tree = ast.parse(source)
        fn = tree.body[0]
        end = next(i for i,node in enumerate(fn.body) if isinstance(node,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='c_l' for t in node.targets))
        fn.body = fn.body[:end+1] + [ast.Return(ast.Name('c_l',ast.Load()))]
        ast.fix_missing_locations(tree)
        values = dict(function.__globals__); exec(compile(tree,'<actual-charge-prefix>','exec'),values)
        return values['__call__']
    source = __import__('textwrap').dedent(inspect.getsource(original))
    class Linear:
        def __init__(self, weight): self.weight = weight
        def __call__(self, values): return values @ self.weight.T
    charge = jnp.asarray([-2,-1,0,1,2],dtype=jnp.int32)
    if batched: charge = jnp.stack([charge,charge])
    native_dtype = getattr(jnp,dtype)
    features = {'ref_charge':charge,'ref_pos':jnp.zeros((*charge.shape,3),dtype=native_dtype),
                'ref_mask':jnp.ones(charge.shape,dtype=native_dtype),
                'ref_space_uid':jnp.zeros(charge.shape,dtype=jnp.int32),
                'atom_to_token_idx':jnp.zeros(charge.shape,dtype=jnp.int32)}
    layer = SimpleNamespace(has_coords=False,input_feature_keys_ordered=[],
        linear_no_bias_ref_pos=Linear(jnp.zeros((2,3),dtype=native_dtype)),
        linear_no_bias_ref_charge=Linear(jnp.ones((2,1),dtype=native_dtype)))
    clone_source = clone_audit['reference_source']
    got = prefix(namespace['__call__'],clone_source)(layer,features)
    expected = jnp.arcsinh(charge.astype(native_dtype))[...,None] * jnp.ones((2,),dtype=native_dtype)
    assert str(got.dtype) == dtype
    np.testing.assert_array_equal(np.asarray(got),np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(features['ref_charge']),np.asarray(charge))
    if dtype=='float32':
        np.testing.assert_array_equal(np.asarray(prefix(original,source)(layer,features)),np.asarray(got))
    audit = __import__('protenix_precision_native').audit_jaxpr
    if dtype=='float64':
        graph = x64.make_jaxpr(lambda values: prefix(namespace['__call__'],clone_source)(layer,dict(features,ref_charge=values)))(charge)
        assert audit(graph)['floating_operations_float64'] is True
