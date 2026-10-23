"""Mandatory allocated pretrained checks for the new DBMol graph objective.

No optimizer updates or altered finite-difference policies are implemented here.
The original X/E raw scales are gating for every component and the total. A
synthetic worker is useful only for allocated negative controls of this checker.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


COMPONENTS = ("dbmol_affinity_loss", "dbmol_contact_loss",
              "dbmol_confidence_loss", "dbmol_total_loss")
EPSILONS = (0.01, 0.003, 0.001)
RELATIVE_TOLERANCE = 0.15
PARITY_ATOL = 2e-5
PINNED = {
    "source": "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc",
    "checkpoint_sha256": "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e",
    "structural_checkpoint_sha256": "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1",
    "pocket_definition_sha256": "9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43",
    "context_file_sha256": "a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093",
    "context_digest": "fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4",
    "affinity_model_digest": "395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16",
    "protocol": "mosaic-boltz-dbmol-v1", "profile": "mosaic-dbmol-fixed-jq1-v1",
    "objective_arithmetic": "contact_head_and_assembly_float64_v1",
    "scalar_dtype": "float64", "graph_dtype": "float32", "gradient_dtype": "float32",
    "diffusion_gradient": False, "recycling_steps": 0, "n_atoms": 25,
    "optimized_output": "dbmol_total_loss", "physical_trunks_per_forward": 2,
    "affinity_mw_correction": False, "contact_weight": 2.5, "confidence_weight": 2.5,
}


class DerivativeCheckError(AssertionError):
    def __init__(self, message, report):
        self.report = report
        super().__init__(message)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def graph_digest(x, e):
    return hashlib.sha256(x.tobytes() + e.tobytes()).hexdigest()


def _graph_arrays(x, e):
    x, e = np.asarray(x), np.asarray(e)
    if x.shape != (25, 6) or e.shape != (300, 5) or x.dtype != np.float32 or e.dtype != np.float32:
        raise ValueError("Exact FP32 X[25,6]/E[300,5] graph is required")
    if not np.isfinite(x).all() or not np.isfinite(e).all():
        raise ValueError("Nonfinite graph")
    for array in (x, e):
        if (array < 0).any() or (array > 1).any() or not np.allclose(array.sum(-1), 1, atol=2e-6, rtol=0):
            raise ValueError("Categorical graph must remain on its original simplex")
    return x, e


def check_raw_finite_differences(primal, x, e, gx, ge, *, objective):
    """Use predeclared raw-probability secants and unmodified native gradients.

    `primal(x,e)` must execute a fresh independent native forward with backward
    disabled. No result from a gradient call can substitute for a secant endpoint.
    Directions are selected solely from the baseline raw derivative before any
    finite difference, matching the existing Mosaic raw X/E contract.
    """
    x, e = _graph_arrays(x, e)
    report = {"status": "running", "objective": objective,
              "epsilons": list(EPSILONS), "relative_tolerance": RELATIVE_TOLERANCE,
              "required_adjacent_passes": 2, "rows": [],
              "independent_primal": True, "raw_unclipped_gradients": True}
    try:
        for field, values, gradient in (("X", x, gx), ("E_upper", e, ge)):
            gradient = np.asarray(gradient)
            if gradient.shape != values.shape or gradient.dtype != np.float32 or not np.isfinite(gradient).all():
                raise AssertionError(f"Malformed/nonfinite raw {field} gradient")
            spreads = gradient.max(-1) - gradient.min(-1)
            row = int(np.argmax(spreads))
            source, target = int(np.argmin(gradient[row])), int(np.argmax(gradient[row]))
            analytic = float(gradient[row, target] - gradient[row, source])
            direction = np.zeros_like(values)
            direction[row, source], direction[row, target] = -1, 1
            entry = {"field": field, "row": row, "source": source, "target": target,
                     "analytic": analytic, "probes": []}
            report["rows"].append(entry)
            if not np.isfinite(analytic) or abs(analytic) <= 1e-7:
                raise AssertionError(f"No finite nonzero raw {field} categorical contrast")
            for epsilon in EPSILONS:
                minus, plus = (values + sign * epsilon * direction for sign in (-1, 1))
                if any((point < 0).any() or (point > 1).any() for point in (minus, plus)):
                    # Do not silently select another direction, shrink epsilon,
                    # clip/project or redraw an initialization to pass the gate.
                    entry["infeasible_epsilon"] = epsilon
                    raise ValueError("Predeclared epsilon leaves the simplex interior")
                endpoints = []
                endpoint_hashes = []
                for point in (minus, plus):
                    px, pe = (point, e) if field == "X" else (x, point)
                    scalar = float(primal(px, pe))
                    if not np.isfinite(scalar):
                        raise FloatingPointError("Nonfinite independent native primal")
                    endpoints.append(scalar)
                    endpoint_hashes.append(graph_digest(px, pe))
                fd = (endpoints[1] - endpoints[0]) / (2 * epsilon)
                agrees = bool(np.isfinite(fd) and analytic * fd > 0
                              and abs(fd - analytic) <= RELATIVE_TOLERANCE * max(abs(fd), abs(analytic)))
                entry["probes"].append({"epsilon": epsilon, "analytic": analytic,
                                         "finite_difference": fd, "agrees": agrees,
                                         "primal_endpoints": endpoints,
                                         "endpoint_graph_sha256": endpoint_hashes})
            probes = entry["probes"]
            if not any(left["agrees"] and right["agrees"] for left, right in zip(probes, probes[1:])):
                raise AssertionError(f"DBMol {objective} raw {field} finite differences failed")
        typed = np.asarray(ge)[:, 1:]
        contrast = float((typed.max(-1) - typed.min(-1)).max())
        if not np.isfinite(contrast) or contrast <= 1e-7:
            raise AssertionError(f"DBMol {objective} has no raw present-bond-type contrast")
        report.update(status="passed", typed_bond_contrast_nonzero=True,
                      typed_bond_maximum_contrast=contrast)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise DerivativeCheckError(str(error), report) from error
    return report


def _request(worker, action, x=None, e=None, **kwargs):
    if x is not None:
        _graph_arrays(x, e)
        kwargs.update(X=x.tolist(), E=e.tolist(), graph_sha256=graph_digest(x, e))
    result = worker.request(action, **kwargs)
    if x is not None and result.get("graph_sha256") != graph_digest(x, e):
        raise AssertionError("Native response changed graph byte identity")
    return result


def _identity(reply):
    keys = ("model_digest", "affinity_model_digest", "structural_model_digest",
            "context_digest", "full_context_digest", "context_file_sha256",
            "checkpoint_sha256", "structural_checkpoint_sha256")
    if reply.get("frozen") is not True or any(key not in reply for key in keys):
        raise AssertionError("Missing independently reported frozen model/context identity")
    result = {key: reply[key] for key in keys}
    if any(not isinstance(value, str) or len(value) != 64 for value in result.values()):
        raise AssertionError("Malformed frozen identity")
    return result


def validate_native_components(worker, graph, key, directory, *, hard_graph, make_loss):
    """Validate exact native primal and the actual first-order JAX transport.

    `make_loss(objective)` creates the production DBMol bridge for the selected
    objective. All four declared objectives are checked without changing their
    coefficients or the raw graph. The total is additionally recomposed from
    independent live primitives inside the native worker. Checkpoint/source/
    resource bindings remain the launcher's separate mandatory responsibilities.
    """
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Native component validation requires allocated DTU HPC")
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.state import RelaxedGraph
    if not jax.config.jax_enable_x64:
        raise RuntimeError("DBMol FP64 scalar transport requires JAX x64")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    x, e = _graph_arrays(graph.X, graph.E_upper)
    hx, he = _graph_arrays(hard_graph.X, hard_graph.E_upper)
    if not all(np.isin(v, (0., 1.)).all() for v in (hx, he)):
        raise ValueError("Hard parity graph must be independently authenticated one-hot native seed")
    report = {"status": "running", "policy": "dbmol-native-raw-fd-v1",
              "scope": "fixed occupied graph/pose/context; no diffusion or optimizer update",
              "components": {}, "input_graph_sha256": graph_digest(x, e),
              "hard_graph_sha256": graph_digest(hx, he),
              "can_recertify_saved_failed_run": False}
    raw_arrays = {"X": x, "E_upper": e, "key_data": np.asarray(jax.random.key_data(key))}
    try:
        if any(worker.identity.get(name) != value for name, value in PINNED.items()):
            raise AssertionError("Native worker source/assets/scope/arithmetic policy differs")
        report["worker_identity"] = dict(worker.identity)
        before = _identity(_request(worker, "verify_identity"))
        for name in ("model_digest", "affinity_model_digest", "structural_model_digest",
                     "context_digest", "full_context_digest", "context_file_sha256"):
            if worker.identity.get(name) != before[name]:
                raise AssertionError("Native ready and fresh frozen identity differ")
        for name in ("checkpoint_sha256", "structural_checkpoint_sha256"):
            if before[name] != PINNED[name]:
                raise AssertionError("Fresh frozen asset identity differs")
        report["frozen_before"] = before
        native_reference = _request(worker, "native_reference")
        if native_reference.get("native_integer_oracle") is not True or native_reference.get("diffusion_called") is not False:
            raise AssertionError("Independent hard oracle lacks explicit native/no-diffusion provenance")
        hard_primal = _request(worker, "evaluate", hx, he, gradient=False,
                               objective="dbmol_total_loss")
        reference_metrics = native_reference.get("metrics", native_reference.get("outputs"))
        if not isinstance(reference_metrics, dict) or not isinstance(hard_primal.get("metrics"), dict):
            raise AssertionError("Hard parity lacks all independently computed native metrics")
        metric_names = worker.identity.get("metric_order")
        if not isinstance(metric_names, list) or len(metric_names) != len(set(metric_names)) or not set(COMPONENTS).issubset(metric_names):
            raise AssertionError("Declared primitive metric inventory missing")
        if set(reference_metrics) != set(metric_names) or set(hard_primal["metrics"]) != set(metric_names):
            raise AssertionError("Hard oracle primitive inventory differs from declared worker")
        differences = {}
        for name in metric_names:
            first, second = float(reference_metrics[name]), float(hard_primal["metrics"][name])
            if not np.isfinite(first) or not np.isfinite(second) or abs(first - second) > PARITY_ATOL:
                raise AssertionError(f"Independent hard/native {name} parity failed")
            differences[name] = abs(first - second)
        report["hard_native_parity"] = {"status": "passed", "atol": PARITY_ATOL,
                                         "absolute_differences": differences}
        evaluated = _request(worker, "evaluate_components", x, e)
        metrics, gradients = evaluated.get("metrics"), evaluated.get("gradients")
        if not isinstance(metrics, dict) or not isinstance(gradients, dict):
            raise AssertionError("Native component/raw-gradient inventory missing")
        recomposed = evaluated.get("recomposed_total")
        if not isinstance(recomposed, dict):
            raise AssertionError("Independent live primitive recomposition is mandatory")
        for name in COMPONENTS:
            selected = gradients[name]
            gx, ge = (np.asarray(selected[k], dtype=np.float32)
                      for k in ("gradient_X", "gradient_E"))
            raw_arrays[f"{name}_gradient_X"], raw_arrays[f"{name}_gradient_E_upper"] = gx, ge
            baseline = float(metrics[name])
            if not np.isfinite(baseline):
                raise FloatingPointError("Nonfinite component baseline")
            def independent_primal(px, pe, objective=name):
                reply = _request(worker, "evaluate", px, pe, gradient=False, objective=objective)
                if reply.get("objective") != objective:
                    raise AssertionError("Native primal objective identity changed")
                value = reply["metrics"][objective]
                if float(reply["values"][0]) != float(value):
                    raise AssertionError("Worker scalar differs from declared live metric")
                return value
            standalone = float(independent_primal(x, e))
            if standalone != baseline:
                raise AssertionError("Independent no-backward primal differs from component AD forward")
            entry = check_raw_finite_differences(independent_primal, x, e, gx, ge, objective=name)
            entry["baseline"] = baseline
            report["components"][name] = entry
            loss = make_loss(name)
            (value, _), bridge_gradient = jax.value_and_grad(loss, has_aux=True)(graph, key=key)
            if np.asarray(value).dtype != np.float64 or float(value) != baseline:
                raise AssertionError("JAX bridge narrowed or changed the native FP64 scalar")
            for actual, expected in ((bridge_gradient.X, gx), (bridge_gradient.E_upper, ge)):
                np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=2e-6)
            entry["production_jax_transport"] = "passed"
            _write(path / "report.json", report)
        total_grad = gradients["dbmol_total_loss"]
        if float(recomposed["value"]) != float(metrics["dbmol_total_loss"]):
            raise AssertionError("Independent live primitive total scalar differs")
        for name in ("gradient_X", "gradient_E"):
            np.testing.assert_array_equal(np.asarray(recomposed[name], dtype=np.float32),
                                          np.asarray(total_grad[name], dtype=np.float32))
        report["live_primitive_recomposition"] = "passed_exact_scalar_and_raw_gradients"
        total_loss = make_loss("dbmol_total_loss")
        # Native key is not consumed. A different JAX key must leave the fixed
        # context scalar unchanged; it cannot become hidden model sampling.
        other, _ = total_loss(graph, key=jax.random.fold_in(key, 1))
        if float(other) != float(metrics["dbmol_total_loss"]):
            raise AssertionError("Fixed native score unexpectedly depends on optimization key")
        # Additive categorical-logit row shifts are mathematical nulls. These
        # supplemental checks use the SAME graph and do not replace raw FD.
        lx, le = jnp.log(graph.X), jnp.log(graph.E_upper)
        def logit_total(ax, ae):
            point = RelaxedGraph(jax.nn.softmax(ax, -1), jax.nn.softmax(ae, -1))
            return total_loss(point, key=key)[0]
        lvalue, logit_gradients = jax.value_and_grad(logit_total, argnums=(0, 1))(lx, le)
        for gradient in logit_gradients:
            array = np.asarray(gradient)
            if not np.isfinite(array).all() or not np.allclose(array.sum(-1), 0, atol=2e-5, rtol=0):
                raise AssertionError("Categorical logit row-shift gradient null failed")
        for field in ("X", "E_upper"):
            sx, se = lx, le
            if field == "X": sx = sx.at[0].add(jnp.float32(0.25))
            else: se = se.at[0].add(jnp.float32(0.25))
            shifted = float(logit_total(sx, se))
            if abs(shifted - float(lvalue)) > PARITY_ATOL:
                raise AssertionError("Categorical logit row-shift scalar null failed")
        report["key_and_logit_row_shift_nulls"] = "passed"
        after = _identity(_request(worker, "verify_identity"))
        report["frozen_after"] = after
        if after != before:
            raise AssertionError("Frozen model/context changed during gradient validation")
        report.update(status="passed", all_components_and_total_raw_fd_passed=True,
                      frozen_unchanged=True)
    except Exception as error:
        if isinstance(error, DerivativeCheckError):
            report["components"][error.report["objective"]] = error.report
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        np.savez_compressed(path / "raw-gradients.npz", **raw_arrays)
        _write(path / "report.json", report)
    return report
