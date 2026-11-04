"""Actual independent-primal finite differences of the declared fixed-pose score."""

from pathlib import Path
import json

import jax.numpy as jnp
import numpy as np

from ..optimizers import evaluate_graph
from ..state import RelaxedGraph


def check_affinity_gradients(loss, graph, key, directory):
    """Require finite, nonzero atom AND bond-type derivatives at two adjacent scales.

    These are continuous input derivative probes, not hard molecules. The native
    network and context remain identical in backward and every independent primal.
    No failed finite difference is replaced by an AD-only acceptance contract.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "scope": "quantitative affinity; fixed pose/masks; first-order raw graph derivatives",
        "rows": [],
    }
    try:
        raw = evaluate_graph(loss, graph, key)
        primal, _ = loss(graph, key=key)
        if abs(float(primal) - float(raw.value)) > 2e-5:
            raise AssertionError("Native AD/primal baseline differs")
        for field in ("X", "E_upper"):
            values = np.asarray(getattr(graph, field))
            grad = np.asarray(getattr(raw.gradients, field))
            row = int(np.argmax(grad.max(-1) - grad.min(-1)))
            source, target = int(grad[row].argmin()), int(grad[row].argmax())
            direction = np.zeros_like(values)
            direction[row, source], direction[row, target] = -1, 1
            analytic = float(grad[row, target] - grad[row, source])
            probes = []
            for epsilon in (0.01, 0.003, 0.001):
                endpoints = []
                for sign in (-1, 1):
                    changed = jnp.asarray(values + sign * epsilon * direction)
                    point = (
                        RelaxedGraph(changed, graph.E_upper)
                        if field == "X"
                        else RelaxedGraph(graph.X, changed)
                    )
                    scalar, _ = loss(point, key=key)
                    endpoints.append(float(scalar))
                fd = (endpoints[1] - endpoints[0]) / (2 * epsilon)
                agrees = bool(
                    np.isfinite(fd)
                    and abs(analytic) > 1e-7
                    and analytic * fd > 0
                    and abs(fd - analytic) <= 0.15 * max(abs(fd), abs(analytic))
                )
                probes.append(
                    {
                        "epsilon": epsilon,
                        "finite_difference": fd,
                        "analytic": analytic,
                        "agrees": agrees,
                        "primal_endpoints": endpoints,
                    }
                )
            report["rows"].append(
                {
                    "field": field,
                    "row": row,
                    "source": source,
                    "target": target,
                    "probes": probes,
                }
            )
            (directory / "report.json").write_text(json.dumps(report, indent=2))
            if not any(a["agrees"] and b["agrees"] for a, b in zip(probes, probes[1:])):
                raise AssertionError(
                    f"Native affinity {field} finite differences failed"
                )
        # Contrast present bond categories: unlike Protenix this is not a null.
        grad = np.asarray(raw.gradients.E_upper)[:, 1:]
        if not np.isfinite(grad).all() or np.max(grad.max(-1) - grad.min(-1)) <= 1e-7:
            raise AssertionError("Native affinity has no observed typed-bond contrast")
        report.update(
            status="passed", baseline=float(primal), typed_bond_contrast_nonzero=True
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
