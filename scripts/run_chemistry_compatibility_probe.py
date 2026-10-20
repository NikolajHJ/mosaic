"""Allocated investigation of saved gradients; no Boltz execution or optimization."""

from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not os.environ.get("LSB_JOBID"):
        raise SystemExit("Chemistry investigation requires an allocated CPU job")
    import jax
    import jax.numpy as jnp
    import numpy as np

    from chemistry_compatibility_probe import (
        hard_cap_loss,
        loss,
        valid_margin_loss,
    )
    from mosaic.small_molecule.chemistry.state import GraphLogits
    from mosaic.small_molecule.io import hard_readout

    assert not jax.config.jax_enable_x64
    study, output = map(Path, sys.argv[1:])
    assert sha(study / "summary.json") == (
        "7f49087f12efc762a46691581f80bc42c43395c00b0cc4778a9e04a6679aae7c"
    )
    case_path = study / "case-0/report.json"
    assert sha(case_path) == (
        "13ca896450a8be2e298858226e9f11dac2be9e15f0d1cc092d6e4edf3707dad2"
    )
    states = json.loads(case_path.read_text())["states"]
    report = dict(
        status="running",
        job_id=os.environ["LSB_JOBID"],
        native_job="29759212",
        scope="experimental saved-gradient CPU probes; not native validation or an optimization trajectory",
        script_sha256=sha(Path(__file__)),
        cases=[],
        limitations=[
            "upper-valence compatibility only; no connectivity or aromatic validity guarantee",
            "aromatic bonds count as one; sulfur's generous cap can be favored",
            "one-step proposals use saved gradients; native scalar not reevaluated or accepted",
        ],
    )
    try:
        for step in (0, 1000, 1467, 8294, 9639, 9998, 9999, 10000):
            saved = states[step]
            path = study / "case-0" / saved["snapshot"]
            assert sha(path) == saved["snapshot_sha256"]
            with np.load(path, allow_pickle=False) as z:
                arrays = {name: z[name].copy() for name in z.files}
            a, b = map(jnp.asarray, (arrays["atom_logits"], arrays["bond_logits"]))
            reference_path = study / "case-0/references" / f"step-{step:04d}.json"
            reference = json.loads(reference_path.read_text())
            receipt = {
                k: v
                for k, v in reference.items()
                if k
                not in ("arrays", "identity", "preparation_seconds", "search_seconds")
            }
            identity = hashlib.sha256(
                json.dumps(
                    {"arrays": reference["arrays"], "receipt": receipt},
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            assert identity == reference["identity"] == saved["preparation_identity"]
            assert reference["arrays"]["route"][0] in (0, 1, 2)
            target_a = np.asarray(reference["arrays"]["target_atoms"][0]) - 1
            matrix = np.asarray(reference["arrays"]["target_bonds"][0])
            ii, jj = np.triu_indices(6, 1)
            target_b = matrix[ii, jj]
            hard_labels = np.argmax(arrays["X"], axis=-1)
            ng_parts = []
            for pkey, gkey in [
                ("X", "affinity_gradient_X"),
                ("E_upper", "affinity_gradient_E_upper"),
            ]:
                p, g = arrays[pkey], arrays[gkey]
                ng_parts.append(p * (g - (p * g).sum(-1, keepdims=True)))
            ng = np.concatenate([g.ravel() for g in ng_parts])
            item = dict(
                step=step,
                saved_readout=saved["readout"],
                saved_repair=saved["components"]["chemistry_loss"],
                snapshot_sha256=sha(path),
                reference_sha256=sha(reference_path),
                candidates=[],
            )
            report["cases"].append(item)
            functions = dict(
                soft_local_compatibility=loss,
                frozen_hard_cap=partial(hard_cap_loss, hard_labels=hard_labels),
                frozen_valid_target_margin=partial(
                    valid_margin_loss, target_atoms=target_a, target_bonds=target_b
                ),
            )
            for name, fn in functions.items():
                forward = jax.jit(fn)
                value, gradients = jax.jit(jax.value_and_grad(fn, argnums=(0, 1)))(a, b)
                ga, gb = (np.asarray(v) for v in gradients)
                assert (
                    np.isfinite(float(value))
                    and np.isfinite(ga).all()
                    and np.isfinite(gb).all()
                )
                g = np.concatenate([ga.ravel(), gb.ravel()])
                record = dict(
                    name=name,
                    value=float(value),
                    atom_gradient_norm=float(np.linalg.norm(ga)),
                    bond_gradient_norm=float(np.linalg.norm(gb)),
                    total_gradient_norm=float(np.linalg.norm(g)),
                    native_gradient_norm=float(np.linalg.norm(ng)),
                    native_dot=float(np.dot(g, ng)),
                    finite_differences=[],
                    proposals=[],
                    atom0_gradient=ga[0].tolist(),
                    edge_1_3_gradient=gb[1].tolist(),
                )
                item["candidates"].append(record)
                for field in ("atom", "bond"):
                    da, db = np.zeros_like(ga), np.zeros_like(gb)
                    if field == "atom":
                        da[0, 5], da[0, 0] = 1.0, -1.0
                    else:
                        db[1, 1], db[1, 0] = 1.0, -1.0
                    ad = float((ga * da).sum() + (gb * db).sum())
                    probes = []
                    for eps in (0.01, 0.003, 0.001):
                        plus = float(forward(a + eps * da, b + eps * db))
                        minus = float(forward(a - eps * da, b - eps * db))
                        fd = (plus - minus) / (2 * eps)
                        passed = abs(ad - fd) <= 2e-5 + 0.15 * max(abs(ad), abs(fd))
                        probes.append(
                            dict(
                                eps=eps,
                                autodiff=ad,
                                independent_primal_fd=fd,
                                passed=passed,
                            )
                        )
                    adjacent = any(
                        x["passed"] and y["passed"] for x, y in zip(probes, probes[1:])
                    )
                    assert adjacent, (step, name, field, probes)
                    record["finite_differences"].append(
                        dict(field=field, probes=probes, two_adjacent_passed=adjacent)
                    )
                if name == "soft_local_compatibility":
                    assert np.linalg.norm(ga) > 1e-9 and np.linalg.norm(gb) > 1e-9
                # These are unaccepted trial proposals, NOT native Boltz predictions.
                for rate in (0.01, 0.03):
                    for weight in (0.1, 1.0, 3.0):
                        pa = a - rate * (arrays["gradient_atom_logits"] + weight * ga)
                        pb = b - rate * (arrays["gradient_bond_logits"] + weight * gb)
                        graph = GraphLogits(pa, pb).graph()
                        readout = hard_readout(graph)
                        record["proposals"].append(
                            dict(
                                maximum_rate=rate,
                                added_coefficient=weight,
                                readout=readout,
                                compatibility_after=float(forward(pa, pb)),
                                edge_1_3=np.asarray(graph.E_upper[1]).tolist(),
                                atom0=np.asarray(graph.X[0]).tolist(),
                            )
                        )
        report["status"] = "recorded"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                status=report["status"], cases=len(report["cases"]), sha256=sha(output)
            )
        )
    )


if __name__ == "__main__":
    main()
