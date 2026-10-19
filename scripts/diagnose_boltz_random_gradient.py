"""Allocated, evidence-only replay of the failed random Boltz bond derivative.

``replay`` uses the original JAX environment to reconstruct the graph actually
tested: softmax(saved logits), rather than the original Dirichlet probabilities.
``diagnose`` uses the isolated native Torch environment and the unchanged source02
scorer. Neither mode supplies a readiness certificate or submits molecular runs.
Additional resolutions and checkpoint controls are diagnostic observations only.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time


INITIALIZATION_SHA = "3513c7b792994de8683243315288dde33a387cd54b0b5a730c227c10b6907b07"
SOURCE_SHA = "5eb6e17ea8d19e52c99a589243228710994dcf137f59a6b9a6659a4fe719a7f4"
CONTEXT_SHA = "254d5b7f11605fb41db6f5abf7015f9cc856a19dfe1d4b506ccaea9007200770"
MODEL_DIGEST = "395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16"
CONTEXT_DIGEST = "495002ffa2a52cc85b58ef34424622763dfc4ca556139a98297e8789ae287aa5"
ORIGINAL_EPSILONS = (0.01, 0.003, 0.001)
EXTRA_EPSILONS = (0.0003, 0.0001)
FAILED_DIRECTION = {"field": "E_upper", "row": 2, "source": 4, "target": 0}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def graph_sha(x, edges):
    return hashlib.sha256(x.tobytes() + edges.tobytes()).hexdigest()


def authenticate_source(root, imported_file):
    """Require the exact sealed scientific module; launchers verify all files."""
    root = Path(root).resolve()
    if sha(root / "source_manifest.json") != SOURCE_SHA:
        raise ValueError("Diagnostic requires the authenticated source02 manifest")
    manifest = json.loads((root / "source_manifest.json").read_text())
    imported_file = Path(imported_file).resolve()
    if not imported_file.is_relative_to(root):
        raise ValueError("Imported Mosaic module came from a different release")
    relative = imported_file.relative_to(root).as_posix()
    if manifest["files"].get(relative) != sha(imported_file):
        raise ValueError("Imported scientific module differs from source02")


def original_gate(analytic, endpoints):
    """Report the unchanged three-scale check, never a readiness approval.

    Fine-resolution observations, repeats and direct-block controls cannot become
    replacement scales. This decision describes this newly recorded replay only;
    the authenticated historical failure remains a failure.
    """
    probes = []
    for epsilon in ORIGINAL_EPSILONS:
        minus, plus = endpoints[epsilon]
        fd = (plus - minus) / (2 * epsilon)
        agrees = bool(
            math.isfinite(fd)
            and math.isfinite(analytic)
            and abs(analytic) > 1e-7
            and analytic * fd > 0
            and abs(fd - analytic) <= 0.15 * max(abs(fd), abs(analytic))
        )
        probes.append(
            {
                "epsilon": epsilon,
                "analytic": analytic,
                "finite_difference": fd,
                "agrees": agrees,
                "primal_endpoints": [minus, plus],
            }
        )
    passed = any(a["agrees"] and b["agrees"] for a, b in zip(probes, probes[1:]))
    return {
        "status": "passed" if passed else "failed",
        "probes": probes,
        "relative_tolerance": 0.15,
        "required_adjacent_scales": 2,
        "readiness_certificate": False,
    }


def replay(args):
    # Deliberately no Torch import in this environment.
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry import state

    authenticate_source(args.scientific_source, state.__file__)
    if sha(args.initialization) != INITIALIZATION_SHA:
        raise ValueError("Saved random initialization SHA differs")
    versions = {
        name: importlib.metadata.version(name) for name in ("jax", "numpy", "equinox")
    }
    if versions != {"jax": "0.11.1", "numpy": "2.5.3", "equinox": "0.13.8"}:
        raise ValueError("Replay requires the original JAX/NumPy/Equinox versions")
    if jax.config.jax_enable_x64 or any(d.platform != "cpu" for d in jax.devices()):
        raise ValueError("Replay requires the original FP32 CPU JAX execution")
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.initialization, allow_pickle=False) as saved:
        logits = state.GraphLogits(
            jnp.asarray(saved["atom_logits"]), jnp.asarray(saved["bond_logits"])
        )
        graph = logits.graph()
        x, edges = np.asarray(graph.X), np.asarray(graph.E_upper)
    if (
        x.dtype != np.float32
        or edges.dtype != np.float32
        or x.shape != (6, 6)
        or edges.shape != (15, 5)
    ):
        raise ValueError("Replayed graph differs from the six-slot FP32 contract")
    direction = np.zeros_like(edges)
    direction[2, 4], direction[2, 0] = -1, 1
    points = [{"name": "baseline", "epsilon": 0.0, "sign": 0, "original_gate": False}]
    point_x, point_edges = [x], [edges]
    for epsilon in ORIGINAL_EPSILONS + EXTRA_EPSILONS:
        for sign in (-1, 1):
            # Same NumPy arithmetic and JAX conversion as the original checker.
            changed = np.asarray(jnp.asarray(edges + sign * epsilon * direction))
            name = f"bond-{epsilon:g}-{'minus' if sign < 0 else 'plus'}"
            points.append(
                {
                    "name": name,
                    "epsilon": epsilon,
                    "sign": sign,
                    "original_gate": epsilon in ORIGINAL_EPSILONS,
                }
            )
            point_x.append(x)
            point_edges.append(changed)
    for point, px, pe in zip(points, point_x, point_edges):
        if not np.isfinite(px).all() or not np.isfinite(pe).all():
            raise FloatingPointError("Replayed point is nonfinite")
        point["graph_sha256"] = graph_sha(px, pe)
    np.savez_compressed(
        args.output / "replay.npz",
        X=x,
        E_upper=edges,
        point_X=np.stack(point_x),
        point_E_upper=np.stack(point_edges),
    )
    write_json(
        args.output / "replay.json",
        {
            "status": "recorded",
            "readiness_certificate": False,
            "job_id": os.environ["LSB_JOBID"],
            "initialization_sha256": INITIALIZATION_SHA,
            "source_manifest_sha256": SOURCE_SHA,
            "versions": versions,
            "reconstruction": "GraphLogits(saved atom_logits, saved bond_logits).graph()",
            "direction": FAILED_DIRECTION,
            "graph_sha256": graph_sha(x, edges),
            "replay_sha256": sha(args.output / "replay.npz"),
            "points": points,
        },
    )


@contextlib.contextmanager
def direct_blocks(model):
    """Temporarily restore the exact upstream methods captured by source02.

    This is an explicitly labeled diagnostic control, not a production scorer
    edit, alternative validated model or permission to waive any derivative gate.
    """
    stacks = [model.msa_module, model.pairformer_module]
    heads = (
        [model.affinity_module1, model.affinity_module2]
        if model.affinity_ensemble
        else [model.affinity_module]
    )
    stacks.extend(head.pairformer_stack for head in heads)
    changed = []
    try:
        for stack in stacks:
            for block in stack.layers:
                wrapper = block.forward
                original = (getattr(wrapper, "__kwdefaults__", None) or {}).get(
                    "_original"
                )
                if original is None:
                    raise ValueError(
                        "Source02 checkpoint wrapper could not be authenticated"
                    )
                changed.append((block, wrapper))
                block.forward = original
        yield len(changed)
    finally:
        for block, wrapper in changed:
            block.forward = wrapper


def diagnose(args):
    # Deliberately no JAX or chemistry imports in this environment.
    import numpy as np
    import torch
    from mosaic.small_molecule.boltz2 import native

    authenticate_source(args.scientific_source, native.__file__)
    if not torch.cuda.is_available() or sha(args.context) != CONTEXT_SHA:
        raise ValueError("Require allocated CUDA and the exact failed-study context")
    if sha(args.replay / "replay.npz") != args.replay_sha256:
        raise ValueError("Replayed graph archive changed")
    metadata = json.loads((args.replay / "replay.json").read_text())
    if (
        metadata["replay_sha256"] != args.replay_sha256
        or metadata["initialization_sha256"] != INITIALIZATION_SHA
        or metadata["source_manifest_sha256"] != SOURCE_SHA
        or metadata["direction"] != FAILED_DIRECTION
    ):
        raise ValueError("Replayed graph provenance differs")
    with np.load(args.replay / "replay.npz", allow_pickle=False) as saved:
        xs, es = saved["point_X"], saved["point_E_upper"]
    points = metadata["points"]
    expected_pairs = [(0.0, 0)] + [
        (e, s) for e in ORIGINAL_EPSILONS + EXTRA_EPSILONS for s in (-1, 1)
    ]
    if [(p["epsilon"], p["sign"]) for p in points] != expected_pairs:
        raise ValueError("Diagnostic point set or ordering differs")
    if (
        xs.shape != (11, 6, 6)
        or es.shape != (11, 15, 5)
        or xs.dtype != np.float32
        or es.dtype != np.float32
    ):
        raise ValueError("Replayed point schema differs")
    for point, x, e in zip(points, xs, es):
        if (
            graph_sha(x, e) != point["graph_sha256"]
            or not np.isfinite(x).all()
            or not np.isfinite(e).all()
        ):
            raise ValueError("Replayed point bytes or finite values differ")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    context = torch.load(args.context, map_location="cuda", weights_only=True)
    native.validate_context(context)
    model = native.load_model(args.checkpoint).to("cuda")
    before = native.tensor_digest(model.state_dict())
    context_before = native.tensor_digest(
        context["features"] | {"coordinates": context["coordinates"]}
    )
    if before != MODEL_DIGEST or context_before != CONTEXT_DIGEST:
        raise ValueError("Frozen model/context tensor identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "readiness_certificate": False,
        "scope": "evidence-only fixed-graph replay; no optimization or readiness approval",
        "historical_gate_status": "failed",
        "job_id": os.environ["LSB_JOBID"],
        "source_manifest_sha256": SOURCE_SHA,
        "replay_sha256": args.replay_sha256,
        "context_sha256": CONTEXT_SHA,
        "checkpoint_sha256": native.AFFINITY_SHA256,
        "model_digest_before": before,
        "context_digest_before": context_before,
        "direction": FAILED_DIRECTION,
        "repeats": args.repeats,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "threads": torch.get_num_threads(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "evaluations": [],
        "original_gate_replays": [],
    }
    heads = (
        [model.affinity_module1, model.affinity_module2]
        if model.affinity_ensemble
        else [model.affinity_module]
    )
    per_head = []

    def capture_head(module, inputs, output):
        del module, inputs
        pair = [
            float(output[name].detach().reshape(()).cpu())
            for name in ("affinity_pred_value", "affinity_logits_binary")
        ]
        if not all(math.isfinite(v) for v in pair):
            raise FloatingPointError("Nonfinite native head output")
        per_head.append(
            {"affinity_pred_value": pair[0], "affinity_logits_binary": pair[1]}
        )

    hooks = [head.register_forward_hook(capture_head) for head in heads]

    def rng_digest():
        values = {"cpu": torch.get_rng_state()}
        values.update(
            {f"cuda-{i}": v for i, v in enumerate(torch.cuda.get_rng_state_all())}
        )
        return native.tensor_digest(values)

    def evaluate(variant, point, x_array, e_array, mode, repeat):
        per_head.clear()
        started, rng_before = time.perf_counter(), rng_digest()
        enabled = mode != "independent_primal"
        x = torch.tensor(x_array, device="cuda", requires_grad=enabled)
        e = torch.tensor(e_array, device="cuda", requires_grad=enabled)
        with torch.set_grad_enabled(enabled):
            values = native.score(model, context, x, e)
            gradients = (
                torch.autograd.grad(values[0], (x, e)) if mode == "autodiff" else None
            )
        torch.cuda.synchronize()
        if (
            not torch.isfinite(values).all()
            or gradients is not None
            and any(not torch.isfinite(g).all() for g in gradients)
        ):
            raise FloatingPointError("Nonfinite native diagnostic value/derivative")
        name = f"{variant}-{point['name']}-{mode}-{repeat}.npz"
        arrays = {"values": values.detach().cpu().numpy()}
        if gradients is not None:
            arrays.update(
                gradient_X=gradients[0].detach().cpu().numpy(),
                gradient_E=gradients[1].detach().cpu().numpy(),
            )
        np.savez_compressed(args.output / name, **arrays)
        result = {
            "variant": variant,
            "point": point["name"],
            "epsilon": point["epsilon"],
            "sign": point["sign"],
            "mode": mode,
            "repeat": repeat,
            "graph_sha256": graph_sha(
                x.detach().cpu().numpy(), e.detach().cpu().numpy()
            ),
            "values": arrays["values"].tolist(),
            "per_head": list(per_head),
            "raw_artifact": name,
            "raw_sha256": sha(args.output / name),
            "rng_before": rng_before,
            "rng_after": rng_digest(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if result["graph_sha256"] != point["graph_sha256"]:
            raise ValueError("Native diagnostic graph transport changed bytes")
        if gradients is not None:
            result["analytic_directional"] = float(
                gradients[1][2, 0] - gradients[1][2, 4]
            )
        report["evaluations"].append(result)
        write_json(args.output / "report.json", report)

    try:
        for variant in ("source02_checkpoint", "diagnostic_direct_blocks"):
            control = (
                contextlib.nullcontext(0)
                if variant == "source02_checkpoint"
                else direct_blocks(model)
            )
            with control as changed:
                if variant == "diagnostic_direct_blocks":
                    report["direct_blocks_restored"] = changed
                for point, x, e in zip(points, xs, es):
                    for repeat in range(args.repeats):
                        for mode in (
                            "independent_primal",
                            "autodiff",
                            "gradient_enabled_forward_only",
                        ):
                            evaluate(variant, point, x, e, mode, repeat)
                for repeat in range(args.repeats):
                    rows = [
                        r
                        for r in report["evaluations"]
                        if r["variant"] == variant and r["repeat"] == repeat
                    ]
                    analytic = next(
                        r["analytic_directional"]
                        for r in rows
                        if r["sign"] == 0 and r["mode"] == "autodiff"
                    )
                    endpoints = {
                        eps: [
                            next(
                                r["values"][0]
                                for r in rows
                                if r["epsilon"] == eps
                                and r["sign"] == s
                                and r["mode"] == "independent_primal"
                            )
                            for s in (-1, 1)
                        ]
                        for eps in ORIGINAL_EPSILONS
                    }
                    report["original_gate_replays"].append(
                        {
                            "variant": variant,
                            "repeat": repeat,
                            **original_gate(analytic, endpoints),
                        }
                    )
        report["status"] = "recorded"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for hook in hooks:
            hook.remove()
        try:
            after = native.tensor_digest(model.state_dict())
            context_after = native.tensor_digest(
                context["features"] | {"coordinates": context["coordinates"]}
            )
            if (
                after != before
                or context_after != context_before
                or any(
                    p.requires_grad or p.grad is not None for p in model.parameters()
                )
            ):
                raise ValueError("Frozen diagnostic model/context identity changed")
            report.update(
                model_digest_after=after,
                context_digest_after=context_after,
                frozen_identity_status="passed",
            )
        except Exception as error:
            report.update(
                status="failed",
                frozen_identity_status="failed",
                identity_error=str(error),
            )
        write_json(args.output / "report.json", report)
    if report["status"] != "recorded":
        raise RuntimeError(
            "Diagnostic failed; preserved report is not readiness evidence"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("replay", "diagnose"):
        command = commands.add_parser(name)
        command.add_argument("--scientific-source", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "replay":
            command.add_argument("--initialization", type=Path, required=True)
        else:
            command.add_argument("--replay", type=Path, required=True)
            command.add_argument("--replay-sha256", required=True)
            command.add_argument("--context", type=Path, required=True)
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--repeats", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Both replay and diagnosis require allocated DTU HPC")
    (replay if args.mode == "replay" else diagnose)(args)


if __name__ == "__main__":
    main()
