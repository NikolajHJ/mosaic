"""Allocated native worker entry point; JSON transport contains only six-slot graphs."""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Native work requires allocated DTU HPC")
    import torch
    from .native import (
        AFFINITY_SHA256,
        BOLTZ_COMMIT,
        file_digest,
        load_model,
        native_reference,
        score,
        tensor_digest,
        validate_context,
    )

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production worker requires an allocated CUDA GPU")
    if file_digest(args.context) != args.context_sha256:
        raise ValueError("Prepared context file identity differs")
    context = torch.load(args.context, map_location=args.device, weights_only=True)
    validate_context(context)
    context_before = tensor_digest(
        context["features"] | {"coordinates": context["coordinates"]}
    )
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(args.checkpoint).to(args.device)
    model_before = tensor_digest(model.state_dict())
    print(
        json.dumps(
            {
                "status": "ready",
                "protocol": "mosaic-boltz-affinity-v1",
                "source": BOLTZ_COMMIT,
                "checkpoint_sha256": AFFINITY_SHA256,
                "context_file_sha256": args.context_sha256,
                "context_digest": context_before,
                "model_digest": model_before,
                "pose_policy": "fixed native predicted pose; no diffusion gradients",
                "optimized_output": "affinity_pred_value",
                "recycling_steps": 0,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        request = json.loads(line)
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if request["action"] == "evaluate":
                    x = torch.tensor(
                        request["X"],
                        dtype=torch.float32,
                        device=args.device,
                        requires_grad=request["gradient"],
                    )
                    edges = torch.tensor(
                        request["E"],
                        dtype=torch.float32,
                        device=args.device,
                        requires_grad=request["gradient"],
                    )
                    if not torch.isfinite(x).all() or not torch.isfinite(edges).all():
                        raise FloatingPointError("Nonfinite input graph")
                    with torch.set_grad_enabled(request["gradient"]):
                        values = score(model, context, x, edges)
                        gradients = (
                            torch.autograd.grad(values[0], (x, edges))
                            if request["gradient"]
                            else None
                        )
                    if (
                        not torch.isfinite(values).all()
                        or gradients is not None
                        and any(not torch.isfinite(g).all() for g in gradients)
                    ):
                        raise FloatingPointError("Nonfinite native output/gradient")
                    graph_sha = hashlib.sha256(
                        x.detach().cpu().numpy().tobytes()
                        + edges.detach().cpu().numpy().tobytes()
                    ).hexdigest()
                    if graph_sha != request["graph_sha256"]:
                        raise ValueError("Transport changed exact input graph bytes")
                    result = {
                        "values": values.detach().cpu().tolist(),
                        "graph_sha256": graph_sha,
                    }
                    if gradients is not None:
                        result.update(
                            gradient_X=gradients[0].detach().cpu().tolist(),
                            gradient_E=gradients[1].detach().cpu().tolist(),
                        )
                elif request["action"] == "native_reference":
                    result = {"values": native_reference(model, context).cpu().tolist()}
                elif request["action"] == "verify_identity":
                    after = tensor_digest(model.state_dict())
                    c_after = tensor_digest(
                        context["features"] | {"coordinates": context["coordinates"]}
                    )
                    if (
                        after != model_before
                        or c_after != context_before
                        or any(
                            p.requires_grad or p.grad is not None
                            for p in model.parameters()
                        )
                    ):
                        raise ValueError("Frozen native model/context changed")
                    result = {"model_digest": after, "context_digest": c_after}
                else:
                    raise ValueError("Unknown worker action")
            print(
                json.dumps(
                    {
                        "id": request["id"],
                        "status": "passed",
                        "elapsed_seconds": time.perf_counter() - started,
                        **result,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "id": request["id"],
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                ),
                flush=True,
            )
            raise


if __name__ == "__main__":
    main()
