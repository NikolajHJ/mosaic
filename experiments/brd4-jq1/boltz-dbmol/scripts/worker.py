"""Allocated two-checkpoint Boltz DBMol JSON-line worker; no optimizer or diffusion."""

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
    parser.add_argument("--structure-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Native DBMol work requires an allocated DTU HPC job")

    import torch
    from mosaic.small_molecule.boltz2 import native
    from mosaic.small_molecule.boltz2.dbmol import (
        ARITHMETIC, COMPONENTS, CONFIDENCE_WEIGHTS, DBMolScorer, METRICS,
        OBJECTIVES, PROFILE, PROTOCOL, STRUCTURE_SHA256, assemble_components,
        load_structure_model,
    )

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The DBMol pilot requires an allocated CUDA GPU")
    if native.file_digest(args.context) != args.context_sha256:
        raise ValueError("Prepared context file identity differs")
    context = torch.load(args.context, map_location=args.device, weights_only=True)
    with contextlib.redirect_stdout(sys.stderr):
        affinity = native.load_model(args.checkpoint).to(args.device)
        structure = load_structure_model(args.structure_checkpoint).to(args.device)
        scorer = DBMolScorer(affinity, structure, context)

    def identity():
        context_values = context["features"] | {"coordinates": context["coordinates"]}
        full_context = context_values | {key: context[key] for key in (
            "atom_indices", "token_indices", "seed_X", "seed_E")}
        models = {"affinity_model_digest": native.tensor_digest(affinity.state_dict()),
                  "structural_model_digest": native.tensor_digest(structure.state_dict())}
        combined = hashlib.sha256(json.dumps(models, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (any(parameter.requires_grad or parameter.grad is not None for parameter in scorer.parameters())
                or any(value.requires_grad for value in full_context.values())
                or any(module.training for module in scorer.modules())):
            raise ValueError("DBMol weights/context must remain frozen and eval")
        metadata = {key: context[key] for key in ("version", "n_atoms", "pocket")}
        full_digest = hashlib.sha256(json.dumps(
            {"tensors": native.tensor_digest(full_context), "metadata": metadata},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {**models, "model_digest": combined,
                "context_digest": native.tensor_digest(context_values),
                "full_context_digest": full_digest}

    before = identity()
    ready = {
        "status": "ready", "protocol": PROTOCOL, "profile": PROFILE,
        "source": native.BOLTZ_COMMIT,
        "checkpoint_sha256": native.AFFINITY_SHA256,
        "structural_checkpoint_sha256": STRUCTURE_SHA256,
        "context_file_sha256": args.context_sha256, **before,
        "pocket_definition_sha256": context["pocket"]["pocket_definition_sha256"],
        "native_pocket_token_indices": scorer.pocket.nonzero().flatten().tolist(),
        "pose_policy": "fixed original predicted pose; detached; no diffusion call",
        "diffusion_gradient": False, "recycling_steps": 0, "n_atoms": 25,
        "optimized_output": "dbmol_total_loss", "objective_arithmetic": ARITHMETIC,
        "scalar_dtype": "float64", "graph_dtype": "float32", "gradient_dtype": "float32",
        "metric_order": list(METRICS), "supported_objectives": list(OBJECTIVES),
        "contact_weight": 2.5, "confidence_weight": 2.5,
        "confidence_signed_weights": CONFIDENCE_WEIGHTS,
        "physical_trunks_per_forward": 2, "affinity_mw_correction": False,
    }
    print(json.dumps(ready, allow_nan=False), flush=True)

    def objective_of(request):
        objective = request.get("objective", request.get("component", "dbmol_total_loss"))
        if (objective not in OBJECTIVES
                or "objective" in request and "component" in request
                and request["objective"] != request["component"]):
            raise ValueError("Unknown or conflicting DBMol objective")
        return objective

    def scalars(outputs):
        if set(METRICS) - set(outputs):
            raise ValueError("Incomplete DBMol primitive/component output")
        result = {key: float(outputs[key].detach().reshape(()).cpu()) for key in METRICS}
        if any(not torch.isfinite(outputs[key]).all() for key in METRICS):
            raise FloatingPointError("Nonfinite DBMol scalar")
        return result

    def graph(request, gradient):
        x = torch.tensor(request["X"], dtype=torch.float32, device=args.device, requires_grad=gradient)
        edges = torch.tensor(request["E"], dtype=torch.float32, device=args.device, requires_grad=gradient)
        graph_sha = hashlib.sha256(x.detach().cpu().numpy().tobytes()
                                  + edges.detach().cpu().numpy().tobytes()).hexdigest()
        if graph_sha != request["graph_sha256"]:
            raise ValueError("Transport changed exact FP32 graph bytes")
        return x, edges, graph_sha

    def derivative(output, x, edges, *, retain_graph=False):
        if not output.requires_grad:
            raise ValueError("Detached DBMol derivative objective")
        gradients = torch.autograd.grad(output, (x, edges), retain_graph=retain_graph)
        if any(value.dtype != torch.float32 or not bool(value.isfinite().all()) for value in gradients):
            raise FloatingPointError("Malformed/nonfinite raw DBMol derivative")
        return {"gradient_X": gradients[0].detach().cpu().tolist(),
                "gradient_E": gradients[1].detach().cpu().tolist()}

    for line in sys.stdin:
        request = {}
        started = time.perf_counter()
        try:
            request = json.loads(line)
            if type(request.get("id")) is not int:
                raise ValueError("A unique integer request ID is required")
            with contextlib.redirect_stdout(sys.stderr):
                action = request.get("action")
                if action in ("evaluate", "evaluate_components"):
                    want_gradient = request.get("gradient", action == "evaluate_components")
                    if type(want_gradient) is not bool:
                        raise ValueError("gradient must be an explicit boolean")
                    objective = objective_of(request)
                    x, edges, graph_sha = graph(request, want_gradient)
                    with torch.set_grad_enabled(want_gradient):
                        outputs = scorer(x, edges)
                        metrics = scalars(outputs)
                        result = {"values": [metrics[objective], metrics["affinity_probability_binary"],
                                              metrics["affinity_logits_binary"]],
                                  "objective": objective, "metrics": metrics,
                                  "graph_sha256": graph_sha,
                                  "contact_selection": dict(scorer.last_contact_selection)}
                        if action == "evaluate_components":
                            if not want_gradient:
                                raise ValueError("Component audit requires gradients")
                            # Reconstruct from LIVE primitives, not rounded stored gradients.
                            recomposed = assemble_components(outputs)
                            keys = (*COMPONENTS, "affinity_pred_value", "affinity_logits_binary")
                            gradients = {key: derivative(outputs[key], x, edges, retain_graph=True)
                                         for key in keys}
                            regrad = derivative(recomposed["dbmol_total_loss"], x, edges)
                            original = gradients["dbmol_total_loss"]
                            exact = (torch.equal(outputs["dbmol_total_loss"], recomposed["dbmol_total_loss"])
                                     and original == regrad)
                            if not exact:
                                raise ValueError("Live total/component derivative recomposition differs")
                            result.update(gradients=gradients, reassembled_total_gradient=regrad,
                                          recomposed_total={"value": float(recomposed["dbmol_total_loss"].detach().cpu()), **regrad},
                                          recomposition={"scalar_exact": True, "gradient_X_exact": True,
                                                         "gradient_E_exact": True},
                                          exact_live_recomposition=True,
                                          component_scalars={key: metrics[key] for key in COMPONENTS})
                        elif want_gradient:
                            result.update(derivative(outputs[objective], x, edges))
                elif action == "native_reference":
                    objective = objective_of(request)
                    outputs = scorer.native_reference()
                    metrics = scalars(outputs)
                    result = {"values": [metrics[objective], metrics["affinity_probability_binary"],
                                          metrics["affinity_logits_binary"]],
                              "objective": objective, "metrics": metrics,
                              "contact_selection": dict(scorer.last_contact_selection),
                              "native_integer_oracle": True, "diffusion_called": False}
                elif action == "verify_identity":
                    after = identity()
                    if after != before or native.file_digest(args.context) != args.context_sha256:
                        raise ValueError("Frozen checkpoint/context identity changed")
                    result = {**after, "frozen": True,
                              "context_file_sha256": args.context_sha256,
                              "structural_checkpoint_sha256": STRUCTURE_SHA256,
                              "checkpoint_sha256": native.AFFINITY_SHA256}
                else:
                    raise ValueError("Unknown DBMol worker action")
            print(json.dumps({"id": request["id"], "status": "passed",
                              "elapsed_seconds": time.perf_counter() - started,
                              **result}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"id": request.get("id"), "status": "failed",
                              "error": f"{type(error).__name__}: {error}"}), flush=True)
            raise


if __name__ == "__main__":
    main()
