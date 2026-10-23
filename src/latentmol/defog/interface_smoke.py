"""HPC-only native DeFoG GuacaMol forward-derivative feasibility check.

The fixed hard target makes this a derivative test of a denoising CE scalar,
not an optimization objective for a free molecule. No categorical transition,
chemical repair, or source017 candidate is differentiated here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import torch
from torch.nn import functional as F

from latentmol.data import GraphDataset, collate_graphs
from latentmol.defog.codec import to_guacamol_hard
from latentmol.defog.assets import DEFOG_COMMIT, GUACAMOL_SHA256
from latentmol.experiments import ExperimentRun, file_sha256, write_json


def _verify_assets(source: Path, checkpoint: Path) -> None:
    if file_sha256(checkpoint) != GUACAMOL_SHA256:
        raise ValueError("Official GuacaMol checkpoint SHA256 differs from acquired asset")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != DEFOG_COMMIT:
        raise ValueError(f"DeFoG source commit differs: {commit}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("External DeFoG source checkout is modified")


def _load_native_model(source: Path, checkpoint: Path, device: torch.device):
    # The source and checksum are verified before unpickling the trusted official
    # Lightning checkpoint. The public CLI imports graph_tool eagerly; importing
    # this module directly avoids that unrelated evaluation dependency.
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source))
    from graph_discrete_flow_model import GraphDiscreteFlowModel

    model = GraphDiscreteFlowModel.load_from_checkpoint(
        str(checkpoint), map_location="cpu", strict=False,
        train_metrics=None, sampling_metrics=None, visualization_tools=None,
    )
    # The official checkpoint saves four sampling-metric buffers, while the
    # inference-only model intentionally omits that evaluator. Audit every key
    # and tensor after Lightning's non-strict load so only those buffers may be
    # ignored; no denoiser parameter or other buffer may be missing or altered.
    saved_state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    loaded_state = model.state_dict()
    expected_unused = {
        "sampling_metrics.n_target_dist",
        "sampling_metrics.node_target_dist",
        "sampling_metrics.edge_target_dist",
        "sampling_metrics.valency_target_dist",
    }
    unused = set(saved_state) - set(loaded_state)
    missing = set(loaded_state) - set(saved_state)
    if unused != expected_unused or missing:
        raise ValueError(f"Checkpoint state mismatch: unused={sorted(unused)}, missing={sorted(missing)}")
    for name, value in loaded_state.items():
        if (value.dtype != saved_state[name].dtype
                or value.shape != saved_state[name].shape
                or not torch.equal(value, saved_state[name])):
            raise ValueError(f"Checkpoint tensor differs after load: {name}")
    print(f"Audited {len(loaded_state)} native DeFoG state tensors; only four sampling-metric buffers omitted", flush=True)
    del saved_state, loaded_state
    model.eval().requires_grad_(False).to(device)
    if model.cfg.dataset.name != "guacamol" or model.cfg.model.transition != "marginal":
        raise ValueError("Checkpoint is not the expected GuacaMol marginal model")
    if (model.output_dims["X"] != 12 or model.output_dims["E"] != 5
            or len(model.limit_dist.X) != 12 or len(model.limit_dist.E) != 5):
        raise ValueError(f"Unexpected DeFoG output/noise vocabulary: {model.output_dims}")
    return model


def _first_neutral_13_train_graph(dataset: GraphDataset):
    neutral = dataset.spec.formal_charges.index(0)
    exclusions = {"not_size_13": 0, "charged": 0, "other_incompatible": 0}
    for index in range(len(dataset)):
        sample = dataset[index]
        active = sample["atoms"] != 0
        if int(active.sum()) != 13:
            exclusions["not_size_13"] += 1
            continue
        if bool((sample["charges"][active] != neutral).any()):
            exclusions["charged"] += 1
            continue
        graph = collate_graphs([sample])
        try:
            mapped = to_guacamol_hard(graph, dataset.spec)
        except ValueError:
            exclusions["other_incompatible"] += 1
            continue
        return index, dataset.smiles[index], mapped, exclusions
    raise ValueError("No charge-neutral, 13-atom GuacaMol-compatible training control")


def _noisy_data(x: torch.Tensor, e: torch.Tensor, node_mask: torch.Tensor, time: float):
    return {"X_t": x, "E_t": e, "node_mask": node_mask,
            "y_t": x.new_zeros((x.shape[0], 0)),
            "t": x.new_full((x.shape[0], 1), time)}


def _loss(model, x, e, node_mask, time, target_x, target_e, *, frozen_extra=None):
    # Reject accidental off-simplex finite-difference probes. DeFoG encodes no
    # categorical value on the edge diagonal, which must remain all zero.
    diagonal = torch.arange(e.shape[1], device=e.device)
    off_diagonal = ~torch.eye(e.shape[1], device=e.device, dtype=torch.bool)
    off_diagonal_edges = e[:, off_diagonal, :]
    if ((x < -1e-7).any() or (e < -1e-7).any()
            or not torch.allclose(x.sum(-1), torch.ones_like(x[..., 0]), atol=1e-5)
            or not torch.allclose(off_diagonal_edges.sum(-1),
                                  torch.ones_like(off_diagonal_edges[..., 0]), atol=1e-5)
            or bool(e[:, diagonal, diagonal, :].any())
            or not torch.allclose(e, e.transpose(1, 2), atol=1e-6)):
        raise ValueError("DeFoG input or finite-difference probe is off simplex")
    noisy = _noisy_data(x, e, node_mask, time)
    extra = model.compute_extra_data(noisy) if frozen_extra is None else frozen_extra
    pred = model.forward(noisy, extra, node_mask)
    n = x.shape[1]
    upper = torch.triu_indices(n, n, offset=1, device=x.device)
    atom_loss = F.cross_entropy(pred.X[0], target_x.argmax(-1)[0])
    bond_loss = F.cross_entropy(pred.E[0, upper[0], upper[1]],
                                target_e[0, upper[0], upper[1]].argmax(-1))
    total = atom_loss + bond_loss
    return total, atom_loss, bond_loss


def _directional_checks(model, x_base, e_base, node_mask, time, target_x, target_e,
                        *, frozen_extra=None):
    x = x_base.detach().clone().requires_grad_(True)
    e = e_base.detach().clone().requires_grad_(True)
    loss, atom_loss, bond_loss = _loss(model, x, e, node_mask, time, target_x, target_e,
                                       frozen_extra=frozen_extra)
    grad_x, grad_e = torch.autograd.grad(loss, (x, e))
    if not all(torch.isfinite(t).all() for t in (loss, grad_x, grad_e)):
        raise ValueError("Non-finite native DeFoG derivative")
    # Perturb within both categorical simplices and preserve edge symmetry.
    carbon = (target_x[0, :, 0] == 1).nonzero(as_tuple=False).flatten()
    if not carbon.numel():
        raise ValueError("Chosen control lacks carbon")
    node = int(carbon[0])
    upper = torch.triu_indices(e.shape[1], e.shape[2], offset=1, device=e.device)
    single = (target_e[0, upper[0], upper[1], 1] == 1).nonzero(as_tuple=False).flatten()
    if not single.numel():
        raise ValueError("Chosen control lacks a single bond")
    left, right = (int(upper[0, single[0]]), int(upper[1, single[0]]))
    dx = torch.zeros_like(x)
    dx[0, node, 0], dx[0, node, 1] = -1, 1  # carbon to nitrogen
    de = torch.zeros_like(e)
    de[0, left, right, 1] = de[0, right, left, 1] = -1
    de[0, left, right, 0] = de[0, right, left, 0] = 1  # single bond to NONE
    analytic_x = float((grad_x * dx).sum().detach().cpu())
    analytic_e = float((grad_e * de).sum().detach().cpu())
    checks = {}
    # The short steps are the predeclared gate. Longer steps diagnose whether
    # float32 cancellation or curvature explains a failed short-step check.
    gate_scales = (0.01, 0.005)
    diagnostic_scales = (0.02, 0.04)
    for epsilon in (*gate_scales, *diagnostic_scales):
        for name, direction_x, direction_e, analytic in (
            ("node", dx, torch.zeros_like(e), analytic_x),
            ("edge", torch.zeros_like(x), de, analytic_e),
        ):
            # The rare nitrogen probability can be below 0.04 at t=0.8;
            # larger probes are only valid for this selected bonded edge.
            if name == "node" and epsilon in diagnostic_scales:
                continue
            with torch.no_grad():
                plus = _loss(model, x_base + epsilon * direction_x,
                             e_base + epsilon * direction_e, node_mask, time,
                             target_x, target_e, frozen_extra=frozen_extra)[0]
                minus = _loss(model, x_base - epsilon * direction_x,
                              e_base - epsilon * direction_e, node_mask, time,
                              target_x, target_e, frozen_extra=frozen_extra)[0]
            numeric = float(((plus - minus) / (2 * epsilon)).cpu())
            error = abs(analytic - numeric)
            passes = (math.isfinite(numeric) and abs(analytic) >= 1e-3
                      and abs(numeric) >= 1e-3 and analytic * numeric > 0
                      and error <= 5e-6 + 0.02 * max(abs(analytic), abs(numeric)))
            checks[f"{name}_eps_{epsilon}"] = {
                "analytic": analytic, "central_difference": numeric,
                "absolute_error": error,
                "relative_error": error / max(abs(analytic), abs(numeric), 1e-6),
                "passed": passes,
            }
    gradient_gate_passed = (
        float(grad_x.norm().detach().cpu()) > 1e-8
        and float(grad_e.norm().detach().cpu()) > 1e-8
        and all(checks[f"{name}_eps_{epsilon}"]["passed"]
                for name in ("node", "edge") for epsilon in gate_scales)
    )
    return {"loss": float(loss.detach().cpu()),
            "atom_ce": float(atom_loss.detach().cpu()),
            "bond_ce": float(bond_loss.detach().cpu()),
            "node_gradient_norm": float(grad_x.norm().detach().cpu()),
            "edge_gradient_norm": float(grad_e.norm().detach().cpu()),
            "gradient_gate_passed": gradient_gate_passed,
            "minimum_gradient_norm": 1e-8,
            "minimum_directional_derivative_magnitude": 1e-3,
            "finite_difference_tolerance": "matching sign and absolute_error <= 5e-6 + 0.02 * max(abs(analytic), abs(numeric)); both scales per direction",
            "gate_scales": gate_scales,
            "edge_only_diagnostic_scales": diagnostic_scales,
            "directions": {"node": node, "edge": [left, right]},
            "finite_differences": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defog-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The native DeFoG gradient smoke requires an HPC GPU allocation")
    config = {"external_source_commit": DEFOG_COMMIT,
              "external_checkpoint_sha256": GUACAMOL_SHA256,
              "data": str(args.data), "time": 0.8,
              "objective": "fixed-reference atom+bond denoising cross entropy; derivative check only",
              "categorical_sampler_in_gradient_path": False,
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    with ExperimentRun(args.run_root, "stage3-native-defog-interface-smoke", config,
                       Path.cwd()) as run:
        _verify_assets(args.defog_source, args.checkpoint)
        dataset = GraphDataset(args.data, "train")
        run.annotate(data_provenance={
            "dataset_sha256": dataset.manifest["dataset_sha256"],
            "split_sha256": dataset.manifest["split_sha256"],
            "schema_fingerprint": dataset.manifest["spec_fingerprint"],
            "split": "train"})
        index, smiles, hard, exclusions = _first_neutral_13_train_graph(dataset)
        device = torch.device("cuda")
        model = _load_native_model(args.defog_source, args.checkpoint, device)
        x_hard, e_hard, node_mask = (hard.x.to(device), hard.e.to(device),
                                    hard.node_mask.to(device))
        time = 0.8
        prior_x = model.limit_dist.X.to(device)
        prior_e = model.limit_dist.E.to(device)
        x_soft = time * x_hard + (1 - time) * prior_x[None, None, :]
        e_soft = time * e_hard + (1 - time) * prior_e[None, None, None, :]
        diagonal = torch.arange(e_soft.shape[1], device=device)
        e_soft[:, diagonal, diagonal, :] = 0
        with torch.no_grad():
            hard_extra = model.compute_extra_data(_noisy_data(x_hard, e_hard, node_mask, time))
            hard_loss = _loss(model, x_hard, e_hard, node_mask, time,
                              x_hard, e_hard, frozen_extra=hard_extra)[0]
        result = {"control": {"train_index": index, "smiles": smiles,
                              "heavy_atoms": 13, "formal_charges": "all zero",
                              "selection_exclusions_before_match": exclusions},
                  "hard_input_loss": float(hard_loss.cpu()),
                  "soft_input_policy": "expected q_t(X_t|x_1) at t=0.8; off-manifold relative to categorical training inputs",
                  "auxiliary_feature_policies": {}}
        # Native extras may contain argmax and cycle assertions. Test their full
        # forward first, recording failure, then also test the explicitly partial
        # derivative with extras frozen at a realizable hard graph.
        try:
            result["auxiliary_feature_policies"]["native_soft"] = {
                "status": "completed", "derivative": _directional_checks(
                    model, x_soft, e_soft, node_mask, time, x_hard, e_hard)}
        except (AssertionError, RuntimeError, ValueError) as error:
            result["auxiliary_feature_policies"]["native_soft"] = {
                "status": "failed", "error_type": type(error).__name__,
                "message": str(error)[:500],
                "traceback": traceback.format_exc(limit=8)[-2000:]}
        result["auxiliary_feature_policies"]["hard_reference_frozen"] = {
            "status": "completed", "derivative": _directional_checks(
                model, x_soft, e_soft, node_mask, time, x_hard, e_hard,
                frozen_extra=hard_extra)}
        write_json(run.path / "summary.json", result)
        run.annotate(summary=result)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
