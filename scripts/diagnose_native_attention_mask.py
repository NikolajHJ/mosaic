"""HPC-only arithmetic probes of pinned native Boltz triangle attention.

This deliberately tiny diagnostic tests a possible float32 failure mechanism:
adding finite logits to a wholly masked ``-1e9`` row can round those logits
away, while autograd still differentiates the additions. It does not load a
Boltz checkpoint, replace model mathematics, approve a failed derivative gate,
or establish that this mechanism caused any molecular optimization failure.

Optional authenticated prepared contexts report exposure to wholly masked
token rows; counts alone cannot measure their contribution to the final score.
Run from an immutable release in a DTU CPU or GPU allocation. No local numerical
test is implied by this script's presence.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch

from latentmol.boltz.contracts import PreparedComplex
from latentmol.boltz.prepare import installed_boltz_provenance
from latentmol.experiments import ExperimentRun, file_sha256, write_json


EPSILONS = (0.0001, 0.001, 0.01, 0.03, 0.1)
CLAIM_LIMIT = (
    "Toy native arithmetic and context mask exposure only; no learned weights, "
    "molecular score, production patch, gradient-gate approval or confirmed cause."
)


def _inputs(point: torch.Tensor, mask_kind: str, mask_scale: float,
            policy: str):
    """Declare one query/head, three keys, and scalar-valued key/value channels.

    Two valid keys in the mixed control give a nonzero derivative; using only
    one valid key would make that control trivially constant. The stable shift
    subtracts a row-constant bias before native attention. Softmax is invariant
    to this shift in real arithmetic, but the rounded native outputs can change.
    That comparison is diagnostic, never a proposed native-parity-preserving fix.
    """
    query = point.reshape(1, 1, 1, 1)
    key = point.new_tensor([-1.0, 1.0, 2.0]).reshape(1, 1, 3, 1)
    value = point.new_tensor([-1.0, 1.0, 0.5]).reshape(1, 1, 3, 1)
    padding = {
        "fully_masked": [1.0, 1.0, 1.0],
        "two_valid_one_masked": [0.0, 0.0, 1.0],
        "unmasked": [0.0, 0.0, 0.0],
    }[mask_kind]
    mask_bias = -mask_scale * point.new_tensor(padding).reshape(1, 1, 1, 3)
    if policy == "stable_row_shift":
        mask_bias = mask_bias - mask_bias.amax(dim=-1, keepdim=True)
    elif policy != "native_bias_order":
        raise ValueError("Unknown arithmetic comparison policy")
    triangle_bias = torch.zeros_like(mask_bias)
    return query, key, value, [mask_bias, triangle_bias]


def _measure(attention, *, dtype, device, mask_kind, mask_scale, policy,
             grad_enabled, backward=False, displacement=0.0):
    point = torch.tensor(0.125 + displacement, dtype=dtype, device=device,
                         requires_grad=grad_enabled)
    with torch.set_grad_enabled(grad_enabled):
        output = attention(*_inputs(point, mask_kind, mask_scale, policy)).reshape(())
        derivative = float(torch.autograd.grad(output, point)[0]) if backward else None
    return {"value": float(output.detach()), "derivative": derivative,
            "point": float(point.detach()), "grad_enabled": grad_enabled}


def probe(attention, *, dtype, device, mask_kind, mask_scale, policy):
    settings = dict(dtype=dtype, device=device, mask_kind=mask_kind,
                    mask_scale=mask_scale, policy=policy)
    analytic = _measure(attention, **settings, grad_enabled=True, backward=True)
    plain = _measure(attention, **settings, grad_enabled=False)
    rows = []
    for epsilon in EPSILONS:
        for grad_enabled in (False, True):
            plus = _measure(attention, **settings, grad_enabled=grad_enabled,
                            displacement=epsilon)
            minus = _measure(attention, **settings, grad_enabled=grad_enabled,
                             displacement=-epsilon)
            # Measure the actual representable scalar displacement, not merely
            # its nominal epsilon. This cannot recover rounded attention logits.
            realized_span = plus["point"] - minus["point"]
            central = (plus["value"] - minus["value"]) / realized_span
            rows.append({"epsilon": epsilon, "grad_enabled": grad_enabled,
                         "plus": plus, "minus": minus,
                         "actual_point_span": realized_span,
                         "central_difference": central,
                         "absolute_vjp_difference": abs(central - analytic["derivative"])})
    return {"dtype": str(dtype), "device": str(device), "mask": mask_kind,
            "mask_scale": mask_scale, "policy": policy, "analytic": analytic,
            "no_grad": plain, "grad_minus_no_grad": analytic["value"] - plain["value"],
            "finite_differences": rows}


def context_exposure(path: Path, expected_sha256: str):
    """Inspect exact native token masks without evaluating any Boltz weights."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Prepared context file differs from declared SHA-256")
    context = PreparedComplex.load(path, device="cpu")
    feats = context.features
    pad = feats["token_pad_mask"].bool()
    receptor = (feats["mol_type"] == 0) & pad
    ligand = feats["affinity_token_mask"].bool() & pad
    cross = ((ligand[:, :, None] & receptor[:, None, :])
             | (receptor[:, :, None] & ligand[:, None, :])
             | (ligand[:, :, None] & ligand[:, None, :]))
    empty = ~cross.any(dim=-1)
    result = {"path": str(path.resolve()), "sha256": actual_sha256,
              "tokens": pad.shape[-1], "real_tokens": int(pad.sum()),
              "padded_tokens": int((~pad).sum()),
              "affinity_wholly_masked_start_rows": int(empty.sum()),
              "affinity_wholly_masked_end_rows": int((~cross.any(dim=-2)).sum()),
              "wholly_masked_real_rows": int((empty & pad).sum()),
              "claim_limit": "Mask exposure is not a measured score/gradient contamination effect."}
    if file_sha256(path) != actual_sha256:
        raise ValueError("Prepared context changed during mask inspection")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prepared", nargs=2, action="append", default=[],
                        metavar=("PATH", "SHA256"))
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        parser.error("Require a DTU allocation and immutable source archive identity")
    if len(args.prepared) > 8:
        parser.error("At most eight explicitly authenticated context inspections")
    provenance = installed_boltz_provenance()
    from boltz.model.layers.triangular_attention.primitives import _attention
    native_path = Path(inspect.getfile(_attention))
    settings = {"native_install": provenance, "native_attention_path": str(native_path),
                "native_attention_sha256": file_sha256(native_path),
                "device": args.device, "epsilons": EPSILONS,
                "baseline_scalar": 0.125, "key": [-1.0, 1.0, 2.0],
                "value": [-1.0, 1.0, 0.5], "triangle_bias": 0.0,
                "mask_scales": [1e6, 1e9], "prepared": args.prepared,
                "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(args.run_root, "native-attention-mask-arithmetic", settings,
                       Path.cwd()) as experiment:
        report = {"claim_limit": CLAIM_LIMIT, "probes": [], "contexts": []}
        for dtype in (torch.float32, torch.float64):
            for mask_scale in (1e6, 1e9):
                for mask_kind in ("fully_masked", "two_valid_one_masked", "unmasked"):
                    for policy in ("native_bias_order", "stable_row_shift"):
                        report["probes"].append(probe(_attention, dtype=dtype,
                            device=args.device, mask_kind=mask_kind,
                            mask_scale=mask_scale, policy=policy))
                        write_json(experiment.path / "attention-mask.json", report)
        for path, sha256 in args.prepared:
            report["contexts"].append(context_exposure(Path(path), sha256))
        if file_sha256(native_path) != settings["native_attention_sha256"]:
            raise ValueError("Native attention source changed during diagnosis")
        write_json(experiment.path / "attention-mask.json", report)
        experiment.annotate(toy_arithmetic_completed=True,
                            prepared_contexts_inspected=len(report["contexts"]),
                            production_model_changed=False,
                            failed_gates_not_reclassified=True)
        print(experiment.path, flush=True)


if __name__ == "__main__":
    main()
