"""Prove the saved native Boltz frame-layout mismatch in a DTU allocation.

This is a CPU tensor-contract diagnostic on authenticated prepared complexes.
It loads no learned weights, draws no molecules, and invokes no diffusion or
chemistry repair. The native single-ensemble frame layout is compared with its
legacy view using the pinned native frame helper. A passed report establishes
this specific stored-data contract correction, not a successful binding run,
usable optimization gradients, or chemically valid DeFoG outputs.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import inspect
import io
import os
from pathlib import Path
import sys
import traceback

import torch

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.contracts import PreparedComplex
from latentmol.boltz.dbmol_model import validate_frame_context
from latentmol.boltz.prepare import installed_boltz_provenance
from latentmol.boltz.provenance import context_digest
from latentmol.experiments import file_sha256, source_manifest, utc_now, write_json
from latentmol.multimodel.study import verified_file


# These are the byte identities of the pinned native sources retrieved while
# diagnosing source076. Package metadata alone does not certify edited files.
NATIVE_SOURCE_SHA256 = {
    "confidence_utils.py": "50cc0fe50e6088127c04ae715875e612c0f8a690ee8114b86a5ad41aebc63db7",
    "featurizerv2.py": "af1f9e6ec0c3d7289eb1f7503c3c1c3ee0d9f03436778dd4a7d412b8f7bd9f94",
}
CLAIM_LIMIT = (
    "Authenticated native stored-data frame-layout proof only; no learned "
    "weights, Boltz score/gradient validation, DeFoG denoising, optimization "
    "efficacy or chemical-validity claim."
)


def tensor_metadata(value: torch.Tensor) -> dict:
    """Describe actual saved tensors without writing another tensor snapshot."""
    result = {"shape": list(value.shape), "dtype": str(value.dtype),
              "device": str(value.device), "requires_grad": value.requires_grad}
    if value.numel():
        result.update(minimum=value.min().item(), maximum=value.max().item())
        if value.is_floating_point():
            result["all_finite"] = bool(value.isfinite().all())
    return result


def frame_comparison(context: PreparedComplex, compute_frame_pred) -> dict:
    """Use a view in a separate dictionary; preserve all original saved tensors.

    compute_frame_pred clones its frame input and returns reconstructed frames
    plus nondegeneracy masks. Exact equality across both supported layouts is
    stronger than merely accepting their dimensions in our validator. Capture
    its stdout because the pinned helper catches a ligand-frame assignment
    exception internally; such a fallback must not masquerade as success.
    """
    features = context.features
    frames = features["frames_idx"]
    pad = features["token_pad_mask"]
    coordinates = context.coordinates
    old_coordinate_shape_supported = coordinates.shape in (
        (1, features["atom_pad_mask"].shape[1], 3),
        (1, 1, features["atom_pad_mask"].shape[1], 3),
    )
    old_frame_shape_supported = frames.shape == (*pad.shape, 3)
    report = {
        "old_frame_shape_supported": old_frame_shape_supported,
        "old_coordinate_shape_supported": old_coordinate_shape_supported,
        "old_shape_clause_would_fail": not (
            old_frame_shape_supported and old_coordinate_shape_supported),
        "old_expected_frame_shape": [*pad.shape, 3],
        "saved_single_ensemble_frame_shape": [pad.shape[0], 1, pad.shape[1], 3],
    }
    # A legacy snapshot is accepted by production, but would not reproduce the
    # particular native four-dimensional mismatch that this diagnostic probes.
    if frames.shape != (pad.shape[0], 1, pad.shape[1], 3):
        raise ValueError("Selected native snapshot lacks its single ensemble frame axis")
    validate_frame_context(features, coordinates)
    report["fixed_validator_passed"] = True
    legacy = {**features, "frames_idx": frames[:, 0]}
    validate_frame_context(legacy, coordinates)
    report["legacy_validator_passed"] = True
    flat_coordinates = coordinates.reshape(1, -1, 3)
    native_stdout, legacy_stdout = io.StringIO(), io.StringIO()
    with torch.no_grad(), redirect_stdout(native_stdout):
        native_frames, native_mask = compute_frame_pred(
            flat_coordinates, frames, features, 1, inference=True)
    with torch.no_grad(), redirect_stdout(legacy_stdout):
        legacy_frames, legacy_mask = compute_frame_pred(
            flat_coordinates, legacy["frames_idx"], legacy, 1, inference=True)
    report.update(
        native_helper_stdout=native_stdout.getvalue(),
        legacy_helper_stdout=legacy_stdout.getvalue(),
        output_frames=tensor_metadata(native_frames),
        output_mask=tensor_metadata(native_mask),
        frame_outputs_equal=torch.equal(native_frames, legacy_frames),
        frame_masks_equal=torch.equal(native_mask, legacy_mask),
        nondegenerate_token_frames=int(native_mask.sum()),
    )
    token_mask = pad.bool()
    different_chains = features["asym_id"][:, :, None] != features["asym_id"][:, None, :]
    pairs = (native_mask.reshape_as(token_mask).bool()[:, :, None]
             & token_mask[:, :, None] & token_mask[:, None, :] & different_chains)
    report["native_nonempty_interface_frame_pairs"] = int(pairs.sum())
    report["passed"] = bool(
        report["old_shape_clause_would_fail"] and report["frame_outputs_equal"]
        and report["frame_masks_equal"] and report["native_nonempty_interface_frame_pairs"]
        and not report["native_helper_stdout"] and not report["legacy_helper_stdout"])
    return report


def inspect_context(count: int, asset: dict, compute_frame_pred) -> dict:
    """Authenticate before loading and detect both file and in-memory mutation."""
    path = verified_file(asset["path"], asset["sha256"])
    context = PreparedComplex.load(path, device="cpu")
    before = context_digest(context)
    report = {"count": count, "path": str(path), "sha256": asset["sha256"],
              "context_sha256_before": before, "saved_metadata": context.metadata,
              "prepared_complex_validation_passed": True,
              "occupied_slots": int(context.active.sum()),
              "coordinates": tensor_metadata(context.coordinates),
              "features": {key: tensor_metadata(context.features[key]) for key in (
                  "frames_idx", "token_pad_mask", "atom_pad_mask", "asym_id",
                  "atom_to_token", "mol_type")}}
    if int(context.active.sum()) != count:
        raise ValueError("Prepared count differs from its authenticated context-bank key")
    rows = context.features["frames_idx"][:, 0]
    active_rows = rows[context.features["token_pad_mask"].bool()]
    report["active_frame_index_bounds"] = tensor_metadata(active_rows)
    report["comparison"] = frame_comparison(context, compute_frame_pred)
    after = context_digest(context)
    report["context_sha256_after"] = after
    report["original_snapshot_unchanged"] = before == after and file_sha256(path) == asset["sha256"]
    report["passed"] = report["comparison"]["passed"] and report["original_snapshot_unchanged"]
    return report


def native_helper_identity() -> tuple[object, dict]:
    """Verify the real installed helpers rather than copied diagnostic fixtures."""
    provenance = installed_boltz_provenance()
    from boltz.data.feature import featurizerv2
    from boltz.model.layers.confidence_utils import compute_frame_pred
    paths = {"confidence_utils.py": Path(inspect.getfile(compute_frame_pred)),
             "featurizerv2.py": Path(featurizerv2.__file__)}
    sources = {}
    for name, path in paths.items():
        verified_file(path, NATIVE_SOURCE_SHA256[name])
        sources[name] = {"path": str(path), "sha256": NATIVE_SOURCE_SHA256[name]}
    return compute_frame_pred, {"installation": provenance, "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New diagnostic directory; existing paths are rejected")
    parser.add_argument("--counts", nargs="+", type=int, default=[22],
                        help="Declared saved context counts; default is failed native count 22")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        parser.error("Require a DTU allocation and immutable source archive identity")
    if (not args.counts or len(set(args.counts)) != len(args.counts)
            or len(args.counts) > 8 or any(count < 1 for count in args.counts)):
        parser.error("Require one to eight distinct positive saved context counts")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at": utc_now(), "passed": False,
              "claim_limit": CLAIM_LIMIT, "job_id": os.environ["LSB_JOBID"],
              "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"],
              "source": source_manifest(Path(__file__).resolve().parents[1]),
              "command": sys.argv, "python": sys.version,
              "config": {"path": str(args.config.resolve()), "sha256": args.config_sha256},
              "declared_counts": args.counts, "contexts": []}
    write_json(args.output / "report.json", report)
    try:
        verified_file(args.config, args.config_sha256)
        config = ActiveSetConfig.load(args.config)
        report["context_manifest"] = config.context_manifest
        helper, report["native"] = native_helper_identity()
        for count in args.counts:
            if count not in config.contexts:
                raise ValueError(f"No authenticated native context for count {count}")
            report["contexts"].append(inspect_context(count, config.contexts[count], helper))
            write_json(args.output / "report.json", report)
        verified_file(args.config, args.config_sha256)
        for asset in report["native"]["sources"].values():
            verified_file(asset["path"], asset["sha256"])
        report["passed"] = all(row["passed"] for row in report["contexts"])
        if not report["passed"]:
            raise ValueError("Native stored-data layout proof did not satisfy all gates")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        (args.output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["finished_at"] = utc_now()
        write_json(args.output / "report.json", report)
    print(args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
