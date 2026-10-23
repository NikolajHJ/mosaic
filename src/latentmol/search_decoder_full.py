"""Full-data, two-seed comparison of matched control and search-guided decoder.

All encoder parameters/buffers stay frozen; no activity model or learned reward
is trained. The shared search engine provides probability-aware detached valid
witnesses, identical pool anchors, native FD/smoke gates and held-out evaluation.
No result is validated until its actual allocated HPC receipts are retained.
"""
import argparse
import json
import os
from pathlib import Path

from latentmol.decoder_finetune import FineTuneConfig
from latentmol.experiments import file_sha256, write_json
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.search_decoder_protocol import read_full_protocol, segment_protocol, validate_full_protocol
from latentmol.search_decoder_study import informative_native_gradient, run_study


def audit_segment(directory, index, summary, expected_protocol=None):
    """Authenticate saved scientific identity and completed endpoint metadata.

    Checkpoint bytes are hashed without loading a model. Native execution has
    separately validated the checkpoint's epoch and parameter/encoder state.
    A stale summary or metadata from a different seed/source cannot mark this
    full study complete. Missing/invalid artifacts are retained as audit errors.
    """
    try:
        selected = json.loads((directory / "protocol.json").read_text())
        base = validate_full_protocol({key: value for key, value in selected.items() if key != "seed"})
        if selected["seed"] != base["seeds"][index] or (expected_protocol is not None and base != expected_protocol):
            raise ValueError("Segment seed or scientific protocol differs")
        gate = json.loads((directory / "native_engineering_gate.json").read_text())
        if gate["status"] != "passed" or not informative_native_gradient(gate["native_search_coverage"], gate["native_gradient_probe"]):
            raise ValueError("Missing informative native GPU gate")
        cases = summary.get("cases", [])
        if summary.get("complete") is not True or [case.get("arm") for case in cases] != base["arms"] or any(case.get("status") != "completed" for case in cases):
            raise ValueError("Both prescribed scientific arms must be completed")
        records = []
        for case in cases:
            fit = case["fit"]
            run_dir, checkpoint = Path(fit["run_dir"]), Path(fit["checkpoint"])
            if not run_dir.resolve().is_relative_to(directory.resolve()) or checkpoint.parent.resolve() != run_dir.resolve():
                raise ValueError("Endpoint belongs to another segment")
            if file_sha256(checkpoint) != fit["sha256"] or checkpoint.name != "epoch-010.pt":
                raise ValueError("Final checkpoint identity differs")
            record = json.loads((run_dir / "run.json").read_text())
            identity = record["config"]
            if (record["status"] != "completed" or record.get("completed_epochs") != 10
                    or identity["seed"] != selected["seed"] or identity["train_count"] != 215000
                    or identity["initializer_sha256"] != base["initializer_sha256"]
                    or identity["dataset_sha256"] != base["dataset_sha256"]
                    or identity["split_sha256"] != base["split_sha256"]
                    or identity["config"] != base["fine_tune"]
                    or identity["extra_guidance"]["name"] != case["arm"]["name"]
                    or identity["extra_guidance"]["repair_weight"] != case["arm"]["repair_weight"]):
                raise ValueError("Completed training metadata differs from full protocol")
            records.append({"source": record["source_archive_sha256"], "encoder": fit["encoder_sha256"],
                            "pool": fit["extra_guidance"]["panel_sha256"], "rng": fit["terminal_rng"],
                            "final_panel": case["prior"]["panel_sha256"]})
        for key in ("source", "encoder", "pool", "rng", "final_panel"):
            if records[0][key] != records[1][key]:
                raise ValueError("Matched arm identity differs: " + key)
        return {"complete": True, "protocol": base, "protocol_file_sha256": file_sha256(directory / "protocol.json"),
                "seed": selected["seed"], "shared": {key: value for key, value in records[0].items() if key != "rng"}}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return {"complete": False, "segment": index, "error": repr(error)}


def aggregate_segments(root, expected_protocol=None):
    """Keep each training seed's paired latent-cluster result separate.

    The4096 panel is shared across models/seeds. It is not8192 independent
    latent points; two trained seeds cannot support population-seed inference.
    Incomplete segments are retained explicitly, including when LSF killed a
    previous allocation before it could write its final summary.
    """
    segments = []
    for index in (0, 1):
        path = root / f"segment-{index}" / "summary.json"
        segments.append(json.loads(path.read_text()) if path.exists() else
                        {"complete": False, "status": "missing_segment_summary", "segment": index})
    audits = [audit_segment(root / f"segment-{index}", index, summary, expected_protocol)
              for index, summary in enumerate(segments)]
    complete = all(row["complete"] for row in audits)
    if complete and (audits[0]["protocol"] != audits[1]["protocol"] or audits[0]["shared"] != audits[1]["shared"]):
        complete = False
        audits.append({"complete": False, "error": "Cross-seed protocol/source/encoder/panels differ"})
    return {"complete": complete, "segments": segments, "integrity_audits": audits,
            "scope": "Two fixed training seeds; per-seed paired latent-cluster CIs; do not pool repeated panel IDs",
            "checkpoint_restart": "Unsupported for search teacher state; partial endpoints not compared"}


def run_full_segment(protocol, root, source_root, segment, *, device="cuda"):
    """One serial allocation per seed; two fresh ten-epoch endpoints."""
    FineTuneConfig(**protocol["fine_tune"])
    ProbabilityRepairConfig(**protocol["repair"])
    selected = segment_protocol(protocol, segment)
    root.mkdir(parents=True, exist_ok=True)
    segment_root = root / f"segment-{segment}"
    if segment_root.exists():
        raise FileExistsError("Full segment already exists; search checkpoint resume is unsupported")
    try:
        run_study(selected, segment_root, source_root, device=device,
            evaluation_reserve_seconds=protocol["evaluation_reserve_seconds"],
            require_full_data=True, require_native_gradient=True,
            scope=f"Full215000-graph ten-epoch control/search pair, training seed{selected['seed']}; independent4096-point generation; no synthesis/binding claim")
    except Exception as error:
        # Keep gate/controller failure separate from scientific-arm outcomes.
        # Never mutate an existing segment on a duplicate invocation.
        if segment_root.is_dir() and not (segment_root / "summary.json").exists() and not isinstance(error, FileExistsError):
            write_json(segment_root / "controller_failure.json", {"error": repr(error), "segment": segment,
                "seed": selected["seed"], "complete": False})
        raise
    finally:
        if segment == 1:
            write_json(root / "summary.json", aggregate_segments(root, protocol))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--segment", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Full search numerical work requires an allocated DTU job")
    run_full_segment(read_full_protocol(args.config, args.sha256), args.output_root, Path.cwd(), args.segment)


if __name__ == "__main__":
    main()
