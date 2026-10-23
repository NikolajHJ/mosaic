"""Allocated CPU profile and exact teacher parity on the authenticated parent.

This preflight compares identical frozen CPU decoder outputs, not GPU training
performance. Native full-update parity, readiness and runtime gates remain
mandatory. No loss coefficient, candidate sampling or search budget is changed.
"""
import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Repair benchmark requires an allocated DTU CPU job")
    import torch
    from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
    from latentmol.chemistry_continuation import authenticated, digest, read_plan
    from latentmol.data import GraphDataset, collate_graphs, validate_large_cache
    from latentmol.evaluate import load_model
    from latentmol.experiments import write_json
    from latentmol.probability_repair import ProbabilityRepairConfig
    from latentmol.representation import DecoderOutput

    plan = read_plan(args.plan, args.plan_sha256)
    identity = {"plan_sha256": args.plan_sha256, "archive_sha256": plan["archive_sha256"],
                "manifest_sha256": plan["manifest_sha256"], "parent_sha256": plan["parent_checkpoint_sha256"]}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Retain existing profile; require a new sealed attempt")
    report = {"status": "failed", "identity": identity, "job_id": os.environ["LSB_JOBID"],
              "scope": "allocated CPU frozen-parent teacher parity/profile, not native training readiness"}
    try:
        if torch.get_num_threads() != 4 or int(os.environ.get("LSB_DJOB_NUMPROC", "0")) < 16:
            raise ValueError("Require sixteen allocated slots and four parent Torch threads")
        authenticated(plan["parent_checkpoint"], plan["parent_checkpoint_sha256"])
        cache = validate_large_cache(Path(plan["data"]))
        data = GraphDataset(plan["data"], "train", validated_cache=cache)
        if (data.manifest["dataset_sha256"] != plan["dataset_sha256"]
                or data.manifest["split_sha256"] != plan["split_sha256"]):
            raise ValueError("Profile data identity differs")
        model, _ = load_model(Path(plan["parent_checkpoint"]), "cpu", data.spec)
        model.eval().requires_grad_(False)
        generator = torch.Generator().manual_seed(plan["protocol"]["prior_seed"])
        with torch.no_grad():
            hard = collate_graphs([data[index] for index in range(4)])
            mu, logvar = model.encode(hard.to_relaxed(model.spec))
            real_z = mu + torch.randn(mu.shape, generator=generator) * (.5 * logvar).exp()
            prior_z = torch.randn(32, model.config.latent, generator=generator)
            real, prior = model.decode(real_z), model.decode(prior_z)
            decoded = DecoderOutput(*(torch.cat([getattr(real, field), getattr(prior, field)])
                                      for field in ("atoms", "charges", "bonds")))
        ids = list(range(36))
        policy = ProbabilityRepairConfig(**plan["protocol"]["search"])
        def semantic(receipt):
            value = {k: v for k, v in receipt.items() if k not in ("elapsed_seconds", "cache_hits", "search_checks")}
            value["attempts"] = [{k: v for k, v in row.items() if k != "cache_hit"} for row in value["attempts"]]
            return value
        reference_semantic, rows = None, []
        for backend, workers in (("reference", 0), ("optimized", 0), ("optimized", 4), ("optimized", 12)):
            profile = cProfile.Profile()
            with AtomPreservingRepairTeacher(model.spec, policy, backend=backend, workers=workers) as teacher:
                tick = time.perf_counter()
                profile.enable()
                initial = teacher.build_targets(decoded, candidate_ids=ids, update=0)
                profile.disable()
                initial_seconds = time.perf_counter() - tick
                # Empty target cache, same inputs and already initialized CPU
                # processes: distinguish startup from steady-state cost.
                teacher.clear_cache()
                tick = time.perf_counter()
                warmed = teacher.build_targets(decoded, candidate_ids=ids, update=0)
                warm_seconds = time.perf_counter() - tick
            current = semantic(initial.receipt)
            if reference_semantic is None:
                reference_semantic = current
            if current != reference_semantic or semantic(warmed.receipt) != reference_semantic:
                raise ValueError("CPU optimized execution changed exact ordered teacher semantics")
            if initial.receipt["search_checks"] != warmed.receipt["search_checks"]:
                raise ValueError("Warmed execution changed the declared search budget/work")
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(35)
            rows.append({"backend": backend, "workers": workers, "profiled_initial_seconds": initial_seconds,
                         "unprofiled_warmed_seconds": warm_seconds, "semantic_sha256": digest(current),
                         "search_checks": warmed.receipt["search_checks"],
                         "accepted_targets": warmed.receipt["accepted_targets"],
                         "status_counts": warmed.receipt["status_counts"], "profile_cumulative_top35": stream.getvalue()})
        if rows[0]["accepted_targets"] < 1 or rows[0]["search_checks"] < 1:
            raise ValueError("Profile cohort did not exercise a nonzero repair witness/search")
        report.update(status="passed", records=rows, real_points=4, fresh_prior_points=32,
                      note="Diagnostic cohort only; actual training ratio remains exactly1:3",
                      fast_serial_speedup=rows[0]["unprofiled_warmed_seconds"] / rows[1]["unprofiled_warmed_seconds"],
                      parallel12_speedup=rows[0]["unprofiled_warmed_seconds"] / rows[3]["unprofiled_warmed_seconds"])
        write_json(output, report)
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, sort_keys=True))
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}")
        write_json(output, report)
        raise


if __name__ == "__main__":
    main()
