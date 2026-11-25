"""Prepare a BRD4 founder panel and freeze its historical hard-molecule exclusions."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def extend_panel(base, roots, history_only, pocket_sha256):
    """Merge compatible finished runs; history-only inputs never change starting latents."""
    if base.get("version") != 1 or base.get("pocket_sha256", pocket_sha256) != pocket_sha256:
        raise ValueError("Incompatible source founder panel or pocket")
    if len({p.resolve() for p in roots + history_only}) != len(roots + history_only):
        raise ValueError("Duplicate source run")
    ids = set(base["historical_valid_smiles"])
    points = list(base["points"])
    ids.update(p["smiles"] for p in points)
    sources = list(base.get("history_sources", []))
    known_jobs = {str(source["job_id"]) for source in sources}
    for root in roots + history_only:
        run = read(root / "run.json")
        result = read(root / "result.json")
        allowed = {"synthesis_waves"}
        if root in history_only:
            allowed.update(("synthesis_pilot", "matched_search", "matched_width"))
        if (result.get("status") != "completed" or run.get("runner") not in allowed
                or any(run.get(k) != base[k] for k in ("decoder_sha256", "checkpoint_sha256", "graph_spec"))):
            raise ValueError(f"Expected a completed compatible search at {root}")
        job = str(run["job_id"])
        if job in known_jobs:
            raise ValueError(f"History already includes job {job}")
        known_jobs.add(job)
        visited = {p["smiles"] for p in run.get("parents", [])}
        branch_files = set()
        observations_total = 0
        waves = [{"cycle": 1}] if run["runner"] == "synthesis_pilot" else result["cycles"]
        for wave in waves:
            cycle = root / f"cycle-{wave['cycle']:02d}"
            batch = read(cycle / "batch.json")
            if batch["native_identity"].get("pocket_sha256") != pocket_sha256:
                raise ValueError(f"Different protein pocket at {cycle}")
            for branch in batch["branches"]:
                path = (cycle / branch["name"] / "metrics.jsonl").resolve(strict=True)
                if path in branch_files or path.parent.parent != cycle.resolve():
                    raise ValueError(f"Duplicate or misplaced branch record in {cycle}")
                branch_files.add(path)
                observations = 0
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        observation = json.loads(line)
                        if observation["step"] != observations:
                            raise ValueError(f"Missing or repeated observation in {path}")
                        observations += 1
                        if observation["raw_valid"]:
                            if not observation["smiles"]:
                                raise ValueError(f"Missing valid molecule in {path}")
                            visited.add(observation["smiles"])
                if observations != branch["summary"]["updates"] + 1:
                    raise ValueError(f"Incomplete valid-identity history in {path}")
                observations_total += observations
        if len(branch_files) != result["planned"]["branches"]:
            raise ValueError(f"Missing branch records in {root}")
        if root in roots:
            import torch

            final_wave = result["cycles"][-1]["cycle"]
            state = torch.load(root / f"cycle-{final_wave:02d}" / "archive-state.pt",
                               map_location="cpu", weights_only=True)
            if state["wave"] != final_wave or set(state["archives"]) != set(run["variants"]):
                raise ValueError(f"Inconsistent final archive in {root}")
            for arm, archive in state["archives"].items():
                for smiles, p in archive.items():
                    loss = p["fresh_loss"]
                    if smiles != p["smiles"] or not math.isfinite(loss):
                        raise ValueError("Invalid fresh archive identity or score")
                    points.append(dict(smiles=smiles, latent=p["latent"].tolist(),
                        active_slots=p["active"].nonzero().flatten().tolist(), fresh_loss=loss,
                        relaxed_loss=p["relaxed_loss"], branch=p["branch"], step=p["step"],
                        ancestry=list(p["ancestry"]), labels={k: v[0].tolist() for k, v in p["hard"].items()},
                        provenance=dict(run=root.name, job_id=run["job_id"], arm=arm, wave=final_wave)))
        sources.append(dict(run=root.name, job_id=run["job_id"], runner=run["runner"],
                            branches=len(branch_files), observations=observations_total,
                            distinct_valid=len(visited), added_to_history=len(visited - ids),
                            included_parent_archive=root in roots))
        ids.update(visited)
    return {k: base[k] for k in ("version", "checkpoint_sha256", "decoder_sha256", "graph_spec")} | dict(
        pocket_sha256=pocket_sha256, points=points, history_sources=sources, historical_valid_smiles=sorted(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-panel", type=Path, required=True, help="Trusted same-pocket replayable founder panel")
    parser.add_argument("--run", type=Path, action="append", default=[],
                        help="Completed synthesis run with replayable archive and per-step metrics")
    parser.add_argument("--history-only-run", type=Path, action="append", default=[],
                        help="Add every valid visited molecule to exclusions while preserving the founder pool")
    parser.add_argument("--native-config", type=Path, default=Path(__file__).resolve().parents[1] / "native-dtu.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = extend_panel(read(args.base_panel), args.run, args.history_only_run,
                         read(args.native_config)["pocket_sha256"])
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(panel, stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          historical_identities=len(panel["historical_valid_smiles"]), points=len(panel["points"]))))


if __name__ == "__main__":
    main()
