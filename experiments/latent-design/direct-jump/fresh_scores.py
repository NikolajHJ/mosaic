"""Fresh graph-level BRD4 scoring on an allocated GPU in the model's environment."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


CONFIG = Path(__file__).resolve().parents[1] / "protenix-vae" / "flow-followup-fresh-dtu.json"
BOLTZ_RELEASE = Path("/work3/s234818/mosaic/runs/brd4-baselines-30082565-20261118a")
BOLTZ_CACHE = Path("/work3/s234818/latentmol/cache/boltz2")


def write(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def protenix_scores(smiles, work, config):
    from latentmol.design.temperature_fresh import main as score_panel

    # The shared scorer uses the arm/start fields to report role-based deltas.
    # This adapter requests one observation per input, including the founder.
    candidates = [dict(start=i, arm="fresh", role="initial", smiles=s)
                  for i, s in enumerate(smiles)]
    write(work / "fresh-candidates.json", {"candidates": candidates})
    cfg = dict(config)
    cfg["assets_config"] = str((CONFIG.parent / cfg["assets_config"]).resolve())
    cfg["candidate_limit"] = len(candidates)
    write(work / "config.json", cfg)
    score_panel(["--config", str(work / "config.json"), "--run", str(work)])
    summary = read(work / "fresh" / "summary.json")
    if summary["status"] != "completed" or summary["unique_candidates"] != len(smiles):
        raise RuntimeError("Fresh Protenix candidate identities or completion differed")
    return {r["smiles"]: {"loss": r["loss"], "status": r["status"], "reason": r.get("reason")}
            for r in summary["candidates"]}


def boltz_rows(rows, log):
    # The native CLI skips individual inputs that cannot form a 3-D conformer.
    # A missing score from any other cause, including a failed GPU stage, is fatal.
    failed_conformers = set(re.findall(
        r"Failed to process [^\r\n]*?[/\\](mol\d+)\.yaml\. Skipping\. Error: Failed to compute 3D conformer for", log
    ))
    result = {}
    for row in rows:
        if "boltz_affinity" in row:
            result[row["smiles"]] = {"loss": row["boltz_affinity"], "status": "scored"}
        elif row["name"] in failed_conformers:
            result[row["smiles"]] = {"loss": None, "status": "embedding_failed",
                                     "reason": "Native Boltz failed to compute a 3-D conformer"}
        else:
            raise RuntimeError(f"Unexplained missing Boltz result for {row['name']}: {row.get('error')}")
    return result


def boltz_scores(smiles, work, config):
    release = Path(os.environ.get("DIRECT_JUMP_BOLTZ_RELEASE", str(BOLTZ_RELEASE)))
    batch = release / "boltz-rescore" / "batch.py"
    library = release / "scoring" / "lib"
    worker = library / "mosaic" / "small_molecule" / "evaluation" / "native_worker.py"
    pocket = Path(config["pocket"]["path"])
    if hashlib.sha256(pocket.read_bytes()).hexdigest() != config["pocket"]["sha256"]:
        raise ValueError("Fresh BRD4 pocket definition changed")
    records = work / "candidates.smi"
    with records.open("x", encoding="utf-8") as out:
        for i, value in enumerate(smiles):
            out.write(f"{value}\tcandidate-{i:04d}\n")
    run = work / "cofold"
    env = dict(os.environ, PYTHONPATH=str(library))
    subprocess.run([sys.executable, str(batch), "prepare", "--input", str(records),
                    "--pocket", str(pocket), "--cache", str(BOLTZ_CACHE), "--device", "gpu",
                    "--seed", "42", "--work", str(run)], env=env, check=True)
    log_path = work / "native.log"
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(worker), str(run / "request.json")],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(batch), "collect", "--work", str(run)], env=env, check=True)
    rows = read(run / "scores.json")
    if [row["smiles"] for row in rows] != smiles:
        raise ValueError("Boltz input and output identities differ")
    return boltz_rows(rows, log_path.read_text(encoding="utf-8", errors="replace"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["boltz2", "protenix"], required=True)
    for field in ("input", "output", "work"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Fresh native scoring requires an allocated GPU job")
    if args.output.exists():
        raise FileExistsError(args.output)
    smiles = read(args.input)
    if not isinstance(smiles, list) or any(not isinstance(s, str) or not s or
            any(c.isspace() for c in s) for s in smiles):
        raise ValueError("Expected a list of canonical SMILES")
    # Identity-based order keeps the native batch independent of arm iteration order.
    smiles = sorted(set(smiles), key=lambda s: hashlib.sha256(s.encode()).digest())
    work = args.work.resolve()
    work.mkdir(exist_ok=False, parents=True)
    config = read(CONFIG)
    scorer = protenix_scores if args.model == "protenix" else boltz_scores
    details = scorer(smiles, work, config) if smiles else {}
    if set(details) != set(smiles):
        raise ValueError("Missing or unexpected fresh hard-graph result")
    scores = {s: row["loss"] for s, row in details.items()}
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v))
           for v in scores.values()):
        raise ValueError("Nonfinite/nonnumeric native hard-graph score")
    write(work / "scores-and-status.json", details)
    write(args.output, scores)


if __name__ == "__main__":
    main()
