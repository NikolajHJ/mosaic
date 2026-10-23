"""HPC CPU gate for the exact carbon-chain start, before GPU pose preparation.

Loads only the frozen graph AE/VAE. No native Boltz weights, latent optimization,
or data-panel evaluation occurs. Failed reconstruction is retained and is never
replaced by an easier seed selected after consulting an affinity score.
"""

import argparse
import os
from pathlib import Path

import torch

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, write_json
from latentmol.multimodel.study import verified_file
from latentmol.representation import HardGraphBatch
from .active_set import audit_start


def run_preflight(checkpoint: Path, sha256: str, mode: str, count: int,
                  run_root: Path, receipt: Path) -> Path:
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Preflight requires an HPC allocation and immutable source identity")
    if receipt.exists():
        raise FileExistsError(receipt)
    config = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256,
              "mode": mode, "count": count, "temperature": 1.0, "threshold": 0.5,
              "scope": "one exact carbon-chain encoder start; no native Boltz or optimization"}
    with ExperimentRun(run_root, f"{mode}-active-set-start-preflight", config, Path.cwd()) as run:
        model, training = load_model(verified_file(checkpoint, sha256), "cpu")
        if training["train_config"]["mode"] != mode or not 1 <= count <= model.spec.max_nodes:
            raise ValueError("Checkpoint mode or requested carbon count differs")
        model.eval().requires_grad_(False)
        encoded = encode_smiles("C" * count, model.spec)
        seed = HardGraphBatch(*(torch.from_numpy(encoded[key].astype("int64"))[None]
                                for key in ("atom_types", "charges", "bond_types")))
        with torch.no_grad():
            point, _ = model.encode(seed.to_relaxed(model.spec))
            decoded = model.decode(point)
            selected, report = audit_start(decoded, seed, model.spec, temperature=1.0, threshold=0.5)
            hard = decoded.harden(model.spec)
            chemistry = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], model.spec)
            report.update(checkpoint_sha256=sha256, count=count, mode=mode,
                          graph_spec_sha256=model.spec.fingerprint(),
                          hard_smiles=chemistry.smiles, hard_failure=chemistry.failure_reason,
                          run_path=str(run.path.resolve()),
                          source_archive_sha256=os.environ["LATENTMOL_SOURCE_SHA256"])
            torch.save({"z": point.cpu(), "active": selected.active.cpu(),
                        "no_atom": selected.no_atom.cpu()}, run.path / "start.pt")
        write_json(run.path / "start-audit.json", report)
        run.log({"stage": "start_audit", **report})
        if not report["passed"]:
            raise ValueError("Selected decoder does not reproduce the exact carbon-chain start; GPU preparation is blocked")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt, report)
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--mode", choices=("ae", "vae"), required=True)
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(run_preflight(args.checkpoint, args.sha256, args.mode, args.count,
                         args.run_root, args.receipt), flush=True)


if __name__ == "__main__":
    main()
