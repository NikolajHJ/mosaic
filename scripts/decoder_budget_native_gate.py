"""Receipt after CUDA optimizer/stream fixture passes; not an efficacy gate."""
import argparse
import json
import os
from pathlib import Path
from latentmol.experiments import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--sha256", required=True)
parser.add_argument("--create", action="store_true")
args = parser.parse_args()
if not os.environ.get("LSB_JOBID"):
    raise RuntimeError("DTU allocation required")
identity = {"source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"], "config_sha256": args.sha256,
            "status": "passed", "scope": "Actual CUDA tiny stable decoder optimizer/stream resume equivalence fixture"}
path = args.output_root / "native_resume_gate.json"
if args.create:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA fixture cannot be credited from a skipped CPU run")
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_json(path, identity)
elif json.loads(path.read_text()) != identity:
    raise ValueError("Native resume gate belongs to another source/config")
