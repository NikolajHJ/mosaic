"""Original Lightning/runtime DiffSBDD replay, in its own allocated process.

This entry point runs before the combined worker owns CUDA. It instantiates the
unchanged upstream Lightning class, strictly restores the entire checkpoint,
and exports native high-level/step observations. The only constructor override
disables an unavailable *metric-only* training-SMILES lookup; model/schedule
hyperparameters and all tensors remain unchanged and authenticated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch

from .native import authenticate_native_source, _file_sha256
from .native_checks import export_native_reference
from .pocket import prepare_pocket


def run(config_path, output, device):
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Original native replay requires an allocated HPC job")
    version = importlib.metadata.version("pytorch-lightning")
    if not torch.__version__.startswith("2.0.1") or version != "1.8.4":
        raise RuntimeError("Reference must use original-compatible Torch2.0.1 / Lightning1.8.4")
    config_path = Path(config_path).resolve(strict=True)
    cfg = json.loads(config_path.read_text())
    source_root = config_path.parents[1]
    native = cfg["upstream"]
    native_root = source_root / native["source_relative"]
    provenance = authenticate_native_source(native_root, native["source_manifest_sha256"])
    checkpoint_path = source_root / native["checkpoint_relative"]
    if _file_sha256(checkpoint_path) != native["checkpoint_sha256"]:
        raise RuntimeError("Native reference checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = checkpoint["hyper_parameters"]
    hp = hp if isinstance(hp, dict) else vars(hp)
    eval_params = deepcopy(hp["eval_params"])
    original_smiles = eval_params.get("smiles_file") if isinstance(eval_params, dict) else getattr(eval_params, "smiles_file", None)
    if isinstance(eval_params, dict):
        eval_params["smiles_file"] = None
    else:
        eval_params.smiles_file = None
    sys.path.insert(0, str(native_root))
    try:
        module = importlib.import_module("lightning_modules")
        model = module.LigandPocketDDPM.load_from_checkpoint(
            str(checkpoint_path), map_location="cpu", strict=True, eval_params=eval_params)
    finally:
        sys.path.remove(str(native_root))
    # Exact all-state restoration, including model and schedule buffers, checked
    # independently of the pure-core loader used in the combined runtime.
    restored = model.state_dict()
    if set(restored) != set(checkpoint["state_dict"]):
        raise RuntimeError("Original Lightning checkpoint key inventory changed")
    for key, value in restored.items():
        target = checkpoint["state_dict"][key].to(value.dtype)
        if not torch.equal(value.cpu(), target.cpu()):
            raise RuntimeError("Original native strict restore differs: " + key)
    model.eval().requires_grad_(False).to(device)
    pocket, definition = prepare_pocket(source_root / cfg["pocket"]["receptor_relative"],
                                      model.dataset_info["atom_decoder"], device=device)
    count = torch.tensor([cfg["pocket"]["atom_count"]], device=device, dtype=torch.long)
    fixture = export_native_reference(model.ddpm, pocket, count, seed=17)
    fixture["runtime_qualification"] = {
        "version":"diffsbdd-original-lightning-reference-v1", "passed":True,
        "torch":str(torch.__version__), "lightning":version,
        "constructor":"unchanged upstream LigandPocketDDPM.load_from_checkpoint(strict=True)",
        "all_state_keys":sorted(restored), "source_manifest_sha256":provenance["manifest_sha256"],
        "checkpoint_sha256":native["checkpoint_sha256"],
        "metric_only_override":{"eval_params.smiles_file":{"saved":str(original_smiles),"used":None}},
        "pocket_definition":definition, "device":str(device)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(fixture, stream)
    receipt = {"passed":True,"output":str(output),"sha256":_file_sha256(output),
               "runtime_qualification":fixture["runtime_qualification"]}
    with output.with_suffix(".json").open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config,args.output,args.device), indent=2))


if __name__ == "__main__":
    main()
