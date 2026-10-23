"""Independent allocated-HPC prototype; never resumes or writes VAE training.

CPU contracts precede native GPU injection checks and a bounded supervised fit.
Every artifact is new. Frozen files are hash-authenticated, and numerical work
requires an LSF allocation. A successful fit is not an optimization approval.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET


VERSION = "epoch36-native-embedding-prototype-v1"


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verified(asset):
    from latentmol.experiments import file_sha256
    path = Path(asset["path"])
    if file_sha256(path) != asset["sha256"]:
        raise ValueError("Asset hash mismatch: " + str(path))
    return path


def load_vae(config, device):
    from latentmol.evaluate import load_model
    from .embedding_training import EPOCH36_SHA256
    if config["vae"]["sha256"] != EPOCH36_SHA256:
        raise ValueError("Only the original completed epoch-36 VAE is authorized")
    model, checkpoint = load_model(verified(config["vae"]), device)
    if checkpoint["epoch"] != 36 or checkpoint["train_config"]["mode"] != "vae":
        raise ValueError("Frozen checkpoint is not the original epoch-36 VAE")
    model.eval().requires_grad_(False)
    model.embedding_bridge_checkpoint_identity = dict(epoch=36, sha256=EPOCH36_SHA256)
    return model, checkpoint


def verify_dataset(config):
    if Path(config["dataset_manifest"]["path"]).resolve() != (Path(config["dataset"]) / "manifest.json").resolve():
        raise ValueError("Dataset path differs from authenticated manifest")
    verified(config["dataset_manifest"])


def load_native(config, device):
    from .model import load_affinity_model
    from .dbmol_model import load_confidence_model, FrozenBoltzDBMol
    from .dbmol_objective import DBMolObjectiveConfig
    from .contracts import PreparedComplex
    from .embedding_bridge import NativeProjectionBank
    from .embedding_injection import native_embedding_models
    affinity = load_affinity_model(verified(config["affinity"]))
    objective = DBMolObjectiveConfig.from_mapping(config["dbmol"], Path("/"))
    structure = load_confidence_model(objective.confidence_checkpoint)
    scorer = FrozenBoltzDBMol(affinity, structure, objective, recycling_steps=0).to(device).eval()
    bank = NativeProjectionBank.from_models(native_embedding_models(scorer),
        checkpoint_sha256={"affinity": config["affinity"]["sha256"],
                           "structure": objective.confidence_checkpoint["sha256"]},
        confidence_models={"structure"}).to(device).eval().requires_grad_(False)
    prepared = PreparedComplex.load(verified(config["prepared"]), device)
    objective.verify_reference(prepared, config["prepared"]["sha256"])
    return scorer, prepared, bank


def cpu_stage(config, output, deadline):
    import torch
    from latentmol.experiments import file_sha256
    from latentmol.train import state_digest
    result = {"status": "running_not_passed"}
    tests = config["cpu_tests"]
    command = [os.sys.executable, "-m", "pytest", "-q", *tests,
               "--junitxml=" + str(output / "cpu-tests.xml")]
    run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, timeout=max(1, deadline - time.monotonic()))
    (output / "cpu-tests.log").write_text(run.stdout)
    suites = list(ET.parse(output / "cpu-tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.get(key, "0")) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    failed = run.returncode or not counts["tests"] or any(counts[k] for k in ("failures", "errors", "skipped"))
    result.update(test_command=command, test_returncode=run.returncode, junit=counts,
                  status="failed_not_passed" if failed else "running_not_passed")
    write(output / "cpu-gate.json", result)
    if failed:
        raise RuntimeError("Allocated CPU contracts did not all pass")
    model, checkpoint = load_vae(config, "cpu")
    before = state_digest(model.state_dict())
    generator = torch.Generator().manual_seed(config["seed"])
    z = torch.randn((2, model.config.latent), generator=generator)
    with torch.no_grad():
        original = model.decode(z)
        captured, features = model.decode_with_features(z)
    parity = all(torch.equal(getattr(original, key), getattr(captured, key))
                 for key in ("atoms", "charges", "bonds"))
    if not parity or state_digest(model.state_dict()) != before:
        raise RuntimeError("Feature export changed original frozen decoder behavior")
    verify_dataset(config)
    result.update(status="passed", vae_sha256=config["vae"]["sha256"],
        epoch=checkpoint["epoch"], model_config=checkpoint["model_config"],
        vae_state_sha256=before, decoder_output_bitwise_parity=True,
        feature_dimensions={key: list(getattr(features, key).shape) for key in
                            ("atom_hidden", "pair_hidden", "charge_input", "bond_input")},
        junit_sha256=file_sha256(output / "cpu-tests.xml"))
    write(output / "cpu-gate.json", result)
    return result


def gpu_stage(config, output, deadline):
    import torch
    from latentmol.experiments import file_sha256
    from latentmol.train import state_digest
    from .embedding_bridge import HiddenEmbeddingBridge
    from .embedding_training import prepare_dataset, train_bridge
    from .embedding_validation import native_gate, score_learned_bridge
    cpu = output.parent / "cpu" / "result.json"
    prior = json.loads(cpu.read_text())
    if (prior["status"] != "passed" or prior["source_sha256"] != os.environ["LATENTMOL_SOURCE_SHA256"]
            or prior["plan_sha256"] != config["_plan_sha256"]):
        raise ValueError("GPU requires this exact release's authenticated CPU pass")
    if (prior["junit"]["tests"] < 1 or any(prior["junit"][key] for key in ("failures", "errors", "skipped"))
            or file_sha256(cpu.parent / "cpu-tests.xml") != prior["junit_sha256"]):
        raise ValueError("CPU gate lacks authenticated all-pass tests")
    verify_dataset(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated GPU has no usable CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, checkpoint = load_vae(config, "cuda")
    del checkpoint
    scorer, prepared, bank = load_native(config, "cuda")
    if model.spec != prepared.spec:
        raise ValueError("VAE and prepared native context graph schemas differ")
    frozen_digest = state_digest(model.state_dict())
    bank_metadata = bank.metadata()
    write(output / "native-projection-bank.json", bank_metadata)
    native = native_gate(scorer, prepared, bank, model, output / "native-gate.json",
                         config["native_tolerances"], deadline=deadline)
    # The supervised fit needs only frozen projection buffers, not Boltz trunks.
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    data = prepare_dataset(model, config["dataset"], output / "paired-data", config["seed"],
                           **config["data_sampling"], deadline=deadline)
    result = train_bridge(model, bank, output / "paired-data", output / "fit", "cuda",
                          seed=config["seed"], **config["fitting"], deadline=deadline)
    if result["status"] == "partial_runtime_not_passed":
        return result
    # A separate fixed-context diagnostic uses the seed's frozen posterior mean.
    # It is not used for selection or for updating any mapping parameter.
    scorer, prepared, fresh_bank = load_native(config, "cuda")
    saved = torch.load(result["checkpoint_path"], map_location="cuda", weights_only=True)
    if saved["bank_metadata"] != fresh_bank.metadata():
        raise ValueError("Fitted bridge native checkpoint metadata changed")
    for key, value in saved["bank_state"].items():
        if not torch.equal(value, fresh_bank.state_dict()[key]):
            raise ValueError("Fitted bridge native projection bytes changed")
    dims = saved["bridge_configuration"]
    bridge = HiddenEmbeddingBridge(fresh_bank, **{key: dims[key] for key in
        ("atom_input_dim", "pair_input_dim", "charge_input_dim", "hidden_dim")}).to("cuda")
    bridge.load_state_dict(saved["bridge_state"], strict=True)
    bridge.eval().requires_grad_(False)
    with torch.no_grad():
        z, _ = model.encode(prepared.seed.to_relaxed(model.spec))
    # Persist independent diagnostic failure without erasing the completed fit.
    try:
        diagnostic = score_learned_bridge(scorer, prepared, bridge, model, z,
                                          require_gradient=True, deadline=deadline)
        tensors = {key: value for key, value in diagnostic.items() if isinstance(value, torch.Tensor)}
        plain = {key: value for key, value in diagnostic.items() if not isinstance(value, torch.Tensor)}
        # Validation helper may include a HardGraphBatch, retain it tensor-only.
        from latentmol.representation import HardGraphBatch
        for key in list(plain):
            if isinstance(plain[key], HardGraphBatch):
                graph = plain.pop(key)
                tensors[key] = dict(atoms=graph.atoms, charges=graph.charges, bonds=graph.bonds)
        torch.save(tensors, output / "learned-diagnostic.pt")
        write(output / "learned-diagnostic.json", plain)
    except Exception as error:
        diagnostic = {"status": "failed_diagnostic", "error_type": type(error).__name__, "error": str(error)}
        write(output / "learned-diagnostic.json", diagnostic)
    if state_digest(model.state_dict()) != frozen_digest:
        raise RuntimeError("Original frozen VAE changed")
    return dict(status="prototype_fit_completed_not_optimization_validated",
        native_gate_status=native["status"], feature_audit_status=result["status"], fit=result, dataset_manifest_sha256=file_sha256(output / "paired-data" / "manifest.json"),
        learned_diagnostic_path=str(output / "learned-diagnostic.json"),
        frozen_vae_unchanged=True, optimization="not_run", cpu_result_sha256=file_sha256(cpu))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cpu", "gpu"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise RuntimeError("Numerical embedding prototype work requires a sealed LSF allocation")
    from latentmol.experiments import file_sha256
    import torch
    torch.set_num_threads(min(4, int(os.environ.get("LSB_DJOB_NUMPROC", "4"))))
    config = json.loads(args.plan.read_text())
    if config["version"] != VERSION:
        raise ValueError("Unknown prototype plan")
    config["_plan_sha256"] = file_sha256(args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config["internal_seconds"][args.stage]
    provenance = dict(version=VERSION, stage=args.stage, job_id=os.environ["LSB_JOBID"],
        hostname=socket.gethostname(), plan_sha256=config["_plan_sha256"],
        source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"],
        vae_checkpoint=config["vae"], existing_vae_continuation="untouched")
    write(args.output / "result.json", {**provenance, "status": "running_not_passed"})
    try:
        result = (cpu_stage if args.stage == "cpu" else gpu_stage)(config, args.output, deadline)
    except BaseException as error:
        result = dict(status="failed_not_passed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write(args.output / "result.json", {**provenance, **result, "elapsed_seconds": time.monotonic() - started})
        raise
    write(args.output / "result.json", {**provenance, **result, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
