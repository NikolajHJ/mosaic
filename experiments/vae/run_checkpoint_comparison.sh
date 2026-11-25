#!/usr/bin/env bash
# Default: frozen evaluation. The separately authorized bounded invariant pilot
# is opt-in by module name; no workflow automatically resumes parent training.
set -euo pipefail
: "${LSB_JOBID:?Run substantive checkpoint evaluation in an allocated LSF job}"
module load python3/3.11.9
unset PYTHONHOME PYTHONPATH
source /work3/s234818/latentmol/env/boltz311/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
threads=${LATENTMOL_CPU_THREADS:-4}
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CPU thread budget' >&2; exit 2; }
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ulimit -c 0
# Reuse the already deployed standard-library LSF GPU resolver before importing
# Torch. The native training source and other jobs remain untouched.
exec python - "$@" <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
if arguments.count("--config") != 1:
    raise ValueError("Exactly one comparison config is required")
config = json.loads(Path(arguments[arguments.index("--config") + 1]).read_text())
helper = Path(config["hpc_cuda_ownership_helper"])
if hashlib.sha256(helper.read_bytes()).hexdigest() != config["hpc_cuda_ownership_sha256"]:
    raise ValueError("Pinned GPU routing helper changed")
spec = importlib.util.spec_from_file_location("_comparison_gpu_routing", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
proof = module.resolve_visibility(os.environ)
os.environ["CUDA_VISIBLE_DEVICES"] = proof["selected_cuda_visible_devices"]
print(json.dumps(dict(event="cuda_visibility_resolved", job_id=os.environ["LSB_JOBID"], **proof)), flush=True)
evaluation_module = os.environ.get("LATENTMOL_EVALUATION_MODULE", "latentmol.evaluation.checkpoint_comparison")
if evaluation_module not in ("latentmol.evaluation.checkpoint_comparison", "latentmol.evaluation.interpolation",
                              "latentmol.training.invariant_pilot", "latentmol.training.interpolation_pilot",
                              "latentmol.training.chembl_comparison", "latentmol.training.combined_continuation",
                              "latentmol.training.compact_benchmark", "latentmol.training.compact_train",
                              "latentmol.training.compact_decoder_comparison",
                              "latentmol.training.projected_continuation", "experiments.vae.motif_pilot",
                              "experiments.vae.encoder_upgrade_pilot", "experiments.vae.early_encoder_arms",
                              "experiments.vae.encoder_next_leg", "experiments.vae.attention_next_leg",
                              "experiments.vae.training_data_eda.shared_ring_pilot",
                              "experiments.vae.training_data_eda.protected_ring_pilot",
                              "latentmol.evaluation.combined_vae"):
    raise ValueError("Unsupported VAE workflow module")
os.execv(sys.executable, [sys.executable, "-m", evaluation_module, *arguments])
PY
