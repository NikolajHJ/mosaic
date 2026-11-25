#!/usr/bin/env bash
# DTU's noninteractive profile includes optional probes that can return nonzero.
set +e
set +o pipefail
source /etc/profile
set -eo pipefail
type module >/dev/null
: "${LSB_JOBID:?An LSF allocation is required}"
root=$1
model=$2
cohort=$3
cd "$root/source"
exp=$PWD/experiments/latent-design/direct-jump
run=$root/$model-$cohort
mkdir -- "$run"

# Translate task-local LSF device assignments before either CUDA runtime starts.
cuda_helper=/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-release-26/source/scripts/cuda_ownership.py
visibility=$(/usr/bin/python3 - "$cuda_helper" "$run/device.json" <<'PY'
import hashlib, importlib.util, json, os, sys
from pathlib import Path
helper, receipt = map(Path, sys.argv[1:])
if hashlib.sha256(helper.read_bytes()).hexdigest() != "578d2b6c9453ad14ce6778d306aa6c1d926acb442dcf8ae9f9a75bccc6b2e3cb":
    raise ValueError("CUDA allocation helper identity differs")
spec = importlib.util.spec_from_file_location("cuda_ownership", str(helper))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
proof = module.resolve_visibility(os.environ)
with receipt.open("x") as stream:
    json.dump(dict(job=os.environ["LSB_JOBID"], **proof), stream)
print(proof["selected_cuda_visible_devices"])
PY
)
export CUDA_VISIBLE_DEVICES=$visibility PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export DIRECT_JUMP_BOLTZ_RELEASE=$root/boltz-release
ops=/work3/s234818/mosaic/runs/stage3-protenix-defog-lr60-3seeds-20261112a/projection-kkt-retry-20261117a/source/src/mosaic/small_molecule/Stage2
args=(--model "$model" --cohort "$cohort" --run "$run" --founders "$root/source/founders.json")
native() (
    unset PYTHONPATH PYTHONHOME
    echo "Starting $model cohort $cohort phase $1"
    if [[ $model == protenix ]]; then
        source "$exp/protenix_env.sh"
    elif [[ $model == boltz2 ]]; then
        module load python3/3.11.9
        unset PYTHONHOME PYTHONPATH
        source "$exp/boltz_env.sh"
    else
        exit 2
    fi
    export PYTHONPATH="$ops:$PYTHONPATH"
    "$DIRECT_JUMP_PYTHON" -u "$exp/study.py" "$1" "${args[@]}"
)
for phase in train eval; do
    native "$phase"
    bash "$exp/fresh_env.sh" --model "$model" --input "$run/$phase/requests.json" \
        --work "$run/$phase/fresh-run" --output "$run/$phase/scores.json"
done
native collect
