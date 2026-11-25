#!/usr/bin/env bash
# Run the matched trajectories and fresh discrete rescore on the same allocation.
set -euo pipefail
: "${LSB_JOBID:?Run the study inside an allocated GPU job}"
if [[ $# -ne 9 ]]; then
    echo 'Expected Torch and Mosaic interpreters, checkpoint, three configs, output, plan and plan SHA' >&2
    exit 2
fi
torch_python=$1
mosaic_python=$2
checkpoint=$3
study_config=$4
native_config=$5
fresh_config=$6
output=$7
plan=$8
plan_sha=$9
wrapper=scripts/hpc/protenix_latent_run.sh

bash "$wrapper" "$torch_python" -m latentmol.design.temperature_study \
    --checkpoint "$checkpoint" --config "$study_config" --output "$output" \
    --mosaic-config "$native_config" --mosaic-python "$mosaic_python" \
    --plan "$plan" --plan-sha256 "$plan_sha"

# The JAX worker is closed before the Torch native scorer takes the GPU.
assets=/work3/s234818/latentmol/protenix_occupancy
exec bash "$wrapper" /usr/bin/env \
    "PYTHONPATH=$PWD/src:$assets/environment-attempt003/deps" \
    "PROTENIX_ROOT_DIR=$assets/data" LAYERNORM_TYPE=torch \
    "$torch_python" -m latentmol.design.temperature_fresh --config "$fresh_config" \
    --run "$output" --plan "$plan" --plan-sha256 "$plan_sha"
