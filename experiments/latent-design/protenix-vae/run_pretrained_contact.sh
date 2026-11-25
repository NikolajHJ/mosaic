#!/usr/bin/env bash
# Run the graph-VAE contact pilot and freshly featurize its discrete candidates.
set -euo pipefail
: "${LSB_JOBID:?Submit inside a DTU GPU allocation}"
if [[ $# != 3 ]]; then
    echo "Usage: $0 NEW_OUTPUT GPU_PLAN GPU_PLAN_SHA256" >&2
    exit 2
fi
out=$1
plan=$2
plan_sha=$3
root=/work3/s234818/latentmol
assets="$root/assets/graph-lda-zinc-591abff0"
native="$root/protenix_occupancy"
torch_python="$root/env/boltz311/bin/python"
experiment=experiments/latent-design/protenix-vae
wrapper=scripts/hpc/protenix_latent_run.sh
runner="$experiment/pretrained_contact.py"
ownership=(--plan "$plan" --plan-sha256 "$plan_sha")

# Each stage exits before the next one opens its GPU model.
bash "$wrapper" "$torch_python" "$runner" \
    --config "$experiment/graph-lda-contact.json" --source "$assets/upstream" \
    --checkpoint "$assets/pigvae_best_model.pt" --native-config "$experiment/native-dtu.json" \
    --mosaic-python /work3/s234818/mosaic/envs/protenix-tiny/bin/python \
    --output "$out" "${ownership[@]}"
bash "$wrapper" /usr/bin/env "PYTHONPATH=$PWD/src:$native/environment-attempt003/deps" \
    "PROTENIX_ROOT_DIR=$native/data" LAYERNORM_TYPE=torch "$torch_python" \
    -m latentmol.design.temperature_fresh --config "$experiment/flow-fresh-dtu.json" \
    --run "$out" "${ownership[@]}"
exec bash "$wrapper" "$torch_python" "$runner" --summarize "$out" "${ownership[@]}"
