#!/usr/bin/env bash
# One allocation. Shut down the Mosaic/JAX VJP worker before native FILE rescoring.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "Expected training output, fresh edit output, scheduler plan, plan SHA" >&2
    exit 2
fi
training=$1
out=$2
plan=$3
sha=$4
wrapper=scripts/hpc/protenix_latent_run.sh
python=/work3/s234818/latentmol/env/boltz311/bin/python
mosaic=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
here=experiments/latent-design/protenix-vae
config="$here/compact-edits.json"
assets=/work3/s234818/latentmol/protenix_occupancy
bash "$wrapper" "$python" -m latentmol.design.decoder_comparison sweep \
    --training "$training" --output "$out" --config "$config" \
    --mosaic-config "$here/native-dtu.json" --mosaic-python "$mosaic" \
    --plan "$plan" --plan-sha256 "$sha"
bash "$wrapper" /usr/bin/env "PYTHONPATH=$PWD/src:$assets/environment-attempt003/deps" \
    "PROTENIX_ROOT_DIR=$assets/data" LAYERNORM_TYPE=torch "$python" \
    -m latentmol.design.temperature_fresh --config "$config" --run "$out" \
    --plan "$plan" --plan-sha256 "$sha"
bash "$wrapper" "$python" -m latentmol.design.decoder_comparison report --output "$out" \
    --plan "$plan" --plan-sha256 "$sha"
