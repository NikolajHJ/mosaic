#!/usr/bin/env bash
# Train and score sequentially so only one process owns the GPU at a time.
set -euo pipefail
: "${LSB_JOBID:?Requires an allocated GPU}"
if [[ $# -ne 3 ]]; then
    echo 'Expected a fresh experiment directory, GPU plan and GPU plan SHA-256' >&2
    exit 2
fi
output=$1
mkdir "$output"
LATENTMOL_CPU_THREADS=2 LATENTMOL_EVALUATION_MODULE=latentmol.training.compact_decoder_comparison \
    bash experiments/vae/run_checkpoint_comparison.sh \
        --config experiments/vae/compact-decoder-comparison.json --output "$output/training"

unset DBMOL_LAUNCH_READY DBMOL_LAUNCH_NONCE
exec bash experiments/latent-design/protenix-vae/run_decoder_comparison.sh \
    "$output/training" "$output/edits" "$2" "$3"
