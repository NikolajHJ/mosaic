#!/usr/bin/env bash
# UNTESTED. Run inside a fresh DTU GPU allocation and immutable source release.
# Usage: bash scripts/hpc/boltz_active_set_prepare.sh NEW_ATTEMPT_DIR AE_CHECKPOINT 11 12 13 14 15
# Each count gets its own native ligand prediction and fixed reference pose.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${BOLTZ_CACHE:?Invoke through the pinned Boltz environment}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source archive identity required}"
: "${BOLTZ_EXAMPLE_ASSETS:?Pinned official example cache required for offline preparation}"
TASK_ATTEMPT="${1:?New unique attempt directory required}"
TASK_AE="${2:?Graph-schema checkpoint required}"
shift 2
if [ "$#" -lt 1 ] || [ -e "$TASK_ATTEMPT" ]; then
    printf 'Require at least one count and a new attempt directory\n' >&2
    exit 2
fi
mkdir -p "$TASK_ATTEMPT"
for COUNT in "$@"; do
    if ! [[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || [ "$COUNT" -gt 50 ]; then
        printf 'Unsupported count: %s\n' "$COUNT" >&2
        exit 2
    fi
    DIR="$(printf '%s/count-%02d' "$TASK_ATTEMPT" "$COUNT")"
    if [ -e "$DIR" ]; then
        printf 'Duplicate count: %s\n' "$COUNT" >&2
        exit 2
    fi
    mkdir -p "$DIR"
    python -m latentmol.boltz.prepare chain-count-example \
        --output "$DIR/input" --count "$COUNT" --assets-dir "$BOLTZ_EXAMPLE_ASSETS"
    STEM="$(printf 'carbon_chain_%02d' "$COUNT")"
    # Short schedules prepare an engineering pose. They do not establish binding.
    boltz predict "$DIR/input/$STEM.yaml" \
        --cache "$BOLTZ_CACHE" --out_dir "$DIR/prediction" \
        --recycling_steps 0 --sampling_steps 20 --diffusion_samples 1 \
        --sampling_steps_affinity 20 --diffusion_samples_affinity 1 \
        --num_workers 0 --preprocessing-threads 1 --no_kernels --seed 17 \
        --subsample_msa --num_subsampled_msa 128
    python -m latentmol.boltz.prepare snapshot \
        --prediction-dir "$DIR/prediction/boltz_results_$STEM" \
        --cache "$BOLTZ_CACHE" --output "$DIR/prepared.pt" \
        --ae-checkpoint "$TASK_AE" \
        --input-manifest "$DIR/input/input_manifest.json"
done
python -m latentmol.boltz.prepare count-catalog \
    --root "$TASK_ATTEMPT" --counts "$@" --output "$TASK_ATTEMPT/contexts.json"
