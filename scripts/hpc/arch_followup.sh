#!/usr/bin/env bash
# Run in a CPU allocation dependent on one completed immutable GPU workflow.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
TASK_PARENT_JOB="${1:?Provide exact parent LSF job ID}"
TASK_PARENT_SHA="${2:?Provide exact parent source archive SHA256}"
TASK_VARIANT="${3:?Provide decoder variant}"
TASK_MODE="${4:?Provide ae or vae}"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name "followup-${TASK_VARIANT}-${TASK_MODE}-parent-${TASK_PARENT_JOB}" -- \
    python -m latentmol.architecture_followup \
    --parent-job-id "$TASK_PARENT_JOB" --parent-source-sha256 "$TASK_PARENT_SHA" \
    --variant "$TASK_VARIANT" --mode "$TASK_MODE" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --molecules 128 --draws-per-molecule 16 --device cpu
