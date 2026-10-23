#!/usr/bin/env bash
# UNTESTED HPC campaign launcher. Supply one pinned study YAML per AE/VAE
# checkpoint/seed. Model combinations and controls live in each YAML's arms.
# Different latent spaces require predeclared radius/scale calibration; this
# launcher never claims that identical raw-z radii make an AE/VAE comparison fair.
set -euo pipefail
: "${LSB_JOBID:?Run inside an LSF allocation}"
if [ "$#" -eq 0 ]; then echo 'Pass at least one pinned study YAML' >&2; exit 2; fi
for TASK_CONFIG in "$@"; do
  # Each child creates a unique ExperimentRun. A failed native gate stops the
  # campaign; earlier child results remain intact with exact provenance.
  bash scripts/hpc/multimodel_job.sh pipeline "$TASK_CONFIG"
done
