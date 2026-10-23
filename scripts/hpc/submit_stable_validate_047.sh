#!/usr/bin/env bash
#BSUB -J stable-deep-validate-047
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 01:00
#BSUB -o /work3/s234818/latentmol/logs/stable-deep-validate-047.%J.out
#BSUB -e /work3/s234818/latentmol/logs/stable-deep-validate-047.%J.err
set -euo pipefail
cd /work3/s234818/latentmol/releases/source-047-stable-deep-pair
exec bash scripts/hpc/run_job.sh bash scripts/hpc/arch_validate.sh
