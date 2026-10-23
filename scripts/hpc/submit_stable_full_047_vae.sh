#!/usr/bin/env bash
#BSUB -J stable-deep-full-vae-047
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 15:00
#BSUB -w "done(29533163)"
#BSUB -o /work3/s234818/latentmol/logs/stable-deep-full-vae-047.%J.out
#BSUB -e /work3/s234818/latentmol/logs/stable-deep-full-vae-047.%J.err
set -euo pipefail
cd /work3/s234818/latentmol/releases/source-047-stable-deep-pair
exec bash scripts/hpc/run_job.sh bash scripts/hpc/arch_full.sh stable_deep_pair_refine vae
