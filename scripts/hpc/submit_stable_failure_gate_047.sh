#!/usr/bin/env bash
#BSUB -J stable-deep-failure-gate-047
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o /work3/s234818/latentmol/logs/stable-deep-failure-gate-047.%J.out
#BSUB -e /work3/s234818/latentmol/logs/stable-deep-failure-gate-047.%J.err
set -euo pipefail
cd /work3/s234818/latentmol/releases/source-047-stable-deep-pair
exec bash scripts/hpc/run_job.sh python scripts/hpc/test_stable_failure_snapshot.py \
    --snapshot /work3/s234818/latentmol/diagnostics/nonfinite-046/failing-batch.pt \
    --historical-config configs/arch_deep_pair_refine_vae_full.yaml \
    --stable-config configs/arch_stable_deep_pair_refine_vae_full.yaml \
    --report /work3/s234818/latentmol/diagnostics/stable-deep-047/failure-gate.json
