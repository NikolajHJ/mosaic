#!/usr/bin/env bash
#BSUB -J arch-deep-vae-nonfinite-044
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o /work3/s234818/latentmol/logs/arch-deep-vae-nonfinite-044.%J.out
#BSUB -e /work3/s234818/latentmol/logs/arch-deep-vae-nonfinite-044.%J.err
set -euo pipefail
cd /work3/s234818/latentmol/releases/source-040-global-bond-context
exec bash scripts/hpc/run_job.sh python \
    /work3/s234818/latentmol/releases/diagnostic-044-nonfinite-replay/scripts/hpc/diagnose_nonfinite_resume.py \
    --config configs/arch_deep_pair_refine_vae_full.yaml \
    --data /work3/s234818/latentmol/data/zinc215k-seed17-v1 \
    --run-root /work3/s234818/latentmol/runs \
    --resume-from /work3/s234818/latentmol/runs/20260929T173057Z_vae-train_c128eaf6/last.pt \
    --report /work3/s234818/latentmol/diagnostics/nonfinite-044/report.json
