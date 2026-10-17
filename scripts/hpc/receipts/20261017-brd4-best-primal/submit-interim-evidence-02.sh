#!/bin/bash
set -euo pipefail
base=/work3/s234818/mosaic/releases/20261017-brd4-interim-02
root=/zhome/95/e/206121/mosaic-runs
echo "5d3ba07b6f937c3e18ae1a5acdac488a2eae6c62badcfeb5aa5f04587b76f732  $base/source.tar.gz" | sha256sum -c -
tar -xzf "$base/source.tar.gz" -C "$base/source"
python3 "$base/source/scripts/verify_stage0_release.py" "$base/source" 7e1bfe4751758f258749fd2a5a3e07b0167ceb544618160f5c6d022fd9d58246
bash -n "$base/source/scripts/hpc/brd4_interim_evidence.lsf"
chmod -R a-w "$base/source"
run="$root/20261017-brd4-interim-evidence-02"
mkdir "$run"
bsub -J mosaic-brd4-interim-evidence -q hpc -n 2 -R 'span[hosts=1]' \
  -R 'rusage[mem=4GB]' -W 01:00 \
  -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_interim_evidence.lsf" "$base/source" "$run" \
  7e1bfe4751758f258749fd2a5a3e07b0167ceb544618160f5c6d022fd9d58246 \
  | tee "$run/submission.txt"
