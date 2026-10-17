#!/bin/bash
set -euo pipefail
base=/work3/s234818/mosaic/releases/20261017-brd4-parallel-01
root=/zhome/95/e/206121/mosaic-runs
echo "b0a35eed62c51b53285607d21a0c04c1f541c3b0bd7af888392da43c2ddcc76f  $base/source.tar.gz" | sha256sum -c -
tar -xzf "$base/source.tar.gz" -C "$base/source"
python3 "$base/source/scripts/verify_stage0_release.py" "$base/source" 1cf635c222bf750c298c7b4bcac30ec36eeb47e40f74480466d813cd64041504
for file in brd4_parallel_readiness.lsf brd4_continue.lsf brd4_parallel_report.lsf; do
  bash -n "$base/source/scripts/hpc/$file"
done
chmod -R a-w "$base/source"
[[ "$(bjobs -noheader -o stat 29745472 | tr -d '[:space:]')" == EXIT ]]
[[ "$(bjobs -noheader -o stat 29745475 | tr -d '[:space:]')" == EXIT ]]
bjobs -l 29745472 > "$root/20261017-brd4-chain-best-primal-01/job-cancelled-for-parallel.txt"
bjobs -l 29745475 > "$root/20261017-brd4-best-primal-report-01/job-cancelled-for-parallel.txt"
run="$root/20261017-brd4-parallel-readiness-01"
mkdir "$run"
bsub -J mosaic-brd4-parallel-ready -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 \
  -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_parallel_readiness.lsf" "$base/source" "$run" 1cf635c222bf750c298c7b4bcac30ec36eeb47e40f74480466d813cd64041504
