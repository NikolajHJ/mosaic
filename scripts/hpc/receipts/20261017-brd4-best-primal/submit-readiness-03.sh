#!/bin/bash
set -euo pipefail
base=/work3/s234818/mosaic/releases/20261017-brd4-best-primal-04
root=/zhome/95/e/206121/mosaic-runs
echo "8377d9dc2f5996c2e81e4ef1fd24519e034905a9a7a299cd5e452e9bc326d8c7  $base/source.tar.gz" | sha256sum -c -
tar -xzf "$base/source.tar.gz" -C "$base/source"
python3 "$base/source/scripts/verify_stage0_release.py" "$base/source" ec278a14e0847939f667dc2a1965f735da627684b3e7f38e0e59e624473e9873
for file in brd4_continuation_readiness.lsf brd4_continue.lsf brd4_continuation_report.lsf; do
  bash -n "$base/source/scripts/hpc/$file"
done
chmod -R a-w "$base/source"
run="$root/20261017-brd4-best-primal-readiness-03"
mkdir "$run"
bsub -J mosaic-brd4-best-primal-ready -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 \
  -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_continuation_readiness.lsf" "$base/source" "$run" ec278a14e0847939f667dc2a1965f735da627684b3e7f38e0e59e624473e9873
