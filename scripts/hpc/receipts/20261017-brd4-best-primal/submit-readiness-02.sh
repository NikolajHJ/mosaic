#!/bin/bash
set -euo pipefail
base=/work3/s234818/mosaic/releases/20261017-brd4-best-primal-03
root=/zhome/95/e/206121/mosaic-runs
echo "fedfc43426cc40361fcdd9fa36eca10834f314bb4475341e22cd3dd1083e099e  $base/source.tar.gz" | sha256sum -c -
tar -xzf "$base/source.tar.gz" -C "$base/source"
python3 "$base/source/scripts/verify_stage0_release.py" "$base/source" cb3f850acd5f0810bf6eb318869dc33f5657eb6c22f3f0c37e9c95327c424903
for file in brd4_continuation_readiness.lsf brd4_continue.lsf brd4_continuation_report.lsf; do
  bash -n "$base/source/scripts/hpc/$file"
done
chmod -R a-w "$base/source"
run="$root/20261017-brd4-best-primal-readiness-02"
mkdir "$run"
bsub -J mosaic-brd4-best-primal-ready -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 \
  -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_continuation_readiness.lsf" "$base/source" "$run" cb3f850acd5f0810bf6eb318869dc33f5657eb6c22f3f0c37e9c95327c424903
