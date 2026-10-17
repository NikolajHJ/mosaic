#!/bin/bash
set -euo pipefail
base=/work3/s234818/mosaic/releases/20261017-brd4-best-primal-01
root=/zhome/95/e/206121/mosaic-runs
echo "28606ac1ad01bb6332aa05b7ce50100d4e943c5b041b2ef048c2c6c49b5dbf69  $base/source.tar.gz" | sha256sum -c -
tar -xzf "$base/source.tar.gz" -C "$base/source"
python3 "$base/source/scripts/verify_stage0_release.py" "$base/source" c47d520f8cf0f7b41c86ced392baaa0a05236a401eeb334026e740fed389396e
for file in brd4_continuation_readiness.lsf brd4_continue.lsf brd4_continuation_report.lsf; do
  bash -n "$base/source/scripts/hpc/$file"
done
chmod -R a-w "$base/source"
run="$root/20261017-brd4-best-primal-readiness-01"
mkdir "$run"
bsub -J mosaic-brd4-best-primal-ready -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 \
  -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_continuation_readiness.lsf" "$base/source" "$run" c47d520f8cf0f7b41c86ced392baaa0a05236a401eeb334026e740fed389396e
