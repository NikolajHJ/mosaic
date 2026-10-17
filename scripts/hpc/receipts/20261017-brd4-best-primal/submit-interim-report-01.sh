#!/bin/bash
set -euo pipefail
base=/work3/s234818/mosaic/releases/20261017-brd4-interim-01
root=/zhome/95/e/206121/mosaic-runs
echo "c153c3ecb412443de0f8a7b4f8a5b0b502b2389b3b871119ce9db06a56c21266  $base/source.tar.gz" | sha256sum -c -
tar -xzf "$base/source.tar.gz" -C "$base/source"
python3 "$base/source/scripts/verify_stage0_release.py" "$base/source" d01c386aba5121af454a76868d49b840a99ac1cf3fea66ca9a2687e9699ed490
bash -n "$base/source/scripts/hpc/brd4_interim_report.lsf"
chmod -R a-w "$base/source"
run="$root/20261017-brd4-interim-report-01"
mkdir "$run"
bsub -J mosaic-brd4-interim-report -q hpc -n 2 -R 'span[hosts=1]' \
  -R 'rusage[mem=4GB]' -W 03:00 \
  -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_interim_report.lsf" "$base/source" "$run" \
  d01c386aba5121af454a76868d49b840a99ac1cf3fea66ca9a2687e9699ed490 \
  | tee "$run/submission.txt"
