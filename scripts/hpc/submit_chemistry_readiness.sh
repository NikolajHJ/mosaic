#!/bin/bash
# After a fresh quota/queue check and authenticated SCP of the sealed archive.
set -euo pipefail
base=$1
manifest=$2
archive_sha=$3
tag=$4
[[ "$base" == /work3/s234818/mosaic/releases/* && "$tag" =~ ^[a-z0-9-]+$ ]]
[[ "$manifest" =~ ^[0-9a-f]{64}$ && "$archive_sha" =~ ^[0-9a-f]{64}$ ]]
echo "$archive_sha  $base/source.tar.gz" | sha256sum -c -
mkdir "$base/source"
tar -xzf "$base/source.tar.gz" -C "$base/source"
python3 "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
for name in brd4_chemistry_readiness brd4_chemistry brd4_chemistry_report; do
  bash -n "$base/source/scripts/hpc/$name.lsf"
done
chmod -R a-w "$base/source"
run="/zhome/95/e/206121/mosaic-runs/$tag-readiness"
mkdir "$run"
bsub -J mosaic-brd4-chem-ready -q hpc -n 4 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 01:00 \
  -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_chemistry_readiness.lsf" "$base/source" "$run" "$manifest" \
  | tee "$run/submission.txt"
