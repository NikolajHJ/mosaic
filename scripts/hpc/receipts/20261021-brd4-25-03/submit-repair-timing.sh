#!/bin/bash
set -euo pipefail
tag=20261021-brd4-25-repair-timing-01
archive=/work3/s234818/mosaic/$tag.tar.gz
base=/work3/s234818/mosaic/releases/$tag
root=/zhome/95/e/206121/mosaic-runs/$tag
manifest=04d4fd4eaaa8e07c02c58a04b9f15c02e0a47fba9a9ec5dbb1e182084c641100
science=/work3/s234818/mosaic/releases/20261021-brd4-25-03/source
science_manifest=1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b
campaign=/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03
printf '%s  %s\n' 1776307af4e202d8b805c8c8c272902863ecb7abee30f61132288ba10af313f6 "$archive" | sha256sum -c -
mkdir "$base" "$root"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
bash -n "$base/source/scripts/hpc/campaign25_repair_timing.lsf"
chmod -R a-w "$base/source"
bsub -q hpc -n 4 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:20 -J mosaic25-repair-timing \
  -w 'done(29768991)' -oo "$root/stdout.%J.log" -eo "$root/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/campaign25_repair_timing.lsf" "$base/source" "$manifest" \
  "$science" "$science_manifest" "$campaign" "$root" | tee "$root/submission.txt"
