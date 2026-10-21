#!/bin/bash
set -euo pipefail
tag=20261021-brd4-25-release-07
base=/work3/s234818/mosaic/releases/$tag
archive=/work3/s234818/mosaic/$tag.tar.gz
root=/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04
science=/work3/s234818/mosaic/releases/20261021-brd4-25-04/source
sm=e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79
tm=80274d8d5f066af3e364f0356182ab0aa124302613800aa8b68fe3ed3cdec964
printf '%s  %s\n' 61d23c4d5f6c7e34b59cdfec60947c1a001c838994fd175d5dc07a5a44c8127b "$archive" | sha256sum -c -
mkdir "$base" "$base/source" "$root/launcher07-cpu" "$root/launcher07-bootstrap"
tar -xzf "$archive" -C "$base/source"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$tm"
"$python" "$science/scripts/verify_stage0_release.py" "$science" "$sm"
bash -n "$base/source/scripts/hpc/campaign25_release_stage.lsf"
bash -n "$base/source/scripts/hpc/campaign25_release_cpu.lsf"
chmod -R a-w "$base/source"
cpu_reply=$(bsub -q hpc -n 1 -R 'span[hosts=1]' -R 'rusage[mem=2GB]' -W 00:10 -J mosaic25-release-cpu -oo "$root/launcher07-cpu/stdout.%J.log" -eo "$root/launcher07-cpu/stderr.%J.log" bash -l "$base/source/scripts/hpc/campaign25_release_cpu.lsf" "$base/source" "$tm" "$science" "$root/launcher07-cpu")
printf '%s\n' "$cpu_reply" | tee "$root/launcher07-cpu/submission.txt"
pattern='Job <([0-9]+)>'
[[ "$cpu_reply" =~ $pattern ]]
cpu=${BASH_REMATCH[1]}
bsub -q hpc -n 1 -R 'span[hosts=1]' -R 'rusage[mem=2GB]' -W 00:20 -J mosaic25-release-bootstrap -w "ended($cpu)" -oo "$root/launcher07-bootstrap/stdout.%J.log" -eo "$root/launcher07-bootstrap/stderr.%J.log" bash -l "$base/source/scripts/hpc/campaign25_release_stage.lsf" "$base/source" "$tm" bootstrap "$science" "$root" "$sm" bootstrap none "$root/launcher07-bootstrap" "$cpu" | tee "$root/launcher07-bootstrap/submission.txt"
bjobs -l 29769177 > "$root/old-protenix-callback-after.txt"
