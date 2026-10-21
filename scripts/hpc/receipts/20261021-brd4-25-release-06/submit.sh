#!/bin/bash
set -euo pipefail
tag=20261021-brd4-25-release-06
base=/work3/s234818/mosaic/releases/$tag
archive=/work3/s234818/mosaic/$tag.tar.gz
root=/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04
science=/work3/s234818/mosaic/releases/20261021-brd4-25-04/source
sm=e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79
tm=396bcec01579aabe50646d4a78b50e20cd344c273a44aa275f11e75a8f79ff04
printf '%s  %s\n' fe4c25379dce26b7803712a6e41eb260a061347a2db103dd4be5488fc339cdae "$archive" | sha256sum -c -
mkdir "$base" "$base/source" "$root/launcher06-cpu" "$root/launcher06-bootstrap"
tar -xzf "$archive" -C "$base/source"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$tm"
"$python" "$science/scripts/verify_stage0_release.py" "$science" "$sm"
bash -n "$base/source/scripts/hpc/campaign25_release_stage.lsf"
bash -n "$base/source/scripts/hpc/campaign25_release_cpu.lsf"
chmod -R a-w "$base/source"
cpu_reply=$(bsub -q hpc -n 1 -R 'span[hosts=1]' -R 'rusage[mem=2GB]' -W 00:10 -J mosaic25-release-cpu -oo "$root/launcher06-cpu/stdout.%J.log" -eo "$root/launcher06-cpu/stderr.%J.log" bash -l "$base/source/scripts/hpc/campaign25_release_cpu.lsf" "$base/source" "$tm" "$science" "$root/launcher06-cpu")
printf '%s\n' "$cpu_reply" | tee "$root/launcher06-cpu/submission.txt"
pattern='Job <([0-9]+)>'
[[ "$cpu_reply" =~ $pattern ]]
cpu=${BASH_REMATCH[1]}
bsub -q hpc -n 1 -R 'span[hosts=1]' -R 'rusage[mem=2GB]' -W 00:20 -J mosaic25-release-bootstrap -w "ended($cpu)" -oo "$root/launcher06-bootstrap/stdout.%J.log" -eo "$root/launcher06-bootstrap/stderr.%J.log" bash -l "$base/source/scripts/hpc/campaign25_release_stage.lsf" "$base/source" "$tm" bootstrap "$science" "$root" "$sm" bootstrap none "$root/launcher06-bootstrap" "$cpu" | tee "$root/launcher06-bootstrap/submission.txt"
bjobs -l 29769177 > "$root/old-protenix-callback-after.txt"
