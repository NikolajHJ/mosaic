#!/bin/bash
set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-25-03.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-25-03
root=/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03
manifest=1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b
printf '%s  %s\n' bfc6ad6f0af3268551258de55ca1bd196bb7ff35f6f690a305bf64a0d34fcff7 "$archive" | sha256sum -c -
mkdir "$base" "$root"
mkdir "$base/source" "$root/cpu" "$root/after-cpu"
tar -xzf "$archive" -C "$base/source"
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
bash -n "$base/source/scripts/hpc/campaign25_cpu.lsf"
bash -n "$base/source/scripts/hpc/campaign25_stage.lsf"
chmod -R a-w "$base/source"
ls -ld /work3/s234818/mosaic/cache/mosaic/protenix
getquota_zhome.sh > "$root/quota-at-submission.txt"
reply=$(bsub -q hpc -n 4 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 01:00 -J mosaic25-cpu -oo "$root/cpu/stdout.%J.log" -eo "$root/cpu/stderr.%J.log" bash -l "$base/source/scripts/hpc/campaign25_cpu.lsf" "$base/source" "$root/cpu" "$manifest")
printf '%s\n' "$reply" | tee "$root/cpu/submission.txt"
pattern='Job <([0-9]+)>'
[[ "$reply" =~ $pattern ]]
cpu=${BASH_REMATCH[1]}
reply=$(bsub -q hpc -n 1 -R 'span[hosts=1]' -R 'rusage[mem=2GB]' -W 00:20 -J mosaic25-after-cpu -w "ended($cpu)" -oo "$root/after-cpu/stdout.%J.log" -eo "$root/after-cpu/stderr.%J.log" bash -l "$base/source/scripts/hpc/campaign25_stage.lsf" "$base/source" "$root" "$manifest" after-cpu none "$root/after-cpu" "$cpu")
printf '%s\n' "$reply" | tee "$root/after-cpu/submission.txt"
[[ "$reply" =~ $pattern ]]
callback=${BASH_REMATCH[1]}
printf 'CPU_JOB=%s\nCALLBACK_JOB=%s\n' "$cpu" "$callback" | tee "$root/initial-job-ids.txt"
bjobs -l "$cpu" "$callback" > "$root/initial-scheduler.txt"
cat "$root/initial-job-ids.txt"
