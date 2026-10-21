#!/bin/bash
set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-25-04.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-25-04
root=/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04
manifest=e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79
printf '%s  %s\n' b62ef75f6ede08fbed8854ff4efe167571be6eb050c1b222278e19e15b3dc879 "$archive" | sha256sum -c -
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
