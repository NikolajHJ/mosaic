#!/bin/bash
set -euo pipefail
archive=/work3/s234818/mosaic/20261021-brd4-jq1-25-01.tar.gz
base=/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-01
root=/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01
manifest=2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c
printf '%s  %s\n' 5e4bf8601f1b09e244a7763dc6c6d8af5ddc76bfd3490830993d7cfd0b2c0906 "$archive" | sha256sum -c -
mkdir "$base" "$root"
mkdir "$base/source" "$root/cpu" "$root/after-cpu"
tar -xzf "$archive" -C "$base/source"
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" "$base/source/scripts/verify_stage0_release.py" "$base/source" "$manifest"
bash -n "$base/source/scripts/hpc/campaign25_pocket_cpu.lsf"
bash -n "$base/source/scripts/hpc/campaign25_pocket_stage.lsf"
chmod -R a-w "$base/source"
ls -ld /work3/s234818/mosaic/cache/mosaic/protenix
getquota_zhome.sh > "$root/quota-at-submission.txt"
reply=$(bsub -q hpc -n 4 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 01:00 -J mosaic-jq1-25-cpu -oo "$root/cpu/stdout.%J.log" -eo "$root/cpu/stderr.%J.log" bash -l "$base/source/scripts/hpc/campaign25_pocket_cpu.lsf" "$base/source" "$root/cpu" "$manifest")
printf '%s\n' "$reply" | tee "$root/cpu/submission.txt"
pattern='Job <([0-9]+)>'
[[ "$reply" =~ $pattern ]]
cpu=${BASH_REMATCH[1]}
reply=$(bsub -q hpc -n 1 -R 'span[hosts=1]' -R 'rusage[mem=2GB]' -W 00:20 -J mosaic-jq1-25-after-cpu -w "ended($cpu)" -oo "$root/after-cpu/stdout.%J.log" -eo "$root/after-cpu/stderr.%J.log" bash -l "$base/source/scripts/hpc/campaign25_pocket_stage.lsf" "$base/source" "$root" "$manifest" after-cpu none "$root/after-cpu" "$cpu")
printf '%s\n' "$reply" | tee "$root/after-cpu/submission.txt"
[[ "$reply" =~ $pattern ]]
callback=${BASH_REMATCH[1]}
printf 'CPU_JOB=%s\nCALLBACK_JOB=%s\n' "$cpu" "$callback" | tee "$root/initial-job-ids.txt"
bjobs -l "$cpu" "$callback" > "$root/initial-scheduler.txt"
cat "$root/initial-job-ids.txt"
