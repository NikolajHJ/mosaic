set -euo pipefail
project=/work3/s234818/mosaic
archive="$project/brd4-5000-20261016-03.tar.gz"
echo '13bf30466bc786338c88df4b615f0acf44ebb7a897b8dcece84a610b452c9f6d  /work3/s234818/mosaic/brd4-5000-20261016-03.tar.gz' | sha256sum -c -
base="$project/releases/20261016-brd4-5000-03"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
chmod -R a-w "$base/source"
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-5000-report-01"
mkdir "$run"
bsub -J mosaic-brd4-5000-report -q hpc -n 1 -R 'rusage[mem=4GB]' -W 01:30 -w 'ended(29736600)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" <<'JOB'
#!/bin/bash
set -euo pipefail
module load python3/3.12.11
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
release=/work3/s234818/mosaic/releases/20261016-brd4-5000-03/source
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-5000-report-01"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
mkdir "$run/payload"
export MPLCONFIGDIR="$run/payload/mplcache"
"$python" "$release/scripts/verify_stage0_release.py" "$release" d6d4bbc17545df536e8dcea70f50d863f966b67ec25fb0a3b3f04f86739a9f9a
"$python" "$release/scripts/hpc/zhome_quota_preflight.py" --run-dir "$run/payload" --minimum-free-gib 5
"$python" "$release/scripts/render_protenix_trajectory.py" --study "$root/20261016-brd4-5000-02/payload/study" --output "$run/payload/report" > "$run/payload/render.log" 2>&1
JOB
