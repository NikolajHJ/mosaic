set -euo pipefail
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-assessment-01"
mkdir "$run"
bsub -J mosaic-brd4-assessment -q hpc -n 1 -R 'rusage[mem=2GB]' -W 00:10 -w 'done(29736576)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" <<'JOB'
#!/bin/bash
set -euo pipefail
module load python3/3.12.11
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
release=/work3/s234818/mosaic/releases/20261016-brd4-5000-02/source
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-assessment-01"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
mkdir "$run/payload"
"$python" "$release/scripts/verify_stage0_release.py" "$release" 80bb2413904b955cc4ebc25a0c01e1940fd185b3bddfb6cbd3afed76ad75a4f2
"$python" "$release/scripts/assess_known_ligand_pilot.py" --pilot "$root/20261016-brd4-pilot-01/payload/study" --config "$release/examples/configs/brd4_5000.json" --output "$run/payload/assessment" --storage zhome --report-reserve-mib 512 > "$run/payload/assessment.log" 2>&1
JOB
