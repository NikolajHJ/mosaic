set -euo pipefail
base=/work3/s234818/mosaic/releases/20261016-known-ligand-1000-02
run=/work3/s234818/mosaic/runs/20261016-known-ligand-pilot-01
fixture=/work3/s234818/mosaic/runs/20261016-known-ligand-readiness-01/payload/prepared/fixture.json
bjobs -l 29736014 > /work3/s234818/mosaic/runs/20261016-known-ligand-readiness-01/job-final.txt
mkdir "$run"
bsub -J mosaic-known-pilot -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' -W 01:00 -w 'done(29736014)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_1000.lsf" "$base/source" "$run" 6 4060ecaae546339cab71991ba6f42813c996451540eacdc9218b36e071c79cc1 examples/configs/known_ligand_pilot.json "$fixture"
