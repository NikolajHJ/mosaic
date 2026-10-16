set -euo pipefail
project=/work3/s234818/mosaic
base="$project/releases/20261016-known-ligand-1000-04"
run="$project/runs/20261016-known-ligand-1000-01"
test -f "$project/runs/20261016-known-ligand-assessment-02/payload/assessment/assessment.json"
mkdir "$run"
bsub -J mosaic-known-1000 -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' -W 06:00 -w 'done(29736114)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_1000.lsf" "$base/source" "$run" 6 8df25aaa80a129dd56ef9fdbf8e12a2f7c4f37bd8668d72f172e544d205fb990 examples/configs/known_ligand_1000.json "$base/source/tests/data/t4_lysozyme_181l/fixture.json" "$project/runs/20261016-known-ligand-pilot-01/payload/study" 45fa41a40b6bf2e3a548b819b29aa1ad4d0b4b8e61067f3aed877a989d0cbdfa
