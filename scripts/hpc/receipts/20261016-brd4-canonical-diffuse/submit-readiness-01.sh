set -euo pipefail
project=/work3/s234818/mosaic
archive="$project/brd4-canonical-diffuse-20261016-01.tar.gz"
echo 'f6efb3c2c8889e6f792482cd1d666c38631afe9b2bf78bcda5ba942e81f5e4c9  /work3/s234818/mosaic/brd4-canonical-diffuse-20261016-01.tar.gz' | sha256sum -c -
base="$project/releases/20261016-brd4-canonical-diffuse-01"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/brd4_uniform_readiness.lsf"
bash -n "$base/source/scripts/hpc/brd4_contact.lsf"
bash -n "$base/source/scripts/hpc/brd4_combined_report.lsf"
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-canonical-diffuse-readiness-01"
mkdir "$run"
bsub -J mosaic-brd4-canonical-ready -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_uniform_readiness.lsf" "$base/source" "$run" 421213580f84a038cde5e47bb4048a659bfa4daee46645deccbb2e7568d7ead4 examples/configs/brd4_canonical_diffuse_5000.json 3 21 3
