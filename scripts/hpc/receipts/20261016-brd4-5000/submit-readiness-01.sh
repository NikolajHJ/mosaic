set -euo pipefail
project=/work3/s234818/mosaic
archive="$project/brd4-5000-20261016-01.tar.gz"
echo 'b3011e5be4502c383567c1f6253ae3aa774cf05fa841234c09912f428112a14f  /work3/s234818/mosaic/brd4-5000-20261016-01.tar.gz' | sha256sum -c -
base="$project/releases/20261016-brd4-5000-01"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/brd4_readiness.lsf"
bash -n "$base/source/scripts/hpc/brd4_contact.lsf"
run="$project/runs/20261016-brd4-readiness-01"
mkdir "$run"
bsub -J mosaic-brd4-readiness -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_readiness.lsf" "$base/source" "$run" 0d020949dc2430c294197bae4fe4a65e53dcde7f8acb1adc387057dfcc73bd89
