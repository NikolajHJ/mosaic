set -euo pipefail
project=/work3/s234818/mosaic
archive="$project/brd4-uniform-5000-20261016-01.tar.gz"
echo '058c9dd851e33657897ed3f8c442e94cde6f3c4697c38518a98eab4810dd098c  /work3/s234818/mosaic/brd4-uniform-5000-20261016-01.tar.gz' | sha256sum -c -
base="$project/releases/20261016-brd4-uniform-5000-01"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/brd4_uniform_readiness.lsf"
bash -n "$base/source/scripts/hpc/brd4_contact.lsf"
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-uniform-readiness-01"
mkdir "$run"
bsub -J mosaic-brd4-uniform-ready -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_uniform_readiness.lsf" "$base/source" "$run" cefff19ac9b68d4b546427565b1036fc625c0cde42f22a5157116c930d8252fd
