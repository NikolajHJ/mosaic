set -euo pipefail
project=/work3/s234818/mosaic
archive="$project/brd4-noisy-uniform-5000-20261016-01.tar.gz"
echo '4d36c3d8d0bb945553eca3e9adf5c53fce754bdf832c896c09866030e51cde21  /work3/s234818/mosaic/brd4-noisy-uniform-5000-20261016-01.tar.gz' | sha256sum -c -
base="$project/releases/20261016-brd4-noisy-uniform-5000-01"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/brd4_uniform_readiness.lsf"
bash -n "$base/source/scripts/hpc/brd4_contact.lsf"
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-noisy-uniform-readiness-01"
mkdir "$run"
bsub -J mosaic-brd4-noisy-uniform-ready -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_uniform_readiness.lsf" "$base/source" "$run" ea4ceab78cb7605fdd715cb79bebacefc03ec8c8a5947834fa58540d5028c02e examples/configs/brd4_noisy_uniform_5000.json 2 14
