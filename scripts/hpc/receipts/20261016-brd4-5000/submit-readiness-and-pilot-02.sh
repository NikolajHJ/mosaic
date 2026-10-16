set -euo pipefail
project=/work3/s234818/mosaic
archive="$project/brd4-5000-20261016-02.tar.gz"
echo 'cd14cc6377e8422a8abb12b639f8c9a34edf989fe097332e958bf60402b5e2c5  /work3/s234818/mosaic/brd4-5000-20261016-02.tar.gz' | sha256sum -c -
base="$project/releases/20261016-brd4-5000-02"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/brd4_readiness.lsf"
bash -n "$base/source/scripts/hpc/brd4_contact.lsf"
root=/zhome/95/e/206121/mosaic-runs
mkdir -p "$root"
run="$root/20261016-brd4-readiness-02"
mkdir "$run"
response=$(bsub -J mosaic-brd4-readiness-home -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_readiness.lsf" "$base/source" "$run" 80bb2413904b955cc4ebc25a0c01e1940fd185b3bddfb6cbd3afed76ad75a4f2)
echo "$response"
pattern='Job <([0-9]+)>'
[[ $response =~ $pattern ]]
cpu=${BASH_REMATCH[1]}
pilot="$root/20261016-brd4-pilot-01"
mkdir "$pilot"
bsub -J mosaic-brd4-pilot -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' -W 01:00 -w "done($cpu)" -oo "$pilot/stdout.%J.log" -eo "$pilot/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_contact.lsf" "$base/source" "$pilot" 80bb2413904b955cc4ebc25a0c01e1940fd185b3bddfb6cbd3afed76ad75a4f2 examples/configs/brd4_pilot.json "$run/payload/prepared/fixture.json"
