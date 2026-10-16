set -euo pipefail
root=/zhome/95/e/206121/mosaic-runs
base=/work3/s234818/mosaic/releases/20261016-brd4-5000-02
pilot="$root/20261016-brd4-pilot-01"
assessment="$root/20261016-brd4-assessment-01/payload/assessment/assessment.json"
echo "cfc44f1e6619a0d0e97bdb761c006be76b4bbad0547dd3cd74656a504ac93cbd  $assessment" | sha256sum -c -
echo "7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4  $pilot/payload/study/summary.json" | sha256sum -c -
bjobs -l 29736575 > "$root/20261016-brd4-readiness-02/job-final.txt"
bjobs -l 29736576 > "$pilot/job-final.txt"
bjobs -l 29736579 > "$root/20261016-brd4-assessment-01/job-final.txt"
run="$root/20261016-brd4-5000-01"
mkdir "$run"
bsub -J mosaic-brd4-5000-lr003 -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -R 'select[hname!="n-62-18-5"]' -gpu 'num=1:mode=exclusive_process' -W 16:00 -w 'done(29736579)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_contact.lsf" "$base/source" "$run" 80bb2413904b955cc4ebc25a0c01e1940fd185b3bddfb6cbd3afed76ad75a4f2 examples/configs/brd4_5000.json "$base/source/tests/data/brd4_3mxf/fixture.json" "$pilot/payload/study" 7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4 "$assessment" cfc44f1e6619a0d0e97bdb761c006be76b4bbad0547dd3cd74656a504ac93cbd
