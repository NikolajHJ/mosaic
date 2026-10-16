set -euo pipefail
root=/zhome/95/e/206121/mosaic-runs
base=/work3/s234818/mosaic/releases/20261016-brd4-canonical-diffuse-01
pilot="$root/20261016-brd4-pilot-01/payload/study"
readiness="$root/20261016-brd4-canonical-diffuse-readiness-01"
assessment="$readiness/payload/assessment/assessment.json"
[[ "$(bjobs -noheader -o stat 29736979 | tr -d '[:space:]')" == DONE ]]
[[ "$(bjobs -noheader -o stat 29736911 | tr -d '[:space:]')" == PEND ]]
echo "4ad7e7f04c740815d248ee405f3979f4ba648f2cdaca872d48d290a387c53821  $assessment" | sha256sum -c -
echo "7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4  $pilot/summary.json" | sha256sum -c -
bjobs -l 29736979 > "$readiness/job-final.txt"
run="$root/20261016-brd4-canonical-diffuse-5000-01"
mkdir "$run"
response=$(bsub -J mosaic-brd4-canonical-5000 -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' -W 16:00 -w 'done(29736979)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_contact.lsf" "$base/source" "$run" 421213580f84a038cde5e47bb4048a659bfa4daee46645deccbb2e7568d7ead4 examples/configs/brd4_canonical_diffuse_5000.json "$base/source/tests/data/brd4_3mxf/fixture.json" "$pilot" 7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4 "$assessment" 4ad7e7f04c740815d248ee405f3979f4ba648f2cdaca872d48d290a387c53821 1)
echo "$response"
pattern='Job <([0-9]+)>'
[[ $response =~ $pattern ]]
native=${BASH_REMATCH[1]}
report="$root/20261016-brd4-combined-report-02"
mkdir "$report"
bsub -J mosaic-brd4-combined-four -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 03:00 -w "done(29736979) && ended(29736629) && ended(29736735) && ended(29736761) && ended($native)" -oo "$report/stdout.%J.log" -eo "$report/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_combined_report.lsf" "$base/source" "$report" 421213580f84a038cde5e47bb4048a659bfa4daee46645deccbb2e7568d7ead4 20261016-brd4-canonical-diffuse-5000-01
# Replace only the still-pending, superseded three-study report. No GPU or
# individual report job is cancelled. Preserve actual asynchronous receipts.
bkill 29736911
state=unknown
for attempt in $(seq 1 15); do
  state=$(bjobs -noheader -o stat 29736911 | tr -d '[:space:]')
  [[ "$state" == EXIT ]] && break
  sleep 2
done
[[ "$state" == EXIT ]]
bjobs -l 29736911 > "$report/superseded-report-29736911.txt"
