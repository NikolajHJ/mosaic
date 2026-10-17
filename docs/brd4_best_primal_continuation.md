# BRD4 best-primal continuation — 17 October 2026

The user authorized resubmission with a changed optimization policy and resume
from the last saved molecule. Historical strict runs and their failed outcomes
remain unchanged. Implementation and CPU/GPU execution have separate receipts
in the experiment ledger; this plan is not evidence of submission or a pass.

## Decisions and scope

- **Budget:** 5000 total molecular updates per case, including the saved prefix,
  following the existing 5000-update campaign. These are graph optimization
  updates, not epochs of model training. The canonical diffuse case already
  passed all 5000 and is reused unchanged.
- **Selection:** try the existing strict dual-loss decrease rule, starting at
  LR=0.03 with ten halving candidates. If all ten fail, accept the candidate
  with the lowest independent standalone-primal loss. Ties select the first
  evaluated candidate. If every candidate increases loss, take the smallest
  increase. The current molecule is not an eleventh candidate.
- **Retained checks:** finite graph, loss, gradients and native outputs are
  mandatory. The fallback does not accept NaN/Inf or bypass native model/context
  identities. A completed exploratory budget does not certify monotonic
  descent, convergence, chemical validity or binding.
- **Resume:** restore the exact saved X/E probabilities at chain 15% update
  4998, uniform 3208 and noisy-uniform 3818; retain seed-17's actual saved key,
  six slots and the frozen BRD4 context. This optimizer has no momentum or
  accumulated state. Replay the resumed graph's AD/primal loss and gradients
  against its checkpoint before updating. Replay tolerances: scalar absolute
  2e-5, gradient absolute 2e-5/relative 1e-4. Record replay values explicitly.
- **Native derivative evidence:** inherit the authenticated initial FD/null/
  parity checks for resumed prefixes. Do not perturb a boundary molecule into
  the simplex interior to repeat them. New chain 30% and 50% cases start from
  their original recipes and pass all initial native checks themselves.
- **Lineage:** a new allocated CPU readiness plan pins parent summary, failed
  case report, initial RNG artifact, initial probe artifacts and every prefix
  snapshot by SHA-256. Resume eligibility is limited to the recorded strict
  descent-stall failure. Copy saved prefix snapshots to unique new run paths;
  retain the failed parent report and separate strict/fallback policy segments.
  Failed parents did not reach their final frozen-identity check. New entry
  replay and before/after checks cover the continuation; they do not create a
  retroactive completed-parent certificate.
- **Report:** six complete 5001-state trajectories are still required. Preserve
  old failed runs in evidence. Allow the declared strict/canonical versus
  exploratory continuation policy comparison explicitly, while keeping all
  model/context/fixture/seed/rate/budget guards. Plot actual loss increases and
  continuation boundaries. Every graph is its raw, unrepaired saved output.
  Policy exposure differs between saved prefixes and fresh cases, so this is
  an exploratory comparison, not an isolated initialization-effect study.

## Execution checklist

- [x] Implement explicit fallback and authenticated continuation tooling.
- [x] Author CPU selection, nonfinite, resume/identity and report-policy tests.
- [x] Seal/upload immutable source; pass allocated CPU tests, parent/source
  identity and 20 GiB aggregate free-home-space readiness.
- [x] Submit three unique GPU studies with a passed readiness dependency and
  fresh per-job quota/source/parent checks. Leave historical jobs unchanged.
- [x] Submit the dependent allocated CPU six-case report/evidence job.
- [ ] Authenticate final native/report outcomes, download/verify and browser-QA
  the actual report before delivery.

The aggregate 20 GiB readiness bound conservatively reserves three full prior
three-case native/report artifact budgets (about 3.324 GiB each), an additional
3 GiB combined/ancestry archive allowance and at least 4 GiB headroom. No HPC
data is deleted. GPU entries recheck 8 GiB free space; report packaging checks
actual bytes and retains its reserve. All tests/rendering/numerical execution
remain on allocated DTU HPC; local work is editing, static review and packaging.

## Readiness attempts

Source 01 / CPU 29745455 failed one JSON-container comparison in the engineering
test fixture (195 passed); its evidence remains intact. Source 02 was sealed
locally only and was not uploaded or submitted. Source 03 adds explicit runtime
numerical-profile authentication and contains the corrected persisted-prefix
test, copied plan provenance and per-frame fallback/increase labels.
Source 03 / CPU **29745459 EXIT** after 196 passed tests: the plan builder
incorrectly demanded equality of re-serialized context container hashes.
Source 04 authenticates each container's own hashes, then requires exact decoded
array values/dtypes and static metadata equality, on both CPU and GPU entry.
No changed feature is accepted. Two added context regressions passed.

Source 04 / CPU **29745466 DONE**, `n-62-11-50`: **198 passed**, three native
items deselected, two existing dtype warnings, 65.27s. Source/parent/context/
prefix checks passed. The 20 GiB storage gate passed with 22 GiB conservatively
available. Plan SHA-256:
`4837faf4a35e49927c8fe828b65dfb91ee871da9028ade9db27e6bfb9db1ade1`.
The plan pins 12,075 case files, including all 12,027 prefix snapshots.

## Submitted continuation campaign

| Study | Native job | New run under `/zhome/95/e/206121/mosaic-runs/` |
| --- | --- | --- |
| Chain 15% resumed at 4998; chain 30%/50% fresh | 29745472 | `20261017-brd4-chain-best-primal-01` |
| Uniform resumed at 3208 | 29745473 | `20261017-brd4-uniform-best-primal-01` |
| Noisy uniform resumed at 3818 | 29745474 | `20261017-brd4-noisy-best-primal-01` |

All three request one exclusive A10, four CPU slots/32GB total host RAM and
16 hours, with dependency `done(29745466)`. Each rechecks source, quota, pinned
parent files, actual numerical profile, frozen identities and native replay.
At submission all were **PEND**; native completion is not yet established.

CPU combined report/evidence **29745475 submitted**, two CPU slots/8GB total
host RAM, three hours, waiting for readiness DONE and all three new native jobs
to end. It combines these continuations with unchanged canonical **29736988**.
Output `20261017-brd4-best-primal-report-01/payload/report/`, archive
`20261017-brd4-best-primal-report-01.tar.gz`, receipt `.tar.receipt.json`.
All paths are under the same home Mosaic run root. The existing hourly
completion heartbeat is active with the new IDs, acceptance semantics and
verification instructions. Native/report outcomes, download and browser QA
remain pending. Exact successful operator:
`scripts/hpc/receipts/20261017-brd4-best-primal/submit-native-and-report-02.sh`.
