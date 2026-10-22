# BRD4 best-primal continuation — 17 October 2026

> **Scope and access note (22 October 2026):** this document contains an
> experiment protocol and dated operator evidence. Concrete account paths,
> environments, jobs and source hashes describe that operator/attempt; they
> are not shared-account defaults or a fresh status/acceptance claim. Use
> [your own HPC setup](HPC_GUIDE.md). Keep the scientific gates and historical
> outcomes below intact when preparing a separately validated candidate.
> Canonical working home: [brd4/best-primal-continuation](../experiments/brd4/best-primal-continuation/README.md).


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
- [x] Authenticate final native/report outcomes and all six saved study identities.
- [x] Download the report and verify its 13 files against the evidence manifest.
- [x] Finish the complete archive's extracted-file integrity pass (186,506 files).
- [ ] Visually inspect playback in a browser; automated inspection is unavailable
  after the browser transport disconnected while loading the large HTML.

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

## Independent chain jobs — user-directed scheduling update

The user requested separate 15%/30%/50% jobs so they can be scheduled in
parallel. On the 17 October 13:16 Europe/Paris check, chain 29745472 and report
29745475 were still PEND, with no allocated worker or new chain updates.
Both cancellations were accepted and their final EXIT states authenticated;
this is intentional, not a numerical failure. Uniform 29745473, noisy 29745474,
completed canonical 29736988 and every historical artifact remain unchanged.
The completion monitor was paused during replacement and is now active with
the independent jobs below.

The new `--case-index` runner option projects the authenticated chain plan onto
one original case and records its original index. Each job has its own context
verification, output directory and completion/failure record. The 15% job
retains its 4998-update prefix; 30% and 50% start fresh. There are no dependencies
between these three GPU jobs. Simultaneous execution depends on scheduler GPU
availability; independent jobs remove the sequential dependency.

Readiness requires the full allocated CPU engineering suite, exact equality of
all `src/` hashes with passed source 04, the same pinned continuation plan and
parent files, and the 20 GiB aggregate quota gate. The existing budget already
reserved nine cases' worth of native artifacts for the five incomplete cases;
splitting their allocations does not add another optimization trajectory.
The replacement report will consume six independent native input studies
(three chain cases, uniform, noisy and canonical) and retain six selectors and
the same six 5001-state curves. Original case indices stay in its receipts.

- [x] Implement independent case selection and test its recipe/prefix mapping.
- [x] Pass allocated CPU source/parent/selection/quota readiness.
- [x] Submit three independent GPU jobs and the replacement combined report.
- [x] Update monitor, ledger and versioned scripts with actual job receipts.

Immutable parallel source 01 contains 293 files. Archive SHA-256
`b0a35eed62c51b53285607d21a0c04c1f541c3b0bd7af888392da43c2ddcc76f`;
manifest `1cf635c222bf750c298c7b4bcac30ec36eeb47e40f74480466d813cd64041504`.
Remote source:
`/work3/s234818/mosaic/releases/20261017-brd4-parallel-01/source`.
Allocated CPU readiness **29745498 DONE**, worker `n-62-29-16`: **205 passed**,
three native items deselected, two existing dtype warnings, 78.92 seconds.
Exact numerical-library identity and all 12,075 pinned parent case files
passed; the same plan SHA above is retained. The 20 GiB aggregate quota gate
passed with 22 GiB conservatively available (quota can lag; not a reservation).
Assessment SHA-256:
`9a38ff0ea26e9ae3e03dc479696db5e05d1dc49c0377b732f983a8b58c7929b8`.

| Initialization | Independent native job | New run under `/zhome/95/e/206121/mosaic-runs/` |
| --- | --- | --- |
| Chain 15%, resume at 4998 | **29745501** | `20261017-brd4-chain15-best-primal-01` |
| Chain 30%, fresh | **29745502** | `20261017-brd4-chain30-best-primal-01` |
| Chain 50%, fresh | **29745503** | `20261017-brd4-chain50-best-primal-01` |

All three were PEND at authenticated submission. Each requests one exclusive
A10 in `gpua10`, four colocated CPU slots/32GB total host RAM and only
`done(29745498)`. The 15% job has a one-hour bound for model restore, replay
and its two remaining updates; 30% and 50% retain the 16-hour bound. Each
targets **5000 total updates**, with unchanged LR=0.03 and fallback/native
gates. The selected original source indices are respectively 0, 1 and 2.
Unchanged uniform/noisy jobs 29745473/29745474 still depend on passed readiness
29745466 and remain independently queued.

Replacement report/evidence **29745504 submitted**, hpc two CPU slots/8GB
total host RAM, three hours. It waits for readiness 29745498 DONE and all five
current native jobs to end. It still requires every input to pass complete
native checks before rendering. Output:
`20261017-brd4-best-primal-report-02/payload/report/`; archive
`20261017-brd4-best-primal-report-02.tar.gz` and receipt `.tar.receipt.json`.
The active hourly monitor follows these IDs and retains completed canonical
29736988. Native outcomes and final report delivery remain pending.
Exact submission operators and local authenticated readiness/scheduler
receipts are recorded in the experiment ledger.

## Progress check — 2026-10-17, approximately 21:03 Europe/Paris

Chain 15% **29745501** completed successfully: persisted native status passed,
5001 saved states through update 5000, source index 0, final independent-primal
loss -0.4594642818. Uniform **29745473** also completed successfully: passed,
5001 saved states through update 5000, final independent-primal loss
-0.5648135543. Both persisted summaries retain the expected identical
before/after frozen model and context digests. LSF completion records confirm
successful execution; the uniform job has aged out of `bjobs`, so its saved
LSF stdout completion record was used. Full evidence download and snapshot
integrity verification remain part of final report delivery.

Chain 30% **29745502** is RUN, approximately 2520/5000 saved updates; noisy
uniform **29745474** is RUN, approximately 4754/5000. Chain 50% **29745503**
is PEND and report **29745504** is PEND awaiting its inputs. These live counts
are snapshots, not completed-case certificates. No experiments were changed.

At the subsequent approximately 21:38 Europe/Paris check, noisy uniform
**29745474** is DONE with native status passed and 5001 saved states. Chain
30% is RUN at approximately 4418/5000; chain 50% and the report remain PEND.
The ledger records provisional runtime estimates separately from outcomes.

At approximately 23:00 Europe/Paris, chain 30% **29745502** is confirmed
complete: saved LSF successful completion at 21:49:27, native status passed,
5001 states through update 5000, source index [1], expected frozen before/after
identities. Chain 50% **29745503** is still PEND with scheduler reason
`Job slot limit reached: 1 host`; report **29745504** is still PEND. The 50%
case is the only unfinished native trajectory; no complete combined report
exists yet. All job settings remain unchanged.

## Final native and report completion — 2026-10-18

The 50% case **29745503** completed at 01:25:10 Europe/Paris: native checks
passed, source index [2], 5001 states through update 5000, frozen model/context
before/after identities unchanged. Final independent-primal loss -0.5394011736.
All six requested native trajectories have now reached their 5000-update
budgets. This is exploratory full-budget completion under their recorded
strict/best-primal policy segments; it does not retroactively complete the
failed strict parents or establish monotone descent/chemical validity/binding.

Full report/evidence **29745504** successfully completed at 02:18:20
Europe/Paris. Its archive receipt and local delivery verification are recorded
in the ledger. The earlier interim report remains a separate saved artifact.

The all-six HTML is downloaded and served at
`http://127.0.0.1:8771/report.html`. All six input IDs, 5001 contiguous states
per case, six atom slots, original chain source indices and frozen model/context
identities passed local stdlib saved-file checks. The report's 13 files and all
186,506 archived evidence files match the authenticated archive manifest.
The final verification receipt is recorded in the ledger. The 417,532,320-byte HTML may take time to load;
HTTP availability is confirmed, while visual/playback QA remains unconfirmed.
