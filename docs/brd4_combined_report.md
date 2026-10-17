# Combined BRD4 5000-update report

Requested 2026-10-16; expanded after the section 4.2 request. The current
final report is job **29736989**, report-02, comparing four separately
submitted studies and six trajectories. The earlier three-study report job
29736911 was intentionally cancelled while pending; dated receipts below
remain historical evidence. Current inputs:

- Original carbon-chain mixtures: 15%, 30%, 50%; GPU job 29736629.
- Exact uniform: one case; GPU job 29736735.
- Half uniform plus half simplex-projected Gaussian: one case; GPU job 29736761.
- Section 4.2 carbon-heavy diffuse prior with small additive noise, N=6; GPU 29736988.

Every case retains all 5001 saved states, raw argmax molecules (including
invalid/disconnected outputs), fixed readable element/slot labels and sizes
based on saved probabilities. Loss and change plots overlay all six cases
with six distinct colors; the explorer selects a case and accepted update.
Show initial/final readouts and probability heatmaps for each case, as well
as endpoint and intermediate molecule galleries. This compares starts at
fixed target/model/loss/seed/rate/budget; it is not replicated chemistry or
binding validation. No molecule repair/filtering or model rerun is allowed.

## Comparison and execution gates

An explicit renderer option permits initialization differences only. All
other configuration, fixture/scaffold, frozen model/context and numerical
profile identities must match. Each study must have passed, the frozen
digests must be unchanged, each saved recipe must match its config, and all
5000 updates must be present. Duplicate recipes are rejected. Preserve each
input summary path/hash, config hash and native job ID; a derived combined
display config is not assigned a fictitious native configuration hash.

Allocated CPU tests cover the five-case mapping, source receipts, unchanged
originals, explicit comparison authorization and rejection of seed/rate,
context, failed study, duplicate recipe and saved-recipe mismatch. An end-to-end
render test uses visibly synthetic saved graphs and fabricated losses; it
checks report construction only and is not native or chemistry evidence.
All tests/rendering stay on allocated DTU HPC. Local work is source editing,
static checks, immutable packaging, file transfer and byte-integrity checks.

The initial readiness gate requests 17 GiB free on home: the previous three
study budgets (about 13.9725 GiB with 4 GiB reserve) plus about 3 GiB for the
combined report/archive. The combined CPU job rechecks at least 8 GiB and
archive packaging checks actual bytes plus reserve. Use unique run/release
paths and stop on quota or numerical failure. Existing native and individual
report jobs remain unchanged. A failed/incomplete study must not be silently
omitted or repaired; preserve its evidence and report the blocker instead of
calling a subset the complete three-study comparison.

Queue the report after all three native jobs end and new CPU readiness passes.
The renderer requires all three complete/passed studies before producing the
combined report. Archive the three native run directories plus the combined
report payload, including a per-file byte/hash manifest and scheduler receipts.
A quiet follow-up in this chat will check completion, download the archive,
verify bytes locally with `verify_hpc_evidence.py`, inspect the report in the
browser, record actual outcomes and deliver the local report link. Completion
of submission alone does not establish any trajectory, finished report or
download. Pause the follow-up after successful delivery or an actionable failure.


Combined-report source 01: 266 files, archive 38,981,862 bytes, SHA
`e5b047d437da7d1bfb9c23d55d5b20be6c12f147289188132aca62daa9b6b45c`,
manifest `40f9864f21f9c03c811492a15e5a7e2b3dee9e2f6b597ec3909f343dbeaf2075`.
Git reference fe66025fc6a0a598a2b90688cfd9add372763e71 plus manifested dirty
bytes. Remote immutable source
`/work3/s234818/mosaic/releases/20261016-brd4-combined-report-01/source`;
local `results/releases/brd4-combined-report-20261016-01/`.
Uploaded archive SHA and both Bash launcher syntax checks passed. Local
static Ruff and bundled Node `--check` passed; no local numerical work.

CPU report readiness **29736903 submitted**, hpc two CPU slots/4GB per slot,
15-minute bound, run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-comparison-readiness-01`.
Exact operator `scripts/hpc/receipts/20261016-brd4-combined-report/submit-readiness-01.sh`;
local receipt `results/hpc/20261016-brd4-combined-report/submit-readiness-01.txt`.
It checks the sealed source, at least 17 GiB free home space and all 13 report
loader/guard/synthetic-render tests. Submitted does not establish a passed
readiness check or actual combined BRD4 report.


CPU comparison readiness **29736903 DONE**, worker n-62-30-25: **13 passed
in 38.87s**. This includes a full five-case synthetic HTML/plot render;
its fabricated losses/graphs are engineering fixtures, not native outcomes.
The 17 GiB home gate passed with 25 GiB conservatively available.
Downloaded logs/quota/final scheduler receipt under
`results/hpc/20261016-brd4-combined-report/`.

Combined report/evidence **29736911 submitted**, hpc two CPU slots/4GB per
slot (8GB total), 150-minute bound. Dependency:
`done(29736903) && ended(29736629) && ended(29736735) && ended(29736761)`.
Run `/zhome/95/e/206121/mosaic-runs/20261016-brd4-combined-report-01`.
Uses exactly tested immutable combined-report source 01 and the manifest
above. Requires all three complete passed studies; overlays five cases and
retains every saved state without model reruns. Rechecks free home space,
then archives all three native run directories and the combined report payload
with per-file hashes and scheduler receipts. Archive:
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-combined-report-01.tar.gz`;
receipt has suffix `.tar.receipt.json`. Local exact operator
`scripts/hpc/receipts/20261016-brd4-combined-report/submit-combined-report-01.sh`.
Submission receipt and final compact state snapshot are in the local campaign
directory. All three native jobs and combined report were PEND; earlier
individual report jobs were not modified. Actual BRD4 comparison rendering,
archive, download verification and browser QA remain pending.

Created active hourly in-chat completion heartbeat
`finish-combined-brd4-trajectory-report`, named "Finish combined BRD4 trajectory report".
It makes a compact scheduler check, stays quiet on unchanged pending/running
state, downloads and verifies the completed archive locally, checks all three
source jobs/five 5001-state cases and actual report presentation, records and
pushes evidence and delivers the report. No GPU experiment changes or local
numerical work are permitted. Failed native/identity gates remain blockers.
The follow-up pauses after delivery or an actionable blocker. This scheduling
receipt is not completed report/download evidence.


## Current four-study replacement (2026-10-16)

[The canonical protocol](brd4_canonical_diffuse_5000.md) and ledger record the
requested six-slot section 4.2 extension. CPU readiness 29736979 passed all
186 selected checks. Native 29736988 is separate and preserves all prior
GPU experiments. Combined report/evidence 29736989 waits for all four native
runs and requires their passed complete trajectories; output is
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-combined-report-02/payload/report/`.
The old pending combined job 29736911 was intentionally cancelled after the
replacement queued, with EXIT confirmed; this is not a scientific failure.
The existing hourly follow-up now downloads/verifies report-02 and checks
four input studies, six cases, 5001 states each and fixed N=6. All current
GPU runs and the replacement report remain pending; no final report exists
yet. The earlier source/job receipts above are retained without alteration.

## Authenticated outcomes — 2026-10-17

SSH access was restored for the user-requested check at 12:42 Europe/Paris.
The four GPU allocations have ended. Saved LSF completion records and native
reports establish the following outcomes; the earlier pending snapshots above
are historical.

| Study / native job | Accepted updates | Outcome |
| --- | ---: | --- |
| Chain mixtures / 29736629 | 4998 of 5000 for the 15% case | Failed descent acceptance; the 30% and 50% cases were not reached |
| Uniform / 29736735 | 3208 of 5000 | Failed descent acceptance |
| Half uniform plus projected noise / 29736761 | 3818 of 5000 | Failed descent acceptance |
| Section 4.2 canonical diffuse / 29736988 | 5000 of 5000 | Native report passed; 5001 saved states, six atom slots |

The three failed runs record `AssertionError: No measured loss-decreasing
projected step`. The existing acceptance rule requires both autodiff and
independent primal loss to decrease by more than 1e-7 within ten halving
backtracks. Exhausting this rule is a failed run under the sealed protocol,
not a completed 5000-update study or a convergence certificate. Invalid raw
graphs are retained; chemical invalidity was not the stopping condition.

Combined report job **29736989 EXIT, code 1**, at 12:42:14. Its input gate
correctly rejected the failed studies with `Render completed authenticated
trajectories only`. This is a downstream native-completeness blocker, not a
report-only engineering defect. No combined HTML, archive or receipt was
produced, and no successful six-case comparison can be delivered. Do not omit
the failed/unreached cases or weaken the native gate to create one.

Downloaded and byte-verified status evidence is under
`results/hpc/20261016-brd4-combined-report/status-20261017T104405Z/`.
Full raw snapshots and trial evidence remain in the original HPC run
directories. The canonical study's final model/context digests match their
initial identities; the failed studies have no completed final digest check.
No model job or individual report job was changed or resubmitted. The hourly
completion heartbeat was **paused** on this native failure, as instructed.

## Authorized continuation superseding the completion blocker

The user subsequently authorized best-primal fallback after exhausted strict
search and resumption from the saved molecules. See
[the continuation protocol](brd4_best_primal_continuation.md). Allocated CPU
readiness 29745466 passed 198 engineering tests and authenticated the parent
prefixes/contexts. New native jobs 29745472/29745473/29745474 and dependent
six-case report 29745475 were submitted, targeting 5000 total updates per case
and reusing completed canonical 29736988. They are queued, not completed.
The new report will identify strict prefixes, fallback segments and loss
increases explicitly. Historical failures above remain unchanged. The hourly
completion monitor now follows this new campaign; no final report exists yet.
