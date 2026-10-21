# Separate combined JQ1-pocket 25-atom report

21 October 2026. The user explicitly requested leaving all three current GPU
experiments running unchanged and making a **new combined report when all are
done**. This extends the completed submission phase; it does not authorize a
scientific rerun, local benchmark, cancellation or objective/gate change.

Inputs are exactly Protenix **29770179**, Boltz-2 **29770093** and
Boltz-2 + chemistry **29770124**. Their original sources, configurations,
readiness assessments, frozen model/context identities and pocket definition
are pinned in `scripts/jq1_report_contract.py` and the submission receipts.
See `docs/brd4_jq1_25_atom_comparison.md` for the scientific scope and history.

## Authenticity and completion

Small allocated CPU completion collectors run after each native job ends. They
capture its actual final scheduler outcome and LSF success promptly, before
short live-record retention expires. They require the original exact job,
configuration, source file map, pocket/no-diffusion identity, model/context
digests, hard parity, initial/terminal raw derivative reports, original passed
readiness proofs, coupled check, native summary/case agreement, 10,000 contiguous
updates and 10,001 contiguous states with exact JSONL equality. Native counts
remain honest, including fallback, increases and no-ops. The chemistry arm
retains its readiness low/high coupled checks and final coefficient 50.

No failed or partial arm can become a complete comparison. Preserve failure
evidence, notify the user and stop report promotion if any native study fails;
do not omit that arm or rerun an experiment to obtain a nicer report. Report
engineering failures are separate and may be corrected with a new immutable
report-only release and allocated CPU checks.

## Saved-only viewer

The old chemistry report loader intentionally assumes six atom slots. The new
`render_jq1_25_report.py` is a separate family, preserving every earlier report.
It requires 25 by 6 atom probabilities and 300 by 5 unordered bond probabilities,
FP32/simplex/finite checks, exact snapshot hashes, saved scalar equality and
argmax/readout correspondence. It never invokes a model, molecular converter,
repair search, sampler or missing-probability inference.

All 30,003 frames are kept. Exact little-endian FP32 arrays are stored in chunks
of at most 100 frames, with SHA-256 for each binary/metadata asset. The small
HTML viewer loads at most three chunks, verifies their hashes and decodes the
saved values without quantization. This avoids a giant in-memory HTML payload.
Repeated pair indices are verified against canonical unordered-slot order and
stored once by the display convention; no chemical identity is changed.

The viewer provides three colors/selectors, initial/5000/final shortcuts,
playback, zoom, a stable 25-slot display grid, element/slot labels, probability
dependent atom sizes and bond widths, all raw bonds, invalid/disconnected graphs
and scrollable exact probability heatmaps. The grid is presentation geometry,
not a new predicted pose. Repaired targets never replace the current molecule.

Plot Protenix contact separately from Boltz affinity: their units/objectives
differ. Overlay the two Boltz affinity curves at their identical frozen context,
and show logged binder probability separately. Include the actual saved
chemistry coefficients, raw/weighted repair, within-update frozen-target loss
changes, fallback labels and saved sanitization/connectivity. Do not rank
changing repair totals as one fixed objective or infer binding/activity from a
saved valid graph. All scalar curves include every saved point and increases.

## Execution and delivery

Report source is sealed independently of both live scientific sources. Allocated
CPU fixtures test exact 25-slot packing, missing/modified/escaping data, scalar
and argmax mismatches, contiguity, full-budget/JSONL contracts, scheduler
retention failures, safe HTML embedding and legacy report compatibility. Local
work remains static AST/LF/JSON inspection, packaging/transfers and standard-
library byte/schema verification; no local numerical tests or model execution.

Planned report identity: `20261021-brd4-jq1-25-report-01`. Allocated CPU tests
and their final certificate precede reporting. Three completion collectors have
only native-ended dependencies. The report waits for all three collectors and
the CPU certificate to succeed; it does not affect GPU scheduling or settings.
Use a fresh quota gate before output/packaging. Produce the report/evidence in
`/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-report-01/combined-report/`.

After actual report scheduler DONE and LSF success, verify its passed assessment
and receipt, then download the exact archive and sibling `.tar.receipt.json` to
a fresh directory beneath `results/hpc/20261021-brd4-jq1-25-report/`. Check exact
remote path, byte length and SHA-256, safely extract and verify every manifest
file using `scripts/verify_hpc_evidence.py`. Confirm only the three native jobs,
10,001 frames each, N=25, all frozen identities and the original chemistry ramp.

Publish only fully verified report assets into this new family's `viewer/`,
HTML last. Use `scripts/serve_protenix_report.py` on a fresh available loopback
port, preserving all old servers. Verify HTTP 200 plus exact HTML/asset hashes.
Open the actual report and inspect all three selectors, playback, endpoints,
readable labels, sizes, all 300 pair rows, loss overlays and raw invalid graphs.
Notify the user with the verified link and concise saved-evidence findings.

The existing hourly monitor is reused, quiet while state is unchanged. It makes
one SSH attempt during an inaccessible follow-up and notifies only on meaningful
completion, failure or required action. Pause after verified report delivery or
an unresolved blocker. Authorization, code preparation and queued dependencies
are not numerical passes or completed trajectories. Actual release hashes,
submission IDs and outcomes are recorded below and in the experiment ledger.


## User-authorized interim editions, 21 October 2026

The user requested progress snapshots at update 1500 and every further 1000
updates. Preserve all three GPU jobs unchanged. First capture when any arm has
reached 1500, then refresh when any arm crosses 2500, 3500, ..., 9500 or 10000.
Track milestones per arm so a slower arm still triggers updates after a faster
one finishes. Poll every ten minutes; scheduler and report CPU queue delays mean
publication is shortly after a milestone, rather than synchronous with a GPU step.

`render_jq1_25_interim.py` is a separate incomplete-report path. It reads only
newline-committed contiguous JSONL prefixes and uses the common update/state
prefix if a producer appends between reads. An incomplete final line is excluded;
corrupt completed lines are failures. Retain prefix path/length/SHA, native job
scheduler evidence, original source/configuration/readiness and objective pins,
passed raw/coupled entry checks, pocket/no-diffusion identity and snapshot SHA.
Aged-out scheduler records require authenticated actual completion evidence.
No terminal native pass follows from an interim capture.

Include every scalar point through each arm's captured update, but transfer
exact probabilities only for state 0, every reached requested milestone and the
latest committed state. Preserve their original step numbers, raw invalid graphs
and lossless FP32 values. Queued/preparing arms remain explicit disabled entries;
never invent an initial molecule. Mark the entire edition IN PROGRESS, give each
arm's own included update, and state that terminal identity/derivative gates are
pending. The strict final renderer still requires all 10001 states per arm and
all original completion gates.

Report source01 CPU 29770499 and certificate 29770500 actually DONE: 56 fixtures
passed without failures/errors/skips. Inspecting actual live case records exposed
a report-only schema mismatch: case.initial_checks is a directions list, while
status and identity are in combined_initial_checks/report.json. Source02 checks
passed status and exact direction-list equality; it does not weaken a native gate.
The fixture now uses that actual schema. Preserve source01 tests as historical
fixture evidence and replace only verified-pending collectors/final report with
new source02 jobs after allocated report checks pass. No scientific source changes.

Interim editions use the same three-selector viewer with sparse snapshot indices
mapped to their actual saved steps. Exact milestone snapshots remain available in
later editions. Publish verified immutable edition folders beneath this NEW
family's viewer directory, then atomically replace the small latest-report page.
Preserve every older edition and server. Final full-trajectory edition uses the
same stable link. Only allocated HPC performs rendering and report fixture tests.


## Actual source03 and first published edition

Source03 authenticated 513 immutable files at
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-report-03/source`.
Manifest `3b1ba56f8ddd0dc163333d9dabccd1119e323553a160a0e7b6b41790e73d7252`;
archive 39,510,050 bytes SHA
`3434ab4cabbac75441adfbf8be7d50fdbc5381d0740bf13c0ce92244076891e0`.
Source02's new aged-out-record contract failed (68 passed/one failed) and is
preserved. Source03 fixes only report status routing and completion-proof filename
handling. CPU 29770614 and certificate 29770615 actually DONE: **70 passed**,
no failures/errors/skips. Raw XML, scheduler/LSF and all copied CPU proof hashes
were verified locally in the first edition's verification record.

Current final completion collectors are **29770616 / 29770617 / 29770618** for
Protenix/Boltz/Boltz+chemistry, and strict full report **29770619**. Their root is
`/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-report-03/`.
Original source01/source02 report-only collectors/final/interim dependencies were
retired only while verified PEND, before promotion; actual EXIT receipts are saved.
Source02 certificate 29770599 EXIT is failed-test propagation. None is a native
scientific failure. No native GPU job was changed or resubmitted.

First interim **29770620** is actually DONE/passed; included updates are
**2275 / 1133 / 996**. Exact snapshot steps: [0,1500,2275], [0,1133], [0,996].
All scalar points through those prefixes are retained. The 197,826-byte archive
SHA is `e82677ec2d1135e450707ff5b8e47e1e47e6f77594865447142b7661748123e2`.
All 36 evidence files were safely extracted/byte verified at
`results/hpc/20261021-brd4-jq1-25-report/edition-0001/verification.json`.
Source/input/CPU proof identities and actual final scheduler/LSF success were
checked separately; archive packaging's in-job scheduler RUN was not relabelled.

Published immutable edition is `viewer/edition-0001/`. Stable link:
**http://127.0.0.1:8776/report.html**. Detached server is rooted at this new report
family's viewer, preserving older servers. The small latest page checks its
atomic latest.json pointer every minute. `publish_jq1_report.py` is local stdlib
publication metadata and copied-byte verification only, with all scientific
plots/probabilities already rendered on allocated HPC. Future editions use new
immutable folders; complete final edition replaces only the latest pointer.

HTTP200 plus every served file's exact byte digest passed. Actual browser QA
covered all three selectors/colors, initial/latest endpoints, Protenix1500,
playback, readable element/slot labels, probability-dependent sizes and exact
25-row/300-pair heatmaps. Saved graphs displayed so far remain sanitized carbon
chains; invalid/disconnected fixture preservation passed on allocated CPU.
Terminal native checks and complete budgets remain pending. The existing monitor
now checks every ten minutes and publishes only on new per-arm milestones,
quiet when nothing actionable changes. Every new published edition is meaningful
progress; final completion still requires all three native studies to pass.
