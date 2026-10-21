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


## User-reported timeline issue and report-only correction

On 21 October the user reported being unable to slide through changes over time.
Edition0001's progress timeline contained only 2 or 3 frames per arm, not every
saved update. Snapshot jumps and unchanged hard carbon-chain labels made it
appear unresponsive. Replace that interim sampling policy with every committed
state through each arm's captured prefix, still exact saved probabilities in
bounded100-frame chunks. Publication remains at the authorized1000-update
milestones; this changes presentation sampling only, not GPU capture or science.
Keep edition0001 immutable. Final full-budget and native gates remain unchanged.

Add an exact-update input and previous/next saved-state buttons, pause playback
when scrubbing, show the selected update and available frame count immediately,
label the interim endpoint Latest captured and disable the5000 shortcut until it
exists. The full committed prefix is checked by the same final-frame writer;
new allocated regression covers intermediate frames across a chunk boundary.
No interpolation or invented intermediate molecules/probabilities is permitted.
Scientific rendering and fixtures still run on allocatedCPU; local byte/schema
checks, static editing and actual browser drag/keyboard presentation QA only.
Seal a new immutable report-only release, retain all historical passes/failures,
and leave current native jobs29770179/29770093/29770124 unchanged.


### Actual source04 timeline fix and edition0002

Report-only source04 authenticated518files at
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-report-04/source`, manifest
`5c3f123748f573dcb5e3d4532659441fc6f5666a1768461a459c94cf208ba31a`;
archive39,518,412bytes SHA
`d1f13f8c440948cb003e3185d3a8daaf3b957985941799d2248b2bce70e57601`.
CPU **29770704** and certificate **29770705** actually DONE: **71passed**,
no failures/errors/skips in36.83s. CPU assessment SHA
`a2654b958374b19cb9f42113032ad3a3a4776c38a88fb387d6409da37ce872d0`.
New completion collectors **29770706 / 29770707 / 29770708** and final report
**29770709** replace source03's verified-pending report-only predecessors after
source04CPU/certificate success. Actual intentional EXIT receipts are preserved.
All scientific GPU jobs/configuration/settings remain unchanged.

New interim **29770710** actually DONE/LSFsuccess and assessmentpassed; report
assessment SHA `f33a28fdfc86966262791fc7b5be4803c38ab2dddf223be87ee8fb1f47dc3d33`.
It includes every saved update0..3634 /0..1227 /0..1148, **3635/1228/1149frames**
(**6012total**), captured21October13:34:09Paris. No interpolation or resampling.
Archive24,677,266bytes SHA
`af2ae8c68c44329211bbf1ed676920cfe3dfa885103155f5690509f9d4a15831`.
All154files safe-extracted/byte-verified, exact CPU XML/proofs, input/source
identities, per-arm frame contiguity/counts/chunk sizes and actual final scheduler
were authenticated locally using stdlib only. Verification is at
`results/hpc/20261021-brd4-jq1-25-report/edition-0002/verification.json`.

Published immutable `viewer/edition-0002/` at the same stable8776link. Every
served asset returned HTTP200/exact bytes. Real browser mouse dragging selected
Protenix1860then658 across multiple chunks; exact numeric1500 worked. Boltz600,
chemistry777→778, playback advancement and manual-scrub pause at500 all worked,
with corresponding saved scalar/readout updates. Screenshot `slider-browser-QA.png`
preserves visible controls. Native hard argmax can remain a carbon chain while
its probabilities and score change. The ten-minute milestone monitor now uses
source04/every-frame progress captures, preserving earlier editions and final
full-budget/native/identity gates.

### Saved-gradient and effective-step audit, 21 October

The user asked whether the apparently static molecules were receiving gradients
and changing. Read-only allocated CPU **29770931** actually DONE with LSF success
at14:08:05Paris. It analyzed the fixed interim-0002 prefixes, not later live data:
Protenix5900, Boltz1409 and Boltz+chemistry1314 updates. No inference, new gradient
calculation, repair, optimization or native setting changed. The script and
campaign are in `scripts/hpc/receipts/20261021-brd4-jq1-gradient-audit-01/`.

All three retained nonzero saved atom/bond gradients and probability changes.
For the last100 updates the median accepted logit step was **0.03 /0.001875
/0.0009375**: Protenix always accepted the nominal rate, while Boltz and chemistry
were typically16x/32x below it through ordinary backtracking. Last100 median
combined logit-gradient L2 norms were0.006517/0.302833/0.389493. Mean absolute
probability changes per update were about0.8e-6 to1.7e-6 across the atom and bond
arrays. Saved logit updates replayed with maximum residual about1.2e-7, consistent
with FP32 rounding. All100 recent steps per arm were strict improving, with zero
fallbacks/no-ops. This descriptive audit is not a new native readiness result.

From initialization, the largest atom probability shifts were0.03075/0.20975
/0.20948, and the largest bond shifts0.05823/0.25488/0.25473. Nevertheless no hard
atom or bond argmax label changed at these captured endpoints; all still display
the same25-carbon chain. Thus visible identity is hiding real soft changes.
Protenix pocket-contact loss moved-0.077953 to-0.085375; Boltz affinity scalar
moved1.103662 to0.332911, and chemistry arm1.103662 to0.339293. Contact and affinity
are distinct objectives, and none is experimental binding evidence. Chemistry
loss at audited0/1000/1314 points was zero with weight0.001; its ramp has not begun.

The optimizer is unnormalized gradient descent on categorical logits, not Adam:
the probability gradient is pulled back through softmax, then logits change by
the accepted rate times that pulled-back gradient. Therefore0.03 is not a direct
0.03 change in probabilities. Increasing nominal Boltz LR alone need not increase
the accepted rate; decreasing Protenix LR would likely slow its already small
steps. These are interpretations of saved evidence, not authorized run changes.

Audit archive11,347bytes SHA
`f84014400a1c3f380a891d2c69068f81d73cf6fc9c5fffe701792c6c6e72cf3c`;
all11files safely extracted and byte-verified locally. Audit JSON SHA
`c09198bde08ddba6b8b23a64ecb675b58ccf9de9a00a66d0bf6e53a5119e9708`.
Verification: `results/hpc/20261021-brd4-jq1-gradient-audit-01/verification.json`.
Local work was stdlib byte/schema verification only; numerical analysis ran on
allocated HPC. Native GPU jobs and original scientific/terminal gates are intact.

### Edition0003 verified and published

Report-only interim **29770891** actually DONE with LSF success and a passed
assessment. Captured at 14:02:25 Paris on 21 October: **5900 / 1409 / 1314**
updates and **5901 / 1410 / 1315** saved states, retaining all **8626 frames**.
Milestone progress is Protenix5500, Boltz0, chemistry0. This is IN PROGRESS;
terminal completion checks remain pending for each arm.

The 33,764,779-byte archive SHA is
`4e744970c5b095b9a2dfdf4c606abf266328b98535e78c546138744f2dbffd07`.
All208files passed safe extraction and byte checks; actual final scheduler and
LSF success, CPU71-pass rawXML/proofs, native/source/config/frozen identities,
per-frame metadata contiguity and chunk lengths were checked separately.
Report assessment SHA:
`933716fc8f6f40dcc61eecfd0b0ea6ba2381d7a00968c07c80ee9e2315e6d697`.

Immutable edition0003 is published at the stable8776link; older editions intact.
All187served files returned HTTP200 with exact length/hash. Actual browser QA
covered all three selectors and latest endpoints, mouse drag to3020, exact1500,
next/playback at chemistry777 and manual-scrub pause at500, expanded probability
heatmaps, readable element/slot labels, probability sizes and the raw graph.
Screenshot and verification are under the local edition0003 directory.
Next unique capture is remote interim-0003 / local edition0004, only on a new
per-arm milestone. No native job or scientific setting changed.

### Edition0004 verified and published

Report-only interim **29771001** actually DONE/LSF success/passed. Captured
21 October 14:21:34 Paris at **7175 / 1570 / 1439 updates**, with all
**7176 / 1571 / 1440 states (10,187 frames)** retained. All 236 evidence files
were safely extracted and byte verified locally. Archive39,386,991bytes SHA
`dd54d783a35a52e571edd731abea859af19f66075393f2b6985bcd3664a8973c`.
AssessmentSHA `eff9ea20cb86ceaebd643dd40907a444aeec8f1a8d498de70c3e455789417964`.
Actual final scheduler/LSF, raw CPU71-pass XML/proofs, source/config/model/context
and readiness pins, N25, contiguous frame metadata and exact chunk lengths/hashes
passed separate authentication. No local numerical/model/chemistry work occurred.

Published immutable edition0004 at the stable8776link, preserving all older
editions/servers. All215served files returned HTTP200 with exact bytes/hash.
Actual browser QA covered all3selectors/endpoints, mouse drags to3673then1299,
numeric1500/next1501, playback777to780 and manualscrubpause500, probability
heatmaps, readable labels/sizes and the raw graph. Verification/screenshot:
`results/hpc/20261021-brd4-jq1-25-report/edition-0004/`.
Saved endpoint contact loss is-0.0867367, Boltz affinity scalar0.278051 and
chemistry-arm affinity scalar0.325720; contact and affinity remain distinct.
All checked raw endpoint graphs still show the25-carbon chain; chemistry loss0
at captured1439 with coefficient0.001. This is IN PROGRESS, not full-budget or
terminal native certification. Published milestones6500/1500/0; next unused
capture parent interim-0004 / local edition0005. Native jobs remain unchanged.

### Matched-window gradient scale analysis

The user requested a direct comparison of gradient sizes rather than learning
rates alone. Allocated read-only CPU **29772035** actually DONE with LSF success;
its outcome is descriptive `recorded`, not a new native readiness pass. It reads
the already authenticated edition0004 capture (updates7175/1570/1439), compares
the first100, updates900–999, updates1300–1399 and each arm's latest100, and
measures raw native gradients, row-tangent gradients, softmax pullbacks and
consecutive saved logit/probability differences. No scorer, new derivative,
repair, candidate update or scientific job changed. All calculations and the
figure ran on allocated HPC; local work was stdlib byte/schema verification.

For updates **1300–1399**, medians were:

| Quantity | Protenix | Boltz-2 | Boltz-2 + chemistry |
| --- | ---: | ---: | ---: |
| Saved logit-gradient L2 | 0.005920 | 0.303586 | 0.303586 |
| Accepted LR | 0.03 | 0.001875 | 0.001875 |
| Measured logit-step L2 | 0.00017753 | 0.00053153 | 0.00053153 |
| Measured probability-step RMS | 1.292e-6 | 3.977e-6 | 3.977e-6 |

Thus Boltz's gradient is **51.28x larger**, its accepted LR16x smaller, but
its measured parameter step is still **2.99x larger** and probability-step
RMS **3.08x larger**. These are medians of measured per-update quantities,
not products of independently aggregated medians. In the first100 updates both
used LR0.03; Boltz's gradient was16.22x larger, measured logit steps16.23x larger
and probability movement15.11x larger. Updates900–999 still accepted LR0.03
in both, with20.76x gradient and19.79x probability movement differences.

The difference is already present before the softmax pullback. At initialization
raw atom-gradient RMS was0.001243 for Protenix versus0.033963 for Boltz, and raw
bond-gradient RMS0.000434 versus0.005407. Pullback/tangent norm ratios were
0.133/0.148 for atoms and0.289/0.246 for bonds: similar orders of attenuation,
not evidence of a uniquely vanishing Protenix softmax connection. Its native
contact objective is a negative mean of per-ligand best-pocket contact
probabilities, bounded[-1,0]; Boltz's affinity-head scalar has different units
and reductions. Gradient magnitude therefore compares numerical scaling, not
model quality, native correctness or experimental binding strength.

Both Boltz arms have identical measured values in the matched windows above;
the repair term is zero at the sampled valid-chain states, with coefficient0.001
before the delayed ramp. The largest-median-LR comparison previously mixed
different stages. Protenix's small gradients explain its small steps despite
accepting the full nominal rate. A future controlled Protenix LR0.1 pilot is a
reasonable first comparison with0.03, retaining all original native/identity
checks and actual step-size inspection; no such pilot or change is submitted.
Do not blindly multiply LR by51 or infer identical loss scales across models.

Analysis-only attempt02 **29772023** EXIT1 because it incorrectly assumed exact
cross-model initial array equality. Preserve its scripts/logs/receipt. New
immutable attempt03 measures the difference: atom arrays exactly equal, bond
maximum difference5.9604645e-8 (consistent with FP32 rounding), RMS2.7476384e-8.
No native gate was changed or relabelled. Later matched update indices represent
different molecular probability states, not identical scorer inputs.

Attempt03 Python SHA `82f1be70f73ac7e86c8d0449d688c2a43c8ced19f29cd795ad649ec62364bef7`;
wrapper SHA `feb66e7906d33471f2366054c8c9f3ead20c4af20ccecb5e604df37886340c4b`.
Archive585931bytes SHA
`19501ab905634cafb6acddcd9560d3cf8aa81a5cec9051c0ab91811c549a24f7`,
all14files safely extracted/byte verified. Analysis JSON SHA
`7f95f1a3692fc5380f39909d6dde975a368fd40b4172eacab1709dfff5a9454c`;
figure SHA `6bb952970750a5ac3f6470721df4efac4081c14b5eee903ff0d28a7f789443aa`.
Verification, JSON and figure are under
`results/hpc/20261021-brd4-jq1-gradient-audit-03/`; exact commands/source/proofs
in the corresponding `scripts/hpc/receipts/` campaign. Existing report editions,
pending report29771969 and all native studies are unchanged.

### Edition0005 verified and published

Interim **29771969** actually DONE/LSF success/passed, captured21October15:02:18
Paris at **9425 / 1859 / 1726 updates** and **9426 / 1860 / 1727 states**.
All **13,013 frames** retained. Exact archive49,698,929bytes SHA
`ef0f74b678f8ded9caa333cfa28d0a313076eafc99cdc7bcee94c2b1bce473a3`,
294files safely extracted/byte verified. Binary SSH streaming downloaded the
unchanged archive directly to an exclusive local file; exact receipt path,
length, archiveSHA and every manifest file/path/hash gates were retained.
Actual final scheduler/LSFsuccess, report assessment
`4e9886989e697b1ae66e0b1c08f97f254170fb2d1c054206a973211c741e3632`,
rawCPU71-pass XML/proofs, all native/source/config/frozen/readiness identities,
every-frame metadata contiguity and chunk lengths/hashes passed authentication.

Published immutable edition0005 at the stable8776link. All273served files
HTTP200/exactbytes. BrowserQA all3selectors/endpoints, actual drags4825then1707,
numeric1500/next1501, playback777to779/manualpause500, exact probability tables,
readable slot/element labels and probability-sized raw graphs passed. Screenshot
and verification under `results/hpc/20261021-brd4-jq1-25-report/edition-0005/`.
Captured contact loss-0.0882124; Boltz affinity scalars0.229913 and0.234419;
contact and affinity remain distinct. Checked endpoint graphs remain C25 chains;
chemistry loss0 at1726, weight0.001. Report is INPROGRESS, terminal/full-budget
checks pending; native jobs/configurations and older report editions unchanged.
Latest compact15:09Paris status showed all3RUN at9946/1918/1776, no terminal
summaries. Do not infer Protenix completion from its proximity to10000.
Published milestones8500/1500/1500; next unused interim-0005/localedition0006.

### 21 October 2026 — Edition0006 verified and published

Report-only interim **29772558** actually DONE/LSF success/passed, captured
21 October 15:15:42 Paris through **10000 / 1960 / 1812 updates** and
**10001 / 1961 / 1813 states**. All **13,775 saved probability frames** retained.
Exact archive52,544,644bytes SHA
`f283da0ffb883987fa5385b16a215abca4d4c0918b8dfbee99f7e3ab0cc30518`;
all310files safely extracted/byte verified. Transfer8126 completed successfully.
Actual final scheduler/LSF success, report assessment
`ca440b1dc46f87dad88fee291605f9dfe3fba50cfe0a5c642ba4d9bf8562c486`,
rawCPU71-pass XML/proofs, native/source/config/frozen/readiness identities and
all contiguous frame/chunk metadata, lengths and hashes authenticated.
All scientific rendering remained allocated HPC; local stdlib byte/XML/schema only.

Published immutable edition0006 at http://127.0.0.1:8776/report.html; all289served
files HTTP200/exactbytes. Actual browserQA covered all3selectors/endpoints,
mouse drags5119then1811, numeric1500/next1501, playback777to779/manualpause500,
exact atom/bond probability tables, overlays and readable raw graph labels/sizes.
Screenshot and verification under
`results/hpc/20261021-brd4-jq1-25-report/edition-0006/`.
Captured endpoint contact loss-0.08846936, Boltz affinity scalars0.22692089 and
0.23105149; these remain different metrics. Checked endpoint argmax graphs remain
C25 chains. Chemistry loss0 at1812, coefficient0.001; delayed ramp not yet begun.

The combined report remains IN PROGRESS. Separately, native Protenix29770179
and terminal collector29770706 are actuallyDONE, with persisted passed collector
assessment/full native gates established on allocated HPC. Local full-summary/
compact-proof transfer51862 remains active into protenix-completion-01; full local
native proof verification is still pending, not inferred from schedulerDONE alone.
The15:39Paris compact query showed Boltz29770093 RUN at2152 and chemistry29770124
RUN at1970; their collectors/finalreport remainPEND. Native runs/settings unchanged.
Published milestones10000/1500/1500; no new milestone beyond this edition, so no
new interim submitted. Next unused remoteinterim-0006/localedition0007. Older
editions/servers and unrelated PDF/OneNote changes preserved.
