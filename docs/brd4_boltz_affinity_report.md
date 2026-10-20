# Standalone Boltz mix50 trajectory report

20 October 2026. The user requested a new interactive report for completed
Boltz mix50 run **29759212**. Allocated report job **29760530 DONE** at
**09:16:29 Europe/Paris**, with actual scheduler/LSF success authenticated.
All 35 tests passed; the renderer authenticated all 10,001 saved snapshots
and 147 readiness proof hashes, then produced a complete affinity-primary
single-run report and 451-file archive. Download, safe extraction and all 451
file checks passed. The new detached viewer at
**http://127.0.0.1:8774/report.html** returned HTTP 200 with exact HTML length/
SHA. Full browser QA passed the single mix50 selector, playback, endpoints,
coefficient ramp, probability heatmaps and readable final raw graph/labels/
sizes. The report is marked deliverable and left paused at state 10000.

## Scope and authoritative input

Create a new report family containing only the source02 Boltz mix50 run:
10,000 updates and 10,001 saved states. Preserve every existing report,
server and artifact family. Do not overlay earlier Protenix experiments or
invent a random-run selector; the separate random diagnostic is not a
completed molecular optimization trajectory.

The completed native run and its configuration/source/model/context identity
remain unchanged. Authenticated completion metadata establishes contiguous
updates 0..9999 and states 0..10000, six occupied neutral slots, matching
JSONL/case/summary records and passed native terminal checks. The existing
21-file completion bundle verifies only three selected snapshot byte sets
at 0/5000/10000; it is not full-trajectory byte verification. The allocated renderer authenticated the full saved input and all 10,001
snapshots, extending beyond that compact metadata/selected-snapshot proof.

All model values, graph probabilities and trajectory observations come from
the saved run. Rendering must not rerun models, evaluate or repair chemistry,
resample molecules, derive missing probabilities or fabricate absent states.
Failed/invalid saved graphs remain visible. In particular, the saved final
raw graph has chlorine valence 2 and null canonical SMILES; preserve that
observation rather than sanitizing it into a different molecule or depicting
it as chemically valid.

## Presentation and interpretation

Extend the shared viewer to select native **affinity** as its primary metric
for this report, preserving contact as the default for existing callers.
Quantitative affinity and native binder probability remain separate plots
and labels. The affinity scalar is the fixed-context Boltz quantitative
output, not the binder probability, a new hard-molecule prediction or an
experimental binding measurement. Its lower-is-better convention and
log10(IC50 in micromolar) units must be explicit.

Retain playback across all 10,001 saved states, clear update selection,
initial/final molecule snapshots, readable element/slot labels, probability-
dependent atom radii and bond widths, and saved atom/bond probability
heatmaps. Graph images and heatmaps must use the exact probabilities stored
at the selected state. Preserve invalid valences and disconnected graphs.
The new report has one actual run, with no placeholder or historical control.

Show affinity, chemistry components, coefficient ramp and recorded optimization
behavior over time. The chemistry coefficient remains 0.001 through applied
update 5000, then follows the saved geometric ramp to 50 at 10000. Explain that
detached repair targets and coefficients are fixed within each state's
trials but can change between states, so changing total repair objectives
are not one globally comparable loss. The recorded 10,000 strict-improving
updates refer to those within-state comparisons.

The frozen native predicted BRD4/hexane pose and auxiliary context remain
fixed during graph optimization. Label the resulting affinity trajectory
as a partial-gradient fixed-pose surrogate. It does not establish fresh-pose
affinity, experimental affinity or useful binding of the final invalid graph.

## Allocated rendering and delivery gates

Use a unique immutable tooling source and an allocated DTU HPC CPU job for
tests, complete saved-state checks, scientific rendering and packaging.
Report-only failures must preserve the native run and scientific outcome;
correct tooling in a new sealed release rather than changing native gates.
Local work is static editing, packaging, file transfers, standard-library
byte/schema verification and browser presentation QA only.

Before publishing, authenticate actual scheduler/LSF completion, saved input
identities and complete trajectory, report metadata, exact receipt archive
path/length/SHA and every downloaded file. Use safe extraction and a fresh
local family. Inspect the actual rendered report in a browser, including
playback endpoints, all plots, labels, probability sizes/heatmaps and the
raw invalid final graph. Start a fresh loopback server without disturbing
old servers, verify the served artifact and deliver its actual working URL.

Record actual source hashes, CPU jobs/test outcomes, artifact paths, archive
verification, browser checks and publication URL here and in the experiment
ledger as they occur. Actual execution and delivery records follow below;
only completed checks are marked passed.
See [the native campaign plan](brd4_boltz_affinity_10000.md) for preserved
scientific identities, failures and arm-specific adoption.

## Authenticated tooling and completed allocated report

Report tools01 authenticated all **12 files** at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-report-01/source`.
Manifest SHA
`7cb5b82c04634f3b3b276943d8d7988bc29b698df2e315bf12ea7915ec82bf65`;
archive 34,412 bytes, SHA
`f9b2bca789af2a455863c66b37bfbf1761dea94cf90ee3aeea647dddf35a2537`.
Scientific source02/configuration remained unchanged; entry authentication
passed its 393-file identity, native proofs and fresh quota reserve 8 GiB.
Only saved native artifacts were used, without inference or a new experiment.

Actual report **29760530** submitted 09:14:27, started 09:14:30 and ended
**09:16:29 Europe/Paris on 20 October 2026**, 119 seconds. Resources were hpc,
two CPU slots, 4 GiB per slot, 60 minutes, zero GPUs. Scheduler DONE and LSF
Successfully completed are authenticated. All 35 tests passed in 26.39 seconds:
26 new fixtures, three existing chemistry and six ramp tests, zero failures/
errors/skips. Assessment SHA
`ff68b20b5264f89e5a9c79fec921c11830cce3104c225cfcdb04c4cf030fb1a1`.
Output:
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-mix50-report-01/payload/`.

The actual renderer validated all 10,001 snapshots and 147 native readiness
proof hashes. Its receipt is complete, with primary metric `affinity`, input
jobs `[29759212]` and saved-state counts `[10001]`. The produced HTML is
40,089,420 bytes, SHA
`57759bd10fb79a5e142f2757c8d8df86c3dc7522c2f39408d9ed241f89a03974`.
These allocated output records now match the locally verified and served
HTML. Full browser presentation QA passed.

Remote archive:
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-mix50-report-01.tar.gz`,
with its `.tar.receipt.json` receipt. Archive 24,432,529 bytes, 451 files, SHA
`7bb0849eff0daeaa2296f50b1c7516e751404c1cd5825f3d57ad54f3ed266a48`.
Download and local receipt/byte/schema verification passed all 451 files at
`results/hpc/20261020-brd4-boltz-affinity-report/edition-01/`, using
`scripts/verify_hpc_evidence.py` and standard-library checks. Exact archive
path/length/SHA, safe extraction, full file set and every-file checks passed.
Downloaded manifest SHA
`b32f255b2937f3244d7fe1337fcfee63206e145e4e02fadf1e0e6479712accce`.

Verified report artifacts were published HTML last to the new family's
`viewer/`; a new detached loopback 8774 server serves
**http://127.0.0.1:8774/report.html**. Actual HTTP 200/exact 40,089,420-byte
length and HTML SHA above passed. Existing report families and servers are
preserved. The actual report opened in the in-app browser; states 0 and 10000
match, and final raw invalid chlorine-valence-2 graph is visible with readable
Cl1/C2/C3/C4/C5/S6 labels and probability-dependent sizes.

Full browser QA passed in the actual in-app report: one correct mix50
selector; state 0 hexane `CCCCCC`; state 5000 displayed coefficient 0.0010021663
and previous applied coefficient 0.001; state 10000 coefficient 50, raw
chlorine-valence-2 invalid graph and null canonical SMILES. Saved heatmaps
have six atom rows and 15 unordered pairs. Playback advanced 0 to 22 and Pause
worked. Labels and probability sizes remained readable. The report is left
paused at 10000 and marked deliverable. All older reports/servers and all
molecular experiments remain unchanged.

Quantitative affinity, separate binder probability and raw/weighted repair
overlay plots were visually checked and readable in the actual report.
