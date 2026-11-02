# Saved-result report entry point

**Default workflow for agents and contributors:** use this command for supported
saved trajectory reports and updates. Select runs or edit the small selection
JSON; reuse the qualified engine and verified cache. A data-only refresh does not
need a new controller, renderer, release archive or fixture-suite rerun. Use
`--force` only when a fresh edition is intentionally needed despite unchanged
input. If a format is unsupported, document the gap and extend the canonical
owner with the required validation. Do not clone historical campaign pipelines.

`report.py` forwards to the single canonical implementation in
`experiments/reporting/jq1-live-progress/scripts/report.py`; no report logic is
duplicated here. This entry point serves saved Protenix, Boltz, DBMol and delayed
chemistry trajectories supported by that experiment's saved-data schema.

From the repository root:

```powershell
python scripts/reporting/report.py
python scripts/reporting/report.py --run protenix --run boltz --run dbmol
python scripts/reporting/report.py --run /absolute/HPC/run-a --run /absolute/HPC/run-b
python scripts/reporting/report.py --runs my-runs.json --output results/hpc/my-report
```

The default selection is `current-completions.json`: the existing trajectory
histories plus registered chemistry comparisons. Its `completed_candidates`
entries are included once their complete production summary is available;
unfinished/failed candidates are explicitly deferred. Completed inputs still
receive all existing source/LSF/frame validation. Add one candidate row to
register another chemistry run; no renderer change is needed. A JSON selection
can also group ordered continuation segments.

Completed chemistry-screen continuations automatically include their
authenticated saved parent prefixes. For interrupted native histories, an
explicit `through_global_update` selects an actual retained checkpoint before
the next segment; later saved tails remain preserved. The captured selection
is fixed before submission, so another experiment finishing while the report
waits does not invalidate that request.

If a completed render's live scheduler record has expired, use
`--recover-delivery --resume <original-request>` with its original profile.
This authenticates persisted same-job LSF/history, source and delivery bytes;
it delivers the existing result without submitting another job.

For the current workstation and stable report at port8780, repeat:

```powershell
.venv\Scripts\python.exe scripts/reporting/report.py --profile experiments/reporting/jq1-live-progress/configs/dtu-report-work3-exploratory.json
```

This picks up newly completed registered runs, preserves existing content and
reuses the engine/cache. An unchanged selection reuses the published report.
See `experiments/reporting/jq1-live-progress/configs/current-completions.json`.
The current engine supports terminal JQ1 N25 fixed-step/noisy saved trajectories,
including chemistry and stride20 retained matrices, plus scheduled-time DeFoG
with every saved matrix validated and stride20/endpoints selected for display. A queued/running segment
without its original terminal summary is outside this version's supported input.
Connection/storage settings are explicit profiles; the supplied DTU profile is
account-bound, not portable by changing environment variables.

All scientific frame validation/rendering and fixtures run on allocated HPC CPU.
The local command performs metadata/byte handling and publication. It cannot
submit model/optimizer jobs. Unchanged data reuses the existing report; changed
data gets one render job using a previously qualified engine and verified cache.
Historical editions and scientific outcomes remain unchanged.


### Scheduled-time Protenix–DeFoG support (28 October2026)

The canonical engine now includes the terminal30k scheduled-time DeFoG schema.
Its allocated-only report_defog.py adapter validates every original saved NPZ,
including non-displayed frames, exact FP32 probabilities/argmax/time/query/teacher
bindings and saved transitions. All scalar records/readouts are retained. The
viewer selects every20th plus endpoint matrix for display and states this
selection explicitly; it does not claim those other matrices were unsaved.
Teacher KL, flow time, prior weight, drift and saved validity remain separate
from the Protenix contact proxy. The original combined terminal atom failure and
zero valid identities remain failed, even when presentation validation passes.
The old six-arm scientific renderer bytes are unchanged and their verified cache
remains reusable. Model/optimizer/chemistry evaluation is absent.

The default dtu-report-home.json profile stages bounded report outputs under the
authorized HPC HOME mosaic-runs root, independently checking HOME quota/caps and
4GiB reserve. WORK3 originals are read-only; this report requests zero WORK3
output bytes and grants no scientific campaign admission. Historical WORK3
storage checks remain unchanged for workloads that write there.
The old WORK3 profile/blocked receipt and every historical edition are preserved.
No cleanup or scientific relocation occurs. See the [adapter run record](../../experiments/reporting/jq1-live-progress/runs/20261028-defog-report-01/README.md) for actual qualification/render/publication status.
