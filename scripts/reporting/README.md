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

For the current workstation and stable report at port8780, repeat:

```powershell
.venv\Scripts\python.exe scripts/reporting/report.py --profile experiments/reporting/jq1-live-progress/configs/dtu-report-work3-exploratory.json
```

This picks up newly completed registered runs, preserves existing content and
reuses the engine/cache. An unchanged selection reuses the published report.
See `experiments/reporting/jq1-live-progress/configs/current-completions.json`.
The current engine supports terminal JQ1 N25 fixed-step/noisy saved trajectories,
including chemistry and stride20 retained matrices. A queued/running segment
without its original terminal summary is outside this version's supported input.
Connection/storage settings are explicit profiles; the supplied DTU profile is
account-bound, not portable by changing environment variables.

All scientific frame validation/rendering and fixtures run on allocated HPC CPU.
The local command performs metadata/byte handling and publication. It cannot
submit model/optimizer jobs. Unchanged data reuses the existing report; changed
data gets one render job using a previously qualified engine and verified cache.
Historical editions and scientific outcomes remain unchanged.
