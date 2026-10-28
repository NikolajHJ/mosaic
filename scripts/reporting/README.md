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

The default selection is the six current trajectories. A JSON selection can group
ordered continuation segments. See the canonical experiment's README and configs.
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
