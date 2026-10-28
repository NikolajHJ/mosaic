# Saved report tools

Saved JSONL/NPZ validation and report construction run only in allocated DTU HPC
CPU jobs. Source/input/proof bindings, raw fixture XML, actual scheduler/LSF and
archive/member bytes are authenticated before immutable publication. No model,
optimization, gradient evaluation, repair or resampling belongs to this renderer.

## One-command reports

The public entry point is `python scripts/reporting/report.py` from the repository
root. It forwards here without duplicating report logic. With no options it
updates the six current trajectories at http://127.0.0.1:8780/report.html.

```powershell
python scripts/reporting/report.py --run protenix --run boltz --run dbmol
python scripts/reporting/report.py --run /absolute/HPC/study-a --run /absolute/HPC/study-b --output results/hpc/my-report --port 8781
python scripts/reporting/report.py --runs my-runs.json
```

An input can be a study directory, a job parent or its terminal summary. A JSON
file supports simple paths or explicitly grouped continuation segments:

```json
{
  "version": "mosaic-report-selection-v1",
  "runs": [
    "/absolute/HPC/study-a",
    {
      "id": "continued-boltz",
      "label": "Boltz continuation",
      "segments": [
        {"run": "/absolute/HPC/segment-0000/payload/study"},
        {"run": "/absolute/HPC/segment-10000/payload/study"}
      ]
    }
  ]
}
```

Run IDs must be unique. One to24 runs are supported. Terminal saved JQ1 N25
fixed-step/noisy formats and stride20 matrices are supported; queued/running
segments without a terminal summary are rejected. Original scientific failures
remain visible. Config/rate/range/input/job provenance is discovered rather than
copied into the selection. Presentation annotations are optional historical
notes, not an automatic account-wide scheduler monitor.

`report.py` handles local stdlib metadata, transfers, byte verification, atomic
publication and localhost delivery. `quick_inputs.py` authenticates saved inputs
and cache keys. `quick_runtime.py` provides only allocated CPU validation/render
phases. Its immutable content-addressed engine gets the full fixture suite once
per exact code/template/runtime-profile hash. Run selection changes do not change
that hash. Subsequent reports reuse the certificate, rehash original inputs and
retained matrices, validate changed arms with the unchanged scientific renderer,
and download only missing assets. Verified reports register themselves as future
cache sources. If nothing changed, no render job is submitted.

The command waits for actual same-job DONE/hpc/LSF success and validates raw CPU
XML, proof/source pins, archive and every reconstructed member before publication.
Full experiment projections, independent report caps,4GiB reserve and100,000-file
headroom remain mandatory. It does not call a model, optimizer or cleanup tool.
`offload_report_assets.py` is a separate operator-only backup/verification helper;
the report command never invokes it.

Useful options:

- `--dry-run`: print selection/output without SSH or submission.
- `--setup-only`: qualify/install the exact engine without rendering data.
- `--force`: deliberately publish a fresh edition even if data is unchanged.
- `--output PATH --port PORT`: keep a separate report at a separate localhost port.
- `--no-serve`: publish files without starting/checking a localhost server.
- `--profile PATH`: use an explicit account/runtime/capacity profile.
- `--resume PATH`: resume the exact recorded request printed by the command.

Every request records source, submissions, proofs, transfers and publication in
`results/hpc/20261028-reusable-report-01/requests/`, plus a compact owner journal
and experiment-ledger events. Never delete a request lock or submission marker to
retry an unknown outcome. Inspect the recorded evidence and use `--resume` with
the same profile/output/serving options. The engine does not schedule recurring
updates; the experiment monitor remains paused. The browser only checks the
published latest pointer every60seconds.

The supplied profile and base-source archive describe this existing DTU account.
Other accounts need their own authenticated profile/guard/runtime; changing an
SSH environment variable is not a supported migration. Keep the pinned local
base-source archive available for deterministic engine packaging.

## Historical manual preparation

`refresh_capsule.py` is a local standard-library preparation tool. It does not
connect to HPC, submit a job, poll results, or publish a report. It authenticates
the exact sealed refined source04 archive and every member, then prepares a new
`results/hpc/20261025-jq1-live-report-01/sourceNN/` capsule. The scientific
renderer, qualification policy, refined HTML template, bond-aware display layout
and all141 fixture cases keep their exact source04 bytes. Only the saved-input plan and controller literals
for the new root, source, job prefix and archive edition change. Each new
capsule repeats the allocated CPU suite and the original whole-storage guard.

Provide a complete fresh bound plan, with the same four arm IDs, scientific
settings and absolute goals as source04. Its edition must be the requested new
edition and its timezone-aware `captured_at` must be within four hours. Preserve
`monitor_remains_paused: true` and `automatic_optimizer_submission: false`.
Original terminal segment statuses remain `passed` or `failed`; this controller
does not admit a running segment. Each binding is exactly
`{"path":"/absolute/Mosaic/path","bytes":123,"sha256":"64 lowercase hex"}`.
Use fresh authenticated file metadata rather than guessing a saved prefix.

Each arm also needs explicit counts:

```json
"expected_counts": {
  "committed_updates": 10000,
  "scalar_states": 10001,
  "retained_frames": 501,
  "latest_global_step": 20000
}
```

Every segment needs `expected_counts` with the first three fields. At a segment
boundary the duplicate initial frame/state is counted once in the arm total,
after exact boundary-array/RNG validation on allocated CPU. The helper adds
the publisher's flat `expected_committed_updates` and
`expected_retained_frames` aliases and rejects conflicting aliases. Absolute
Protenix progress remains `global_start + committed_updates`, so10,000 new
updates after the original10,000 are absolute20,000 against a50,000 goal.

For example, after independently authenticating `fresh-plan.json` and its SHA:

```powershell
python experiments/reporting/jq1-live-progress/scripts/refresh_capsule.py --plan results/hpc/20261025-jq1-live-report-01/fresh-plan.json --plan-sha256 <actual-plan-sha256> --generation 05 --edition edition-0004
```

This only writes a new package and prints the next commands. Inspect its
`preparation.json`, `source_manifest.json`, `plan.json` and `remote-program.py`.
After checking the prepared capsule, the explicit one-shot launch command is:

```powershell
python results/hpc/20261025-jq1-live-report-01/source05/start_prepared.py --execute
```

That command can submit only the report CPU job and its own-ended callback;
the callback can submit only the saved-only CPU renderer after raw fixture and
same-job terminal proofs pass. The remote startup additionally verifies exact
newline-committed scalar/retention counts before submitting. The generated
launcher records an exclusive local `launch-05` directory and remote claim.
An existing or interrupted outcome must be inspected; never replay it or delete
its markers to retry. No model or optimizer job is in this pipeline.

Use a fresh observation name on every manual check:

```powershell
python experiments/reporting/jq1-live-progress/scripts/observe.py --name observation-refresh05-01 --generation 05 --edition edition-0004
```

Only after that observation establishes the actual render DONE/LSF success and
passed archive receipt, publish with the matching generation and edition:

```powershell
python experiments/reporting/jq1-live-progress/scripts/publish.py --observation observation-refresh05-01 --generation 05 --edition edition-0004
```

Independent archive/member verification, exact served HTTP bytes and browser QA
are still required. Publication creates a new immutable edition and atomically
changes the existing viewer's latest pointer. Its60-second browser refresh
checks for published editions; it does not poll HPC or generate reports. These
instructions describe an unexecuted preparation path, not a new submitted job
or a passed check.

Generation04 is the separate presentation redesign and is reserved. Later
refreshes require generation05 or higher. The historical source03 launch prefix
is separately authenticated and rewritten to the new generation; source04
controller constants use a separate04-to-new-generation mapping. The immutable
refined template and graph geometry are therefore retained by future refreshes.

## Current six-arm update (28 October)

`capture_current.py` collected the exact terminal saved segments for the current
six-arm cohort using one read-only metadata query. Its completed capture is bound
under `results/hpc/20261028-jq1-current-report-05/capture01/`; do not replay that
one-shot capture or its old root-absence guard. `prepare_current.py` authenticates
that capture and source04, adds the two chemistry selectors, current continuation
labels and the current full-campaign storage guard. It performs stdlib packaging
only. The source08 package is already staged and submitted: **do not rerun its
starter, callback or claim**. Its exact commands and outcomes belong to
[run08](../runs/20261028-jq1-current-report-08/README.md).

Source05 failed one obsolete viewer literal fixture (147 passes); no renderer
ran. Source06 is a preserved local-only draft. Source07 corrects the fixture and
explicitly charges the failed05 upload along with every owned source/root byte.
Source07 passed all148 allocated cases and rendering, but final review found a
stale four-run heading. Source08 preserves07, charges all its terminal bytes and
uses cohort-agnostic wording with an additional viewer fixture. It repeats all149
allocated cases:19 qualification,73 data,16 orchestration and41 viewer contracts.
Existing scientific frame/transition/boundary checks remain
unchanged. The independent report allowance is1GiB, in addition to all current
experiment forecasts and the unchanged4GiB/100,000-file reserves.

Use fresh observation names with `--generation 08 --edition edition-0004`.
`publish.py` accepts only the exact planned four- or six-arm cohort and validates
the new allocated CPU/render proofs, archive and every member before publishing.
The older display-only equality audit applies only to generation04. Updating
this report never submits a scientific run or resumes the paused monitor.
