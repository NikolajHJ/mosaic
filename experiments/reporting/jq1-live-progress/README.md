# Updating saved JQ1 progress report

User-authorized on25October2026, refreshed on28October: create updating, saved-only reports for the
Protenix native-only continuation, plain Boltz30k, DBMol binding10k and noisy
Protenix1k pilot, now including random Protenix and Boltz with delayed chemistry.
The report separates trajectory progress, presentation and
scientific validation. Original runs, settings, failed gates and older reports
remain unchanged. The monitor remains paused.

## Saved data and interpretation

Every newline-committed scalar record, component and trial is available. Molecule
matrices appear only at actual retained frames (initial, every20 updates and
segment endpoint). Unsaved steps explicitly clear the graph and heatmaps. There
is no interpolation, repair, resampling, inference or new scoring. Original raw
validity/readout observations remain visible.

The Protenix continuation starts at absolute update10,000. Its plot contains the
additional10,000 saved updates through20,000; the original parent10k is preserved
in its historical report. The absolute goal is50,000. Plain Boltz is at20k/30k;
DBMol is at7.5k/10k across three2,500-update segments. The shared boundary is removed
only after exact saved arrays/key equality, with both scalar observations retained.
The separate noisy pilot completed its requested1,000 updates. Random Protenix
with delayed chemistry has10k saved toward50k; Boltz with delayed chemistry has
10k toward30k. Their continuation blockers remain distinct from saved progress.

Protenix contact, Boltz affinity, binder probability and the DBMol component losses
have different meanings. Fixed-pose proxy scores and valid graphs do not establish
binding/activity. Presentation can pass while the original scientific run failed.

## Ownership and execution

- `scripts/report_data.py`: allocated HPC CPU validation of original FP32 matrices,
  simplex/scalar/argmax/hash correspondences and contiguous scalar prefixes;
  deterministic bond-aware display geometry retaining every raw atom and bond.
- `templates/report.html`: report controls, scalar curves, raw graphs and heatmaps.
- `scripts/report_ops.py`, `report_stage.lsf`: account-bound immutable CPU fixture,
  actual terminal certificate and report-render controller. No optimizer phase.
- `scripts/inspect_inputs.py`, `inspect_noise.py`: historical read-only input audits.
- `scripts/observe.py`, `publish.py`, `verify_http.py`: local stdlib evidence transfer,
  byte/XML/schema publication and HTTP delivery checks. These delivery utilities
  are maintained separately from the immutable remote render release.
- `tests/`: allocated positive/negative saved-data, metadata and browser-structure
  fixtures. Real browser interactions are checked separately before delivery.
- `runs/20261025-jq1-live-report-01/`: exact immutable sources, jobs, outcomes,
  original failure preservation, publication and QA receipts.
- `scripts/prepare_refined.py`: explicit generation04 presentation redesign
  packaging, distinct from an unchanged-template data refresh.
- `scripts/capture_current.py`, `prepare_current.py`: explicit six-arm metadata
  capture and immutable package preparation; no scientific execution locally.
- `runs/20261025-jq1-refined-report-01/`: redesign provenance, allocated checks,
  original-data equality audit and delivery QA.
- `runs/20261028-jq1-current-report-05/` through `-08/`: preserved fixture failure,
  local-only draft, passed render with a stale heading, and corrected delivery.

All numerical fixtures and frame validation/rendering run on allocated DTU HPC.
Local work is stdlib metadata/AST/XML/byte/packaging, transfers, publication and
browser presentation QA. The source caps, full existing experiment projections,
4GiB byte reserve and100,000-file headroom remain mandatory.

## Updating

Use the reusable entry point from the repository root:

```powershell
python scripts/reporting/report.py
python scripts/reporting/report.py --run protenix --run boltz --run dbmol
python scripts/reporting/report.py --run /absolute/HPC/run-a --run /absolute/HPC/run-b --output results/hpc/my-report --port 8781
python scripts/reporting/report.py --runs my-runs.json
```

The default is the current six trajectories. Choose one to24 named runs or saved
run paths; the command discovers the original summary, config, driver, source,
job and retained-frame bindings. Ordered continuation segments can be grouped in
a small JSON file. See [the command guide](scripts/README.md#one-command-reports)
and [selection/profile files](configs/README.md).

The exact report engine is qualified once on allocated HPC CPU and reused by
hash. An unchanged selection returns its existing edition without a render job.
Changed selections use one allocated CPU job, reusing previously validated arms
only after rechecking their original input/frame hashes. The transfer includes
only assets missing from the verified local cache. The command checks actual
scheduler/LSF success, raw CPU evidence, archive/member hashes and served HTTP
bytes, then publishes an immutable edition through the atomic latest pointer.
It has no model or optimizer submission branch. No per-refresh source release,
manual hash ledger or manual transfer/publication sequence is needed.

This version supports these terminal JQ1 N25 saved fixed-step/noisy trajectories,
including failed terminal scientific gates and stride20 retained matrices. It
does not fabricate a terminal summary for queued/running jobs, import arbitrary
other experiment schemas or turn a failed scientific result into a pass.
Continuation/status annotations in a selection are explicit dated presentation
metadata; the command does not infer a new scheduler state from those words.

The stable viewer reads `latest.json` every60seconds and preserves experiment and
absolute-update selection when a new verified edition is published. Each edition
is immutable; the latest pointer is replaced atomically. The viewer does not poll
HPC or submit experiments. Its manual refresh button checks only the publication
pointer. Never replay a previous launch or unknown submission outcome. Requests
and exact resume instructions are recorded automatically; an interrupted request
must use its recorded `--resume` path. Historical manual capsule tools remain
available for provenance, not the normal update workflow.

Endpoint qualification is a separate [diagnostic experiment](../../validation/endpoint-qualification/README.md).
Neither report availability nor supplementary diagnostic agreement converts an
original failed gate to a pass or authorizes a continuation.

## Current delivery

The reusable command published **edition-0005** at
[the same viewer](http://127.0.0.1:8780/report.html) on28October2026 at12:20UTC.
The engine passed186 allocated fixtures once; the refresh reused its certificate
and all40 saved-data assets. Actual render29840605 DONE/hpc/LSF, every file and
48 served HTTP files passed. All six selectors, scalar/chemistry plots, numeric
navigation, actual slider drag, arrows, playback/manual pause, heatmaps and raw
graphs passed browser QA. The58,500 scalar records/2,931 retained frames and all
original scientific outcomes are byte-identical to edition0004.

Measured end-to-end cached refresh:153s, including31s rendering, with a29,775-byte
download instead of the prior70MB complete archive. The unchanged rerun returned
edition0005 in33s and submitted no render job. These are observations, not queue
time guarantees; new saved data still receives its required allocated checks.
See [the reusable command record](runs/20261028-reusable-report-01/README.md).

### Previous baseline delivery

Verified on28October2026: [updating viewer](http://127.0.0.1:8780/report.html),
historical immutable edition-0004. CPU29839663/callback29839664 and renderer29839701 all
actual DONE/hpc/same-job LSF success;149 raw fixtures passed. All46 archive
members,130 served HTTP files and actual browser interactions passed. The capture
at09:54:35UTC contains58,500 update records and2,931 retained frames across all
six trajectories. Protenix is at absolute20k/50k, plain Boltz20k/30k, DBMol7.5k/10k,
random Protenix chemistry10k/50k and Boltz chemistry10k/30k; the noisy pilot is
complete at1k. Scientific outcomes remain separate from presentation and progress.

Source08 preserves the failed05 fixture attempt, local-only06 and passed07
render with a stale heading. It retains the refined layout and exact scientific
saved-data validation, with six selectors, continuation labels and a full current
storage guard. See [delivery record](runs/20261028-jq1-current-report-08/README.md)
for source/input/job/archive/byte/XML/HTTP/browser evidence and exact commands.

The refined view has fixed primary loss curves, a compact DBMol component plot,
larger bond-aware molecule view and one shared timeline. Matrices and provenance
start closed. Both chemistry selectors now show their actual saved chemistry
curve alongside the relevant native loss; Boltz chemistry's saved zero curve is
kept as recorded. The first four runs do not acquire an invented chemistry term.

The stable shell applies its historical CSS adjustment only to older documents.
All old editions and scientific values remain unchanged. Edition0003's historical
141-case render and exact display-only equality audit remain preserved. The new dated run record
binds source, checks, equality audit, HTTP/browser QA and preparation-only future
refresh instructions. Historical failures remain preserved; the monitor stays paused.
