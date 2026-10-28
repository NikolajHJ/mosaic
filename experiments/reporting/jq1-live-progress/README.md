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

The stable viewer reads `latest.json` every60seconds and preserves experiment and
absolute-update selection when a new verified edition is published. Each edition
is immutable; the latest pointer is replaced atomically. The viewer does not poll
HPC or submit experiments. Its manual refresh button checks only the publication
pointer. A future refresh must bind the actual new saved prefix/segment/source/
config/job/LSF inputs, pass the allocated report pipeline and complete independent
archive/member, HTTP and browser checks before publication. Never replay a previous
launch or unknown submission outcome. The refresh preparation helper and exact
operator commands are described in [scripts/README.md](scripts/README.md).

Endpoint qualification is a separate [diagnostic experiment](../../validation/endpoint-qualification/README.md).
Neither report availability nor supplementary diagnostic agreement converts an
original failed gate to a pass or authorizes a continuation.

## Current delivery

Verified on28October2026: [updating viewer](http://127.0.0.1:8780/report.html),
immutable edition-0004. CPU29839663/callback29839664 and renderer29839701 all
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
