# Updating saved JQ1 progress report

User-authorized on25October2026: create updating, saved-only reports for the
Protenix native-only continuation, plain Boltz30k, DBMol binding10k and noisy
Protenix1k pilot. The report separates trajectory progress, presentation and
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
in its historical report. The absolute goal is50,000. Plain Boltz is at10k/30k;
DBMol is at5k/10k across two2,500-update segments. The shared boundary is removed
only after exact saved arrays/key equality, with both scalar observations retained.
The separate noisy pilot completed its requested1,000 updates.

Protenix contact, Boltz affinity, binder probability and the DBMol component losses
have different meanings. Fixed-pose proxy scores and valid graphs do not establish
binding/activity. Presentation can pass while the original scientific run failed.

## Ownership and execution

- `scripts/report_data.py`: allocated HPC CPU validation of original FP32 matrices,
  simplex/scalar/argmax/hash correspondences and contiguous scalar prefixes.
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

Verified on25October2026: [updating viewer](http://127.0.0.1:8780/report.html),
immutable edition-0002. CPU29812102/callback29812103 and renderer29812120 all
actual DONE/LSF success;123 raw fixtures passed. Archive/member, all56 HTTP files
and actual browser interaction QA passed. The original scientific statuses remain
failed/failed/failed/passed. See the dated run record for all hashes and limits.

A stable-shell responsive CSS adjustment affects layout only. Immutable edition
resources and every scientific scalar/frame byte remain unchanged. Historical
source01 metadata rejection and source02 browser-layout failure are preserved.
