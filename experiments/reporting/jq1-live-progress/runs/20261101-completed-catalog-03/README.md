# Completion-aware report update

User requested a report for new results and an easy path for future additions.
At 1 November20:23UTC,13 chemistry strategy arms have complete20k results.
Neighbor17, adaptive17/23 and temperature2-17 are new relative to edition0014.
Original terminal summaries record frozen identities preserved; all13 record
zero strict-valid argmax states. Actual full source/LSF/frame/sample validation
runs on allocated report CPU. The four structural comparison jobs remain queued;
they have no results to present yet. Model/optimizer jobs and saved data stay
unchanged.

The existing canonical command now prefers `current-completions.json`, preserving
seven original history panels and registering all22 strategy plus four structural
comparisons. `completed_candidates` is metadata discovery only: absent, running
and failed summaries are explicitly deferred; corrupt claimed completions fail.
Selected completions still pass every existing provenance/LSF/FP32/frame check.
The bounded panel limit is64, accommodating the33 currently registered panels.
The renderer and viewer are unchanged. Six focused metadata controls are added
to the existing allocated report suite; no model or chemistry is evaluated.

One repeatable update command:

```powershell
.venv\Scripts\python.exe scripts/reporting/report.py --profile experiments/reporting/jq1-live-progress/configs/dtu-report-work3-exploratory.json
```

Add one entry under `completed_candidates` to register a future chemistry-screen
producer payload. Existing fixed/noisy continuations use ordered `segments` as
before. Rerunning the command adds newly completed candidates and reuses verified
cached assets. It does not schedule itself or submit optimization jobs.

The request-specific `selection.json` and `profile.json` are immutable inputs.
Current request: `request-20261101T202542Z-8f626be6`. The first CPU validation
29870693 retained261 passes and one obsolete exact32-panel fixture failure.
The corrected immutable engine8456066f6f5b passed262 actual cases on CPU29870703;
no renderer/viewer/profile bytes changed. All20 selected panels were captured
with the existing source/LSF/count checks. New render29870705 is the sole
accepted render job and targets edition0015.

At20:34UTC every DTU queue, including `hpc`, reported `Open:Inact_Adm`;
the exact render pending reason is **The queue is inactivated by the
administrator**. There is no active alternate queue. No resubmission or local
scientific rendering was attempted. Edition0014 remains authoritative; new
publication/HTTP/browser QA have not occurred. Public HPC news provided no
current reopening time. The client remains waiting; if it exits, resume this
same request rather than creating another:

```powershell
.venv\Scripts\python.exe scripts/reporting/report.py --resume results/hpc/20261028-reusable-report-01/requests/request-20261101T202542Z-8f626be6 --profile experiments/reporting/jq1-live-progress/runs/20261101-completed-catalog-03/profile.json
```

[Readable new-results summary](results-summary.md) and its original scalar
bindings are available now, independently of the queued interactive render.
Read [campaign.json](campaign.json) before recovery; never duplicate an accepted
or unknown submission. Historical editions, running experiments and saved
results stay intact. After this publication finishes, use the normal update
command above to pick up later completions automatically.

### 2November recovery of the queued report

Old render29870705 actually EXIT1: its dynamic completed-candidate selection grew while queued. The old request/failed LSF and stale viewer lock were preserved and reconciled. One local fresh attempt stopped before SSH/submission at that lock; after proving no active report CLI and actual old EXIT, its exact new request was resumed. Replacement29871060 is RUN with a fixed23-panel selection (7 existing histories +16 completed comparisons), using engine8456066/CPU262 cache. No publication is claimed yet. See campaign.json and exact request `request-20261102T003338Z-190b45a9`; do not duplicate it.
