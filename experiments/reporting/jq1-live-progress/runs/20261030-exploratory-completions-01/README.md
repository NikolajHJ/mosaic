# Completed exploratory results,30 October2026

Question: add completed exploratory continuation outputs to the existing saved-result report without model evaluation or changing saved runs.

Plain Protenix job29855339 finished50,000 total updates at16:48 UTC: its final segment applied22,825 updates from27,175, with22,826 committed states, same model/context digests before/after, actual scheduler DONE and successful same-job LSF. No device-smoke states are adopted. DBMol29853835 was still queued at that observation; the three other experiment producers were running.

The canonical `scripts/reporting/report.py` command now accepts completed exploratory-v1/v2 wrappers through a versioned input adapter. It binds the original submission/plan/config/source and reuses the exact existing renderer and cache. The new panel explicitly covers global27175–50000; historical seven panels are preserved. Completion remains **unqualified** for derivatives/native correctness/binding.

Engine931a3fcd4be1093d9b7285405c897f711138b81949e9e8084ca225a25849e494 passed218 allocated CPU cases in job29856729. Its predecessor report test job29856718 had217 passes and one missing-output-directory fixture failure; that evidence is retained. No model/check algorithm was changed. Original scientific renderer SHA346eda6860d0a7b2ba02992b669e16cc0d9f4a751b26ed105573d8bc449822e7 is unchanged.

Command:
```powershell
py -3.13 scripts/reporting/report.py --profile experiments/reporting/jq1-live-progress/configs/dtu-report-home.json --runs results/hpc/20261030-exploratory-report-01/completed-selection.json
```

Request `request-20261030T164856Z-31fdc9a3` produced edition0008 below. Its completed request must not be resubmitted. Local artifacts are under `results/hpc/20261030-exploratory-report-01` and the existing reusable-report request directory. Budget: existing allocated4CPU report runner,16GiB,1h, bounded1GiB output; no GPUs or optimizer jobs submitted for reporting.

Published immutable edition0008 at16:54:24 UTC. Render job29856757 actual DONE/hpc/same-job LSF success; seven cached arms reused, only Protenix50k newly processed. All78served files matched exact bytes. Eight selectors, keyboard numeric39995, next40015, manual pause39975, slider27195/latest50000 and probability heatmaps passed actual browser QA. New delta20,357,697bytes;57local assets reused. See `publication.json`.

Boltz job 29855430 completed 30,000 total updates, verified at 20:05:18 UTC. Its final 20,000 to 30,000 segment has 10,000 applied updates and 10,001 contiguous states, identical frozen model/context digests and actual DONE/successful same-job LSF. See `boltz-completion.json`. Both chemistry jobs were running; DBMol and the new noise arm were queued. No optimizer was changed or resubmitted.

The canonical refresh published edition 0009 at 20:17:43 UTC, render job 29857654 DONE. Its new Boltz panel retains all 10,000 scalar records and 501 saved molecule frames; eight prior panels came from verified cache. All 85 HTTP assets and actual browser controls passed: numeric 25000, next 25020, previous 25000, slider 20020, playback, manual pause 25040, latest 30000, all nine selectors and both probability heatmaps. See `publication-0009.json`. The 13,336,668-byte delta is independently byte verified. Scientific scope remains unqualified exploratory; old failures remain visible.

Two report metadata defects were fixed: a cached exploratory header lost the version needed to select its adapter, and the HOME path recharged existing files against live quota while applying a legacy reserve. New HOME exploratory jobs use the shared remaining-output byte/file policy; explicit legacy accounting remains available. Known quota/disk/file shortfalls still fail. Engine CPU job 29857649 passed 225 cases, including the cache regression and six focused storage controls. The original scientific renderer bytes are unchanged. Both failed requests were rejected before submission and remain available as evidence. No experiment results/settings changed and no files were deleted.

Latest refresh command:
```powershell
py -3.13 scripts/reporting/report.py --profile experiments/reporting/jq1-live-progress/configs/dtu-report-home.json --runs results/hpc/20261030-exploratory-report-02/selection.json
```

On 31 October the completed Prochem 50k input exposed an adapter mismatch: its original producer has snapshot.json, not the later checksums.sha256. The adapter now verifies all named original manifest files/driver/plan and binds that actual proof; it writes nothing into the sealed source. Five portable controls passed. New engine CPU validation29860463 is PEND because the hpc queue is administratively inactive; expected230 cases are NOT yet claimed passed. Current request request-20261031T124153Z-03f8102b (local controller PID70968) will continue automatically; do not launch a second controller while it is alive. No render job/new edition yet; edition0009 remains authoritative.


### Verified31 October2026,21:05UTC

Edition0010 is published at http://127.0.0.1:8780/report.html; see `publication-0010.json`. Protenix+chemistry completed50,000 total updates (new panel10,000-50,000,40,000 applied,2,001 retained frames). Boltz+chemistry completed30,000 (new panel is the final28,986-30,000 recovery:1,014 applied,52 frames). The earlier wall-limited10,000-28,986 Boltz chemistry prefix remains saved separately and is not rendered in that panel. Old reports and failed checks remain unchanged.

The earlier pending CPU29860463 subsequently passed230 cases. Render29860699 completed its saved-frame audit but failed the old1GiB output budget; measured complete output was1,172,625,475 bytes. The report-only budget is now explicitly1.5GiB, with the same actual-capacity and output checks;18 portable metadata controls passed. Fresh immutable engine3af46190cd58 passed230 allocated cases in CPU29860721; render29861968 actually DONE/hpc/LSF succeeded. All109HTTP files and independent14-proof/XML/archive audits passed. All11 selectors, both new chemistry endpoints, numeric/arrows/slider/playback/manual pause and probability heatmaps passed actual browser checks. A source-manifest adapter authenticates original v1 snapshot.json; no scientific renderer/model evaluation or saved-result change. Interrupted DNS/capture requests and the failed cap evidence remain preserved.

Four requested budgets are now complete. DBMol29860458 is RUN at least12,919/30,000 and noise29857358 RUN at least26,250/50,000 (21:05UTC committed-prefix query). No active producer was restarted or changed. Completion remains exploratory, with original derivative warnings: the saved final Prochem argmax is invalid, while the saved final Bolchem argmax sanitizes; neither proves binding.

Current defaults include the completed chemistry panels. Refresh with the existing engine/cache:
```powershell
py -3.13 scripts/reporting/report.py --profile experiments/reporting/jq1-live-progress/configs/dtu-report-home.json
```
