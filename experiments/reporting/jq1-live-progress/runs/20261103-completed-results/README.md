# Completed-results report refresh, 3 November 2026

The remaining Mosaic producers reached their requested budgets after DTU queues
reopened. Add DBMol through60k, the completed chemistry-first23 and structural
constant10 comparisons, and the valid annealed sparse50k result through the
canonical report command/cache. Preserve earlier results and uncertainty labels.

The old engine e2777dbb CPU29874871 failed4 of311 cases: Linux disk usage was
queried before the destination directory existed. The publisher now creates
that directory before checking available space. Existing positive/negative
publication tests cover the fix; allocated CPU29876081 passed all311 cases.
Original failure evidence is retained in [predecessor-failure.json](predecessor-failure.json).
No model, loss, saved state or experiment rerun was involved.

The interrupted DBMol30k child is explicitly included only through its actual
retained30,260 checkpoint; the final child contributes the remaining29,740.
The preserved four-update interrupted tail is not counted twice. The first
capture rejected that incomplete child under the complete-only adapter; the
selection now uses the existing saved-partial adapter with its exact original
source/config/job/LSF bindings. No validation criterion was weakened.

Command:
```powershell
.venv\Scripts\python.exe -B scripts/reporting/report.py --profile experiments/reporting/jq1-live-progress/configs/dtu-report-work3-exploratory.json
```

Current request: request-20261103T083805Z-082e0399. Publication proof is recorded
in campaign.json once complete; until then the prior edition remains active.
