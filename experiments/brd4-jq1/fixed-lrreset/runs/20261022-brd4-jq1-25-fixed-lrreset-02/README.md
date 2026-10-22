# Historical attempt: 20261022-brd4-jq1-25-fixed-lrreset-02

Experiment: [JQ1 lower-rate restart, chemistry endpoint 100](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> Source622 staged/authenticated read-only; CPU and own-ended callback actually submitted. New CPU/native/long passes not yet established.

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261022-brd4-jq1-25-fixed-lrreset-02/campaign.json), SHA-256 `682fc53f2fa8638790952a753ad7b838622003ad2e5fd672baee228b061afd30` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261022-brd4-jq1-25-fixed-lrreset-02) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "initial_jobs": {
    "cpu": "29781363",
    "after_cpu": "29781364"
  },
  "cpu_submitted": true,
  "cpu_passed": false,
  "native_submitted": false,
  "native_passed": false,
  "long_submitted": false
}
```

Saved source identity:

```json
{
  "source_files": 622,
  "manifest_sha256": "832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3",
  "archive_bytes": 39601171,
  "archive_sha256": "306bdf055770e7245bb37de291800cb7da0998af7345aabb7d1b9e0b97057e25",
  "source_root": "/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-02/source"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.


## Later actual observation: 22 October 19:03 Paris

The migration-time snapshot above is historical. All three validation jobs subsequently exited at the entry storage gate before model execution; no long jobs were submitted. See the [dated outcome and exact evidence](status-20261022-1903.md).
