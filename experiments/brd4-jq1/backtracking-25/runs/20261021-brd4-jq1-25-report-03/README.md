# Historical attempt: 20261021-brd4-jq1-25-report-03

Experiment: [JQ1-pocket three-arm comparison](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> 70 allocated report tests passed; first interim DONE/passed/downloaded/verified/published; final collectors/report wait on unchanged natives

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-report-03/campaign.json), SHA-256 `24d5148ec1d2041eafa331671e615d1df16d135ee87abcef92c8667d30f9e609` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-report-03) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "native_jobs": {
    "protenix": "29770179",
    "boltz2": "29770093",
    "boltz2_chemistry": "29770124"
  },
  "jobs": {
    "cpu": "29770614",
    "cpu-certificate": "29770615",
    "complete-protenix": "29770616",
    "complete-boltz2": "29770617",
    "complete-boltz2_chemistry": "29770618",
    "combined-report": "29770619",
    "interim-0001": "29770620"
  },
  "obsolete_report_jobs": "verified PEND before bkill, actual EXIT authenticated; see preserved cancellation receipts",
  "obsolete_pending_report_jobs": "29770616/29770617/29770618/29770619 intentionally retired while verified PEND after source04 CPU71pass; actual EXIT receipts preserved in source04 campaign"
}
```

Saved source identity:

```json
{
  "archive_bytes": 39510050,
  "archive_sha256": "3434ab4cabbac75441adfbf8be7d50fdbc5381d0740bf13c0ce92244076891e0",
  "manifest_sha256": "3b1ba56f8ddd0dc163333d9dabccd1119e323553a160a0e7b6b41790e73d7252"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
