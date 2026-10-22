# Historical attempt: 20261021-brd4-25-release-07

Experiment: [BRD4 25-slot three-model comparison](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> No compact campaign status; consult the preserved submission/status receipts.

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261021-brd4-25-release-07/campaign.json), SHA-256 `a942c15ae5853826d49f23b5bbaec1cf0b52bed0a65bd537e80a4e69449e5d74` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261021-brd4-25-release-07) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "long_chemistry_job": {
    "job_id": "29769540",
    "actual_status_at_capture": "PEND",
    "submitted_paris": "2026-10-21 11:04:34",
    "resources": {
      "queue": "gpul40s",
      "exclusive_gpus": 1,
      "cpu_slots": 4,
      "memory_gib_per_slot": 8,
      "walltime": "23:00",
      "inter_arm_dependency": false
    }
  },
  "long_boltz_job": {
    "job_id": "29769525",
    "actual_status_at_capture": "RUN",
    "submitted_paris": "2026-10-21 11:02:11",
    "started_paris": "2026-10-21 11:04:20"
  }
}
```

Saved source identity:

```json
{
  "manifest_sha256": "80274d8d5f066af3e364f0356182ab0aa124302613800aa8b68fe3ed3cdec964",
  "archive_bytes": 10105,
  "archive_sha256": "61d23c4d5f6c7e34b59cdfec60947c1a001c838994fd175d5dc07a5a44c8127b"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
