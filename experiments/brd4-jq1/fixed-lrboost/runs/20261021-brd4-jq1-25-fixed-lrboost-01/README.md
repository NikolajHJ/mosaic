# Historical attempt: 20261021-brd4-jq1-25-fixed-lrboost-01

Experiment: [JQ1 higher fixed learning rates](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> All3higher-rate readiness and whole2FS/entry storage passed; three fresh runs actually started. Pro/Bol each saved10000updates then failed mandatory terminal gradient gates; chemistry remainsRUN.

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-lrboost-01/campaign.json), SHA-256 `44a715e44e9b9e9dd7964869344cfd2c8da09380d0fd741a1bc320fbb853a8e1` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-lrboost-01) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "stopped_native_jobs": [
    "29778203",
    "29778204"
  ],
  "cpu_submitted": true,
  "native_submitted": true,
  "long_submitted": true,
  "initial_jobs": {
    "cpu": "29778388",
    "after_cpu": "29778391"
  },
  "native_jobs": {
    "protenix": {
      "native": "29778424",
      "callback": "29778425"
    },
    "boltz2": {
      "native": "29778426",
      "callback": "29778427"
    },
    "boltz2_chemistry": {
      "native": "29778428",
      "callback": "29778429"
    }
  },
  "native_readiness_passed": true,
  "long_jobs": {
    "protenix": {
      "job_id": "29779177",
      "learning_rate": 3.0,
      "output_parent": "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01/run-protenix",
      "submitted_utc": "2026-10-21T23:23:09.872347+00:00"
    },
    "boltz2": {
      "job_id": "29779178",
      "learning_rate": 0.05,
      "output_parent": "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01/run-boltz2",
      "submitted_utc": "2026-10-21T23:23:10.281491+00:00"
    },
    "boltz2_chemistry": {
      "job_id": "29779179",
      "learning_rate": 0.05,
      "output_parent": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-fixed-lrboost-01/run-boltz2_chemistry",
      "submitted_utc": "2026-10-21T23:23:10.534368+00:00"
    }
  },
  "scientific_campaign_passed": false
}
```

Saved source identity:

```json
{
  "source_files": 610,
  "manifest_sha256": "9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323",
  "archive_bytes": 39582615,
  "archive_sha256": "e1a100aebf44698515ba161c1a001e338a902b250bf1a737f888df33f55098d1",
  "source_root": "/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
