# Historical attempt: 20261021-brd4-jq1-25-01

Experiment: [JQ1-pocket three-arm comparison](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> No compact campaign status; consult the preserved submission/status receipts.

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-01/campaign.json), SHA-256 `6595ed6487ff82a9b4e06c91a4c6df00f00d5dc48bd44caf911a9d38e2a2a86b` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-01) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "jobs": [
    {
      "job_id": "29769654",
      "phase": "cpu",
      "submitted": true,
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29769655",
      "phase": "after-cpu",
      "submitted": true,
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29769668",
      "phase": "validate-protenix",
      "last_observed_status": "EXIT"
    },
    {
      "job_id": "29769669",
      "phase": "after-ready-protenix",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29769670",
      "phase": "prepare-boltz-pocket",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29769671",
      "phase": "after-prepare",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29770007",
      "phase": "validate-boltz2",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29770008",
      "phase": "after-ready-boltz2",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29770009",
      "phase": "validate-boltz2_chemistry",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29770010",
      "phase": "after-ready-boltz2_chemistry",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29770093",
      "phase": "run-boltz2",
      "last_observed_status": "RUN",
      "submitted": true,
      "no_inter_arm_dependency": true
    },
    {
      "job_id": "29770124",
      "phase": "run-boltz2_chemistry",
      "last_observed_status": "PEND",
      "submitted": true,
      "no_inter_arm_dependency": true
    }
  ]
}
```

Saved source identity:

```json
{
  "archive_bytes": 39462299,
  "archive_sha256": "5e4bf8601f1b09e244a7763dc6c6d8af5ddc76bfd3490830993d7cfd0b2c0906",
  "manifest_sha256": "2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
