# Historical attempt: 20261021-brd4-jq1-25-protenix-02

Experiment: [JQ1 Protenix learning-rate 0.3 study](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> No compact campaign status; consult the preserved submission/status receipts.

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-protenix-02/campaign.json), SHA-256 `4367f345ac9bf8427702d62fb4f46071c810e0189e636e99aeeec72622ce9e15` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-protenix-02) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "jobs": [
    {
      "job_id": "29770038",
      "phase": "cpu",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29770039",
      "phase": "after-cpu-protenix",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29770090",
      "phase": "validate-protenix",
      "last_observed_status": "DONE"
    },
    {
      "job_id": "29770091",
      "phase": "after-ready-protenix",
      "last_observed_status": "DONE",
      "intentional_reversible_hold": "Review whole-three-arm quota across source01/source02 roots before bresume; native job unchanged",
      "hold_released_after": "authenticated fullnative and whole-three-arm quota pass"
    },
    {
      "job_id": "29770179",
      "phase": "run-protenix",
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
  "archive_bytes": 39465640,
  "archive_sha256": "52caceb014e66749d55ad85950e8cacdddd76e2ebeb5feaac6b975dec5f54b8a",
  "manifest_sha256": "40ab2d4532c477b5074d6631587ae15188941d5167016b7421f993257a064438"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
