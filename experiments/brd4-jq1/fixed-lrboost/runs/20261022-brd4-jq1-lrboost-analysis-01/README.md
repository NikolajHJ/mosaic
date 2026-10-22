# Historical attempt: 20261022-brd4-jq1-lrboost-analysis-01

Experiment: [JQ1 higher fixed learning rates](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> First high-rate saved-only report verified/published/browser-QA passed; original scientific failures preserved; separate diagnostic GPUs still pending

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261022-brd4-jq1-lrboost-analysis-01/campaign.json), SHA-256 `4f88461f64cf1ae7a20fdd7c779b68102f492813d52ec0b0c1e422a128c7d8d1` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261022-brd4-jq1-lrboost-analysis-01) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "original_jobs": {
    "protenix": "29779177",
    "boltz2": "29779178",
    "boltz2_chemistry": "29779179"
  },
  "cpu_submitted": true,
  "diagnostics_submitted": true,
  "report_submitted": true,
  "cpu_job": "29780814",
  "after_cpu_job": "29780815",
  "cpu_passed": true,
  "analysis_jobs": {
    "cpu": "29780814",
    "native_jobs": {
      "protenix": "29779177",
      "boltz2": "29779178",
      "boltz2_chemistry": "29779179"
    },
    "report_manifest": "214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc",
    "diagnostics": {
      "protenix": "29780821",
      "boltz2": "29780823"
    },
    "diagnostic_captures": {
      "protenix": "29780822",
      "boltz2": "29780824"
    },
    "collectors": {
      "protenix": "29780825",
      "boltz2": "29780826",
      "boltz2_chemistry": "29780827"
    },
    "active_interim": "29780829",
    "interim_parent": "interim-0001",
    "final": "29780830",
    "cpu_certificate_job": "29780815"
  },
  "report_job": "29781041",
  "report_job_actual_status": "DONE",
  "jobs": {
    "cpu": "29780814",
    "native_jobs": {
      "protenix": "29779177",
      "boltz2": "29779178",
      "boltz2_chemistry": "29779179"
    },
    "report_manifest": "214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc",
    "diagnostics": {
      "protenix": "29780821",
      "boltz2": "29780823"
    },
    "diagnostic_captures": {
      "protenix": "29780822",
      "boltz2": "29780824"
    },
    "collectors": {
      "protenix": "29780825",
      "boltz2": "29780826",
      "boltz2_chemistry": "29780827"
    },
    "active_interim": "29781041",
    "interim_parent": "interim-0002",
    "final": "29780830",
    "cpu_certificate_job": "29780815"
  }
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
