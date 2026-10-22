# Historical attempt: 20261019-brd4-boltz-independent-02

Experiment: [BRD4 frozen Boltz-2 affinity optimization](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> mix50 completed/passed; original diagnostic failed at launcher; replacement diagnostic29763693 submitted; random long arm uncertified

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261019-brd4-boltz-independent-02/campaign.json), SHA-256 `2bec582269a86ebbbcf09f0b8c0e1c2eb60b4bac574e95ad34b8535b2166fc6b` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261019-brd4-boltz-independent-02) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "parent_tools_job": "29759210",
  "cpu_job": "29759211",
  "scientific_cpu_job": "29759068",
  "native_parent_job": "29759069",
  "jobs": [
    {
      "kind": "mix50",
      "job_id": "29759212",
      "run": "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-mix50-independent-01",
      "args": [
        "bsub",
        "-q",
        "gpul40s",
        "-n",
        "4",
        "-R",
        "span[hosts=1]",
        "-R",
        "rusage[mem=8GB]",
        "-gpu",
        "num=1:mode=exclusive_process",
        "-W",
        "1380",
        "-J",
        "mosaic-boltz-mix50",
        "-o",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-mix50-independent-01/stdout.%J.log",
        "-e",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-mix50-independent-01/stderr.%J.log",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-02/source/scripts/hpc/brd4_boltz_run.lsf",
        "/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-02/source",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-mix50-independent-01",
        "5eb6e17ea8d19e52c99a589243228710994dcf137f59a6b9a6659a4fe719a7f4",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-independent-02-cpu/payload/mix50-ready",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-02-cpu/payload",
        "ca400a1477471790bcc5c93cd85a55455da77b331caa504f882b2020d954140e",
        "mix50"
      ],
      "mix50_assessment_sha256": "ca400a1477471790bcc5c93cd85a55455da77b331caa504f882b2020d954140e",
      "tools_cpu_assessment_sha256": "bc76e33927104e88a9e019b706d0ed4c6e1fc8f3bd4460264acad13ca7594448",
      "initial_actual_scheduler_status": "PEND",
      "initial_status_at": "2026-10-19T20:40:18+02:00",
      "latest_actual_scheduler_status": "DONE",
      "started_at": "2026-10-19T20:41:13+02:00",
      "latest_authenticated_at": "2026-10-20T08:59:37+02:00",
      "repeated_native_entry_gradient_status": "passed; not a completed long-run certificate",
      "scientific_run_status": "passed; terminal gates and frozen identities passed",
      "previous_saved_state_observation": {
        "checked_at": "2026-10-19T20:44:01+02:00",
        "states": 42,
        "state_step": 41
      },
      "last_partial_progress": {
        "applied_updates": 3970,
        "checked_at": "2026-10-20T00:35:04+02:00"
      },
      "last_operational_eta": {
        "checked_at": "2026-10-20T00:35:04+02:00",
        "mix_job": "29759212",
        "mix_status": "RUN",
        "applied_updates": 3970,
        "requested_updates": 10000,
        "recent_sample_updates": 50,
        "recent_mean_seconds_per_update": 4.479814045602689,
        "recent_pace_remaining_seconds": 27013.278694984216,
        "recent_pace_finish_estimate": "2026-10-20T08:05:17.278695+02:00",
        "eta_scope": "operational saved-timing extrapolation only; final native checks and later trial/search changes remain pending",
        "random_diagnostic_job": "29759213",
        "random_diagnostic_status": "PEND",
        "random_pending_reason": "physical-GPU reservation unavailable at two hosts; no inter-arm dependency"
      },
      "finished_at": "2026-10-20T08:35:34+02:00",
      "completed_updates": 10000,
      "saved_state_records": 10001,
      "completion_receipt": "scripts/hpc/receipts/20261020-brd4-boltz-mix50-completion-02/completion.json",
      "summary_sha256": "7f49087f12efc762a46691581f80bc42c43395c00b0cc4778a9e04a6679aae7c"
    },
    {
      "kind": "random-diagnostic",
      "job_id": "29759213",
      "run": "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-random-diagnostic-01",
      "args": [
        "bsub",
        "-q",
        "gpul40s",
        "-n",
        "4",
        "-R",
        "span[hosts=1]",
        "-R",
        "rusage[mem=8GB]",
        "-gpu",
        "num=1:mode=exclusive_process",
        "-W",
        "60",
        "-J",
        "mosaic-boltz-random-diagnostic",
        "-o",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-random-diagnostic-01/stdout.%J.log",
        "-e",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-random-diagnostic-01/stderr.%J.log",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261019-brd4-boltz-independent-02/source/scripts/hpc/brd4_boltz_random_diagnostic.lsf",
        "/work3/s234818/mosaic/releases/20261019-brd4-boltz-independent-02/source",
        "848cb213d91be09513439f719ece8e43ec55991bd473f1bc233f3d80eb204c1b",
        "/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-02/source",
        "5eb6e17ea8d19e52c99a589243228710994dcf137f59a6b9a6659a4fe719a7f4",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-02-ready/payload",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-independent-02-cpu/payload",
        "bc76e33927104e88a9e019b706d0ed4c6e1fc8f3bd4460264acad13ca7594448",
        "/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-random-diagnostic-01"
      ],
      "mix50_assessment_sha256": "ca400a1477471790bcc5c93cd85a55455da77b331caa504f882b2020d954140e",
      "tools_cpu_assessment_sha256": "bc76e33927104e88a9e019b706d0ed4c6e1fc8f3bd4460264acad13ca7594448",
      "initial_actual_scheduler_status": "PEND",
      "initial_status_at": "2026-10-19T20:40:18+02:00",
      "latest_actual_scheduler_status": "PEND",
      "latest_authenticated_at": "2026-10-20T08:59:37+02:00",
      "pending_reason": "physical GPU reservation unavailable at one host; no dependency"
    }
  ]
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
