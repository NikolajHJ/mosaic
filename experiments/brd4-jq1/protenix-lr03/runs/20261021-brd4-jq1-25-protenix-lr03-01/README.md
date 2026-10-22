# Historical attempt: 20261021-brd4-jq1-25-protenix-lr03-01

Experiment: [JQ1 Protenix learning-rate 0.3 study](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> Superseded by user-requested fixed-step restart; CPU29776049 EXIT134passed/1failed fixture-path assertion; callback29776050EXIT; noGPU/longsubmitted

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-protenix-lr03-01/campaign.json), SHA-256 `0badb57dd0257590a3fee4e413e7b9a48ecd3634a3401c011d5c027fa78310f2` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261021-brd4-jq1-25-protenix-lr03-01) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "unchanged_live_boltz_jobs": [
    "29770093",
    "29770124"
  ],
  "jobs": {
    "cpu": {
      "submitted_at": "2026-10-21T14:10:56.299073+00:00",
      "args": [
        "bsub",
        "-q",
        "hpc",
        "-n",
        "4",
        "-R",
        "span[hosts=1]",
        "-R",
        "rusage[mem=4GB]",
        "-W",
        "01:00",
        "-J",
        "mosaic-protenix-lr03-cpu",
        "-oo",
        "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01/cpu/stdout.%J.log",
        "-eo",
        "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01/cpu/stderr.%J.log",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-protenix-lr03-01/source/scripts/hpc/protenix_lr03_cpu.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-protenix-lr03-01/source",
        "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01/cpu",
        "d5c1bf07868a6ef8f3c1a96a24afee11b76c8fb2affe334fb924d0a38b7849a0"
      ],
      "returncode": 0,
      "stdout": "Job <29776049> is submitted to queue <hpc>.\n",
      "stderr": "",
      "phase": "cpu",
      "job_id": "29776049"
    },
    "callback": {
      "submitted_at": "2026-10-21T14:10:56.680573+00:00",
      "args": [
        "bsub",
        "-q",
        "hpc",
        "-n",
        "1",
        "-R",
        "span[hosts=1]",
        "-R",
        "rusage[mem=2GB]",
        "-W",
        "00:20",
        "-J",
        "mosaic-protenix-lr03-after-cpu",
        "-oo",
        "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01/after-cpu/stdout.%J.log",
        "-eo",
        "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01/after-cpu/stderr.%J.log",
        "-w",
        "ended(29776049)",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-protenix-lr03-01/source/scripts/hpc/protenix_lr03_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-protenix-lr03-01/source",
        "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01",
        "d5c1bf07868a6ef8f3c1a96a24afee11b76c8fb2affe334fb924d0a38b7849a0",
        "after-cpu",
        "/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01/after-cpu",
        "29776049"
      ],
      "returncode": 0,
      "stdout": "Job <29776050> is submitted to queue <hpc>.\n",
      "stderr": "",
      "phase": "after-cpu",
      "job_id": "29776050"
    },
    "long_run_submitted": false,
    "native_submitted": false
  }
}
```

Saved source identity:

```json
{
  "archive_bytes": 39555775,
  "archive_sha256": "8c50ef45cf1d60d90b380f21e39aa41cebd95fa6aefd01c0fe5e107fde68ba5c",
  "manifest_sha256": "d5c1bf07868a6ef8f3c1a96a24afee11b76c8fb2affe334fb924d0a38b7849a0"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
