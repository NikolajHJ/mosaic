# Historical attempt: 20261021-brd4-25-04

Experiment: [BRD4 25-slot three-model comparison](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> No compact campaign status; consult the preserved submission/status receipts.

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261021-brd4-25-04/campaign.json), SHA-256 `44ac87c50c3c8e8a3eaa8498ee4d9a238fe1d0b21328720a6fb68ed462e9dd6a` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261021-brd4-25-04) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "native_jobs": [
    {
      "submitted_at": "2026-10-21T08:25:49.982867+00:00",
      "args": [
        "bsub",
        "-q",
        "gpua10",
        "-n",
        "4",
        "-R",
        "span[hosts=1]",
        "-R",
        "rusage[mem=8GB]",
        "-W",
        "02:00",
        "-J",
        "mosaic25-validate-protenix",
        "-oo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/validate-protenix/stdout.%J.log",
        "-eo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/validate-protenix/stderr.%J.log",
        "-gpu",
        "num=1:mode=exclusive_process",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source/scripts/hpc/campaign25_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04",
        "e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79",
        "validate",
        "protenix",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/validate-protenix",
        "0"
      ],
      "stdout": "Job <29769176> is submitted to queue <gpua10>.\n",
      "stderr": "",
      "returncode": 0,
      "run": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/validate-protenix",
      "phase": "validate",
      "arm": "protenix",
      "job_id": "29769176"
    },
    {
      "submitted_at": "2026-10-21T08:25:50.471673+00:00",
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
        "mosaic25-after-ready-protenix",
        "-oo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/after-ready-protenix/stdout.%J.log",
        "-eo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/after-ready-protenix/stderr.%J.log",
        "-w",
        "ended(29769176)",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source/scripts/hpc/campaign25_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04",
        "e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79",
        "after-ready",
        "protenix",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/after-ready-protenix",
        "29769176"
      ],
      "stdout": "Job <29769177> is submitted to queue <hpc>.\n",
      "stderr": "",
      "returncode": 0,
      "run": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/after-ready-protenix",
      "phase": "after-ready",
      "arm": "protenix",
      "job_id": "29769177"
    },
    {
      "submitted_at": "2026-10-21T08:25:50.790275+00:00",
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
        "-W",
        "02:00",
        "-J",
        "mosaic25-prepare-none",
        "-oo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/prepare/stdout.%J.log",
        "-eo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/prepare/stderr.%J.log",
        "-gpu",
        "num=1:mode=exclusive_process",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source/scripts/hpc/campaign25_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04",
        "e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79",
        "prepare",
        "none",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/prepare",
        "0"
      ],
      "stdout": "Job <29769178> is submitted to queue <gpul40s>.\n",
      "stderr": "",
      "returncode": 0,
      "run": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/prepare",
      "phase": "prepare",
      "arm": "none",
      "job_id": "29769178"
    },
    {
      "submitted_at": "2026-10-21T08:25:51.072188+00:00",
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
        "mosaic25-after-prepare-none",
        "-oo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/after-prepare/stdout.%J.log",
        "-eo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/after-prepare/stderr.%J.log",
        "-w",
        "ended(29769178)",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source/scripts/hpc/campaign25_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04",
        "e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79",
        "after-prepare",
        "none",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/after-prepare",
        "29769178"
      ],
      "stdout": "Job <29769179> is submitted to queue <hpc>.\n",
      "stderr": "",
      "returncode": 0,
      "run": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/after-prepare",
      "phase": "after-prepare",
      "arm": "none",
      "job_id": "29769179"
    }
  ],
  "long_jobs": []
}
```

Saved source identity:

```json
{
  "archive_bytes": 39425961,
  "archive_sha256": "b62ef75f6ede08fbed8854ff4efe167571be6eb050c1b222278e19e15b3dc879",
  "manifest_sha256": "e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
