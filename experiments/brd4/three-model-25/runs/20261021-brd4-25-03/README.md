# Historical attempt: 20261021-brd4-25-03

Experiment: [BRD4 25-slot three-model comparison](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> superseded after concrete final-response identity dictionary bug; native subchecks not a readiness pass

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261021-brd4-25-03/campaign.json), SHA-256 `f6ee044db3d4835aa691df6ad6ccf8a65cb86443716c854fff011978c9fb26a6` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261021-brd4-25-03) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "native_jobs": [
    {
      "submitted_at": "2026-10-21T08:12:27.609002+00:00",
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
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/validate-protenix/stdout.%J.log",
        "-eo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/validate-protenix/stderr.%J.log",
        "-gpu",
        "num=1:mode=exclusive_process",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-03/source/scripts/hpc/campaign25_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-03/source",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03",
        "1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b",
        "validate",
        "protenix",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/validate-protenix",
        "0"
      ],
      "stdout": "Job <29769026> is submitted to queue <gpua10>.\n",
      "stderr": "",
      "returncode": 0,
      "run": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/validate-protenix",
      "phase": "validate",
      "arm": "protenix",
      "job_id": "29769026"
    },
    {
      "submitted_at": "2026-10-21T08:12:27.917643+00:00",
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
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/after-ready-protenix/stdout.%J.log",
        "-eo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/after-ready-protenix/stderr.%J.log",
        "-w",
        "ended(29769026)",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-03/source/scripts/hpc/campaign25_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-03/source",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03",
        "1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b",
        "after-ready",
        "protenix",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/after-ready-protenix",
        "29769026"
      ],
      "stdout": "Job <29769027> is submitted to queue <hpc>.\n",
      "stderr": "",
      "returncode": 0,
      "run": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/after-ready-protenix",
      "phase": "after-ready",
      "arm": "protenix",
      "job_id": "29769027"
    },
    {
      "submitted_at": "2026-10-21T08:12:28.197208+00:00",
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
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/prepare/stdout.%J.log",
        "-eo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/prepare/stderr.%J.log",
        "-gpu",
        "num=1:mode=exclusive_process",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-03/source/scripts/hpc/campaign25_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-03/source",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03",
        "1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b",
        "prepare",
        "none",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/prepare",
        "0"
      ],
      "stdout": "Job <29769028> is submitted to queue <gpul40s>.\n",
      "stderr": "",
      "returncode": 0,
      "run": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/prepare",
      "phase": "prepare",
      "arm": "none",
      "job_id": "29769028"
    },
    {
      "submitted_at": "2026-10-21T08:12:28.507449+00:00",
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
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/after-prepare/stdout.%J.log",
        "-eo",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/after-prepare/stderr.%J.log",
        "-w",
        "ended(29769028)",
        "bash",
        "-l",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-03/source/scripts/hpc/campaign25_stage.lsf",
        "/work3/s234818/mosaic/releases/20261021-brd4-25-03/source",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03",
        "1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b",
        "after-prepare",
        "none",
        "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/after-prepare",
        "29769028"
      ],
      "stdout": "Job <29769029> is submitted to queue <hpc>.\n",
      "stderr": "",
      "returncode": 0,
      "run": "/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/after-prepare",
      "phase": "after-prepare",
      "arm": "none",
      "job_id": "29769029"
    }
  ],
  "long_jobs": []
}
```

Saved source identity:

```json
{
  "archive_bytes": 39420362,
  "archive_sha256": "bfc6ad6f0af3268551258de55ca1bd196bb7ff35f6f690a305bf64a0d34fcff7",
  "manifest_sha256": "1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b"
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
