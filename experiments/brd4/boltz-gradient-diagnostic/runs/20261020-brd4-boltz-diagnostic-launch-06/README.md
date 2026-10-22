# Historical attempt: 20261020-brd4-boltz-diagnostic-launch-06

Experiment: [BRD4 Boltz gradient diagnostics](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> native diagnostic29763949 recorded/DONE; original readinessFD still failed; saved-recordCPUaudit29764094 complete; random long unsubmitted; followup paused

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261020-brd4-boltz-diagnostic-launch-06/campaign.json), SHA-256 `e4ee52605de0b8d315c4acda855bac8cf7e4e740ea0f3c75e233e75540bb98c2` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261020-brd4-boltz-diagnostic-launch-06) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "mix50_job": "29759212"
}
```

Saved source identity:

```json
{
  "manifest_sha256": "ec45d67b99f8e1645bfca1b228a2ce7b706edb4d9312886a08d74549aa138a4a",
  "archive_bytes": 14708,
  "archive_sha256": "fdbd19455b342aeafff6e5d2b8bca9b9eaa86b7e9533efe0745a0d92a91e53f4",
  "source_files": {
    "scripts/check_boltz_diagnostic_cpu.py": "6033ed57d25cedb3590b2cfaf0fff8ff662fdf31e5d2637ed1a8fd2cb9f80aa5",
    "scripts/diagnose_boltz_random_gradient.py": "251bfa76cb4fab680eb14d653803c3e6677a1573b0d7df56ce9f429c87f23e66",
    "scripts/verify_stage0_release.py": "d3e5373b84b123fe2388db2e9493a5a76d1cfc7edcc53f73ff5746f935c49c02",
    "scripts/hpc/brd4_boltz_diagnostic_cpu.lsf": "ed9d967eb102d18f8abead648c37556c1cc68ec60ea68014f7e6cd72ada5aa09",
    "scripts/hpc/brd4_boltz_random_diagnostic_retry.lsf": "b0882be6ff0bd4c54e4045ecb0a3bdc00f8f2bd4f335a94758681ea4ee24e915",
    "tests/test_boltz_diagnostic_cpu.py": "19887754cfc801c10ec8f724e54f02c4a14c59ec7bc731e5699f9e11cb8f2d0b",
    "tests/test_boltz_gradient_diagnostic_contract.py": "aeabec1d89ca826788f103dab19a41ccc143e8a1dbe46cef18151c3779700e3e",
    "scripts/hpc/brd4_boltz_diagnostic_certificate_cpu.lsf": "10d0526e63f6b7073e99698441829d453cb0ef3f20097aa436ae5d8c770c1c1d"
  }
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
