# Historical attempt: 20261020-brd4-boltz-diagnostic-launch-05

Experiment: [BRD4 Boltz gradient diagnostics](../../README.md).

Record type: migration-time index of saved evidence, not a new execution. Indexed at 2026-10-22T10:07:46.767895+00:00. Actor: Codex organization agent.

## Last saved state and evidence

The following is the existing campaign receipt's statement, not a live scheduler query or a new scientific acceptance assessment:

> diagnostic29763693 EXIT1 before model due incompatible JUnit counts; source05 historical CPU20case pass preserved; replacement source06

[Authoritative campaign receipt](../../../../../scripts/hpc/receipts/20261020-brd4-boltz-diagnostic-launch-05/campaign.json), SHA-256 `fcdd0a0328686456ec1b5e871473a3bb2856d209647963b7a0cb0872dfc2d61b` at indexing. [Preserved receipt directory](../../../../../scripts/hpc/receipts/20261020-brd4-boltz-diagnostic-launch-05) contains the exact commands, job dependencies, source/config/input identities and saved evidence.

Saved job/stage fields (quoted metadata, not independently revalidated):

```json
{
  "mix50_completed_job": "29759212"
}
```

Saved source identity:

```json
{
  "manifest_sha256": "e14c41ada91a0a1750776d57affbb33edb782f6d05cacd3519fa543b338960be",
  "archive_bytes": 14124,
  "archive_sha256": "bd70f72ed4814e2627371b202db5927c5b1034c4b879676f82fa3522bf5c721d",
  "source_files": {
    "scripts/check_boltz_diagnostic_cpu.py": "6033ed57d25cedb3590b2cfaf0fff8ff662fdf31e5d2637ed1a8fd2cb9f80aa5",
    "scripts/diagnose_boltz_random_gradient.py": "251bfa76cb4fab680eb14d653803c3e6677a1573b0d7df56ce9f429c87f23e66",
    "scripts/verify_stage0_release.py": "d3e5373b84b123fe2388db2e9493a5a76d1cfc7edcc53f73ff5746f935c49c02",
    "scripts/hpc/brd4_boltz_diagnostic_cpu.lsf": "87dced728dc71e99de7dda0de286d99ddcc56fa358683af5d3d8065f17122aa8",
    "scripts/hpc/brd4_boltz_random_diagnostic_retry.lsf": "b0882be6ff0bd4c54e4045ecb0a3bdc00f8f2bd4f335a94758681ea4ee24e915",
    "tests/test_boltz_diagnostic_cpu.py": "77dc415c6907956181f1f871893089af853e2d86b14459ffe90d99ef5aabd3b9",
    "tests/test_boltz_gradient_diagnostic_contract.py": "aeabec1d89ca826788f103dab19a41ccc143e8a1dbe46cef18151c3779700e3e"
  }
}
```

Submitted/start/end timestamps, exact interpreter/resource vectors and native outcomes are retained in the linked raw receipts; fields not recorded there remain **unknown**. Read the latest dated status artifact before acting: a compact campaign summary may lag later observations. No command was replayed, no job restarted, no failed/skipped/unreached gate promoted, and no historical output was moved.
