# Restore notes for legacy-offload-03

The [laptop archive](C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/results/hpc/20261024-storage-offload-01/backup-03/legacy-offload-03.tar.gz) is **fully verified**: 8,769,931,895 bytes, SHA256 `ac75713c102dba31b0ea9934c2303ebd5cb529d5c98c422baccfc42afda29ff3`. The [canonical backup verification](C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/experiments/brd4-jq1/boltz-lr0025/runs/20261024-storage-offload-01/backup-verification-03.json) records verification of all 25,413 original files plus the embedded manifest at 2026-10-24 11:47:35 UTC. The [standalone verification](C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/results/hpc/20261024-storage-offload-01/backup-03/verification.json) and separately collected [completed stream receipt](C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/results/hpc/20261024-storage-offload-01/backup-03/stream-receipts-01/stream-complete.json) preserve the exact evidence.

The [canonical approval](C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/experiments/brd4-jq1/boltz-lr0025/runs/20261024-storage-offload-01/approved-selection-03.json) pins the [raw approved plan](C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/results/hpc/20261024-storage-offload-01/backup-03/approved-plan.json):

- Operation: `legacy-offload-03`.
- Plan: 15,558,449 bytes; SHA256 `8abfb724eaa6629ac73d6985d6216e898788868aec70a6784bcce8ba975e57bb`.
- Selected originals: 25,413 single-link files, totaling 15,725,430,139 logical bytes. Linked aliases were excluded and remain protected.
- Operator source04: manifest SHA256 `8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe`; compression level1 changes the compressed container, not original file bytes.

The first archive member must be `offload-manifest.json`, containing the exact raw approved plan. Each plan `files[]` row maps its original absolute `path` to a numbered `member`, beginning `files/00000000`. Use that mapping, original byte length and SHA256 to identify a file; a numbered filename alone does not identify its original location. A complete archive has 25,414 members including the manifest.

To repeat verification later, set `$ReceiptPath` to the completed stream receipt linked above and `$ReceiptSha256` to `e2858b0dcf70da1d33b72e3dc7a015facb7f0bddaa475a0aa5a0a6a69ac7843b`. Set `$FreshVerificationPath` to a new, absent output file; preserve the original verification. The real verifier CLI is:

```powershell
& 'C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' `
  'C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/experiments/brd4-jq1/boltz-lr0025/runs/20261024-storage-offload-01/verify_stream_backup_02.py' `
  --archive 'C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/results/hpc/20261024-storage-offload-01/backup-03/legacy-offload-03.tar.gz' `
  --plan 'C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/results/hpc/20261024-storage-offload-01/backup-03/approved-plan.json' `
  --plan-sha256 '8abfb724eaa6629ac73d6985d6216e898788868aec70a6784bcce8ba975e57bb' `
  --remote-receipt $ReceiptPath --remote-receipt-sha256 $ReceiptSha256 `
  --output $FreshVerificationPath
```

Only `verified_all_members` with the exact approved plan, authenticated receipt and complete archive establishes a usable backup. Preserve the archive, plan, approval, receipt and verification together; old partial archives belong to different operations and cannot substitute for this one.

For a later manual recovery, first repeat verification, then create a **fresh empty laptop directory**. Select the exact numbered member through its plan row, copy its opaque bytes there, and check its recorded length and SHA256 before using it. Keep the original absolute path as mapping metadata; do not automatically restore to that absolute path or overwrite active HPC files. This note supplies no restore tool, remote restore, model loading or scientific validation.

Preparation history: this note originally marked the in-progress archive unverified. The completed verification above supersedes that checkpoint. Only afterward did the exact deleter remove all 25,413 identical remote copies, totaling 15,725,430,139 logical bytes, with zero retained-changed or already-absent paths. The [post-deletion audit](C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/results/hpc/20261024-storage-offload-01/backup-03/post-deletion-audit-03/verification.json) verified every removed path absent and rehashed the protected inputs. Earlier operation01/02 partial archives remain separate historical evidence and are not substitutes for this verified archive.
