# Authorized HPC storage maintenance

## Verified report and ingest offload, 13 November 2026

Completed at 00:33 CET; independent verification passed at 00:34 CET.
Removed **8,722,551,296 allocated bytes / 8.123509 GiB** from 2,340 files,
after downloading and verifying a local recovery archive. The final official
WORK3 quota reports **82.07/100.00 GiB used, 17.93 GiB free**.

| Material offloaded | Removed paths | Allocated GiB released |
| --- | ---: | ---: |
| Old Mosaic HTML viewer assets under `runs/reusable-saved-report-v3/requests/*/viewer/` | 2,290 | 6.963940 |
| Completed PubChem ingest intermediates under `data/pubchem-large-corpus-20261105a/ingested/` | 50 | 1.159569 |

The archive preserves all 2,443 inventoried files, including hard-linked
aliases and the 103 report receipts/metadata files retained on the HPC.
Original scientific run outputs, report engine code/cache metadata, environments,
the canonical PubChem corpus, the packed graph cache and the 60M split remain
in place. No jobs, callbacks, experiment settings or model weights were changed.
Independent verification checked every removed path, all 103 retained file
hashes, five protected manifest/checkpoint hashes (including epochs 45 and 46),
and the packed training graph file's original 13,882,943,501-byte length.

The verified **2,946,022,487-byte / 2.744 GiB** backup is outside OneDrive:

`C:\Users\nikol\HPC_Archives\DBMol_exploratory\2026-11-13\mosaic-report-viewers-and-pubchem-ingest.tar.gz`

Archive SHA-256:
`df631fbb559872f8085a066ccbb67b92e46f1969b98b0de526dfffe776ccd06b`.
Its adjacent `offload-plan.json` maps original WORK3-relative paths and file
hashes; `local-proof.json`, `completed.json` and
`independent-verification.json` retain recovery and removal evidence. Extract
into an empty directory to inspect/restore the old report viewers; archive
members retain the `mosaic/...` and `latentmol/...` directory structure.

The account changed concurrently during the audit: quota moved from 97.61 GiB
at first observation to 91.83 GiB before this cleanup's inventory completed.
Those changes are not attributed to this operation. The table counts only
last-link allocations actually unlinked here, not summed hard-link aliases or
the full account-wide quota difference. Initial broad `du` scans timed out;
bounded category scans and the exact selected-file inventory supplied the
cleanup evidence. Metadata and archival I/O ran on the login host; no numerical
model work was run there.

[Removal receipt](../outputs/hpc-storage/cleanup-20261113a/completed.json),
[independent verification](../outputs/hpc-storage/cleanup-20261113a/independent-verification.json),
and [local archive proof](../outputs/hpc-storage/cleanup-20261113a/local-proof.json).

## Historical models and retired HOME repository offloaded and verified, 27 October 2026

The user authorized moving old model payloads to local storage and removing
retired repositories other than DBMol and Mosaic. This is a separate operation
from the completed narrow cache cleanup below. **Both cold-payload and retired
HOME removals completed and passed independent verification.** The final
read-only category report subsequently failed without changing those completed
outcomes; its failure is recorded below.

The [combined verified summary](../outputs/hpc-cleanup/20261027-history-offload/summary.json)
binds every inventory, local proof, removal, independent verification and
reporting-failure receipt. **Conservative combined allocation removed is
7,341,853,184 bytes /6.83763361 GiB**, comprising **5.95346308 GiB WORK3** and
**0.88417053 GiB HOME**. This excludes the earlier cache cleanup below and the
additional HOME root-directory allocation. The maintenance is complete; the
category-report failures do not require another removal run.

| Verified offload | Removed paths / payloads | Allocated bytes removed |
| --- | --- | ---: |
| Inactive DBMol model/diagnostic payloads on WORK3 | 323 aliases /131 unique payloads | 6,392,482,304 (5.953463 GiB) |
| Retired `relative-weight-models` HOME repository | 6,950 files /577 directories /eight links | 949,370,880 (0.884170532 GiB, conservative non-root accounting) |

The HOME figure belongs to HOME, not WORK3. Its
[saved inode accounting](../outputs/hpc-cleanup/20261027-history-offload/derived-home-block-accounting.json)
identifies another32,768 allocated bytes for the root directory, totaling
949,403,648 including root. These are allocated-block observations, not a
measured HOME quota delta or a guarantee of immediate pool reclamation.

The successful [v6 checkpoint inventory](../outputs/hpc-cleanup/20261027-history-offload/stage-inventory007-remote-inventory.json)
and [transport receipt](../outputs/hpc-cleanup/20261027-history-offload/stage-inventory007.json)
record **131 unique inactive payloads / 323 path aliases**, totaling
**6,392,448,608 logical / 6,392,482,304 allocated bytes (5.953463 GiB allocated)**.
Hard-linked aliases are grouped by their underlying payload rather than counted
as independent storage savings. The inventory's historical
`inventoried_not_removed` status is superseded by the
[complete local proof](../outputs/hpc-cleanup/20261027-history-offload/local-proof.json)
at **15:27:40 CET**, the
[actual removal completion](../outputs/hpc-cleanup/20261027-history-offload/remove001-remote-completion.json)
at **15:35:35 CET**, and the
[independent verification](../outputs/hpc-cleanup/20261027-history-offload/verify001-remote-verification.json)
at **15:36:54 CET**. All323 selected paths are absent, the durable unlink
journal hash matches, all131 payloads have verified local copies and all138
protected hashes remain unchanged. Removal completion SHA256 is
`dc8cc95c2458e99a23dd2c8a9a26c87e42b8611358ad1cb31e6410604e4f42f4`.
Whole run directories, experiment evidence and logs were retained.

The safeguarded [completion workflow](../outputs/hpc-submissions/finish-verified-hpc-offload-20261027.py)
records every phase in
[`completion-workflow001.jsonl`](../outputs/hpc-cleanup/20261027-history-offload/completion-workflow001.jsonl).
It waited for complete `all_local_payloads_verified` proof before exact removal
and independent verification, both returncode0. It then stopped on the final
read-only storage-report failure at **15:37:29 CET**; no guard was waived or
deletion retried. The
[failed final audit001](../outputs/hpc-cleanup/20261027-history-offload/final-audit-001.json)
rechecked all138 protected hashes successfully, then hit
`TypeError: dict() got multiple values for keyword argument 'utc'` while
combining scheduler metadata with the report timestamp. It did not reach quota
or `du` commands. **A new complete category breakdown is therefore unavailable
from this receipt.** Immediate removal and independent-verification quota
observations report **45.80 ->39.84 /100 GiB used,60.16 GiB free on WORK3**.
These account-wide observations can include concurrent activity; the exact
removed allocation is the table's measure, not a quota-delta attribution.

The corrected [read-only audit002](../outputs/hpc-cleanup/20261027-history-offload/final-audit-002.json)
retained a **15:39:36 CET** observation confirming all138 protected hashes and
the same **39.84/100 GiB WORK3 usage**. Its project-root inventory shows only
`latentmol` and `mosaic` directly underWORK3, and only the two HPC guidance
Markdown files under HOME `projects/`; the retired HOME repository is absent.
This covers those explicit roots rather than proving the absence of unrelated
repositories anywhere in the account. The HOME quota command reports
**21.19G of30.00G**, but explicitly caches information for about240minutes and
last updated **14:04:42**, before HOME removal: **do not treat that number as
post-cleanup HOME usage**. Audit002 then lost SSH with
`client_loop: send disconnect: Connection reset`, returned1 and saved partial
evidence without any `du` category records. The separate
[category retry003](../outputs/hpc-cleanup/20261027-history-offload/final-breakdown-003.json)
was rejected before login with `Permission denied (password)` /returncode255;
it made no change. A fresh post-cleanup `du` breakdown remains unavailable.
**The last verified WORK3 quota is39.84 GiB used /60.16 GiB free at15:39:36 CET;
this is a dated observation, not current usage.** Completed offload and
independent-verification evidence remain valid; no cleanup was rerun.

The summary provides the following **derived allocation estimate**, using the
earlier **13:59 CET directory snapshot** and subtracting only the exact verified
6,392,482,304-byte removal from `runs/`. It is **not a fresh `du` measurement**;
other concurrent writes/removals are excluded from this arithmetic.

| DBMol category | Estimated allocated GiB |
| --- | ---: |
| Model / CCD caches | 7.81 |
| Runs, checkpoints and saved experiment outputs | 7.72 |
| Active environment | 5.74 |
| Training data | 4.27 |
| Protenix occupancy work | 1.99 |
| Immutable source releases | 1.11 |
| Other DBMol files, including submissions and preparation | 1.02 |
| **DBMol `latentmol` total, derived** | **29.66** |

Mosaic was protected throughout. Its fresh exact allocation was not measured;
this table and account-wide quota do not establish a new exact Mosaic total.

Cold payloads are preserved outside OneDrive under
`C:\Users\nikol\HPC_Archives\DBMol_exploratory\2026-10-27\cold-models-history006`.
`payloads/` contains all131 byte-verified payloads; `restore-map.json` maps each
SHA256 and logical size to every original alias. Restore by copying a payload
once to its original filesystem and recreating the remaining aliases as hard
links on that filesystem. The map SHA256 is
`eb027d4faa1e20e294b76a2477fd6b9f2e2a9e4af4f78a5467450988c78fde86`;
the local proof SHA256 is
`bbc8e1446393b43b237ab2f805de2acc03bc4bf1df7647ea24c661294ffe549f`.
The backup also retains `remote-inventory.json`, `local-proof-001.json` and
transfer evidence; removal is recoverable offload, not intentional loss of the
older scientific model payloads.
An exact copy of the combined summary is also preserved outside OneDrive at
`C:\Users\nikol\HPC_Archives\DBMol_exploratory\2026-10-27\cleanup-summary.json`.
The inventory excludes live dependencies and checks **138 protected hashes**,
including authentic original epochs35/36, completed epoch40, the trained bridge,
all50 prepared source056 contexts, current source140 and independent embedding
work. All Mosaic roots are protected, and post-removal checks passed. The
inventory's dependency audit inspected
4,203 metadata files / 1,306,858,687 bytes and found no incoming Mosaic symlink
to a selected payload.

The separate retired HOME root is
`/zhome/95/e/206121/projects/relative-weight-models`. This is distinct from the
WORK3 root deleted on23 October. Its
[remote archive receipt](../outputs/hpc-cleanup/20261027-history-offload/home-pack002-remote-home-repository-archive.json)
records **6,950 files / 577 directories / eight symlinks**,
**467,965,198 logical / 949,370,880 allocated bytes (0.884171 GiB allocated)**.
The [complete local tar verification](../outputs/hpc-cleanup/20261027-history-offload/home-local-proof.json)
at **14:37:43 CET** verifies all7,535 entries, including `.git` and symlink
metadata, in the481,607,680-byte archive:

`C:\Users\nikol\HPC_Archives\DBMol_exploratory\2026-10-27\retired-home-relative-weight-models\relative-weight-models.tar`

SHA256 `b8f57d5573dfe077b100e24445f206b86cde94879bbe58ee72996445f63ea5ba`.
External symlinks are preserved as links rather than followed. Restoration is
by extracting into a chosen empty directory. After local verification, the
[removal completion](../outputs/hpc-cleanup/20261027-history-offload/home-remove001-remote-home-removal-completion.json)
at **14:42:54 CET** records
`completed_verified_retired_home_repository_offload`, exact original-root
absence and949,370,880 allocated bytes removed. Its SHA256 is
`131cbc5adeb3cb0bbb7ff1e168deb8387aa76a605476a9ebf2ec000eca06269a`.
The [independent verification](../outputs/hpc-cleanup/20261027-history-offload/home-verify001-remote-home-verification.json)
passed: the project and remote transfer archive are absent and all138 protected
hashes remain unchanged. The complete local tar remains the recovery copy.
This is verified offload, distinct from the intentional WORK3 root deletion
on23 October. The transfer archive was temporary WORK3 staging and its removal
does not establish a net checkpoint-space saving.
The [project inventory](../outputs/hpc-cleanup/20261027-history-offload/projects001.json)
found DBMol's `latentmol` and Mosaic onWORK3 plus this retired HOME repository;
it does not establish that every unrelated repository elsewhere has been found
or removed.

All unsuccessful audits remain recorded, with the broad inventory and staged
checkpoint inventory kept distinct:

- Broad `inventory001.json` reached its read-only audit timeout;
  `inventory002.json` rejected an invalid LSF display width of20000;
  `inventory003.json` completed the selected-project audit, retaining partial
  `du` timeouts as partial observations.
- `stage-inventory001.json` stopped read-only on a16MiB metadata bound;
  `stage-inventory002.json` timed out in SSH;
  `stage-inventory003.json` / `stage-inventory004.json` were denied login;
  `stage-inventory005.json` / `stage-inventory006.json` stopped read-only on
  a256MiB Mosaic summary. The v6 operator streams large metadata instead of
  skipping it; `stage-inventory007.json` then passed.
- `home-pack001.json` stopped read-only on the same256MiB metadata bound;
  `home-pack002.json` completed archival successfully.

These were not failed deletions, and no numerical tests, model training, job
changes or scientific-setting changes are part of this maintenance operation.

## Narrow cache cleanup completed and verified, 27 October 2026

The authorized cleanup removed **451,529,216 allocated bytes**, about
**430.6118 MiB / 0.42052 GiB**, with separate treatment of the recoverable
duplicate archive and disposable generated caches:

| Removed material | Files | Allocated bytes |
| --- | ---: | ---: |
| Exact duplicate `cache/boltz2/mols.tar.gz`, preserved locally | 1 | 406,779,392 |
| Unsealed generated `.pyc`/pytest cache files in inactive releases | 1,310 | 44,749,824 |
| **Total** | **1,311** | **451,529,216** |

The [archive removal](../outputs/hpc-cleanup/20261027-safe-storage/archive-remove001.json)
completed at **12:40:49 CET** and the
[cache removal](../outputs/hpc-cleanup/20261027-safe-storage/cache-remove001.json)
at **12:42:19**, with zero skipped cache files. Independent
[archive verification](../outputs/hpc-cleanup/20261027-safe-storage/archive-verify001.json)
and [cache verification at 12:43:55](../outputs/hpc-cleanup/20261027-safe-storage/cache-verify001.json)
passed: every selected path absent, valid operation journals and all **138
protected hashes unchanged**. The audit also checked active commands, plans,
source and symlinks for dependencies before removal. The completed remote
receipts were copied locally and hash-verified; their exact identities are
recorded in the [combined summary](../outputs/hpc-cleanup/20261027-safe-storage/summary.json).

The removed gzip's **406,779,014 logical bytes** remain recoverable from the
[exact local copy and restoration map](../outputs/hpc-cleanup/20261027-safe-storage/archive-restoration-map.json),
SHA256 `d1e4f47aa69824d37641e70cc60ebae582ccfe6e99d6b4868d791b8e9bc41116`,
confirmed again after cleanup.
Its decompressed payload [matched the retained native archive exactly](../outputs/hpc-cleanup/20261027-safe-storage/archive-proof001.json):
`/work3/s234818/latentmol/cache/boltz2/mols.tar`, SHA256
`39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7`.
**The native CLI requires that `.tar` file**; it, the extracted `mols/` tree and
all weights were retained. The final extracted CCD inventory remained **45,227
files / 1,820,698,819 logical bytes**, matching the initial inventory.

The [12:44:20 CET observation](../outputs/hpc-cleanup/20261027-safe-storage/after001.json)
reported **45.73/100 GiB used, 54.27 GiB free**. This is an account-wide quota
snapshot, not an exact attribution of concurrent cross-project storage changes;
the table gives the cleanup's authenticated removal bytes. Both native DBMol
GPU campaigns remained pending naturally. No job or scientific setting was
changed, and source140/VAE continuation, earlier evidence and Mosaic were
untouched. [Combined cleanup summary](../outputs/hpc-cleanup/20261027-safe-storage/summary.json).

## Historical source056 contexts restored exactly, 23 October 2026

To support the separately authorized matched embedding comparison, **44**
source056 native prepared contexts offloaded on 21 October were restored to
their original HPC paths from the verified local cold2 archive. The
[completion](../outputs/hpc-submissions/source138-context-restore-20261023/completion.json)
at **21:30:43 DTU-local** records **1,451,429,736 raw bytes**, exact size/hash
matches and unchanged protected original epoch36, trained bridge and source136
release archive. The [transport/restoration receipt](../outputs/hpc-submissions/source138-context-restore-20261023/remote-restoration.json)
and [independent verification](../outputs/hpc-submissions/source138-context-restore-20261023/independent-verification.json)
preserve the operation; the latter rehashed all **50** context-catalog entries
at **21:31:52** and passed. No model regeneration, numerical execution,
checkpoint/source mutation or job/setting change occurred during restoration.

The original local cold archive remains preserved with SHA256
`ec6797d064f5219b1279ea12b92a737a314bbfd02ebd8c3be25c690968ce866e`.
Only the requested contexts were restored; all historical cleanup/deletion
receipts remain unchanged. Immediate post-restoration work3 quota was
**64.41/100 GiB used, 35.59 GiB free**; subsequent concurrent storage movement
is not part of this restoration's byte count. This supersedes the historical
absence of those44 paths without changing the original verified-offload result.
[Comparison execution record](78_embedding_latent_comparison.md#exact-context-bank-restored-unchanged-attempt003-cpu-submitted).

## Retired relative-weight-models directory removed, 23 October 2026

The user explicitly authorized further cleanup of old repositories, including
deletion of pretty much all of their contents. The selected exact root was
`/work3/s234818/relative-weight-models`. Its remaining **32,955 regular files,
73 symlinks and 3,418 descendant directories** were removed, followed by the
root itself. Recorded descendant allocation is **15,529,641,984 bytes /
14.463106 GiB**; the root adds 512 allocated bytes. This is intentional
deletion of retired artifacts, not verified offload or a promise of recovery.

| Removed category | Allocated GiB |
| --- | ---: |
| Old virtual environments | 7.390863 |
| Model/download caches | 3.997335 |
| Remaining data/model assets | 3.073013 |
| Old run artifacts | 0.001894 |
| Empty/temporary directory allocation | less than 0.000001 |

The [passing read-only audit002](../outputs/hpc-results/storage-cleanup-20261023-001/audit002.json)
inspected **1,309 active executable/configuration files**, **508,138 entries**
across protected projects, and **806 symlinks**. No active job/source or incoming
symlink dependency on the retired root was found. All **51 protected hashes**
matched, retaining original epochs 35/36/89, source135 proof, source136 release/
plan and the independent source137 archive/manifest/plan. The actual deletion
repeated the live-job, symlink, protected-hash and complete tree-metadata checks
before changing any path. Fresh owner/device/type and exact-root guards passed.

The [actual removal001](../outputs/hpc-results/storage-cleanup-20261023-001/remove001.json)
returned zero at **10:49:53 UTC / 12:49:53 DTU-local**. A durable intent preceded
renaming the authenticated root to the exact temporary quarantine path
`/work3/s234818/.cleanup-relative-weight-models-20261023-001`, then fd-based,
symlink-safe `shutil.rmtree` removed that directory without following links.
Both original and quarantine paths are absent and all 51 protected hashes
matched again. Official work3 usage was **75.55 → 61.03 / 100 GiB**,
**38.97 GiB free** immediately afterward. Quota differences include concurrent
activity; the exact selected allocation above is the removal measure.
The [independent final verification](../outputs/hpc-results/storage-cleanup-20261023-001/verify001.json)
passed at **10:51:41 UTC / 12:51:41 DTU-local**: both removed paths are absent,
all four ordered journal events and receipt hashes match, and all 51 protected
hashes remain unchanged. The remote intent/journal/completion were copied to
[completed-remote001](../outputs/hpc-results/storage-cleanup-20261023-001/completed-remote001/completion.json)
and their exact byte hashes verified locally. The final fresh quota is
**61.04 / 100 GiB used, 38.96 GiB free**. The
[combined summary](../outputs/hpc-results/storage-cleanup-20261023-001/cleanup-summary.json)
has SHA256 `f6efcb0e13ab0d5744933bb2d489252e744c62b848ab7e4a306df031c1126d49`.

| Remaining project/category | Allocated GiB observed |
| --- | ---: |
| `latentmol` total | 47.46 |
| Its saved runs/checkpoints/evidence | 25.96 |
| Its model caches | 8.18 |
| Its active environment | 5.74 |
| Its training data | 4.27 |
| Its Protenix occupancy work | 1.26 |
| Its immutable releases | 1.08 |
| Its other files | 0.98 |
| Active `mosaic` total, approximate | 12.50 |

The `latentmol` `du` returned zero. Mosaic's live `du` returned one because a
writer's `step-2587.json.tmp` disappeared during traversal; its 12.50 GiB is
therefore an approximate concurrent observation, not an immutable verified
total or a cleanup failure. Its largest reported category is 6.69 GiB runs,
then 3.29 GiB releases and 2.46 GiB cache. Do not force these filesystem totals
to equal the separately measured official quota or attribute concurrent file
changes to this removal. No Mosaic path was selected.

`latentmol` and `mosaic` were excluded from deletion. Mosaic has actual running
and queued jobs; it is not treated as a retired repository. No current source,
checkpoint, environment, data, model weight, experiment setting or job under
those protected roots was deleted or changed. Active commands were inspected,
not canceled, modified or submitted.

The broad initial `du` audit exceeded its 65-second read-only cap. The first
dependency audit also stopped without mutation because the old guard could not
resolve the generated source137 bash script's relative executable. The local
operator now tokenizes shell punctuation and resolves that executable within
its explicit release root; audit002 then passed. Both preliminary records and
their errors remain preserved. No failed mutation was automatically retried.
The standard-library operator is
[`retired-repository-cleanup-20261023.py`](../outputs/hpc-submissions/retired-repository-cleanup-20261023.py);
all model/numerical work remains outside this maintenance operation.

## Other-repository datasets removed and independently verified, 22 October 2026

The user explicitly requested cleanup and deletion of datasets belonging to
other repositories. Two exact allowlist operations removed **94 entries: 72
regular files and 22 internal dataset-cache symlinks**, **15,448,249,017 logical /
15,448,273,920 allocated bytes (14.387326 GiB)**. These were intentionally deleted
datasets, not offloaded scientific evidence. No scientific source, checkpoint,
failure record, environment, pretrained model weight or active-job setting was
removed or changed. The earlier cold-checkpoint inventory was read-only; no
checkpoint backup/removal was needed.

| Completed operation | Exact selected scope | Reclaimed allocated space | Immediate official work3 usage |
| --- | --- | ---: | ---: |
| [Phase 1 result](../outputs/hpc-results/storage-cleanup-20261022-001/dataset-cleanup-local-result001.json), 17:32:12 UTC | 38 regular dataset/feature-store files under `relative-weight-models/data/datasets` and `relative-weight-models/cache/huggingface/datasets` | 9.318947 GiB | 85.93 → 76.61 / 100 GiB |
| [Phase 2 result](../outputs/hpc-results/storage-cleanup-20261022-001/dataset-cleanup-local-result002.json), 17:46:37 UTC | 34 regular files and 22 internal links in HF `datasets--ag_news`, `datasets--glue`, `datasets--phiyodr--coco2017`, `datasets--sxj1215--mmimdb` namespaces | 5.068379 GiB | 76.99 → 71.92 / 100 GiB |

The independently queried final quota at **17:47:07 UTC** was **71.92 GiB used /
28.08 GiB free**. The 0.38 GiB growth between operations belongs to concurrent
activity, so the aggregate official change is not the exact bytes reclaimed.
Quota is observational, not a reservation for future training.

The final [sequential depth-2 inventory](../outputs/hpc-results/storage-cleanup-20261022-001/after002.json)
completed at **17:48:33 UTC**. The [combined machine-readable summary](../outputs/hpc-results/storage-cleanup-20261022-001/cleanup-summary.json)
binds every result, verification and durable receipt by SHA256 and retains all
before/after category bytes.

| Project | Before, 17:16 UTC | After, 17:48 UTC |
| --- | ---: | ---: |
| `latentmol` | 47.32 GiB | 47.40 GiB |
| `relative-weight-models` | 28.85 GiB | 14.46 GiB |
| `mosaic` | 9.66 GiB | 8.95 GiB |

Remaining `latentmol` space is primarily **25.96 GiB runs**, **8.27 GiB cache**,
**5.74 GiB environment**, **4.27 GiB data**, **1.12 GiB Protenix work** and
**1.06 GiB releases**. These retained categories include current inputs and
authoritative evidence. No `mosaic` bytes were selected: its observed size
change and `latentmol` growth reflect concurrent activity. Filesystem `du`
allocation totals and official quota are separate observations and need not
match exactly.

[Phase 1 independent verification](../outputs/hpc-results/storage-cleanup-20261022-001/independent-verification001.json)
passed 38 absences and 76 ordered intent/success journal rows. Its 227 preserved
asset entries included HF dataset metadata as well as pretrained assets; they
were not 227 pretrained models. The separately authorized
[phase 2 verification](../outputs/hpc-results/storage-cleanup-20261022-001/independent-verification002.json)
passed 56 absences and 112 ordered journal rows, with the remaining **167 asset
entries and four zero-byte negative-cache markers unchanged**. Both phases
verified all **48 protected scientific/checkpoint/source/readiness hashes**.
This includes original epoch 36, frozen reference 35, original 89 and source135/
source136 archives, manifest bytes, plans and adopted proof artifacts. All HF
`models--*` caches, GPT-2 and pretrained gensim embeddings remain. No Mosaic
dataset was selected; active Mosaic/Protenix inputs were preserved.

The operators re-audited actual live commands, scripts, plans and source inputs
before deletion, including source136's nested training command. The final input
audit covered **802 metadata files / 8,291,058 bytes**; no selected-dataset
dependency was found. Every selected file had exact fresh owner, link count,
type, canonical scope, fingerprint and full SHA checks before any unlink.
Internal links were constrained to authenticated blobs in their own dataset
namespace and removed first. Each unlink has a durable intent/success pair.
Only standard-library metadata/byte inspection ran; tensor archives were
inspected through ZIP/pickle opcodes without unpickling or model execution.

| Durable receipt set | Intent SHA256 | Journal SHA256 | Completion SHA256 |
| --- | --- | --- | --- |
| [Phase 1](../outputs/hpc-results/storage-cleanup-20261022-001/completed-remote001/completion.json) | `363209c8b8fcfb360815ad49b86fb287fa167f069cd3f0104112d7de235caf65` | `aefef1408881550316d055b6aa915b4c17cf9963dbcd510e0165efeed3d9ef5b` | `4b632007c4527ae926403698ba9074442f2a5f71eb116ae72010ea7d4f9c0dc4` |
| [Phase 2](../outputs/hpc-results/storage-cleanup-20261022-001/completed-remote002/completion.json) | `ad744ecdc212533cadcf1a201e9723c769c694912735796f74cbe691201389ff` | `22f54980d83d8771401c22d2f07e35571c1f4c58d7a28acf96ab30f2170d5e63` | `916e3edf161652499a69a9a911cb94a3d4f33d899410e2acdabbcb8a86d7ce54` |

All preliminary failures were read-only/pre-mutation and remain preserved.
They cover legitimate interpreter-alias discovery, nested-shell/release input
discovery and conservative retention of code-looking negative-cache markers.
The first unexecuted operator's semantic-manifest/file-hash mismatch was fixed
in a separate v2 operator before execution. No failed or uncertain deletion was
automatically retried.

## Verified local offload completed, 21 October 2026 18:45:55 UTC

The [v2 operation](../outputs/hpc-results/storage-cleanup-20261021-001/cleanup-operation-result-002.json)
returned **0**, `completed_verified_local_offload`: **260 files** removed,
**4,969,885,698 logical / 4,969,949,184 allocated bytes (4.628626 GiB)**.
Exactly 166 cold payloads and 94 duplicate transfer capsules were unlinked only
after complete local byte/hash preservation and fresh remote input/stat/hash
guards. The [independent verification](../outputs/hpc-results/storage-cleanup-20261021-001/independent-verification002.json)
passed at **18:46:12 UTC**: all 260 paths absent, **520 ordered intent/success
journal rows**, unchanged protected epoch-35/36/89 checkpoint hashes, **11
retained contexts** and **five pinned plans**. No job was stopped/submitted,
scientific setting changed, or numerical/model work performed.

Immediate official work3 usage was **87.10 → 82.47/100 GiB**; the independent
fresh query also returned **82.47 used / about 17.53 GiB free**. The earlier
inventory's **91.53 → 87.10 GiB** movement preceded this cleanup and reflects
concurrent activity; do not attribute the full 9.06 GiB interval to this operation.
Free quota is a snapshot, not reserved future training capacity.

All removed bytes remain recoverable. The [166-file restoration map](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-restoration-map.json)
binds original HPC paths to [cold2-payloads.tar.gz](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-payloads.tar.gz)
(**1,064,073,926 bytes**, SHA256
`ec6797d064f5219b1279ea12b92a737a314bbfd02ebd8c3be25c690968ce866e`),
whose [local proof](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-local-proof.json)
verified all raw member hashes. The [94-capsule restoration map](../outputs/hpc-results/storage-cleanup-20261021-001/capsule-restoration-map.json)
and [capsule proof](../outputs/hpc-results/storage-cleanup-20261021-001/capsule-local-proof.json)
identify their existing complete local copies. Restore the exact recorded
member/file to its original absolute path and verify its SHA256 before historical
replay; no model rerun is needed. Retained examples are **ten proteinC13 contexts
plus one random-study count-10 context**, 11 total. All **eight epoch-9/10
checkpoints across four source075 continuations** remain; no best/last path was
selected, without implying that such basenames exist. Data, environments,
original/current checkpoints, source and failed/raw evidence remain protected.

| Durable remote receipt | SHA256 |
| --- | --- |
| [Intent](../outputs/hpc-results/storage-cleanup-20261021-001/completed-remote002/intent.json) | `c4c398681f6d1d3a8fda211ae2ab5d53e2f57846dccad3ec20f3456b227d1407` |
| [Unlink journal](../outputs/hpc-results/storage-cleanup-20261021-001/completed-remote002/unlinks.jsonl) | `612dae55d102fcc5db548bc06812b46aa3534e39ca7e05fb828a2d89f048da57` |
| [Completion](../outputs/hpc-results/storage-cleanup-20261021-001/completed-remote002/completion.json) | `f74cdb2d0946c71268401faad94fbbaf772e1a83050cee89aaf7d89f529323aa` |

Root used bundled CPython with command tails
`outputs/hpc-submissions/backup-cold-payloads-20261021-v2.py`,
`outputs/hpc-submissions/audit-hpc-cleanup-live-jobs-20261021-v4.py`,
`outputs/hpc-submissions/remove-verified-cold-payloads-20261021-v2.py` and
`outputs/hpc-submissions/verify-hpc-cold-cleanup-20261021.py`.
[Audit 004](../outputs/hpc-results/storage-cleanup-20261021-001/live-job-audit004.json)
passed 3,559 files/four Mosaic input roots and 87 covered internal fixture-link
targets; three mutable trajectory outputs were identified from their writer/
invocation identities. Unrelated running Mosaic jobs were untouched. Earlier
access, tar-stream, scheduler and audit failures below remain preserved;
[audit 003](../outputs/hpc-results/storage-cleanup-20261021-001/live-job-audit003.json)
also stopped on fixture-link handling without changes before corrected audit004.

## Access restored; local preservation verified, deletion pending, 21 October 2026

This earlier preservation-stage observation is superseded by the verified
completion above; its partial/failure receipts remain unchanged.

The user requested another retry and [inventory 004](../outputs/hpc-results/storage-cleanup-20261021-001/inventory004.json)
succeeded (return code 0), followed by detailed/runs/cold-payload inventories.
Official work3 usage was **91.53/100 GiB**, **8.47 GiB free**. This is a fresh
inventory observation, not recovered space. [Cold-offload plan](../outputs/hpc-results/storage-cleanup-20261021-001/cold-offload-plan.json)
SHA256 `21be505ba54377ebfe246a6f9067eed7ae3097a91efe42464cc50550fbd0d2d6`
selects **166 files**, **4,943,670,740 logical / 4,943,711,232 allocated bytes**:
134 old source056/source057 prepared contexts and 32 source075 epoch-1–8
checkpoints. The selected keep set retains ten proteinC13 contexts and one
random-study count-10 context, 11 total (count-13 already absent at inventory),
and all eight source075 epoch-9/10 files across four continuations.
No selected basename is best/last; the inventory does not prove those
names exist. Epoch 36/reference 35/original 89, current data/environments,
scientific source/run evidence and unrelated Mosaic work remain protected.

[Successful backup transfer](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-transfer.json)
retains the complete [cold2-payloads.tar.gz](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-payloads.tar.gz),
**1,064,073,926 bytes**, SHA256
`ec6797d064f5219b1279ea12b92a737a314bbfd02ebd8c3be25c690968ce866e`.
[Local proof](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-local-proof.json)
verifies every one of the **166 raw archive-member hashes** against the plan;
the [restoration map](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-restoration-map.json)
maps original remote paths to exact preserved members. Separately,
[capsule proof](../outputs/hpc-results/storage-cleanup-20261021-001/capsule-local-proof.json)
verifies **94 duplicate transfer capsules**, **26,214,958 logical / 26,237,952
allocated bytes**, in their existing complete local copies, with a separate
[capsule restoration map](../outputs/hpc-results/storage-cleanup-20261021-001/capsule-restoration-map.json).
These preservation proofs are not deletion receipts or numerical validation.

**No removals yet.** The [first streaming backup](../outputs/hpc-results/storage-cleanup-20261021-001/cold-transfer.json)
failed on Python 3.11's tar-stream `compresslevel` argument; its 21-byte
`cold-payloads.tar.gz.part` remains preserved. Backup v2 used explicit
`gzip.GzipFile` around a plain tar stream and produced the separate verified
`cold2` archive. The [first cleanup operator](../outputs/hpc-results/storage-cleanup-20261021-001/cleanup-operation-result.json)
stopped on scheduler authentication before remote control creation/unlink.
[Live-job audit 001](../outputs/hpc-results/storage-cleanup-20261021-001/live-job-audit001.json)
rejected oversized metadata; [audit 002](../outputs/hpc-results/storage-cleanup-20261021-001/live-job-audit002.json)
rejected a growing unrelated Mosaic `states.jsonl`. These maintenance failures
remain retained. Mosaic jobs are now RUN and must not be cancelled or altered;
their active input scope is still being resolved before any removal. No
scientific job, model/test work, gate change or storage recovery is claimed.

## New cleanup audit blocked by SSH access, 21 October 2026

This is the earlier access-failure stage, superseded by successful inventory
004 and byte preservation above; no earlier failure receipt was overwritten.

The user requested another HPC storage cleanup. The first read-only full-tree
inventory returned **255**, with `client_loop: send disconnect: Connection reset`
after the server information banner; [inventory receipt](../outputs/hpc-results/storage-cleanup-20261021-001/inventory.json)
retains the exact command, remote-source SHA256 and stderr. One bounded metadata
retry also returned **255**: `ssh: connect to host login1.hpc.dtu.dk port 22:
Connection timed out` ([retry receipt](../outputs/hpc-results/storage-cleanup-20261021-001/inventory002.json)).
The user then reported restored SSH and explicitly requested another retry.
That bounded attempt also returned **255**, before login:
`banner exchange: Connection to UNKNOWN port -1: Connection aborted`
([third receipt](../outputs/hpc-results/storage-cleanup-20261021-001/inventory003.json)).
It reused the unchanged v2 metadata payload by standard-library AST extraction.
None of the three returned an inventory payload. These are maintenance-access failures,
not scientific/model failures or evidence of a quota result.

Root used bundled CPython
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`
with command tails:

```text
outputs/hpc-submissions/inventory-hpc-storage-20261021.py
outputs/hpc-submissions/inventory-hpc-storage-20261021-v2.py
outputs/hpc-submissions/inventory-hpc-storage-20261021-v3.py
```

All three scripts requested read-only metadata through the recorded command
`ssh -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 dtu-hpc`
and remote login-shell `boltz311/bin/python -I -`. **No files were deleted,
no upload/job submission occurred, and no recovered space is claimed.** No
model/numerical work or network/VPN/SSH-setting change occurred. Read-only local
`Get-Command ssh` and `ssh -G dtu-hpc` checks resolve the normal
`C:\WINDOWS\System32\OpenSSH\ssh.exe`, alias host `login1.hpc.dtu.dk`, port 22,
user `s234818`, with no ProxyJump/ProxyCommand and connection multiplexing
disabled. This is configuration inspection, not a successful connection or a
settings change. Further SSH attempts are paused pending the user's result for
the fresh exact command `ssh dtu-hpc hostname`; local preparation
may continue. Any later cleanup still requires authenticated inventory,
complete verified local preservation/restore mapping before scientific
offload, fresh deletion guards and actual completion/quota receipts. Epoch
36/reference 35/original 89, data/environments and all failed/raw scientific
evidence remain protected.

## Narrow generated-cache cleanup completed, 20 October 2026 15:59 UTC

The [completed operation](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/release-cache-cleanup-result001.json)
removed **6,212 owned, singly-linked regular generated cache files** under
`__pycache__`/pytest caches: **131,252,224 allocated bytes (0.122238 GiB)**.
All removed paths were freshly verified absent. The
[local standard-library receipt verification](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/release-cache-cleanup-local-verification001.json)
confirms every removal had a prior durable intent and verifies saved intent,
before-inventory, journal and completion hashes. Operation SHA256 is
`10696f08c83d78d76045f5fb31f452cb2c69898d98d6cad8825ca2caf06ca10c`;
verification SHA256 is
`21b05b64165d0245d8fb3f32b47cd65d89e354855bea5138b5ef2c9c5435174a`.

The [audited inventory 002](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/release-cache-inventory002.json)
protected active source129, source098, source124 attempt002, source126 and
recursively referenced live-plan inputs; **71 source126 cache files were
skipped/protected**. No directories, source, checkpoints, scientific results,
logs or plans were removed. The unrelated Mosaic job and its files were
untouched; no job action or numerical/model test occurred. The
[failed inventory 001](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/release-cache-inventory001.json)
is retained: LSF rejected width 10000 before traversal or mutation, then the
corrected inventory succeeded.

Official quota moved from **89.37 to 89.24/100 GiB used** between
15:58:57 and 15:59:11 UTC, leaving **10.76 GiB free**. This is an interval
observation, not reserved space or an isolated attribution of all quota movement.
Fresh quota is still required before any future promotion. Source129's
diagnostic remained PEND at the subsequent 16:05:21 UTC status observation.

## Cold intermediate checkpoints offloaded with verified local restore copies, 19 October 2026 08:10 UTC

The completed offload copied **103 cold intermediate epoch checkpoints,
4,281,629,416 logical bytes**, into
[`source124-cold-offload-001/payloads`](../outputs/hpc-results/source124-cold-offload-001/payloads).
SFTP session 73372 exited zero; its [transfer receipt](../outputs/hpc-results/source124-cold-offload-001/transfer-success.json)
records transfer only. Subsequent complete local SHA256 verification passed
for all 103 files ([verifier output](../outputs/hpc-results/source124-cold-offload-001/local-verification-output.json)),
producing [local proof](../outputs/hpc-results/source124-cold-offload-001/local-proof.json)
SHA256 `13bb07ed86a7e0adbeb89f1dfe3d9565c7ce7aef0d1688d7ddb6cfa84ad976c1`.

Only then did the reviewed remote helper recheck every selected file's hash,
stat and active-job protections and unlink all 103, totaling **4,281,656,832
allocated bytes (about 3.9876 GiB)**. The
[operation intent](../outputs/hpc-results/source124-cold-offload-001/remote-operation-intent.json)
and [successful output](../outputs/hpc-results/source124-cold-offload-001/remote-operation-output.json)
record `completed_verified_local_offload`, return code 0, exact allowlist/proof
bindings and before/after quota. The
[restoration map](../outputs/hpc-results/source124-cold-offload-001/restoration-map.json)
maps every original absolute HPC path to its retained local `.pt` file and
SHA256. Restore from those exact bytes to the recorded path when needed,
then verify the original hash; no experiment rerun is needed to recover them.

All best/last/newest-two protected checkpoints and source098/121/123/124
ancestry/current inputs remain retained. Fresh protected hashes before and
after removal are unchanged:

| Protected checkpoint | SHA256 |
| --- | --- |
| Frozen epoch 35 reference | `19f438222e454ef53169f6d81082335ee7e24902b1b7cf5e0f18eced4b8e92ed` |
| Current completed epoch 36 | `3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f` |
| Original epoch 89 | `1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088` |

Official work3 free quota changed **7.44 → 11.43 GiB**; usage after removal
was 88.57/100 GiB at 08:10:42.951 UTC. The recorded 10.0625 GiB minimum was
met, but capacity is not reserved: continuation must run its independent
fresh native/quota checks. Independent [saved remote verification](../outputs/hpc-results/source124-cold-offload-001/remote-evidence-001/local-verification.json)
passed all 15 metadata checks against six exact remote files: 103 ordered
intents, 103 successful journal rows, fresh absence of all 103 remote files,
unchanged protected hashes and 11.43 GiB free. Verification SHA256 is
`5068bd12f8b4d5e2ea6fa49aed4e66dc272884b845bf0f28184fa947906655ea`;
[completion](../outputs/hpc-results/source124-cold-offload-001/remote-evidence-001/completion.json)
SHA256 is `0e8bdf0645b283937851687caf5a7a9ca87c1573b9ee06a4844a3c0b300de7dd`.
No model/numerical test was run locally for transfer/hash verification or this
documentation.

## Verified historical offload completed, 18 October 2026 21:18:13 UTC

All 175 inventoried files, **523,112,700 bytes**, were copied under
[`source123-storage-offload-001/payloads`](../outputs/hpc-results/source123-storage-offload-001/payloads)
and SHA256-verified locally before remote selection/unlinking. Only **81**
selected files were removed: eight obsolete disposable native probes,
legacy exports and older evidence/status transfer capsules. The other 94
current captures/capsules remain remote, including current Boltz-installed
capture evidence. Real checkpoints, original epoch 35/89 models, data,
source releases and scientific raw run outputs were untouched. Removed bytes
remain recoverable using the remote-path/local-name/hash mapping in the
[inventory](../outputs/hpc-results/source123-storage-offload-001/inventory.json)
and [local-copy proof](../outputs/hpc-results/source123-storage-offload-001/local-proof.json).

The [completion](../outputs/hpc-results/source123-storage-offload-001/completion.json)
records **496,897,742 logical /496,919,552 allocated bytes** unlinked, with
official work3 usage 90.24 → 89.77/100 GiB, **10.23 GiB free**. This is a live
quota snapshot, not reserved capacity or a future promotion approval. Recheck
quota before submission. Initial offload/helper attempts used a non-login
PATH and could not locate `getquota_work3.sh`/`bjobs`; they stopped before any
unlink or submission. A login-shell retry succeeded. The local verifier's
initial intent-field assumption (`arguments` versus actual `command`) was
corrected; final saved-byte verification passed without numerical model work.

## Five duplicate transfer capsules removed; selected continuation queued, 18 October 2026 12:16 DTU-local

After the unused-environment cleanup below, exactly five redundant remote
transfer archives under `/work3/s234818/latentmol/evidence` were removed at
10:13:12 UTC. Each archive was freshly matched to its complete verified local
archive/extraction before unlinking. Original HPC scientific payloads, source
releases, model/data/checkpoints and active Boltz–DeFoG inputs/outputs remain
unchanged. The local copies retain every exported byte and can be copied back
and hash-checked if a remote transfer archive is needed again.

| Retained local archive; removed remote basename | Logical bytes | SHA256 |
| --- | ---: | --- |
| [source120a004-staged-003.tar.gz](../outputs/hpc-results/source120a004-staged-003.tar.gz) | 118,803,082 | `69b517fde301989a889b74ab04bb3e45709b003298b004596be6d477bc77a333` |
| [source101-first-suite-003.tar.gz](../outputs/hpc-results/source101-first-suite-003.tar.gz) | 12,798,864 | `c190f99ce3d74077b387c7889799c848d6149089c8de6284458cb23a9624db99` |
| [source109a002-pointwise-round-003.tar.gz](../outputs/hpc-results/source109a002-pointwise-round-003.tar.gz) | 9,778,311 | `7dd705f2f1f86a121b7ea5a1ecb7f93c84cb1081ed30405b84d4f3ad456ff5ff` |
| [source105a002-first-suite-004.tar.gz](../outputs/hpc-results/source105a002-first-suite-004.tar.gz) | 7,519,827 | `1725b668a0fccf46bb4ecb3f4e3f28303ea70bbe055bf98723a1e24bd700712a` |
| [source091-defog-matched-pilot-attempt001-002.tar.gz](../outputs/hpc-results/source091-defog-matched-pilot-attempt001-002.tar.gz) | 5,543,796 | `419dbf3e9cec0c085ea51124254717142a7cf7308ee49346b35dd0447ec0fbaf` |

[Completion](../outputs/hpc-results/verified-capsule-offload-20261018-001/completion.json)
records **154,443,880 logical bytes /154,445,312 allocated bytes** removed;
[local-copy proof](../outputs/hpc-results/verified-capsule-offload-20261018-001/local-proof.json)
and [verification](../outputs/hpc-results/verified-capsule-offload-20261018-001/local-verification.json)
retain the allowlist, source hashes and prior complete member verification.
Completion SHA256 is `bbc22f3b0b4a078e8b87cb742d5f79515d5babb68f62721f38bb5fe426fea5ee`.
The official after-quota read at 10:13:12.501 UTC is 89.44/100 GiB used,
**10.56 GiB free**, above the unchanged 10.125 GiB initial chemistry minimum.
That live quota includes concurrent active-job I/O and ongoing gzip work;
it must not be attributed solely to the five transfer archives.

The additive metadata readiness passed before actual promotion. The selected
source121 segment 001, job 29749078 and serial callback 29749079 were then submitted
at 12:13:40/12:13:42 DTU-local and both observed **PEND** at 12:14:50, not
training started. Its own fresh promotion quota check passed at 10.57 GiB free;
the epoch 35 parent remains the last completed checkpoint.
[Verified submission](../outputs/hpc-results/source121a002-quota-promotion-attempt003/local-verification.json)
records both actual receipts and the unchanged source/parent/plan bindings.

**Separate gzip remains RUN and partially complete:** the frozen 12:16:17
DTU-local snapshot verifies **38 of 132 files**, with 297,980,805 original bytes
retained in 9,002,723 gzip bytes: **288,978,082 logical bytes saved (about
0.269 GiB)**. For those 38, fresh gzip hash/size checks match the completed
recovery receipts and the old original/backup paths are absent. This is a
verified prefix, not terminal completion or a total for all 132 files. The
gzip payloads remain adjacent to the historical original paths on HPC; the
local capsule contains metadata/receipts, not those scientific payloads.
Official quota at 12:16:17.613 is 89.51/100 GiB used, 10.49 GiB free, during
continuing active I/O.
[Progress snapshot](../outputs/hpc-results/historical-json-archive-20261018-002/progress_snapshot.json)
and [verification](../outputs/hpc-results/historical-json-archive-20261018-002/local-verification.json)
bind the 97 safe members/96 verified data files at capsule SHA256
`2487613107ff299b40352470069bc84d592073ff32e29e7cad4764640ea4f3de`,
exported `2026-10-18T10:16:17.613789+00:00`, with `terminal_completed=false`.
See [readiness proof](../outputs/hpc-results/source121a002-quota-readiness-attempt003/local-verification.json)
and [ledger](05_experiment_ledger.md) for the submission and retained failures.

## Unused Python 3.12 environment removed; further archival pending, 18 October 2026

The user requested space recovery and starting the already selected chemistry
continuation. The completed maintenance removed only the obsolete
`/work3/s234818/latentmol/env/boltz` Python 3.12 environment, after fresh
dependency, active-job, symlink, path-file, owner/stat and hard-link audits.
The active Python 3.11 `boltz311`, `defog311` and `ae` environments, scientific
sources/data/checkpoints, all benchmark evidence and current Boltz–DeFoG
inputs/outputs were retained. This is filesystem maintenance, not numerical
model validation or a changed training protocol.

At **12:02:02.963 DTU-local**, completion records **14,396 regular-file unlinks
and four symlink unlinks**, with the exact obsolete target absent. The audited
unique recoverable allocation was **2,166,806,528 bytes (about 2.018 GiB)**.
Another **2,899,337,216 allocated bytes** had external hard links, which remain
retained; the environment's roughly 5.066 GB apparent allocation is therefore
not the space-recovery amount. Official quota during the maintenance interval
changed **92.01 to 89.72/100 GiB used**, leaving **10.28 GiB free**. Concurrent
active-job I/O means that rounded quota delta is not an isolated measurement
of this deletion. The unchanged 10.125 GiB training minimum must be rechecked
at actual promotion; this snapshot does not reserve space.

[Completed receipt](../outputs/hpc-results/unused-boltz312-cleanup-20261018-001/completed.json),
[intent](../outputs/hpc-results/unused-boltz312-cleanup-20261018-001/intent.json),
[unlink journal](../outputs/hpc-results/unused-boltz312-cleanup-20261018-001/unlinks.jsonl),
[audit](../outputs/hpc-results/unused-boltz312-cleanup-20261018-001/audit.json)
and [supplement](../outputs/hpc-results/unused-boltz312-cleanup-20261018-001/supplement.json)
retain the exact target and inventory. Fresh post-cleanup checkpoint hashes
remain unchanged:

- Epoch35 parent `submissions/source104-chemistry-continuation-attempt005/parent.pt`:
  `19f438222e454ef53169f6d81082335ee7e24902b1b7cf5e0f18eced4b8e92ed`.
- Original epoch 89 VAE `runs/20261001T001617Z_vae-train_c008c090/best_validity.pt`:
  `1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.

The reviewed cleanup-v2 operator SHA256 is
`ca2e4efd7b952feb5615059f2a6a58ad545bc4bc0d147d4ab3503aa96773f1c3`.
The actual remote command used a login shell and isolated Python:

```text
ssh -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=1 dtu-hpc bash -lc '/work3/s234818/latentmol/env/boltz311/bin/python -I /work3/s234818/latentmol/submissions/unused-boltz312-cleanup-20261018-001/cleanup-v2.py'
```

[Local verification](../outputs/hpc-results/unused-boltz312-cleanup-20261018-001/verification.json)
matched the five remote audit/intent/completion/journal receipts, the operator
and all 14,396 regular unlink records, confirmed target absence and freshly
hashed the two protected checkpoints. It used Python 3.12.14 and
`outputs/hpc-submissions/verify-unused-boltz312-cleanup-20261018.py`, with
standard-library bytes/JSON/filesystem work only.

Two earlier invocations stopped before intent creation or unlink: the initial
`operator.py` filename shadowed Python's standard-library `operator` import;
isolated Python fixed that import, then recursive `*.pth` inspection mistook a
binary Torch checkpoint for a Python path file. The v2 guard checks Python
path files only in site/dist-packages locations and retains all reference
checks. These are maintenance preflight failures, not scientific model
failures.

**Separate historical JSON archival remains in progress:** CPU job 29748954 was
submitted at 12:01:44 DTU-local for two slots/4 GB total/45 minutes and is RUN
at the latest coordinator read. Its exact 132-file plan covers inactive
source057/source063 evidence, totaling 1,032,704,006 original bytes. Each
original may be removed only after a lossless adjacent gzip, full decompressed
byte/hash verification and durable recovery receipt; no full completion or
storage saving is claimed yet. Planning v1/v2 stopped before mutation at the
8 MiB metadata-file and 512-path traversal bounds. The new immutable v3 plan
keeps reference/stat/hash checks, explicitly bounds reads at 128 MiB/file,
512 MiB aggregate and 16,384 paths, and authenticated1,216 metadata files
(about 424 MB). A runtime estimate can exceed the 45-minute allocation; any
partial completion must retain its actual receipts rather than be called a
complete archive.

[Submission receipt](../outputs/hpc-results/historical-json-archive-20261018-v3-submission-receipt.json)
binds operator SHA256 `664aa40695e1885074bd25c7f252c3e236ff71f9d2127f60a6c38236b6a292f1`
and [plan](../outputs/hpc-results/historical-json-archive-20261018-v3-plan.json)
SHA256 `ecf544ce95f1917383e86ee48be502fabfc96a78871ad2881a9a16509d5194cf`.
[Failed planning record](../outputs/hpc-results/historical-json-archive-20261018-failed-planning.json)
preserves both pre-mutation stops. Training promotion is separate and remains
unsubmitted at this update; see the [ledger](05_experiment_ledger.md).

## Verified duplicate-transfer-capsule offload, 17 October 2026 13:47 DTU-local

Only five redundant `.tar.gz` transfer exports under
`/work3/s234818/latentmol/evidence` were removed after each remote file was
freshly SHA-matched to its preserved local archive, prior complete member
verification and extraction. Underlying runs, raw scientific records, source
releases, model/checkpoint/data files and other projects were untouched. No
Python cache was deleted. This is distinct from the earlier lossless gzip
archival below: these export archives remain byte-identical locally.

| Removed remote basename; retained local archive | Logical bytes | Preserved SHA256 |
| --- | ---: | --- |
| [source079-long-run-investigation-002.tar.gz](../outputs/hpc-results/source079-long-run-investigation-002.tar.gz) | 53,350,530 | `b24a477c6764458fe3cd827bd8d9c47fba1237fd607df8c93013af5c32916163` |
| [status-20261010-check002.tar.gz](../outputs/hpc-results/status-20261010-check002.tar.gz) | 50,855,471 | `f30dce44b5252157feaa4a7ddb2824f4568c4694cbc5f43b8db7ead4a1116784` |
| [source079-long-run-investigation-001.tar.gz](../outputs/hpc-results/source079-long-run-investigation-001.tar.gz) | 41,435,591 | `7c46d0425314d2ed7a93cdad35328334a847f99ef802d018b480cdb6969e4651` |
| [status-20261010-check001.tar.gz](../outputs/hpc-results/status-20261010-check001.tar.gz) | 36,546,601 | `9a78f97740735926d7875b47cdf6f760cd10c51dbd0e0d6484e672a5798035b8` |
| [status-20261009T081813Z.tar.gz](../outputs/hpc-results/status-20261009T081813Z.tar.gz) | 22,472,831 | `280652e517a2265ba7744860364a3d43139c3249e52c67a6584d303f33d957e5` |

The full original remote path is `/work3/s234818/latentmol/evidence/<basename>`.
Local archives remain under `outputs/hpc-results`, with same-stem extracted
directories and their prior `.tar.gz.verification.json` records. If a remote
export is needed again, copy the retained exact local archive back to that path
and verify its SHA256; original HPC runs do not need regeneration or modification.

[Completion002](../outputs/hpc-results/verified-capsule-offload-20261017-002/completion.json),
SHA256 `fff69886749cbade070de75ad63baa3f93e3984772594fca85a38de045abc355`,
records five completed unlink operations between11:47:38 and11:47:39.830291 UTC.
Totals: **204,661,024 logical bytes /204,662,784 allocated bytes**, about195 MiB.
The [active-job input audit](../outputs/hpc-results/verified-capsule-offload-20261017-002/active-job-command-audit.json),
[intent](../outputs/hpc-results/verified-capsule-offload-20261017-002/intent.json),
[local-copy proof](../outputs/hpc-results/verified-capsule-offload-20261017-002/local-proof.json)
and individual completion/stat/hash receipts are retained beside it. Root
independently rechecked the completion hash and all five local archive hashes.

Official [quota-before](../outputs/hpc-results/verified-capsule-offload-20261017-002/quota-before.json)
and [quota-after](../outputs/hpc-results/verified-capsule-offload-20261017-002/quota-after.json)
reads changed89.88→89.69/100 GiB used, leaving **10.31 GiB free**, about0.185 GiB
above the unchanged10.125 GiB initial chemistry reserve. Logical/allocated byte
savings and rounded quota deltas are distinct. This does not reserve space:
the callback must recheck current official quota before promotion. The fresh
[native queue read](../outputs/hpc-results/verified-capsule-offload-20261017-002/native-queue-observation.json)
at11:47:39.828656 UTC still shows GPU29745548/callback29745549 PEND.

Attempt001 failed **before any unlink**, because the bare SSH environment
could not find `bjobs` or `getquota_work3.sh`.
[Failed completion001](../outputs/hpc-results/verified-capsule-offload-20261017-001/completion.json),
SHA256 `138abcf54ae77f3114ad021aeb5063703067f1b03ea61aaa807b82d86b1e2539`,
records zero completed files and zero removed bytes. Attempt002 used the normal
login-shell environment for those metadata tools, keeping the same five-path
and byte/stat/input guards. No VPN/SSH setting, job, model, loss or scientific
gate changed. These are filesystem/metadata maintenance receipts, not new
numerical experiments or native readiness approval.

## Verified archival completion, 16 October 2026 13:38 DTU-local

Allocated **v3 job 29736588 completed 21 files** at
`2026-10-16T11:32:20.019704+00:00`; **v4 job 29736618 completed 52 files** at
`11:37:48.121240 UTC`. Together, **73 exact inactive files** were retained
losslessly as adjacent gzip artifacts. Every completion item records matching
original/decompressed SHA256 and byte size, compressed SHA256/size, original
mode/mtime and its recovery receipt. The original plain files were removed only
after full verification. Checkpoints, models, datasets, extracted molecule
resources and other projects were not changed.

| Completed scope | Original bytes | Gzip bytes | Logical bytes saved |
| --- | ---: | ---: | ---: |
| v3: 20 historical rows logs + redundant `mols.tar` | 5,411,707,430 | 545,743,432 | 4,865,963,998 |
| v4: 52 large historical repair-refresh JSON files | 1,272,572,495 | 37,879,127 | 1,234,693,368 |
| Total: 73 files | 6,684,279,925 | 583,622,559 | 6,100,657,366 |

The [campaign's pre-archival quota](../outputs/hpc-results/source112-first-suite-002/runs/source112-chemistry-search-acceleration-attempt001/initial_quota.json)
at 11:18:48 UTC reports 94.54/100 GiB used, 5.46 GiB free. The logical saving is
about **5.68 GiB**; it is not a direct measurement of quota delta. The
[final official quota snapshot](../outputs/hpc-results/repair-archive-20261016-001/quota_snapshot.json)
at `11:38:45.294072 UTC` reports **89.22/100 GiB used, 10.78 GiB available**,
currently above the unchanged 10.125 GiB initial chemistry-training reserve.
Concurrent account activity explains why before/after charged quota need not
equal the sum of logical saved bytes. The chemistry controller must recheck
actual quota at promotion; this snapshot does not reserve space or approve the
still-pending native benchmark.

[Verified metadata capsule](../outputs/hpc-results/repair-archive-20261016-001.tar.gz)
SHA256 `1e3921064e83ed395ebba99ea05677add4244a1917f751d73d925400b53aee8c`
contains **186 safe members / 185 verified files**, exported
`2026-10-16T11:38:45.294578+00:00`. It retains the failed v2 run, v3/v4 plans,
operator/submission identities, active-input audits, per-item restore receipts,
journals and [v3 completion](../outputs/hpc-results/repair-archive-20261016-001/submissions/historical-repair-archive-20261016-v3-plan.json.artifacts/completed.json)
/ [v4 completion](../outputs/hpc-results/repair-archive-20261016-001/submissions/historical-repair-archive-20261016-v4-plan.json.artifacts/completed.json).
**The scientific gzip payloads are not inside this local metadata capsule.**
They remain on HPC beside each original path as `rows.jsonl.gz`,
`search-refresh-########.json.gz` or `cache/boltz2/mols.tar.gz`; exact paths/hashes
are in the plans/receipts.

To restore one exact historical plain file when needed, use its matching plan
and an original path copied from that plan. These documented commands are not
executed by this report:

```text
python3 /work3/s234818/latentmol/submissions/archive-historical-repair-evidence-20261016-v3.py --restore-plan /work3/s234818/latentmol/submissions/historical-repair-archive-20261016-v3-plan.json --plan-sha256 ff4456ca640d3d2c7b626fcacf3ccd8702d9e4196a97a194d1d4c6daa60797a1 --restore-path <EXACT_ORIGINAL_FROM_V3_PLAN>
python3 /work3/s234818/latentmol/submissions/archive-historical-repair-evidence-20261016-v4.py --restore-plan /work3/s234818/latentmol/submissions/historical-repair-archive-20261016-v4-plan.json --plan-sha256 6da5791bd0bb303d232454673529c16c2c018908c0d734de75c4a6cf2336fbc1 --restore-path <EXACT_ORIGINAL_FROM_V4_PLAN>
```

Restoration verifies the gzip/receipt, recreates one absent original with its
mode/mtime and retains the compressed copy. It requires enough quota for the
restored plain bytes. Local evidence verification and saved-JSON aggregation
used the standard library only; archival executed in allocated HPC CPU jobs.
The earlier v2 stat discrepancy remains unproven as a cross-node device cause;
successful v3/v4 runs do not retroactively identify that missing observation.

## Earlier archival preparation and retained failures, 16 October 2026

This maintenance supports the gated source112 chemistry continuation; it does
not change model weights, training data, chemistry policy or validation gates.
At this earlier preparation stage, execution outcomes and final capacity
accounting still awaited the new authenticated evidence capsule; the completed
update above now provides them. Earlier
10 October deletion work below remains separate historical maintenance.

The first archival plan covers twenty inactive historical `rows.jsonl` files
at least 20 MiB and exactly the redundant `cache/boltz2/mols.tar` source archive.
The extracted `cache/boltz2/mols` directory remains intact. Historical rows are
restricted to these five completed-study roots:

- `runs/source-070-decoder-validity-attempt001-decoder-study`
- `runs/source-071-search-decoder-attempt001-search-study`
- `runs/source-073-decoder-budget-attempt002-recovery`
- `runs/source-075-search-full-attempt001-search-full`
- `runs/source-077-dense-chemistry-attempt001-density-study`

All paths are under `/work3/s234818/latentmol`. Source/checkpoint identities,
models, datasets, extracted molecule resources, current/active runs, small rows
files and other user projects are outside the mutation allowlist. Active job
metadata is checked for references before archival, including unrelated user
jobs; this is a read-only audit, not permission to cancel or edit those jobs.

Compression retains every original byte in a `.gz` artifact. Before removing
an original, the operator must close and fully decompress the gzip, verify its
original SHA256 and size, persist its compressed-byte identity and a durable
per-file recovery receipt, then revalidate source metadata. Restoration checks
that receipt/gzip and exclusively recreates the exact original path, mode and
mtime while retaining the gzip. Partial failures retain originals/backups/gzip
and journals for explicit recovery. Future readers requiring a historical plain
file may need that explicit restoration; archival is not deletion of its
scientific contents.

The [v2 plan](../outputs/hpc-results/historical-repair-archive-20261016-v2-plan.json)
and [v3 plan](../outputs/hpc-results/historical-repair-archive-20261016-v3-plan.json)
record exact allowlists, hashes, inactivity audit and restoration commands.
Preserved failures/changes before new completion evidence:

- The initial read-only plan exceeded its 128-file bound and stopped without
  deleting data. The revised plan excludes small rows files rather than
  expanding the execution scope silently.
- An initial queue helper had a malformed 65-character plan hash and rejected
  before submission; its corrected unique helper submitted v2 job 29736578.
- Allocated v2 job 29736578 stopped at its initial planned-stat guard **before
  compression**. A subsequent login-node reread reported no stat differences.
  The exact discrepancy on the failed worker was not captured, so a cross-node
  mount/device cause is **unproven** and must not be stated as the root cause.
- A new immutable v3 plan/operator records the full per-node stat comparison.
  All compared metadata and content hashes remain strict; only device-number
  changes are allowed when execution host differs from the planning host.
  Job 29736588's reported execution is awaiting the consolidated verified
  maintenance capsule before completed counts/storage savings are recorded here.

A further proposed plan targets only 52 large
`repair-refresh-epoch-*/search-refresh-########.json` evidence files under those
same five inactive roots, with lossless compression and the same protections.
It is separate from the rows/archive plan; it is not a completed maintenance
result or proof of sufficient quota. Concurrent account growth can offset
freed bytes. The chemistry callback must query actual current available quota
against its unchanged 10.125 GiB initial reserve; an improved quota reading below
that threshold is still a failed reserve check.

Local maintenance review/preparation uses AST/bytes/JSON only; all substantive
archival execution is in allocated HPC CPU jobs. The user's current permission
also allows local tests if needed, but no local numerical test has been used
for this performance/maintenance work. Exact source/job identities and eventual
outcomes remain in the [experiment ledger](05_experiment_ledger.md).

## Historical capacity review, 10 October 2026

**Read-only capacity estimate, 10 October 2026:** two bounded SSH inventory
queries failed before login, so no fresh quota or directory inventory is
available. The last verified work3 snapshot, at `2026-10-10T11:36:52+00:00`,
was **82.19/100 GiB used, 17.81 GiB free**; this is historical, not current
free-space certification. No deletion, download, preprocessing, training,
submission or VPN/SSH change occurred in this capacity review.

At the current 50-slot int8 NPZ layout, 1,936,962 molecules require about
**4.69 GiB of uncompressed atom/charge/bond/split labels** (2,601 bytes each).
Fixed-width Unicode SMILES add `4 × count × maximum length` bytes: about
0.72 GiB at a 100-character maximum. Compression is unmeasured. A rough
linear extrapolation from the existing 12.1 MiB data directory to this
cohort gives about 102 MiB, but it cannot predict a different molecular
distribution or maximum SMILES length. Reserve several GiB for dataset
artifacts plus checkpoints and temporary files; the historical quota appears
sufficient for disk storage, but live capacity remains unverified.

Preprocessing and training RAM are separate constraints: the current
preparation builds all records, stacks arrays and buffers the archive in
memory; the loader decompresses its full cache. Neither disk estimate proves
that these paths fit RAM at the larger scale. The [dated ledger entry](05_experiment_ledger.md)
records the failed commands and the estimate's scope. No storage-format or
scientific configuration was changed.

## Historical completed cleanup

Completed at approximately **13:37 DTU-local**. The user explicitly requested a
separate cleanup agent to free space while Boltz–DeFoG debugging continued.
This was filesystem maintenance, with no model imports, numerical experiments,
job submissions/cancellations or VPN/SSH configuration changes.

The final official `/work3` query at cleanup reports **82.19 GiB used of 100 GiB**, leaving
**17.81 GiB**. It reported 98.31 GiB immediately before checkpoint pruning.
The initial inventory had reported 95.64 GiB; intervening account usage and
maintenance metadata mean that earlier number is not the appropriate direct
before/after accounting pair.

## What was removed

| Location | Removed | Recorded release |
|---|---|---|
| Remote home | Six obsolete ELF crash dumps dated 29 May–1 June 2026 | 24,674,758,656 logical bytes; 32,388,063,232 allocated inode bytes |
| Historical AE/VAE training directories | 344 obsolete intermediate epoch checkpoints | 17,309,577,224 logical bytes; 17,309,657,600 allocated inode bytes |

Home allocated-byte accounting differs from its charged quota. `du -x -B1`
decreased from 41,256,091,648 to **8,868,028,416 bytes** (about 8.26 GiB) after
crash-dump removal. The official home quota output still says 27.52G/30.00G
because it is cached to 13:03:56 and updated approximately every four hours.
Do not present that cached number as a post-cleanup measurement or sum the home
inode release with `/work3` charged-quota savings.

The crash-dump plan records exact paths, owner UID 306121, regular-file and
nonsymlink checks, dates, ELF headers, format descriptions and all six content
SHA-256 hashes before deletion. Its process-descriptor check is scoped to the
login host. Protected `sd-pam`/`sshd-session` authentication helpers were
explicitly excluded and recorded; the application-process scan found no open
dump descriptor. These old memory dumps were deleted rather than copied locally.

## Checkpoint selection and preservation

The initial conservative checkpoint inventory proposed 404 non-milestone epoch
files. A remote semantic audit scanned **33,819 JSON/YAML metadata files** across
configs, scientific submission plans, `run.json`/summary records and resume
receipts. It preserved **60 additional epoch paths** with explicit scientific
references. A second live audit before mutation gave the same result. Maintenance
inventories were excluded as inventories rather than runtime checkpoint inputs.

Only the remaining 344 candidates were deleted. Every candidate had a SHA-256,
size, inode/device, modification time and link count recorded in a durable plan
before mutation. Completion journals record each unlink and count allocated
blocks only when the final hard link to an inode was removed.

The cleanup retained all best/last/inherited checkpoint files, ten-epoch
milestones, final saved epochs, selected/evaluated/resume references and the
entire original VAE training directory. It also retained the protected historical
source041 architecture lineage, all recent chemical-decoder training, all
datasets and frozen external-model weights, all DeFoG/Boltz environments and
native caches, all scientific run evidence and saved diagnostic states, and
all active/pending source088/source089 dependencies.

Post-cleanup verification confirms **484 retained checkpoint paths** still have
their original inode/device, size and modification time, and all 344 declared
deletion paths are absent. The untouched original VAE
`runs/20261001T001617Z_vae-train_c008c090/best_validity.pt` remains at SHA-256
`1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.
Pruned intermediate epochs cannot be used for future replay; retained metrics,
milestones and selected models remain available.

The unrelated `/work3/s234818/relative-weight-models` project (about 30.98 GB in
the initial inventory), home project directories, IDE installations and user
configuration were left intact. Old pending job **29510465** has an invalid or
unsatisfied dependency on 29510449; its source041 inputs were preserved and the
job was not canceled. Pending diagnostics/smoke jobs were also not changed.

## Evidence and execution record

The authenticated [cleanup capsule](../outputs/hpc-results/hpc-space-cleanup-20261010-001.tar.gz)
contains **24 safe regular-file members / 23 verified data files**, exported
`2026-10-10T11:37:37.549090+00:00`, at archive SHA-256
`84cc7ce2a9757efca2feb58a4f138988a9c3bee4e9c30be05e0bbf00022ed5ce`.
Its adjacent [local verification receipt](../outputs/hpc-results/hpc-space-cleanup-20261010-001.tar.gz.verification.json)
records bundled Python 3.12.14 / Windows 11 hash and safe-extraction work only.
Key evidence includes:

- [Core deletion plan](../outputs/hpc-results/hpc-space-cleanup-20261010-001/stale-core-deletion-plan.json)
  and [completed unlinks](../outputs/hpc-results/hpc-space-cleanup-20261010-001/stale-core-deletion-completed.jsonl).
- [Checkpoint deletion plan](../outputs/hpc-results/hpc-space-cleanup-20261010-001/historical-checkpoint-deletion-plan.json),
  SHA `b00a6e4f1ce10be940bec856c947c3ba1338077c7105341b205bace21c1c243c`,
  and [completed unlinks](../outputs/hpc-results/hpc-space-cleanup-20261010-001/historical-checkpoint-deletion-completed.jsonl).
- [Checkpoint deletion summary](../outputs/hpc-results/hpc-space-cleanup-20261010-001/historical-checkpoint-deletion-summary.json)
  and [preservation verification](../outputs/hpc-results/hpc-space-cleanup-20261010-001/preservation-verification.json).
- [Official work3 quota](../outputs/hpc-results/hpc-space-cleanup-20261010-001/after-work3_quota.json),
  [cached home quota](../outputs/hpc-results/hpc-space-cleanup-20261010-001/after-home_quota.json),
  [home directory usage](../outputs/hpc-results/hpc-space-cleanup-20261010-001/after-home_du.json)
  and [project directory usage](../outputs/hpc-results/hpc-space-cleanup-20261010-001/after-work3_project_du.json).
- [Execution environment](../outputs/hpc-results/hpc-space-cleanup-20261010-001/maintenance-environment.json)
  and [scheduler snapshot](../outputs/hpc-results/hpc-space-cleanup-20261010-001/after-jobs.json).

All remote maintenance commands used `ssh -o BatchMode=yes -o ConnectTimeout=20
-o ConnectionAttempts=1 dtu-hpc` and native Python/pathlib filesystem operations.
Remote execution used `/usr/bin/python3`, Python 3.9.25, UID 306121 on hpclogin1.
The remote audit root is
`/work3/s234818/latentmol/submissions/hpc-space-cleanup-20261010`.
The decisive commands were:

```text
python3 /work3/s234818/latentmol/submissions/remove-stale-hpc-cores-20261010.py
python3 /work3/s234818/latentmol/submissions/hpc-space-cleanup-20261010/prune-historical-checkpoints-20261010.py --candidate-plan /work3/s234818/latentmol/submissions/hpc-space-cleanup-20261010/checkpoint-pruning-candidate-plan-20261010.json
python3 /work3/s234818/latentmol/submissions/hpc-space-cleanup-20261010/prune-historical-checkpoints-20261010.py --execute-plan /work3/s234818/latentmol/submissions/hpc-space-cleanup-20261010/historical-checkpoint-deletion-plan.json --plan-sha256 b00a6e4f1ce10be940bec856c947c3ba1338077c7105341b205bace21c1c243c
python3 /work3/s234818/latentmol/submissions/hpc-space-cleanup-20261010/export-hpc-cleanup-evidence-20261010.py
```

Two initial core-helper attempts stopped before deletion: `fuser` was unavailable,
then a strict `/proc` inspection encountered protected authentication helpers.
The explicit descriptor-scope implementation above resolved that inspection
limitation. A later narrowly targeted stop-helper found the plan already written
and stopped **no process**; all six hashes/deletions had completed. The unrelated
native `quota -s` query returned no live home accounting. These are maintenance
limitations, not experiment failures or changes to scientific validation.

Local planning and review imported only stdlib metadata helpers. Six maintenance
scripts were AST-parsed with bundled Python 3.12.14, and the evidence exporter was
parsed separately. Local byte verification used this exact command, with no model
or numerical execution:

```text
C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe outputs/hpc-submissions/verify-source077-evidence.py --archive outputs/hpc-results/hpc-space-cleanup-20261010-001.tar.gz --sha256 84cc7ce2a9757efca2feb58a4f138988a9c3bee4e9c30be05e0bbf00022ed5ce --destination outputs/hpc-results/hpc-space-cleanup-20261010-001
```


## 13 November 2026 — old report assets offloaded and concurrent cleanup reconciled

Verified 608 inactive report asset aliases against 305 complete laptop payloads,
then removed 2,039,047,680 allocated bytes (1.90 GiB) from WORK3. Independent
postflight confirmed absence of the selected assets and unchanged pilot summaries
and terminal logs. The final WORK3 quota showed 20.50 GiB free of 100 GiB.

A concurrent authorized cleanup archived both old 28–29 October DeFoG frame
sets during this operation's read-only transfer. The source-change guard stopped
that transfer before any pilot deletion. Both completed laptop archives were
independently checked by whole-archive SHA256 and all 60,002 member hashes.
Their removals are excluded from this operation's 1.90 GiB total. Two redundant
partial transfers were removed locally after confirming complete frame coverage.

The backup and restore index is
`C:/Users/nikol/HPC_Archives/maintenance-20261113-01/README.md`.
The pilot archives are under
`C:/Users/nikol/HPC_Archives/hpc-cleanup-20261113a/archives/`.
Historical report views and DeFoG trajectory rendering may require restoration.
