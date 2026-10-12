# Experiment ledger

Retain submitted, completed, failed, and incomplete attempts with their source/config/input identities and artifact paths. A submission or successful dependency installation is not a completed model experiment. Never turn a skipped check into acceptance evidence.

## 2026-10-10: DTU setup and bounded native Protenix-Tiny example

Status: **completed for the bounded native protein example — GPU job `29674532` DONE, output checks passed, results downloaded and verified**.

The user authorized HPC setup, a small Protenix-Tiny example, quota inspection, and download of results. This entry concerns native **protein** inference through the existing Mosaic interface, not Stage 0 ligand-graph implementation/validation, optimization, training, or the entire Mosaic dependency collection. [The setup receipt](hpc_protenix_tiny.md) contains the current environment, storage, helper, and operating details.

### Source, environment, and input identity

- Mosaic commit: `6e127d60e920f25bcd27ca5c7434cb3b2ac930c9`.
- Immutable source: `/work3/s234818/mosaic/releases/20261010-tiny`, made read-only. Archive includes current tracked files and uncommitted source/helper contents, with a per-file manifest.
- Source archive: local `results/hpc/20261010-protenix-tiny/source.tar.gz`, 38,660,850 bytes, SHA-256 `d31d13da526a7ac58281ad7de6182fa451c95d313bd79be23909d67a19b83920`; remote hash verified. Sibling `source_manifest.json` and `archive-files.txt` record included files and working-tree identity.
- Remote project/environment: `/work3/s234818/mosaic`, `/work3/s234818/mosaic/envs/protenix-tiny`.
- Setup host/allocation: `linuxsh` interactive `n-62-30-8`, helper allocation `714306.hnode41`. No setup compute was performed on the login node.
- Python module `python3/3.12.11`, `uv 0.10.0`; reduced 57-package environment with `jax[cuda12-local]==0.11.1`, Equinox `0.13.8`, NumPy `2.5.3`, and Protenij revision `97cf8172f61829681582f06de7052485bc94eae4`. Torch/unused model stacks were omitted. Exact setup commands, lock and resolved package export are downloaded.
- Native model: `protenix_tiny_default_v0.5.0`; verified assets from `nickrb/protenij` revision `81a257b61a82a17c41a7c20638460c04c29d97ad`. Tiny `.eqx` SHA-256 `19f5f7af00cda104e8b059289e7e641cecc03309d7e960e6e53d82ad6e3f1f57`; full file identities in downloaded `expected_assets.json`/`asset_receipt.json`.
- Input: `ACDEFGHIKLMNPQRSTVWY` (20 residues), seed `0`, no MSA/template, one recycle, two sampling steps. Frozen weights, no optimizer or gradients; source example records sampler settings and output interpretation.
- Run directory: `/work3/s234818/mosaic/runs/20261010-native-tiny-01`. Local results directory: `results/hpc/20261010-protenix-tiny/`, ignored by Git. These documentation files were created after the archived release and do not change its identity.

### Attempt and outcome receipts

| Attempt | Action | Observed outcome | Evidence |
| --- | --- | --- | --- |
| Setup 01 | Install the dedicated reduced environment on the interactive allocation. | `uv` installation exit `0`; initial environment usage 1,573,271,552 bytes. | `setup-commands.sh`, `requirements-tiny.lock`, `environment.freeze.txt`, install/dry-run logs. |
| Packaging check 01 | `uv pip check` against upstream metadata. | Original incompatibility retained: upstream `requires-python ==3.12` does not accept `3.12.11`. | `package-check-before-fix.txt`. |
| Packaging correction/check 02 | Change dedicated-environment requirement to `==3.12.*`; update `METADATA` and `RECORD` only, model source unmodified. | Subsequent `uv pip check` reports all 57 packages compatible. Metadata SHA-256 changed from `674cfe92b81c4efda15b8d09fb7fdf3892269676a33789203980cd910ad21cfc` to `b63c36bb9b96e867f063b7fdfc2e0934cc25b370e618e1ab78930c4866037cbd`. | `fix_python_metadata.py`, `metadata_fix_receipt.json`, `package-check-after-fix.txt`. |
| Static helpers 01 | `bash -n` on the LSF script; `ast.parse` on both Python helpers, on interactive CPU. | Operator reported passed static checks only. Interactive allocation `714306.hnode41` completed by shell exit. | Operator receipt and source manifest; no separate static-check transcript in the downloaded bundle. |
| Storage 01 | Inspect quota before installation. | Operator reported work-pool 6 `95.64 / 100 GiB`; home `27.52 / 30 G`. | Initial operator receipt; no separate pre-install quota file in the bundle. |
| Storage 02 | Inspect quota before asset staging, `2026-10-10 11:23:07 UTC`. | Work-pool 6 `97.44 / 100 GiB`; cached home `27.52 / 30 G`; project usage 1,655,676,928 bytes. | `quota-before-assets.txt`. |
| Assets 01 | Stage five pinned files sequentially with bounded downloads and hashes. | Staging exit `0`; all five verified; total 1,001,183,124 bytes. | `stage-assets.log`, `asset_receipt.json`, `expected_assets.json`. |
| Storage 03 | Inspect quota before GPU submission, `2026-10-10 11:25:44 UTC`. | Work-pool 6 `98.37 / 100 GiB`; cached home `27.52 / 30 G`; project usage 2,656,892,928 bytes. | `quota-before-job.txt`. |
| GPU inference 01 | Native example job `29674532`, `gpua10`, one exclusive GPU, four slots on one host, `4GB` per slot/16 GB host total, 30 minutes. | `DONE`/“Successfully completed”; all arrays finite, native shapes and both 168-atom structure round trips passed. | `submission.txt`, `submit-command.txt`, stdout/stderr, `lsf-job.txt`, `artifacts/summary.json`. |
| Accounting 01 | Query `bacct -l 29674532`. | Permission denied; no retry/permission change. Final stdout and `bjobs -l` retain successful completion/resource accounting. | `accounting-limitation.txt`; `lsf-accounting.txt` is empty. |
| Storage 04 | Inspect quota after job, `2026-10-10 11:31:50 UTC`. | Work-pool 6 `98.43 / 100 GiB` (1.57 GiB headroom), 227,913 / 20,000,000 chunk files; cached home `27.52 / 30 G`; project 2,707,586,560 bytes. | `quota-after-job.txt`; shared `df` 510 T available does not change user quota. |
| Result download 01 | Copy the small output/log/receipt bundle by SCP. | Successful at `2026-10-10T11:32:47.6274266Z`; safe tar paths and all 30 run-manifest file sizes/hashes verified. | Parent `download_receipt.json`, downloaded bundle/run manifest. No local numerical evaluation. |

### Commands and artifact collection

Existing source helpers are [the native example](../examples/hpc_protenix_tiny.py), [the LSF script](../scripts/hpc/protenix_tiny.lsf), and [the sequential asset stager](../scripts/hpc/stage_tiny_assets.py). The exact CLI submission and reproduction instructions are in [the setup receipt](hpc_protenix_tiny.md). Resources came from the `bsub` command line, which invoked `bash -l` with the script/release/run arguments; the file's `#BSUB` comments were not the source of submitted resource options. A repetition must use a fresh run directory and the resident environment/assets.

The downloaded [run directory](../results/hpc/20261010-protenix-tiny/20261010-native-tiny-01/) contains FASTA, NPZ, PDB/mmCIF, summaries/manifests, logs, lock/export, source/asset identities, quota checks and packaging-correction receipts. Remote bundle: `/work3/s234818/mosaic/runs/20261010-native-tiny-01.tar.gz`; local bundle: `results/hpc/20261010-protenix-tiny/20261010-native-tiny-01.tar.gz`, 202,193 bytes, SHA-256 `3e5b3820d68292afc9303562f977b405e3f46cb7f8be8e0adea91a0768b35be4`. [The download receipt](../results/hpc/20261010-protenix-tiny/download_receipt.json) records the exact successful SCP command and integrity checks. Environment and model/data weights remain on HPC and were not downloaded.

Execution node `n-62-18-5` used NVIDIA A10, driver `615.71.09`, 23,028 MiB, JAX `cuda:0`. Summary: coordinates `[1,168,3]`, distogram `[20,20,64]`, pLDDT `[20]`, PAE `[20,20]`, all recorded arrays finite. Elapsed model time 218.214 seconds; loading 2.834 seconds, featurization 21.429 seconds, forward including compilation 170.814 seconds. Final LSF stdout reports 685 CPU seconds, 3,668 MB max memory, 246 seconds run time and 248 seconds turnaround.

`lsf-job.txt` retains a truncated external message beginning “GPU 0: Detected Power error”. The job nevertheless finished successfully. No cause or hardware-health conclusion is drawn from that truncated message, and no retry or administrative action was taken.

### Interpretation

This bounded setup/example and result download are complete. This task removed no unrelated files and used no quota bypass; the user's separate HPC-cleanup agent operates outside this receipt. Preserve this run and record any future attempt separately.

The successful ordinary protein forward validates the selected installation/inference behavior. An arbitrary 20-residue sequence and two sampling steps do not establish a validated fold, binding, molecular graph gradients/feasibility, or full-repository validation. This result is not Stage 0 graph acceptance. Local work was file integrity and receipt inspection only.

## 2026-10-12–13: Stage 0 graph branch implementation and native source audit

Status: **all Work 0-5 code/tests authored, merged and independently reviewed; final static checks and local source packaging completed; numerical gates A-G not executed**. Immutable packaging is recorded at `results/releases/stage0-20261013/`. This is an implementation/history receipt, not a submitted or completed numerical experiment.

The user requested all planned Stage 0 code components with progress markers, pivotal decisions, and supporting tests. HPC is unavailable for this turn. No HPC connection, upload, submission, model/preprocessing run, local numerical test or evaluation is authorized/performed. Existing 2026-10-10 native-protein evidence and the user's moved project PDF are preserved.

| Work/identity | Current receipt |
| --- | --- |
| T1 graph and optimizer | Commit `8cb6f6c` contains shared fixed-occupancy X/E_upper state, symmetric existence reconstruction, host checks, probability-space initialization/diagnostics, strict raw graph gradients, separate projected steps and authored tests. Tests not run. |
| T2/T3 native adapter/model | Native preparation, differentiable injection, frozen geometry hook, pinned resident Tiny path, losses and tests authored and merged as `4168f0f`/`3470f17`. |
| T4 integration/validation | Shared contracts, sealed context/readout/failure IO, immutable checkpoint commit pointers, source/environment binding, lazy exports, full CLI/config/fixtures, validation module, combined/unit/real-checkpoint tests and future HPC/source/quota helpers are authored in main. |
| Upstream source | `escalante-bio/protenij` revision `97cf8172f61829681582f06de7052485bc94eae4`, audit archive SHA-256 `ea7abd10bd9ad2ce1bff0f9bde76969a298d511051fa200fe47cfb5436e43711`; local source inspected without executing/importing it. |
| Assets | `nickrb/protenij` revision `81a257b61a82a17c41a7c20638460c04c29d97ad`, unchanged pinned Tiny/CCD/cluster identities from the historical asset receipt. No new model download. |
| Contract/review | `small_molecule_v0.1`, checkpoint schema version 1; independent modularity/reuse reviewer separate from code owners and documentation owner. |

The [updated plan](stage_0_implementation_plan.md) marks code completion separately from M1 producer/consumer test evidence and M2/M3 native acceptance. [The native contract](stage_0_native_contract.md) records actual source consumers and feature losses: float 128-element embedding and float token-bond existence; separate geometry gates preserve identity/occupancy and native protein geometry; neutral Q000 slot names; serialized sampled conformer; one final recycle; native upper-edge distogram bins and conservative whole-bin contacts. Bond order is retained in graph state but invisible to the native objective. Formal charge, hydrogen slots, variable occupancy, stereo, diffusion/confidence and Stage 1 PAE are outside this implemented geometry-off path. [The usage guide](stage_0_usage.md) records future allocated-HPC setup/test/CLI/launcher commands; they have not run.

Independent review of merged slices prompted fixes for native integer element truncation, nested LossTerm auxiliary composition, immutable step/evaluation commit pointers, raw overflowing proposals/failure retention, dtype rejection, complete CLI settings and source/environment resume bindings, and strict finite trunk/output auxiliary checks. Direct probability-space projected gradient descent reuses host simplex projection; typed fixed/split RNG retains the next unused key; final evaluation and best-state score identity are explicit. Those fixes are code/static-review evidence, not numerical passes.

Earlier static receipts: selected files were parsed with `ast.parse` without importing Mosaic/JAX; Ruff `0.15.12` lint/format and staged whitespace checks were performed. The native slice's original two F401 unused imports were corrected before its clean final lint. Root downloaded only an approximately 11.3 MiB Ruff utility for static checking, without project dependencies/models.

Final combined static receipt, 2026-10-13, reported by the root:

| Command/operation | Scope and observed result |
| --- | --- |
| CPython `3.13.1`, `ast.parse` | All 24 new Python files parsed successfully. No project imports or source execution. |
| Python JSON parsing and `tomllib` parsing | Shared fixture/config JSON and `pyproject.toml` parsed successfully. |
| Ruff `0.15.12`, `ruff check` | Combined 24-file scope: `All checks passed!`. |
| Ruff `0.15.12`, `ruff format --check` | `24 files already formatted`. |
| `git diff --check` | Exit `0`; whitespace check passed. |
| Git Bash, `bash -n scripts/hpc/small_molecule_stage0.lsf` | Exit `0`; shell syntax check passed, script not executed. |
| AST-literal `SOURCE_SHA256` extraction and static file-hash comparison | All 14 guarded native files matched the locally pinned tar contents; zero mismatches. No native module import/model run. |

These static results establish syntax, style and authenticated source-file bindings, not numerical behavior. The final prepared-feature provenance correction was independently reviewed and the full static lint/AST checks repeated successfully. Packaging created `results/releases/stage0-20261013/source.tar.gz`, its manifest and external release receipt; the sealing receipt below was written after that immutable snapshot.

Final review subsequently fixed launcher preservation: atomically create `RUN/payload` and refuse an existing payload; keep scheduler stdout/stderr in the parent and numerical outputs, caches and pytest temporary directories under the payload. This prevents pytest cleanup from erasing earlier run evidence. Source authentication now precedes the quota helper and rejects unmanifested execution files in `src`, `tests`, `examples` and `scripts`. The reviewer inspected these fixes and reported no remaining Work 0-5 implementation omission. Root repeated Ruff, AST parsing of the release helpers and `bash -n` after the changes; all passed. The numerical gates remain unexecuted.

Final prepared-feature identity correction: `native_features_sha256` and `base_features_sha256` now hash the actual dtype-normalized resident feature arrays; the original importer arrays are recorded separately as `native_precast_features_sha256`. This is a provenance correction, not observed model fidelity. All implementation bytes, including uncommitted source, are covered by the local release manifest; the external release receipt at `results/releases/stage0-20261013/release_receipt.json` records sealing outcome and hashes without making this ledger self-referential.

Local source sealing completed with `py -3.13 scripts/package_stage0_release.py --output results/releases/stage0-20261013`. The archive is 38,750,827 bytes and contains 175 authenticated files, including current source and relevant uncommitted contents. Source revision reference: `a7e4ccebd428bc188ac372586082c0243fb8caa0`. Archive SHA-256: `5cd38dd45f9c45825cd10decf38feebc96d560a2fd463117d1a58565d79d2a86`; manifest SHA-256: `909fb95bfe0a8e2550b56af06a405f1c5078edabf86a5b92090b87a4dbbf391b`. A standard-library integrity read independently checked the archive/manifest hashes, every declared archived file, and the embedded manifest; all matched. [The external receipt](../results/releases/stage0-20261013/release_receipt.json) records these results. No HPC transfer or submission occurred. This documentation receipt was appended after sealing and does not redefine the archived bytes.

No HPC run directory, model output, job ID, passed numerical test, derivative value, checkpoint-resume fidelity or scientific result exists for this implementation turn. The future launcher verifies allocation, an externally bound source manifest and quota (4 GiB/10,000 files); missing/insufficient quota exits 75 with tests not run. Historical 2026-10-10 quota is not evidence of current space. The dedicated environment needs a minimal locked pytest extension before future tests; no full dependency/model installation or silent native patch is performed now. Later allocated-HPC validation must retain exact commands, configuration/source/dependency/context/checkpoint identities, outcomes and failures before any gate can pass.
