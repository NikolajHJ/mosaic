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

## 2026-10-16: DBMol-inspired Protenix relaxed-graph gradients

Status: **focused two-case study completed and passed; immutable source-04 job `29735139` DONE; results downloaded and verified**. All three earlier failed jobs and their evidence remain retained. The user requested tests with a relaxed molecule whose gradients flow back through frozen Protenix, similar in idea to DBMol, and explicitly confirmed: “Yes—run on allocated DTU HPC”. Full Stage 0 A–G acceptance remains pending; these results do not turn historical checks into passes or change DBMol repository experiments.

| Final case | Atom/edge finite differences | Standalone primal objective | Actual projected updates |
| --- | --- | --- | --- |
| Uniform mixing `0.15` | Both leaves passed all three declared epsilons. | `1.7413299083709717` → `1.7370867729187012` | Three accepted steps; `eta_X=eta_E=0.01`. |
| Uniform mixing `0.30` | Both leaves passed all three declared epsilons. | `1.6747362613677979` → `1.633281946182251` | Three accepted steps; `eta_X=eta_E=0.01`. |

Actual final counts: **126 unit checks passed (59.66 s), seven GPU feature checks passed (25.55 s), one real-native pytest item covering both cases passed (207.07 s)**. Model/context digests and the same-primal bond-order-null controls passed. These are engineering gradient/update results on a synthetic protein context; no affinity, binding, chemical-validity, full Stage 0 or new scientific target claim follows. Detailed final receipts appear after the retained attempt history below.

The [focused protocol](protenix_relaxed_gradient_tests.md) uses the existing ten-heavy-atom acetanilide fixture and synthetic 20-residue protein context. It tests a DBMol-inspired contact loss composed of pocket-coverage minimum contact NLL and best-pair minimum contact NLL, with an explicit pocket subset and the native conservative upper-edge distogram convention. Only the common graph's neutral fixed-occupancy atom and edge probabilities are mutable. Model weights, protein/context, atom/token maps and ligand geometry-off treatment remain fixed. Native edge features see existence, so present-bond-order contrasts remain invisible. This is a gradient/update engineering study, not a DBMol full-method reproduction or an affinity, binding, chemical-validity or synthesis claim.

Planned allocated checks are finite real-Tiny forward/backward values, separate nonzero atom and edge-existence simplex-tangent derivatives, central finite differences over a declared epsilon sweep, the zero present-bond-order contrast boundary, bounded projected updates with the actual objective, and frozen-model/context/input identity checks. Exact configurations, immutable dirty-source-inclusive release/manifest hashes, assets/environment/device identities, actual quota, submission/job resources, commands, outcomes, failures and download checks will be appended from evidence. No local numerical test is part of preparation. Preparation alone establishes no job ID, gradient value, descent or passed test.

### Source, configuration and preparation

- Source reference is `7937b2af53fe4bff6b40395f7b68a8439ed8233f` plus the current uncommitted implementation, tests and documentation bytes, bound by the per-file manifest. Local sealed source: `results/releases/protenix-gradients-20261016-01/`; remote read-only release: `/work3/s234818/mosaic/releases/20261016-relaxed-gradients-01/source`.
- Source archive: 38,764,078 bytes, SHA-256 `8efcfa957288a5bf461c93186fed62d4959696435cd936d5b92da1f548c2f346`; manifest SHA-256 `96463bd0a58776e07381033b1ce745050fb047cfe633e5f337d7495c2dad8b73`, 181 files. The remote archive hash was verified before extraction. The earlier local sealing receipt says transfer not performed because it predates this subsequent transfer; it is preserved rather than rewritten.
- [The resolved configuration](../examples/configs/protenix_relaxed_gradients.json) uses seed `17`, uniform mixing weights `0.15`/`0.30`, pocket protein ordinals `[4,5,6,7,8]` mapped through prepared native indices, cutoff `8.0` Å, log epsilon `1e-8`, finite-difference epsilons `[0.01,0.003,0.001]`, `atol=0.002`, `rtol=0.15`, tangent floor `1e-7`, three projected updates, initial step size `0.01`, and at most ten halving/backtracking trials per update. Acceptance requires correct finite-difference sign and relative error at most `0.15` as well as the combined tolerance, at two adjacent scales. Accepted updates require measured actual loss decline greater than `1e-7`.
- Resident native source/model/assets retain the pinned 2026-10-13 identities. The numerical objective is frozen Tiny trunk/distogram only; no model training, diffusion/confidence, remote MSA or new model download is performed.

### Actual attempts and submissions

| Attempt | Action | Observed outcome and evidence |
| --- | --- | --- |
| Access/storage inspection | Read current DTU quota before setup/submission. | Reported work-pool 6 `88.49/100 GiB`, `241,957/20,000,000` chunk files; home `4.97/30 G`. Headroom is `11.51 GiB` in the finite work pool. Other queried work pools reported unlimited quotas. These are current inspection values, not a reservation. |
| Storage mapping inspection and conservative decision | `beegfs-ctl --getentryinfo` on the existing Mosaic path; inspect all current work-pool quotas. | Path lookup refused because `connAuthFile` was unavailable. No authentication/configuration bypass or alteration was attempted. Pool 6 is the only finite queried pool, with `11.51 GiB` free; pools 1–5 are unlimited. The most restrictive finite headroom therefore bounds every currently queried work-pool placement without relying on path mapping. Historical same-path pool-6 placement is corroborating only. The allocated preflight repeats full work-quota output and checks pool 6 plus actual filesystem availability against 4 GiB/10,000 files. This is a conservative quota bound, not a newly authenticated path-to-pool mapping or a space reservation. |
| Setup submission attempts 01–02 | Submit the bounded CPU setup command. | Both rejected because DOS-format stdin was sent to `bsub`; no job IDs or allocated work resulted. The next submission used LF input. Rejections are retained as submission failures, not model failures. |
| CPU setup `29735006` | `hpc`, one slot, `2GB`, ten minutes; only extend the dedicated existing Tiny environment. | `DONE` on `n-62-31-22`. Added `pytest==9.0.3`, `iniconfig==2.3.1`, `pluggy==1.7.0`, `pygments==2.21.0`; retained the original 57 packages. Used `uv compile --generate-hashes` and installation with `--require-hashes`. Remote setup directory `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-setup-01` retains before/after JSON, `pytest.lock` and logs. Exact submission: local ignored `results/hpc/20261016-relaxed-gradients/setup-submit.sh`. Setup completion is not a passed numerical/model test. |
| GPU validation `29735009` | `gpua10`, one exclusive-process GPU, four slots on one host, `8GB` per slot, one hour; dependency `done(29735006)`. | Submitted, outcome pending. Run `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-01`; exact command in ignored `results/hpc/20261016-relaxed-gradients/submit-01.sh`. Launcher authenticates the external manifest, claims a fresh payload, checks quota, then runs selected boundary/unit tests and the focused real-Tiny test. No numerical pass, gradient value or descent is claimed from submission. |
| Scheduler inspection | Read pending job and available GPU queues. | Job `29735009` is pending a physical GPU on `gpua10`; both A10 GPUs were busy at inspection. Reserved CPU slots do not mean the exclusive GPU is available. Numerical tests have not run. |
| Queue-change request | `bmod -q gpua100 29735009`, after a read-only check of other GPU queues. | Rejected: only administrators may run `bmod`. The job was not modified, cancelled or replaced, and remains the original `gpua10` submission. No permission/configuration change or bypass/retry was attempted; wait for the original allocation. |
| Conditional cancel/replacement preparation | Prepare a state-guarded cancellation and ordinary `bsub` replacement on `gpul40s`, subject to normal submission checks. | The state guard observed original job `29735009` already `RUN` and refused the transition. No `bkill` or replacement `bsub` occurred; no replacement job exists. Local ignored `results/hpc/20261016-relaxed-gradients/submit-02.sh` preserves the unused conditional transition command. The original A10 job is now running; unit/native outcomes remain pending. |
| First allocated numerical result, `29735009` | Selected unit/boundary phase on the allocated A10 host. | Job `EXIT 1`; pytest reports **119 passed, five failed, three deselected in 101.69 s**. Local retained log `results/hpc/20261016-relaxed-gradients/unit-01.log`; remote `payload/unit.log`/`payload/unit.xml`. The launcher stopped at this failed phase; real-Tiny native contact-gradient study **not run**. This is a retained failed attempt, not a native derivative/descent result. |

The GPU launcher selects ordinary small-molecule tests with `-m 'not slow and not small_molecule_model'`, followed only on success by `tests/test_small_molecule_contact_native.py -m small_molecule_model`; the latter launches the focused CLI in the same allocation. Planned native output is `payload/native-results/contact-gradients/`, with a sealed context, per-case raw gradient/finite-difference/update files and summary. JUnit XML and separate unit/native logs remain under the payload. The first attempt never reached that native phase. Complete artifact-download integrity and a corrected rerun remain pending. This documentation update postdates the sealed source release and does not redefine its archived bytes.

### First-run failures and corrections awaiting a new release

| Failed check | Observed cause and intended correction |
| --- | --- |
| `test_integer_native_element_channel_does_not_truncate_relaxed_values_or_gradients` | A real feature-boundary precision bug: one-hot matrix multiplication used A10 TF32 and rounded graph mass `0.05` to approximately `0.04998779` (maximum error about `1.22e-5`). Replace the sparse one-hot multiplication with indexed scatter so graph probabilities and gradients are preserved without a precision-mode/tolerance change. |
| `test_geometry_hook_masks_zero_distance_constants_before_identity_attention` | Test used a Python list for JAX multidimensional indexing, rejected by this runtime. Use a supported array index; geometry-mask policy is unchanged. |
| `test_fresh_process_resume` | Unit parent and fresh-process child competed for the exclusive GPU, leaving the child unable to initialize CUDA. Run the ordinary unit phase on CPU on the allocated DTU host; keep native model work on the allocated GPU. No local numerical work is introduced. |
| `test_projection_can_hit_boundary_without_deleting_an_atom_slot` | The `eta=1` fixture landed within float32 rounding of the vertex, while the assertion required exact zeros/ones (maximum residual about `5.96e-8`). Use a genuinely overshooting `eta=2` boundary fixture; keep the strict assertion and optimizer behavior. |
| `test_loader_is_resident_and_hook_shares_weights_without_mutating_native_model` | Equinox may rebuild immutable wrapper containers; Python container `is` identity is not a valid frozen-weight sharing contract. Check parameter-array leaf sharing instead, preserving the unmodified-native-model assertion. |

These are identified corrections, not yet a passed rerun. Preserve the original release/run/failure logs and seal corrected bytes into a new immutable identity. Existing setup/environment assets remain unchanged. No acceptance tolerance, fixture molecule, objective, gradient scope or frozen model is silently replaced to make the failed attempt pass.

### Corrected immutable attempt 02

The corrected local release is `results/releases/protenix-gradients-20261016-02/`, containing 181 files in a 38,768,528-byte archive. Archive SHA-256 `88e592c4ec066ef059e4050a5de6facdf5c48b317923292fc624dea2977cce0a`; manifest SHA-256 `14367e2b65f0c19bc0315bace7d0570bf4892a7f2d549e8aa829635787485000`. It was uploaded to `/work3/s234818/mosaic/releases/20261016-relaxed-gradients-02/source`, authenticated and made read-only. The earlier local sealing receipt's pre-transfer state is preserved. Attempt 01 remains intact.

Fresh job **`29735041` submitted** to `gpua10`: one exclusive GPU, four slots on one host, `8GB` per slot/32 GB host total, one hour. Run `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-02`; exact submission in ignored `results/hpc/20261016-relaxed-gradients/submit-corrected-02.sh`. The existing environment/setup job is reused without another install.

The corrected launcher sequences: (1) all 124 selected ordinary unit checks with `JAX_PLATFORMS=cpu` on the allocated DTU host; (2) seven feature checks with `JAX_PLATFORMS=cuda` on the allocated A10, independently testing the exact probability scatter that failed under TF32; (3) the native frozen-Tiny two-case derivative/projected-descent study. All remain fail-fast, with separate XML/logs under the new payload. CPU-unit routing resolves exclusive-GPU subprocess contention without replacing the explicit GPU precision recheck. Objective/configuration, epsilon sweep, strict relative derivative gate and measured descent threshold remain unchanged. **These are the submitted sequence and acceptance requirements, not passing results.**

Actual intermediate outcomes, source 02 unchanged:

| Phase | Observed result | Interpretation/evidence |
| --- | --- | --- |
| Full selected ordinary unit/boundary checks | **124 passed, three deselected, two warnings in 55.77 s** with `JAX_PLATFORMS=cpu` on the allocated DTU host. | The corrected engineering checks pass; `payload/unit.log`/`payload/unit.xml`. The warnings are the existing contract-test integer-to-boolean scatter FutureWarnings, not failed assertions. No local numerical computation. |
| Explicit A10 feature precision/gradient recheck | **Seven passed in 23.85 s** with `JAX_PLATFORMS=cuda` on the allocated A10. | `payload/feature-gpu.log`/`payload/feature-gpu.xml`; this includes exact relaxed-element channel preservation and raw feature-boundary derivatives. It independently verifies the indexed-scatter correction on the hardware that exposed the TF32 bug. It is not a real pretrained-Tiny end-to-end derivative pass. |
| Native two-case study | One focused native pytest item running under job `29735041`. | Outcome pending; no native gradient/descent or full Stage 0 A–G acceptance claimed. |

Subsequent actual native outcome: job **`29735041` EXIT 1**, native pytest item failed after **135.22 s**. First mixing case (`0.15`) produced finite loss `1.741985321`, raw atom-gradient norm `0.335869393` and selected simplex-tangent autodiff derivative `0.18240170925855637`. Retained central differences:

| Epsilon | Finite-difference derivative | Relative error | Acceptance |
| --- | --- | --- | --- |
| `0.01` | `0.1733541489` | approximately `4.96%` | pass |
| `0.003` | `0.0755389531` | approximately `58.59%` | fail |
| `0.001` | `0.7123947144` | approximately `74.40%` | fail |

The required two adjacent agreeing scales were absent. The run stopped at this atom-derivative gate; no edge-direction probe, projected optimization step or second mixing case executed. The finite raw atom gradient demonstrates backpropagation, but the failed numerical gate prevents a claim of validated gradients or descent. Failed native summary and `payload/native-results/contact-gradients/case-0/report.json`, `initial.npz` and the six plus/minus probe NPZ files remain in the unique source-02 run.

### Pivotal precision-profile decision after native failure

The separate reviewer audited the pinned Protenij matrix-product/float32 attention paths and JAX default behavior: no explicit high-precision override was active on the consumed paths. The observed small-scale derivative disagreement is consistent with reduced-precision rounding but does not establish that cause. Preserve the failed native report; do not reinterpret it as a pass or loosen `atol`/`rtol`, signal floors, epsilon scales or required adjacent agreement.

Prepare a **new immutable numerical profile** that explicitly sets `matmul_precision="highest"` before model evaluation and records effective `jax_default_matmul_precision`, x64 state and XLA flags. This changes the declared matrix-product precision, not the frozen model weights/objective, graph storage, seed, epsilon sweep, derivative gates or measured-descent threshold. [JAX's official precision documentation](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html) explains that matrix-product precision can be controlled independently for 32-bit inputs. This is a declared new-profile counterfactual, not silent replacement of source 02's result.

For matching, the new CLI preparation adds `--prepared-study` and `--prepared-summary-sha256`: reuse the authenticated source-02 study summary and exact serialized native context, authenticating summary hash, context bytes, fixture/model identities and native maps. Regenerate the hard fixture graph outside the differentiable model path and check its identity/order against those bindings; avoid generating a new conformer. Later source/release/job IDs and actual outcomes must be recorded separately. **No matched-profile submission or pass is established by this preparation decision.**

### Failed-evidence download and matched-input bindings

Both failed attempts were downloaded and safely extracted using standard-library path/integrity checks only, with no local numerical evaluation. Parent directory: ignored `results/hpc/20261016-relaxed-gradients/`. Each archive has an extracted directory with the same basename and a sibling `*-download.json` receipt.

| Evidence bundle | Size | SHA-256 | Verified declared files |
| --- | --- | --- | --- |
| `20261016-gradient-evidence-01.tar.gz` — setup and first failed attempt | 19,344 bytes | `940576d4ba3e36cc60ae5e93804cf30cd78cd2888f334defc438884dd68fda1d` | 21 |
| `20261016-gradient-evidence-02.tar.gz` — second failure with raw gradients/probes/context and logs | 4,639,112 bytes | `06b1ea901b677463def26f13673b03866779c6daf4e429829e0c2479048cb5d6` | 26 |

The authenticated source-02 study summary SHA-256 is `663835f89a54ba0e69e202f25062a7af12d5c4c42b4f795206652396fffe3577`. It records configuration SHA-256 `96ccced5fb7ab4b970e8c842f6aacc12375066e60934ffdcf092402ca877cd71`, fixture SHA-256 `d36c687fb5d4428414ffc7b14b75082b907e3fd97b3c644803cde3347f34b99c`, context JSON SHA-256 `a0bc4321691ca7de98dcf8d15473a73f54d31e3c6369bec093deb421542f3b4d`, and context NPZ SHA-256 `55471d4f63b67737c01b7a5627d713bc16253663144bea2814bea84e4703cf29`. Actual maps are ligand native tokens `20–29` and selected pocket tokens `[4,5,6,7,8]`.

Runtime evidence in that summary is CPython `3.12.11`, JAX/jaxlib `0.11.1`, Equinox `0.13.8`, NumPy `2.5.3`, NVIDIA A10 with driver `615.71.09`, and `jax_enable_x64=false`, with the complete dependency/native-source/asset receipts retained. The summary records failure before an after-run model/context digest, so do not infer a completed final immutability check from its initial digests. The matched numerical-profile comparison must authenticate and reuse these exact context bytes; these file receipts establish integrity, not a passing derivative gate.

### Matched highest-precision attempt 03 submitted

Local source release `results/releases/protenix-gradients-20261016-03/`: 181 files, archive 38,768,168 bytes, SHA-256 `943f0d8d0f02b4e90d80bc5d7dac76a9996dac7e06217c28fa2101429b1b413b`; manifest SHA-256 `da51aa26b630f43b61d530e08770c67527bf8f3deeb76144ce63e9c72971885f`. The uploaded archive hash was verified; remote source `/work3/s234818/mosaic/releases/20261016-relaxed-gradients-03/source` is read-only. Fresh run `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-03` preserves both prior failed attempts.

Actual GPU job **`29735092` submitted** to `gpua10`, one exclusive GPU, four slots on one host, `8GB` per slot/32 GB total, one hour. Exact operator command: ignored `results/hpc/20261016-relaxed-gradients/submit-highest-03.sh`. It reuses the existing authenticated Tiny environment/model/cache, with no new installation or weight change. The source-02 prepared-study path and independent summary SHA-256 `663835f89a54ba0e69e202f25062a7af12d5c4c42b4f795206652396fffe3577` are passed explicitly.

Replay now fails closed unless every configuration field except `matmul_precision` equals source 02's configuration, the fixture and model identities match, context JSON/NPZ hashes authenticate, and current initial model/context resident-array digests equal the prior initial digests. `matmul_precision="highest"` is set before model imports/tracing and the effective JAX precision/x64/XLA state is recorded. This is not an independent new conformer/fixture comparison. The native pytest wrapper itself imports no JAX, leaving GPU ownership to the native child; ordinary unit checks remain on the allocated host CPU and the independent feature phase on its allocated GPU. No epsilon, tolerance, objective, fixture, mixing weight, seed or descent criterion is relaxed. **Submission is established; attempt-03 numerical outcomes and final identity checks remain pending.**

### Actual matched-profile result and null-control correction

Job **`29735092` EXIT 1**. On the allocated host, all 124 selected ordinary unit checks passed in `57.45 s`, and all seven A10 feature checks passed in `23.86 s`. The native item failed after `191.84 s`, following successful atom and edge directional derivatives in the first (`mixing=0.15`) case:

| Leaf | Raw gradient norm | Selected autodiff derivative | FD at `0.01` | FD at `0.003` | FD at `0.001` |
| --- | --- | --- | --- | --- | --- |
| `X` | `0.3360345716` | `0.1835568994` | `0.1834988594`; `0.0316%` relative error | `0.1834829648`; `0.0403%` | `0.1833438873`; `0.1160%` |
| `E_upper` | `0.2888619756` | `0.1163478047` | `0.1169145107`; `0.4847%` | `0.1162290573`; `0.1021%` | `0.1159310341`; `0.3582%` |

Both leaves passed the unchanged sign/absolute-plus-relative/relative-only gates at all three epsilons, exceeding the two-adjacent-scale requirement. Present-bond-only raw gradient contrast was exactly `0`, as required by the existence-only native feature map. These are valid intermediate engineering results under the explicit highest precision profile, not a completed optimization study or retroactive certification of default-profile attempt 02.

The subsequent null-perturbation output guard measured maximum distogram difference `0.000224113`, above the unchanged `2e-5` threshold, and stopped the attempt. No projected update, second mixing case or final after-run model/context immutability check was reached. Preserve the failed report and raw arrays; the whole attempt remains failed.

Reviewer finding and bounded correction: the existing null guard compared a standalone primal null forward with auxiliary values obtained during the autodiff forward. Those execution paths can compile differently, confounding a graph-invariance check with floating-point-path differences. The corrected checker will persist an unaltered standalone primal baseline, compare the bond-order-null forward against that baseline through the **same** primal path, retain `2e-5`, and persist the autodiff-forward-versus-primal discrepancy separately as a diagnostic. No raw derivative, finite-difference, objective, precision, epsilon, tolerance or descent requirement is removed. This restores a matched control; source 04 sealing/submission and an actual rerun are still pending. The attempt-03 failing measurement is retained and not reclassified.

Source-03 failed evidence was subsequently downloaded to ignored `results/hpc/20261016-relaxed-gradients/20261016-gradient-evidence-03.tar.gz` and its same-basename extracted directory. Archive size **9,242,435 bytes**, SHA-256 `675c497ecdbc3fd5fd55839e5220de70b263e263f27e84c9c3bcae8670607f5c`; **34 declared files verified**. The sibling `20261016-gradient-evidence-03-download.json` records safe extraction/standard-library byte checks only, with no local numerical evaluation.

### Corrected matched-primal attempt 04 submitted

Local source release `results/releases/protenix-gradients-20261016-04/`: **182 files**, archive **38,772,316 bytes**, SHA-256 `220fc886cf276337f14fa10a7fbfd477d68ef5e6d42cf92570ead83559d5eb83`; manifest SHA-256 `407e12a8a76989217d7e97f0cf7047c94f49b2fe1de7be88165d234257c65b31`. Uploaded source `/work3/s234818/mosaic/releases/20261016-relaxed-gradients-04/source` was authenticated and made read-only. Fresh run `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-04` preserves all prior attempts.

Actual job **`29735139` submitted** to `gpua10`, one exclusive GPU, four slots on one host, `8GB` per slot/32 GB total, one hour; exact operator command is ignored `results/hpc/20261016-relaxed-gradients/submit-primal-04.sh`. It retains the source-03 highest precision profile and authenticated source-02 prepared study/summary SHA-256 `663835f89a54ba0e69e202f25062a7af12d5c4c42b4f795206652396fffe3577`, with all nonprecision configuration fields matched to source 02. Existing environment, assets and model weights are reused.

The checker now persists `baseline-primal.npz`, compares unchanged/null atom-and-existence inputs exactly, and compares standalone primal null logits with the standalone primal baseline at the unchanged `2e-5` limit. AD-versus-primal logits remain a recorded diagnostic, while initial scalar parity independently requires `2e-5`. It records `initial_primal_loss` and `final_primal_loss` and additionally requires a standalone-primal final decrease greater than `1e-7`; the three evaluated candidate declines and all finite-difference gates remain required. New regressions in `tests/test_small_molecule_gradient_checks.py` exercise these control boundaries; **126 unit checks are expected, not yet passed**. The pivotal precision/control decisions are now also recorded in [the implementation plan](stage_0_implementation_plan.md). **Attempt-04 outcomes, projected descent and final immutability remain pending; full Stage 0 A-G acceptance remains pending.**

### Actual source-04 completion and final evidence

Job **`29735139` DONE**, source 04 unchanged: **126 selected unit checks passed in 59.66 s**, **seven A10 feature checks passed in 25.55 s**, and **the real native pytest item passed in 207.07 s**, covering both mixing cases. Run host `n-62-18-5`, NVIDIA A10 with driver `615.71.09` and 23,028 MiB; existing authenticated CPython/JAX/Equinox environment and frozen assets were reused. Effective numerical profile: `jax_default_matmul_precision="highest"`, `jax_enable_x64=false`, `XLA_FLAGS=""`; `prepared_context_reused=true` and prior summary SHA-256 `663835f89a54ba0e69e202f25062a7af12d5c4c42b4f795206652396fffe3577`. Final config SHA-256 is `5a216c87afaa9d4283730fbf790e9b09ad2f0bf553edbdbb8a6830d4711d61f0`.

| Mixing/leaf | Raw gradient norm | Selected autodiff tangent | Finite differences at `0.01`, `0.003`, `0.001` | Outcome |
| --- | --- | --- | --- | --- |
| `0.15`, `X` | `0.3360346332` | `0.1835573018` | `0.1835227013`, `0.1836220423`, `0.1837611198` | All three pass. |
| `0.15`, `E_upper` | `0.2888617592` | `0.1163473874` | `0.1169204712`, `0.1162091891`, `0.1161694527` | All three pass. |
| `0.30`, `X` | `0.6477462225` | `0.4851007015` | `0.4849970341`, `0.4849036535`, `0.4848837852` | All three pass. |
| `0.30`, `E_upper` | `1.1079191267` | `0.5728260279` | `0.5729675293`, `0.5726814270`, `0.5734562874` | All three pass. |

Every accepted projected step used `eta_X=eta_E=0.01` on its first trial, with nonzero changes to both arrays. Evaluated objective traces: mixing `0.15`, `1.7413297892` → `1.7399121523` → `1.7384976149` → `1.7370867729`; mixing `0.30`, `1.6747355461` → `1.6614136696` → `1.6476159096` → `1.6332821846`. Independent standalone-primal initial/final objectives are the top results table's values and pass the extra `>1e-7` decline requirement. These different primal/evaluated values are retained with their own states/paths.

In both cases, the present-bond-only raw-gradient contrast, same-primal null-logit difference and same-primal null-scalar difference are exactly `0`. Initial scalar AD/primal discrepancies were `1.1920928955078125e-7` and `7.152557373046875e-7`, passing the `2e-5` parity gate. Cross-path maximum logit discrepancies remain visible diagnostics: `0.00018835067749023438` and `0.0001087188720703125`. The same-primal controls isolate graph invariance without erasing those compilation-path differences.

Final frozen-model resident-array digest **before = after**: `8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2`; prepared-context digest **before = after**: `14f774f37776bcff180a916bf6d4995b3c253dcdfe70255b9957754988cd4e68`. These initial digests also match the authenticated source-02 comparison inputs. Thus the completed highest-profile study retained the same model/context rather than substituting a new conformer or changing frozen weights.

Final evidence archive: ignored `results/hpc/20261016-relaxed-gradients/20261016-gradient-evidence-04.tar.gz`, **26,362,938 bytes**, SHA-256 `7ed2474ad86107a41895bcc79531ddff3d31609a8a9466182e8f71ca76536fb2`; same-basename extracted directory with **63 declared files verified**. Sibling `20261016-gradient-evidence-04-download.json` records standard-library safe extraction/byte integrity only. Complete summary: `20261016-gradient-evidence-04/runs/20261016-relaxed-gradients-04/payload/native-results/contact-gradients/summary.json`. Raw initial/final graphs, gradients, primal baselines, all probes/candidates, context, logs, JUnit and manifests remain available locally. No local numerical evaluation was performed.

Final quota inspection: work pool 6 `89.01/100 GiB`, `250,966/20,000,000` files. No storage issue occurred and no cleanup was performed. The previous failed releases/runs and downloaded evidence remain intact. The focused user-requested gradient/descent test is complete; ligand-only acceptance, all-mode A–G geometry/protein/replay gates and full M3 remain pending.

Final independent review confirmed all 12 finite-difference samples passed, zero bond-order gradient/output/scalar nulls, initial/final model/context digest equality with the authenticated source-02 inputs, manifest/job binding, and all 155 source-file hashes named by the runtime summary matching sealed source 04. The root separately checked byte hashes of 166 current execution files (`src`, `tests`, `examples`, `scripts`, `pyproject.toml`) against the tested source-04 release; all matched. Documentation updates occur after sealing and remain external receipts rather than redefining immutable source bytes. These final local checks inspect receipts/file integrity only; no further numerical computation or model execution occurred locally.

## 2026-10-16: requested 100-update carbon-chain trajectories

Status: implementation prepared; native 100-update runs not yet submitted. The
user requested 100 updates for each of the two relaxation strengths and a view
of loss/molecule changes, starting from a carbon chain. The new hard seed is
neutral decane (`CCCCCCCCCC`), retaining ten occupied slots and mixing 0.15/0.30.
This starts fresh trajectories rather than resuming the earlier acetanilide runs.
See [the protocol](carbon_chain_100.md) and sealed configuration/fixture.

Pivotal bounded choice: preserve seed 17, highest FP32 matmul, fixed RNG/context,
protein/pocket, cutoff/epsilon and initial eta=0.01. Initial derivative/null gates
retain their thresholds. Every proposed update additionally evaluates a native
standalone primal; both AD-evaluated and primal losses must decrease by >1e-7
before acceptance. Up to ten halving trials are allowed. Failed checks stop,
without silently changing tolerances/counts. Save states 0..100, raw gradients,
readouts, rejected candidates and exact source/config/model/environment hashes.
Chemical readouts are detached, unrepaired and may be invalid/disconnected.

Plot setup job **29735600 DONE**, queue hpc, one slot/2GB, 10-minute bound;
actual scheduler turnaround 27 seconds. Unique helper directory:
`/work3/s234818/mosaic/runs/20261016-carbon-chain-plot-setup-01`.
`uv pip compile --generate-hashes --constraint constraints.txt` and
`uv pip install --require-hashes` added only matplotlib 3.11.2, contourpy 1.4.0,
cycler 0.12.1, fonttools 4.67.0, kiwisolver 1.5.1 and pyparsing 3.3.3.
An allocated before/after assertion confirmed every existing package version
unchanged, and the Agg plot/RDKit drawing import smoke passed. Requirements,
constraints, before/after JSON and setup stdout/stderr are retained. No model
weights were downloaded or changed. No numerical work ran locally/on login.
Current quota output: pool6 89.04/100 GiB, 250969/20000000 files; other work
pools unlimited. The source/run preflight will repeat conservative finite-pool
checks and require 4 GiB/10000 files free; no cleanup is planned.

Local Ruff 0.15.12, stdlib AST/JSON and shell syntax checks are static only.
Two new allocated-unit regressions cover per-update snapshots and rejection of
an AD-only improvement unconfirmed by the standalone primal. The authored
visualization includes loss/change plots, fixed-slot raw heavy graphs, endpoint
probability heatmaps and an offline slider report. These are not performed
100-step results yet; immutable source/submission/outcome receipts follow.

### Carbon-chain source 01 submitted; unit checks passed

Actual job **29735673**, queue `gpua10`, one exclusive GPU, four slots on
one host and 8 GB per slot/32 GB total, one-hour bound, dependency
`done(29735600)`. Unique run:
`/work3/s234818/mosaic/runs/20261016-carbon-chain-100-01`.
Immutable source:
`/work3/s234818/mosaic/releases/20261016-carbon-chain-100-01/source`.
Local release `results/releases/carbon-chain-100-20261016-01/` contains
187 files; archive 38,791,628 bytes, SHA-256
`b339d6a70ab024825adb418cd169714b4f29821db2d8f439387de2f21e499289`;
manifest SHA-256
`da6ce2e2dc7e55d9c6c50c842f8591ba54543bed0efdb8e3844eb1deab078569`.
Git reference `7937b2af53fe4bff6b40395f7b68a8439ed8233f` is reference only;
uncommitted source bytes are included. Upload SHA and shell syntax were checked,
then source made read-only. Exact submission is retained in ignored
`results/hpc/20261016-carbon-chain-100/submit-01.sh`.

Allocated **128 unit checks passed**, three deselected, in 61.06 seconds,
including both new trajectory/primal regressions. Native optimization remains
in progress; this receipt does not claim 100 updates or a completed trajectory.
The user subsequently requested a polished report using
`render_protenix_trajectory.py`. Presentation-only revisions will be sealed
separately and run on allocated HPC against the saved study. Running experiment
source 01 will not be mutated or replaced by those report revisions.

### Native carbon-chain run completed; polished report submitted

Job **29735673 DONE**. Both starts completed exactly **100 accepted updates**
and saved all 101 states per case. The standalone primal losses were
`1.8446226119995117` → `1.7735052108764648` (mixing 0.15) and
`1.6099765300750732` → `1.1193751096725464` (mixing 0.30). Raw categorical
readouts remain the connected, sanitized decane chain at every saved state:
one unique slot-labelled hard graph, zero changed atom/bond labels and
101/101 sanitized readouts per case. Probability changes, rather than a new
discrete molecule, account for this measured optimization outcome.
Mean atom total variation from the start is `0.039444055408239365` and
`0.09065239131450653`; mean absolute bond-existence changes are
`0.018457680940628052` and `0.04041802138090134`. No chemistry repair or
fragment selection occurred. Completed study summary SHA-256:
`81ae15d61983690fe8a260360ba856e27c17b0ea90be0da8492f251ed6240d51`.

Presentation-only job **29735743 submitted**, queue hpc, one slot/4GB,
15-minute bound, dependency `ended(29735673)`; renderer itself additionally
requires a passed, complete study. Report source:
`/work3/s234818/mosaic/releases/20261016-carbon-chain-report-01/source`;
unique run `/work3/s234818/mosaic/runs/20261016-carbon-chain-report-01`.
Local release `results/releases/carbon-chain-report-20261016-01/`,
189 files, 38,797,960 archive bytes, archive SHA-256
`f9a6b1951ffc06e07749f35f7473addf30fcafb6b43d42b3ea4e59e65b19b01e`;
manifest SHA-256
`e49129ba0a5a8c3b4e7161b9ca4eb4aace60c68e6f7a316b2f9481b311838ba2`.
This source adds the HTML template and allocated CPU analysis launcher;
the GPU experiment source remains immutable. Static Ruff/format and shell
syntax checks passed before sealing. Exact operator script is ignored
`results/hpc/20261016-carbon-chain-100/submit-report-01.sh`.

### Final report, downloads and retained numerical evidence

Report job **29735743 DONE**. Browser inspection found the controls/plots
working; presentation source 02 adds a direct unchanged-hard-chain finding,
singular/plural text and heatmap slot labels 1..10 consistent with depictions.
Final presentation-only CPU job **29735759 DONE**, hpc, one slot/4GB,
15-minute bound, on `n-62-11-61` (started 10:49:05, finished 10:49:24 DTU
local time). Exact submission:
`results/hpc/20261016-carbon-chain-100/submit-report-02.sh`.
Immutable source `/work3/s234818/mosaic/releases/20261016-carbon-chain-report-02/source`;
unique run `/work3/s234818/mosaic/runs/20261016-carbon-chain-report-02`.
Local `results/releases/carbon-chain-report-20261016-02/` contains 189 files,
38,797,431 archive bytes; archive SHA-256
`27de7274a11a934f883609f3dc16bf00b2db025941843cd4a27b2947d4b862a5`;
manifest SHA-256
`b00593833cb1ecbc73841a22068e8b0da3008ff6f9c9b7265e4baca6a07fdd30`.
The completed input summary hash remains
`81ae15d61983690fe8a260360ba856e27c17b0ea90be0da8492f251ed6240d51`.
Renderer SHA-256 `899a2e97e5a0630cc93facbb43a94a81b2082d6d15d04b144bc065e0eaeb5fb1`;
template SHA-256 `9ebfeecea588584edd9dbe60d08e0afdd623cbbb79346a392953f44305305061`.
Plotting uses Matplotlib 3.11.2, NumPy 2.5.3, Pillow 12.3.0 and RDKit
2026.9.1. No second model run occurred for the presentation revisions.

The native GPU job ran on `n-62-18-5` from 10:38:58 to 10:45:26 DTU local
time; A10, driver 615.71.09, 23,028 MiB. Config SHA-256
`b009546466d29ae202abf4ed0b781b3b438adb58f0cf94c5f79b17406b8d8c2d`;
fixture SHA-256
`655e2199389201d81d9ea5ac2c4ebc7e38da5b8abea126c7862bd49b0735788f`.
The initial derivative gates and same-primal bond-order controls passed for
both starts; raw-gradient, scalar and output null differences are exactly zero.
Initial AD/primal scalar discrepancies `3.5762786865234375e-7` and
`2.384185791015625e-7` pass the unchanged `2e-5` limit. Cross-path logit
differences remain diagnostic (`0.00011682510375976562` and
`0.00015425682067871094`). Each of the 200 accepted updates satisfies both
the AD-evaluated and independently measured primal decline gates.
Frozen model digest before = after:
`8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2`;
new decane prepared-context digest before = after:
`737edd04299752288d68c58655bd9f12a51ae313becaea9bd3a8f45d6010025f`.
Scheduler receipt includes an external `GPU 0: Detected Power error` message;
that message is retained alongside the successful job and numerical gates.
No hardware-cause inference or additional hardware intervention was made.

Downloaded evidence under ignored `results/hpc/20261016-carbon-chain-100/`:

- `20261016-gradient-evidence-carbon-chain-100-01.tar.gz`: 291,872,438 bytes,
  SHA-256 `68d70cd1c4eb55431ab96b22723fabed845e2bfcc9ca4182e9b4ba252d03b181`,
  **699 declared files verified**; setup/native/report-01 logs, locks, quota,
  JUnit, every accepted state/gradient and all probes/candidates retained.
- `20261016-gradient-evidence-carbon-chain-report-02.tar.gz`: 3,058,939 bytes,
  SHA-256 `87f8bf800d388e9061992b4feb860faa5b647d2b1e6aafbbfd4e2d1c2bd0f537`,
  **21 declared files verified**; final report, input summary, source identities,
  quota and scheduler/log evidence retained.

Both same-basename directories were safely extracted and all declared file
sizes/hashes verified by the earlier stdlib-only download verifier. Sibling
`*-download.json` files record the exact local byte-integrity scope. The final
report files are copied to `results/hpc/20261016-carbon-chain-100/report/`;
`report.html` SHA-256
`6734f89cee8c4bde15af3f7fa6a13c0efb8eab9645a047c5220c9d829a36a78a`.
Browser QA confirmed case selection, update-100 controls, keyboard slider
Home/End, advancing playback and no JavaScript console errors. Plots and
depictions were inspected visually; no numerical analysis was run locally.
The self-contained report is open in the app and works offline as a saved file.

Final recorded work pool6 quota: **89.99/100 GiB**, 257161/20000000 files.
No storage issue or cleanup occurred. This completes the requested 100-update
carbon-chain study and report. Full both-mode Stage 0 A–G/M3 and restart/protein
regression remain outside this focused result and pending.

## 2026-10-16: probability-weighted chain report

The user requested atom and bond sizes based on their probabilities and asked
whether the experiment included a protein and binding spot. This is a
presentation-only revision of the completed 100-update study, not a model rerun.
The detached `scripts/protenix_trajectory_depiction.py` helper draws the same
raw categorical graph with atom radius/label size based on its displayed element
probability, and hard-bond width based on bond existence `1-P(NO_BOND)`.
Shared absolute 0..1 size scales apply across both cases and every update;
occupancy stays fixed, hard absent bonds stay absent, and no chemistry repair
or physical atomic-radius interpretation is introduced. Exact SVG hover values
and drawing formulas/policy/source hash are retained in the report metadata.

Protein/pocket interpretation is explicit in the report and protocol:
synthetic sequence `ACDEFGHIKLMNPQRSTVWY`, selected residues F5/G6/H7/I8/K9,
complete protein sequence in context, frozen protein and model, ligand-only
probability updates, conservative whole-bin `<8 Å` contact objective. No
experimental structure or validated biological binding pocket was supplied.
The renderer authenticates the sealed fixture against the completed study's
fixture SHA before presenting that context.

Allocated CPU job **29735815 submitted**, queue hpc, one slot/4GB,
15-minute bound, dependency `ended(29735673)`. Source:
`/work3/s234818/mosaic/releases/20261016-carbon-chain-report-03/source`;
unique run `/work3/s234818/mosaic/runs/20261016-carbon-chain-report-03`.
Local release `results/releases/carbon-chain-report-20261016-03/` contains
191 files, archive 38,804,219 bytes, archive SHA-256
`78e111b17f754092392c7b2804e2d24a6b3e2ca31f3b0cb1cd2984b78b59e662`;
manifest SHA-256
`a6558c367f6c5b0c45b38c699e9e5b20f9682eb72d0234eb67c2ac60831f8f85`.
Git reference `8b08daf28cb7e0731e686b476e6f1a9db5a10cd9` is reference only;
actual dirty bytes are sealed. Upload SHA and shell syntax passed before
submission; source is read-only. Exact operator script:
`results/hpc/20261016-carbon-chain-100/submit-report-03.sh`.
Pre-upload quota: pool6 90.00/100 GiB, 257163/20000000 files. Allocated
preflight repeats the 4 GiB/10000-file headroom gate. Local Ruff/format and AST
checks passed; no local rendering/numerical work. Completion/visual QA and
download receipts remain pending at submission.

### Weighted report 03 completed; inline-label issue retained and corrected

Job **29735815 DONE**, source 03 unchanged, rendered all 202 states and
both probability-weighted PNG/SVG galleries on allocated HPC. Downloaded
`20261016-gradient-evidence-carbon-chain-report-03.tar.gz`, 3,592,478 bytes,
SHA-256 `5f0c0491217ee9aceb690f24ee6e845fc3acfce4c22bf50f350f7f7f9b08db7d`;
**21 declared files verified** by stdlib-only local extraction/hash checks.
Input summary SHA is unchanged. Rendering/depiction/template source hashes are
in the retained `analysis_summary.json`; local copies of the report were
updated from that evidence. Quota: pool6 90.10/100 GiB, 257766/20000000 files.

PNG depictions passed visual inspection, but browser QA found missing visible
SVG atom labels and captions despite working geometry, sizes and hover titles.
Matplotlib's glyph-path xlink references lose their effective namespace when
the XML-serialized SVG is inserted through HTML `innerHTML`. Source 03 and
that unsuccessful inline presentation remain retained. This is a presentation
failure; no model/numerical trajectory was changed or rerun. The user also
explicitly reiterated that element types must remain clear.

Source 04 emits native SVG text instead of glyph references and keeps all
element/slot labels fixed at **10 points, bold**, independent of probability.
Only atom circles and bond line widths vary. Allocated CPU job **29735830
submitted**, same hpc one-slot/4GB/15-minute bounds, same completed input study.
Unique source `/work3/s234818/mosaic/releases/20261016-carbon-chain-report-04/source`;
run `/work3/s234818/mosaic/runs/20261016-carbon-chain-report-04`.
Local release `results/releases/carbon-chain-report-20261016-04/`, 191 files,
38,806,335 archive bytes; archive SHA-256
`638e322a6217934686369fafc941b6c5d4b2f3e0761575f8b4a96648af24debc`;
manifest SHA-256
`d182ed133f8e233bd8357cb83a332ec4eb68840a887e4b3d218f80afb041f0bb`.
Uploaded archive hash and launcher syntax passed, then source was made
read-only. Exact submission:
`results/hpc/20261016-carbon-chain-100/submit-report-04.sh`.
Static Ruff/format/AST passed. New rendering outcome and visible-label browser
QA remain pending at submission; no extra model run or local numerical work.

### Weighted report 04 completed; element labels verified

Job **29735830 DONE**, rendering all 202 states with fixed bold element/slot
labels in both PNG and SVG. Browser QA verified visible `C1` through `C10`
as native SVG text at update 100, probability hover titles, case/slider
controls and no JavaScript console errors. The 30% endpoint hover values
include atom 3 `P(C)=0.59118` and bond 2–3 `P(exists)=0.71462`; sizes vary
while the hard labels and connectivity remain decane. Model execution and
the completed study summary remain unchanged.

Downloaded `20261016-gradient-evidence-carbon-chain-report-04.tar.gz`:
2,040,460 bytes, SHA-256
`cab348bf45aa94b3900c18df354ff2dadbe26fcf05450974ff15bac3c9f290b0`;
**21 declared files verified** by stdlib safe extraction and file hash checks.
Immutable report source hashes: renderer
`b97a0e159d0f49f483e9be486ee2b0b561a802c7ae06e18a9cfc4b150ef9ae23`,
depiction `f3eba2a4462306552cb15a7456778480a8920944424081a35aa6332aa51a91e1`,
HPC template `d78b00dbbb6141e1bdd5e48750560225961917999ebcb851b59046c5a385c97c`.
Downloaded HPC `report.html` SHA-256:
`44f5f5db3edecc8ee583b489e66d41e7a9794d5cfdf2b04dd36def32842ca871`.
All previous renderings/evidence are preserved. Quota at final collection:
pool6 **90.20/100 GiB**, 258371/20000000 files; no storage issue or cleanup.

The user viewed the report in a very narrow app panel. A **local CSS/static
HTML edit only**, permitted by the standing local-editing scope, gives chain
SVGs a minimum readable width with horizontal scrolling, wraps narrow toolbar
and header layouts, and makes small-screen galleries one column. The current
source template includes the same responsive styles for future rendering.
No plotting, numerical analysis or model execution ran locally. The **entire
HTML body, embedded SVGs, probabilities, losses and plots remain byte-for-byte
unchanged** from HPC report 04; only head styles and a presentation-patch meta
tag changed. This avoids rerunning quantitative figures just for CSS.

Exact local static editing script:
`results/hpc/20261016-carbon-chain-100/apply_readability_css.py`.
The report's sibling `presentation_patch_receipt.json` records base/final
hashes and the unchanged-body checksum. Final delivered `report.html` SHA-256:
`1a197905ce5d0e962e77993a5ea9a06ed45b2bf25a5238d2eda66f49f931641f`;
unchanged HTML-body SHA-256:
`b5c16af3106a4bad07cedbeac7117108828be6da4d20bdd09f3af130187841a2`;
current responsive source-template SHA-256:
`47e9ddd17b244daea48a47b164f45cda78bd545e82f54fc25a52b094eff13865`.
Browser inspection after the style edit confirmed readable element labels in
the narrow panel; root separately checked the byte identities above. Final
report and probability-weighted figures are in
`results/hpc/20261016-carbon-chain-100/report/` and the report is open in-app.
This completes the requested presentation update and protein/pocket explanation.

## 2026-10-16: requested 500 updates per carbon-chain initialization

Prepared; not yet submitted or passed. The user requested rerunning the study
with 500 updates. Interpret this as **500 total accepted updates per start**,
fresh from decane at mixing 0.15/0.30. Preserve the 100-update study/report.
The [500-update protocol](carbon_chain_500.md) records the bounded choices.

Pivotal input-comparison decision: reuse completed job 29735673's authenticated
context/fixture/model via summary SHA-256
`81ae15d61983690fe8a260360ba856e27c17b0ea90be0da8492f251ed6240d51`,
rebuild the hard graph and start at step zero. The explicit
`--allow-update-count-change` path requires a passed prior study and permits
only a strictly longer integer `checks.steps`; every other config field must
match, including precision/seed/pocket/objective/eta/derivative tolerances.
Existing precision-only prepared-context replay keeps its former contract.
Six new unit cases exercise policy opt-in, input preservation and rejection
of changes to seed, precision, pocket, eta or a non-longer budget. **134 unit
checks expected**, not yet passed. Numerical gradient-checker/model code is
unchanged. All numerical work remains allocated DTU HPC.

Quota metadata: pool6 **90.20/100 GiB**, 258373/20000000 files. The completed
100-update study has 289168702 raw bytes/644 files; conservative longer-run
planning reserves 4 GiB for new run/archive/render artifacts while preserving
4 GiB free. GPU preflight therefore requires **8 GiB/10000 files** free,
without cleanup or relaxed verification. GPU launcher runs selected units on
its CPU and then the native model; report rendering will use a separate CPU
allocation after success, avoiding holding the GPU for plotting.

The report now substitutes the requested update count into its lead/slider
instead of hardcoding 100; all probability-size cues and fixed bold readable
atom labels remain. Static Ruff/format, AST/JSON and shell syntax checks passed;
source/submission/outcome/download identities follow when actual.

### 500-update source 01 actually submitted

GPU job **29735858 submitted**, gpua10, one exclusive GPU, four slots on one
host, 8GB per slot/32GB total, one-hour bound. Unique source
`/work3/s234818/mosaic/releases/20261016-carbon-chain-500-01/source`;
run `/work3/s234818/mosaic/runs/20261016-carbon-chain-500-01`.
Local release `results/releases/carbon-chain-500-20261016-01/` contains
195 files, archive 38,816,221 bytes; archive SHA-256
`2e27fa7a8a9f10fb2b7806272b5c07d96d3972a693fcbfec5ea8dac3e890c7d6`;
manifest SHA-256
`c64a10d59b8ff5181a498534816e3c45f236b96098f2f6a733b245a50a4087b3`.
Git reference `8b08daf28cb7e0731e686b476e6f1a9db5a10cd9`; dirty bytes are
captured. Uploaded archive SHA and launcher syntax passed, then source was
made read-only. Exact submission is ignored
`results/hpc/20261016-carbon-chain-500/submit-01.sh`. The launcher binds the
completed 100-update input summary SHA explicitly. No dependency/setup/model
download or package update is introduced. Submitted is not passed; 134 unit
checks, both 500-update native trajectories and final identity/plot/download
outcomes remain pending.

### Original 500-update units passed; user adds a 50% start

Job **29735858 RUN**: **134 selected unit checks passed**, three deselected,
in **61.87 seconds**. Native execution is in progress. Static manifest byte
comparison confirms **zero changed `src/` files** versus the completed
100-update source. This is not a completed 500-update result.

The user subsequently requested an additional **50% uniform mixture**. Keep
the running two-case job/source untouched; prepare a separate one-case
500-update allocation, serially after the existing job. It binds the same
passed 100-update context summary and uses explicit longer-budget and
mixing-list replay policies. Only the step count and relaxation strengths may
differ from that reference; model/context/fixture, precision, RNG, protein,
pocket, objective, eta and thresholds remain matched. New guards retain the
explicit opt-in and reject precision/config drift. **141 unit checks expected**
in this source, including per-study report input/identity mismatch checks;
not yet run/passed. The native `src/` model/gradient code remains unchanged.

The report reads the two saved studies directly and retains independent native
summary/config/job receipts. Combining requires shared non-mixture config,
hard graph, model/context digests and numerical profile, passed studies and
distinct mixtures. It does not duplicate raw trajectories or assign a native
hash to a derived combined configuration. Rendering uses teal/orange/purple
for 15%/30%/50%. The three-case storage estimate fits inspected headroom, and
each allocation still requires the conservative 8 GiB/10000-file preflight.
Original job outcomes, additional source/submission/outcomes and final report
remain separate ledger evidence.

### Additional 50% case actually submitted

Job **29735875 submitted**, gpua10, one exclusive GPU, four slots on one
host, 8GB per slot/32GB total, one-hour bound, serial dependency
`ended(29735858)`. This permits the independently authorized 50% attempt after
the original job ends, while preserving its own readiness/numerical gates;
it does not approve a failed result. It reuses passed 100-update job 29735673's
authenticated context, not any incomplete 500-update output.

Immutable source `/work3/s234818/mosaic/releases/20261016-carbon-chain-500-02/source`;
unique run `/work3/s234818/mosaic/runs/20261016-carbon-chain-500-mix50-01`.
Local release `results/releases/carbon-chain-500-20261016-02/` contains
198 files, archive 38,819,796 bytes, archive SHA-256
`71e88aa2f72d432ebe63f2923cba0a34dfcb67cbf902134269649c0e1d4693e7`;
manifest SHA-256
`df33c30a8ab6c5f86db522bc9903bfda7bd0618c7e5ec78d4cb03ca459a68c99`.
Uploaded archive SHA and both launcher syntax checks passed, then source was
made read-only. Exact operator script is ignored
`results/hpc/20261016-carbon-chain-500/submit-mix50-01.sh`.
Ruff/format and static AST/JSON/shell checks passed before sealing; 141 unit
checks and the 50% native outcome remain pending. Original job 29735858 has
started accepting updates after its 134 unit checks; no source mutation or
cancellation occurred.

Combined CPU report job **29735880 submitted**, hpc, one slot/4GB, 15-minute
bound, dependency `done(29735858) && done(29735875)`. It uses immutable
source 02 and reads both actual native run directories directly. Unique run:
`/work3/s234818/mosaic/runs/20261016-carbon-chain-500-report-01`.
Exact operator script:
`results/hpc/20261016-carbon-chain-500/submit-report-queued-01.sh`.
The report's own passed-study/shared-input guards remain required in addition
to scheduler dependencies. No report outcome is claimed at submission.
Root byte-manifest comparison confirms zero changed `src/` files between
source 01/02. Local destination has about 255 GB free; this is filesystem
capacity inspection only, not numerical work.

### Two 500-update trajectories passed; extra case stopped at quota guard

Job **29735858 DONE**, both cases passed 500 accepted updates with 501 saved
states each. Standalone primal loss: mixing 0.15,
`1.8446224927902222` → `0.5082295536994934`; mixing 0.30,
`1.6099762916564941` → `0.553653359413147`. Initial scalar values can differ
by a float32 rounding unit from the prior allocation; they are retained as
measured rather than overwritten with earlier values. Config SHA-256
`133c9916981f541fe668c06ac6bb4329a72f611997d1488ad6db20a99c9e3426`;
fixture SHA remains
`655e2199389201d81d9ea5ac2c4ebc7e38da5b8abea126c7862bd49b0735788f`.
Model digest before = after
`8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2`;
context before = after
`737edd04299752288d68c58655bd9f12a51ae313becaea9bd3a8f45d6010025f`,
matching the authenticated 100-update reference.

50% attempt **29735875 EXIT** at quota preflight, status
`tests_not_run_insufficient_space`; **no unit/native tests ran in this attempt**.
Its available bytes were `8557722337.279999`, just below the unchanged 8 GiB
requirement (8589934592 bytes), with file headroom sufficient. Preserve its run
and quota receipt. Dependent report **29735880** was explicitly cancelled via
`bkill` while pending; actual tool receipt says `is being terminated`. It did
not render a result. These are storage/operational outcomes, not failed native
gradient checks or scientific model instability.

Pivotal bounded storage recovery: verify both local archive hashes and the two
matching remote transfer archives, confirm unpacked source manifests exist,
then remove **only** this campaign's redundant transfer tarballs:
`releases/20261016-carbon-chain-500-01/source.tar.gz` and
`releases/20261016-carbon-chain-500-02/source.tar.gz` (under the Mosaic HPC root).
The exact bytes remain locally under `results/releases/`, and read-only HPC
source trees/manifests/all numerical results are preserved. No prior runs,
weights, caches or another agent's files were removed. Operator script:
`results/hpc/20261016-carbon-chain-500/recover-upload-space-01.sh`.
Post-cleanup quota was **91.71/100 GiB**, 263306/20000000 files, exceeding
the same 8 GiB gate. The protocol/implementation plan records this specific
exception to the initially planned no-cleanup approach; no threshold was loosened.

Retry **29735926 submitted** with the same source-02 manifest, configuration,
model/context binding and numerical/space gates; fresh run
`/work3/s234818/mosaic/runs/20261016-carbon-chain-500-mix50-02`.
Same gpua10 four-slot/one-GPU/32GB/one-hour allocation with already-satisfied
`ended(29735858)` dependency. Exact script:
`results/hpc/20261016-carbon-chain-500/submit-mix50-02.sh`.
This is a new attempt after storage recovery, not approval of the stopped one.
141 unit checks, 50% native outcome and final combined report remain pending.

Replacement comparison job **29735927 submitted**, same one-slot/4GB/15-minute
hpc bounds, new run
`/work3/s234818/mosaic/runs/20261016-carbon-chain-500-report-02`, dependency
`done(29735858) && done(29735926)`. It preserves the cancelled report attempt.
Exact script: `results/hpc/20261016-carbon-chain-500/submit-report-queued-02.sh`.
Actual retry 29735926 passed the **unchanged 8 GiB quota gate**, then
**141 unit checks passed**, three deselected, in **62.57 seconds**. Native
50% execution is in progress; no 500-update 50% result is claimed yet.

### Additional 50% trajectory completed; comparison rendering started

Job **29735926 DONE**: its 141 selected units passed, then all initial native
derivative/null/parity gates and **500 accepted updates** passed at mixing 0.50.
Standalone primal loss `1.5916939973831177` → `0.6682145595550537`;
final AD-evaluated loss `0.668214917`. Config SHA-256
`3980cbec9a573f25cd93511a9a4730ba9d641c4efc5024c7459dca995ad0dd35`.
Model and context digests before/after equal the two completed 500-update cases
and the authenticated 100-update reference (`8d6926d5...b0ab2` and
`737edd04...0025f`, full values above). All three starts completed **1500
accepted updates total**, with 501 saved states per start. Report **29735927
RUN**; no completed comparison analysis/readout statistics claimed yet.

Allocated evidence-packaging job **29735961 submitted**, hpc, one slot/2GB,
15-minute bound, dependency `done(29735927)`. Unique helper
`/work3/s234818/mosaic/runs/20261016-carbon-chain-500-evidence-01`.
It verifies source 02, requires 6.5 GiB free before constructing the roughly
2 GB evidence archive (preserving the 4 GiB reserve), and uses stdlib-only
file hashing/packaging on allocated CPU. No numerical model/analysis is rerun.
Exact submission: `results/hpc/20261016-carbon-chain-500/submit-evidence-01.sh`.
Helper script SHA-256
`af5db930ea68402ec4030f1d0a30d939248694e5db05b00e20590322784f694d`;
collector SHA-256
`6e80d143cb6a5b9a41c7685d24638224bd807f65490a34079d3589af261e3d05`.
Uploaded helper hashes and syntax passed, then both helper files were made
read-only. The bundle will retain the declined preflight, intentional report
cancellation, successful retry and all native snapshots/probes/candidates.

### Three-case comparison and complete evidence download finished

Comparison **29735927 DONE**, allocated CPU node `n-62-31-22`,
2026-10-16 11:29:54–11:33:11 Europe/Paris, 131.3 CPU seconds. Both studies
passed the common-input guards before rendering. Native summary SHA-256:
15%/30% `d232b7f54a725c12e0683d2fac0da7b9f31d20bb1309ffca1159c803142bb89f`;
50% `466eebe4b77e6031d8972ddf77331ea18167cda1677ae62656ffc91860c10de6`.
Input studies/configurations remain separately identified; the combined
display configuration is not a native run configuration. No model execution
was repeated for analysis. GPU job 29735858 ran 11:08:23–11:20:01 and retry
29735926 ran 11:21:47–11:29:51, both on `n-62-18-5`; final scheduler receipts
are retained, including the cancelled 29735880's actual `EXIT` status.

HPC-computed detached raw readouts and change statistics:

| Uniform mixture | Initial → final standalone primal loss | Hard-graph change updates | Sanitized saved readouts | Final fragments | Final canonical SMILES |
| --- | --- | --- | --- | --- | --- |
| 15% | 1.8446224927902222 → 0.5082295536994934 | 264, 370, 428 | 501/501 | 3 | `C.CCCCCCC.CS` |
| 30% | 1.6099762916564941 → 0.553653359413147 | 179, 465 | 501/501 | 2 | `CC.CCCCCCCS` |
| 50% | 1.5916939973831177 → 0.6682145595550537 | 199 | 501/501 | 1 | `CCCCCCCSCC` |

All three final hard graphs change slot 3 from C to S. The first two remove
two and one chain bonds respectively; the 50% endpoint removes none. Final
atom mean total variation from initialization is 0.1727315933 / 0.2246217281 /
0.2212680131, and mean edge-existence L1 change is 0.0744575933 /
0.0918869898 / 0.0884181559 for 15% / 30% / 50%. These values come from the
allocated report's `analysis_summary.json`. Sanitization is not connectivity
or affinity evidence; no repair, fragment selection or chemical constraint
was applied. The complete synthetic peptide is `ACDEFGHIKLMNPQRSTVWY`, with
one-based F5/G6/H7/I8/K9 selected for contact scoring. This is a gradient fixture,
without a named biological target or experimentally established pocket.

The offline report contains all 1503 saved states, probability-sized atom/bond
depictions with fixed bold element/slot labels, 0..500 playback, endpoint/change
plots and three probability heatmaps. Local path:
`results/hpc/20261016-carbon-chain-500/report/report.html`, 34,203,948 bytes,
SHA-256 `2a997a61fc0bb8b60b8932f65fa7544ff56de93b744f2d3144ea965f12bacd13`.
Browser QA verified all three cards, the 50% update-500 endpoint, readable S3,
the third heatmap and absence of JavaScript errors. All ten local report files
match the verified HPC evidence archive; receipt `report-copy-verification.json`.
The earlier 100-update report remains separate. Renderer/depiction/template
source SHA-256, respectively:
`d2e0e2006b01d404862413b0fdf20d7110c74712d221ce2f97458dd65a483b03`,
`f3eba2a4462306552cb15a7456778480a8920944424081a35aa6332aa51a91e1`,
`741572de13f71e4512951fc2134fa1867f57bf87ce79ced0f8e08784880bf771`.

Evidence job **29735961 DONE**, node `n-62-31-23`,
11:33:13–11:35:24 Europe/Paris, 106.4 CPU seconds. Its unchanged 6.5 GiB
packaging guard passed. Remote archive:
`/work3/s234818/mosaic/runs/20261016-gradient-evidence-carbon-chain-500-01.tar.gz`,
**2,020,603,152 bytes**, SHA-256
`dc1e1d948c99d487f5c06897964037cd2130a983f003ad4fc1643d996bad14ba`.
Archive and extracted evidence are downloaded under
`results/hpc/20261016-carbon-chain-500/`. Local stdlib-only command
`py -3.13 results/hpc/20261016-relaxed-gradients/verify_download.py results/hpc/20261016-carbon-chain-500/20261016-gradient-evidence-carbon-chain-500-01.tar.gz dc1e1d948c99d487f5c06897964037cd2130a983f003ad4fc1643d996bad14ba`
exited zero: archive checksum, safe regular-file/directory extraction, and all
**4619 declared file sizes/hashes passed**. The sibling
`20261016-gradient-evidence-carbon-chain-500-01-download.json` records the
result. This was local byte integrity verification, not numerical evaluation.
Separate `evidence-helper/` retains collector scripts, quota preflight, logs,
the packaging job's final scheduler receipt and final quota receipt.

Final measured work3 quota: **94.31/100 GiB**, 266757/20000000 files,
about **5.69 GiB free** after packaging. No further cleanup was performed;
all scientific evidence and both the stopped storage attempt and successful
retry remain retained. The requested three 500-update trajectories, comparison
report and local evidence download are complete. Full Stage 0 A–G acceptance,
both-mode regression/restart and M3 remain pending.

## 2026-10-16: real target/known ligand and 1000 updates requested

User authorized replacing the synthetic fixture with a real protein and known
ligand comparison, increasing the budget to 1000 updates, and pushing all used
scripts. Preserve prior results. Selected T4 lysozyme L99A/benzene, PDB 181L,
with six occupied carbon slots to match the neutral six-heavy-atom reference.
The deposited construct also includes C54T/C97A. See
[the protocol](known_ligand_1000.md) for target/representation boundaries and
the compact serialization decision. Original Stage 1 BRD4/JQ1 gates remain
pending; CCD JQ1's +1 charge is unsupported here, not silently neutralized.

Downloaded raw PDB (transfer only, no local preprocessing), SHA-256
`77018feaaa65bb22dea47c784e8c059b0ccc09cd6dc7442b79cce83f3170985f`,
from `https://files.rcsb.org/download/181L.pdb` into
`tests/data/t4_lysozyme_181l/181L.pdb`. Read-only DTU metadata showed
94.18/100 GiB used, 266757/20000000 files; no Mosaic job remained running.
Other agents' queued jobs were left untouched. The full-artifact 1000-update
study would not fit; a separately tested compact serialization policy keeps
raw graph gradients/states/scalars/RNG and derivative/end-point distograms,
omitting repeated interior distograms and all persisted trunk tensors. It
changes no model/objective/finite checks or numerical acceptance tolerance.
All numerical unit/preparation/model/report work remains allocated DTU HPC.
No 1000-update run is claimed complete or submitted at this point.

### Immutable readiness release, unit/preparation pass and pilot submission

Local source-01 archive was packaged only, never uploaded/submitted:
38,863,147 bytes, archive SHA-256
`af60f97bb8fda427e058c34b653bdcc36f146b96486f95631479d096d0614dad`,
manifest `e0182e04414f571929f8b922c8883a9a93d3d4b0e463d71888327a8d4cbdcc1a`.
Source 02 added the reusable CPU readiness launcher; 207 files,
38,863,153 archive bytes, SHA-256
`ff8dd32ace4d5b75b283dc58a65f9749b5e1b541388eb91b215a5fae463a8d35`,
manifest `4060ecaae546339cab71991ba6f42813c996451540eacdc9218b36e071c79cc1`.
Git reference is `8b08daf28cb7e0731e686b476e6f1a9db5a10cd9`; dirty bytes
are bound by the manifest. Local release directories are
`results/releases/known-ligand-1000-20261016-01/` and `...-02/`.
Remote read-only source:
`/work3/s234818/mosaic/releases/20261016-known-ligand-1000-02/source`.
Archive checksum and launcher syntax passed before submission.

Readiness **29736014 submitted, then DONE**, hpc, two slots on one host,
4GB per slot/8GB total, 15-minute bound; node `n-62-30-25`.
Run `/work3/s234818/mosaic/runs/20261016-known-ligand-readiness-01`.
Its 5 GiB preflight passed, then **146 units passed**, three model items
deselected, two existing dtype warnings, **53.16 seconds**. Tests establish
identical full/compact state histories and stored raw gradients, continued
finiteness checks for omitted tensors, deposited construct identity and
rejection of missing ligand/mismatched residue mapping. These are allocated
CPU engineering checks, not a native GPU result.
Allocated preparation authenticated 181L and generated fixture SHA-256
`6dae87554347d1b14764eeb6a3d10ff3488a145c6db142534a4a3c54a4564d8b`:
164 residues, contact author numbers 78/84/87/88/91/99/102/103/111/118/121/153.
The exact generated JSON was copied to `tests/data/t4_lysozyme_181l/fixture.json`;
local hash matched. Readiness files are downloaded under
`results/hpc/20261016-known-ligand-1000/readiness-01/`.

Native pilot **29736017 submitted**, gpua10, one exclusive A10 GPU, four slots
on one host, 8GB per slot/32GB total, one-hour bound, dependency
`done(29736014)`. Run
`/work3/s234818/mosaic/runs/20261016-known-ligand-pilot-01`.
Uses source 02, `examples/configs/known_ligand_pilot.json`, and the authenticated
allocated fixture above. All three mixtures request three accepted updates;
the known hard benzene reference is evaluated in the same context before them.
Full native derivative/null/parity/identity and per-step AD/primal gates remain
required. No pilot or 1000-update outcome is claimed at submission.
Exact operator files are retained under
`results/hpc/20261016-known-ligand-1000/submit-readiness-01.sh` and
`submit-pilot-01.sh`. Local checks were Ruff/format, source packaging, shell
syntax and byte hashes only; no local numerical tests were run.

The reusable experiment/HPC/render/config/test/package/download-integrity tools
were committed and **pushed to origin/main as `c45b9ba27b997559c85a60697d097add22db2656`**.
Added scoped Git attributes preserve authenticated PDB bytes (including fixed
columns/padding) and LF execution/config bytes on Windows and Linux. The
unrelated local PDF relocation and OneNote export remain outside this commit.

Source 03 adds pilot storage/runtime assessment and known-reference report
presentation, plus the exact allocated fixture. All `src/` files and the native
experiment driver match unit-tested source 02 byte-for-byte. Local release:
`results/releases/known-ligand-1000-20261016-03/`; 213 files, 38,868,328 bytes,
archive SHA-256
`6979ab616a55432888bc67237c490f475a245e7a3bc5c4fd3fb31110cf310b68`,
manifest `83f62cdb6d0265b09b033663f9b30712b8d319c6410a27dae7fe003256ab9507`.
Git reference `c45b9ba27b997559c85a60697d097add22db2656`; manifests include
the new helper/presentation bytes. Remote read-only source:
`/work3/s234818/mosaic/releases/20261016-known-ligand-1000-03/source`.
Upload checksum and launcher syntax passed.

Assessment/report **29736081 submitted**, hpc, one slot/4GB, 15-minute bound,
dependency `done(29736017)`; run
`/work3/s234818/mosaic/runs/20261016-known-ligand-assessment-01`.
It requires the complete native pilot to pass, checks frozen identities and
strict longer-budget configuration parity, then estimates the worst declared
ten-trial output cost plus report/evidence archive and the 4 GiB reserve.
Only a passed quota assessment can lead to the longer GPU submission. It also
renders the pilot's actual saved states with the hard benzene loss/reference
drawing and detached Morgan radius-2/2048-bit similarity; these are engineering
graph comparisons, not binding or pose measurements. Outcome remains pending.
Exact operator: `results/hpc/20261016-known-ligand-1000/submit-assessment-01.sh`.

### Native pilot passed; first long-run resource assessment declined

Pilot **29736017 DONE** passed three updates for all three starts, every
initial atom/bond finite-difference gate, order-null checks, AD/primal scalar
parity, every measured decline, and frozen model/context checks. Standalone
primal losses: 15% `1.9935636520385742` → `1.9927682876586914`;
30% `1.942352294921875` → `1.940629005432129`;
50% `1.9883627891540527` → `1.9863002300262451`.
Hard benzene's same-context primal loss is `2.0979607105255127`, graph SHA
`2538e357b1ea386ff2e1cd7fb898fd867c239101e4b4c8eeda02df90f3e9fa64`.
The chain starts already score lower than the known ligand: this illustrates
why this contact proxy is not an affinity ranking or a recovery guarantee.
Pilot summary downloaded to `results/hpc/20261016-known-ligand-1000/pilot-summary.json`.

Assessment **29736081 EXIT** on node `n-62-31-24`, before rendering/submitting
the long run, because storage was **5.00 GiB free** versus the predeclared
worst-case projection **5.6760657941922545 GiB required**. Quota receipt status
`tests_not_run_insufficient_space`, available bytes 5368709120, used files
271106/20000000. This is a resource stop, not a failed native gate; preserve
its logs/quota receipt and scheduler outcome.

Bounded recovery selected for the new campaign: rehash the verified local
2,020,603,152-byte 500-update evidence archive (matched
`dc1e1d948c99d487f5c06897964037cd2130a983f003ad4fc1643d996bad14ba`)
and confirm its retained 4619-file verification receipt. A separately sealed
allocated CPU helper will verify the identical remote archive and retained raw
500-study summary identities before removing only
`/work3/s234818/mosaic/runs/20261016-gradient-evidence-carbon-chain-500-01.tar.gz`.
All raw HPC runs/source and the exact verified local archive/extracted evidence
remain preserved. No other cleanup or reduced quota/numerical threshold is
authorized by this choice. Recovery and replacement assessment outcomes remain
pending, and no 1000-update run has been submitted yet.

Recovery/assessment **29736114 submitted**, hpc, one slot/4GB, 15-minute bound,
dependency `done(29736017)`, fresh run
`/work3/s234818/mosaic/runs/20261016-known-ligand-assessment-02`.
Uses separately sealed source 04, 214 files, 38,872,951 archive bytes;
archive SHA-256 `9b7aaf4253f9380b694eae353e6ba7fd91aaccf26aea68245a11ecdf96c7c04a`,
manifest `8df25aaa80a129dd56ef9fdbf8e12a2f7c4f37bd8668d72f172e544d205fb990`,
Git reference `b18fd876a593d865266a6d5254ffa96f71b08ab4`.
Local `results/releases/known-ligand-1000-20261016-04/`; remote source
`/work3/s234818/mosaic/releases/20261016-known-ligand-1000-04/source`.
It adds only the narrow archive recovery and persistence of declined resource
projection receipts; numerical model/gradient sources are unchanged. Upload
checksum and launcher syntax passed, source is read-only. Exact operator:
`results/hpc/20261016-known-ligand-1000/submit-assessment-02.sh`.
The earlier failed assessment remains preserved. Recovery and the unchanged
resource threshold must pass; no long-run submission is claimed yet.

### Resource recovery passed; 1000-update GPU run actually started

Recovery/assessment **29736114 DONE**, node `n-62-31-24`. Its receipt confirms
that only the redundant 500-update transfer archive was removed after local
and remote SHA and preserved raw-summary checks; 2,020,603,152 bytes recovered.
Exact local receipt `results/hpc/20261016-known-ligand-1000/transfer_recovery.json`.
The unchanged **5.6760657941922545 GiB** assessment then passed. Allocated
metadata projection: pilot 347274779 bytes; worst-case new run 582461779;
report reserve 134217728; archive bound 1082982436. Mean observed step seconds
by case 4/3/3, projecting 10000 seconds for 3000 updates, excluding new-process
compilation. Assessment/quota receipts downloaded to `assessment.json` and
`assessment-quota.json` in the local campaign directory.
Pilot summary SHA `45fa41a40b6bf2e3a548b819b29aa1ad4d0b4b8e61067f3aed877a989d0cbdfa`;
frozen model digest `8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2`,
new real-target context digest
`04a4c84ab0933b2cd7af81c379127762535322c64f037c55b5b2720fc6192e39`.
The synthetic context is not reused or mislabeled as the real target.

The allocated pilot report completed and was copied locally to
`results/hpc/20261016-known-ligand-1000/pilot-report/`. Browser QA showed all
three 3-update case cards, six occupied slots, the benzene drawing/loss,
per-case detached similarity, exact real-construct/contact annotations and no
JavaScript errors. No local scientific plotting/evaluation was performed.

Long run **29736126 submitted, then RUN**, gpua10, one exclusive A10 GPU,
four slots on one host, 8GB per slot/32GB total, six-hour bound, dependency
`done(29736114)`; actual node `n-62-18-5`.
Run `/work3/s234818/mosaic/runs/20261016-known-ligand-1000-01`.
Source 04/manifest/config `examples/configs/known_ligand_1000.json` and sealed
fixture above; strict longer-budget replay binds the passed pilot summary SHA,
every non-step configuration field and the frozen model/context identities.
All three start at update zero and request **1000 total** updates each.
Exact operator `results/hpc/20261016-known-ligand-1000/submit-1000-01.sh`.
This is an actual submission/start, not a completed 1000-update result.
Presentation/evidence/download completion will have separate receipts.

### 1000 updates per mixture passed; final reporting/evidence queued

Long job **29736126 DONE**: all three starts passed **1000 accepted updates**,
**3003 saved states** total, initial derivative/null/parity gates, per-update
AD and independent primal decreases, and unchanged frozen model/context.
Summary SHA-256 `2a159de33e26f050a8dcb4e8bee8d0059af2a141aa1f3da787cc4219e3ce2fa1`;
config SHA-256 `b0f05ce6735a094e080734dfae8a4bfec08be6886686f6c9899f87723923c3d0`;
fixture SHA-256 `6dae87554347d1b14764eeb6a3d10ff3488a145c6db142534a4a3c54a4564d8b`.
Model/context before/after match the passed pilot values above. Standalone
primal loss: 15% `1.9935636520385742` → `1.8578355312347412`;
30% `1.9423514604568481` → `1.8529831171035767`;
50% `1.9883629083633423` → `1.8166388273239136`.
The same-context hard benzene reference measured `2.0979597568511963` in this
allocation (the pilot's tiny float32 execution difference is retained).
Final 15%/30% hard graphs sanitize as the unchanged `CCCCCC` chain. The 50%
endpoint is **invalid**: fluorine at zero-based atom 4 has explicit valence 2.
Its raw graph and conversion error are retained; no chemistry repair was used.
These trajectories demonstrate contact-gradient optimization, not known-ligand
recovery, binding or chemical feasibility.

Postprocessing source 05: 216 files, 38,880,357 archive bytes, SHA-256
`eae3eb9109ba89e42bde1c262340abe536495f987f156bc73dd34150aa4fdecb`,
manifest `544be74174b9568bb712e866483f4cd054400d80fab3ca880561267d3b23373b`,
Git reference `0373c0234bfd85c03f5467b6d1e209411938c9dc`.
Local `results/releases/known-ligand-1000-20261016-05/`; read-only remote
`/work3/s234818/mosaic/releases/20261016-known-ligand-1000-05/source`.
Upload SHA/launcher syntax passed. This adds reusable evidence packaging,
quota/scheduler receipts and one-shot byte-only completion download; the
native model/gradient driver remains unchanged.

Evidence collector smoke **29736170 DONE**, hpc one slot/2GB, ten-minute
bound, node `n-62-31-4`. It successfully packaged the retained stopped
assessment plus its own execution receipts on allocated CPU. Archive
`/work3/s234818/mosaic/runs/20261016-known-ligand-collection-smoke-01.tar.gz`,
1,059,474 bytes, SHA-256
`fccd1b9ab882cfa6bd5d789893b35a902cbfd74e114b1a5d37bfa6cc48220cd5`.
Downloaded under the local campaign directory; the versioned
`scripts/verify_hpc_evidence.py` exited zero with **113 files verified**.
This local check was stdlib file integrity, not numerical evaluation.

Final report **29736412 submitted and RUN**, hpc one slot/4GB, 15-minute
bound, dependency `ended(29736126)`; run
`/work3/s234818/mosaic/runs/20261016-known-ligand-1000-report-01`.
It uses source 05 and validates the saved study before rendering; an ended
failed model job would be rejected, not treated as successful.
Evidence **29736472 submitted**, hpc one slot/2GB, 15-minute bound, dependency
`ended(29736412)`; helper run
`/work3/s234818/mosaic/runs/20261016-known-ligand-1000-evidence-01`.
Archive target `/work3/s234818/mosaic/runs/20261016-known-ligand-1000-evidence-01.tar.gz`.
It retains readiness, pilot, both resource assessments, long native study,
final report, collector smoke and helper/scheduler/quota receipts, including
failure evidence. Packaging has its own measured-size-plus-4-GiB quota guard.
Exact operators: local `submit-collection-smoke-01.sh`, `submit-report-01.sh`
and `submit-evidence-01.sh` under the campaign directory. Final report,
packaging and complete evidence download remain pending at submission.


### Known-ligand completion download stopped

Evidence job 29736472; CalledProcessError: Command '['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'dtu-hpc', 'bjobs -noheader -o stat 29736472']' returned non-zero exit status 127.. See `C:\Users\nikol\OneDrive\Dokumenter\GitHub\mosaic\results\hpc\20261016-known-ligand-1000\download-01\completion.json`. No successful download or native outcome inferred.

The first local file-transfer helper failed before downloading: its non-login
SSH shell did not initialize LSF's `bjobs` PATH (exit 127). Corrected both
scheduler queries to use `bash -l -c`; the read-only query then returned the
expected unique scheduler state. New output `download-02/` preserves the stopped
`download-01/` receipt. No native job was rerun or numerical setting changed.
Exact submitted operator scripts are additionally preserved in versioned
`scripts/hpc/receipts/20261016-known-ligand-1000/`; the README distinguishes
historical receipts from reusable launchers.

### Matched hard-graph comparison and packaging follow-up

Pivotal comparison decision: a relaxed trajectory score and a hard benzene
score are different input regimes. Add a separate allocated hard-control pass
for hard benzene, the initial hard chain and every raw argmax endpoint, with
the same authenticated full-study context, model, key and objective. This
does not change/repeat the 3000 optimized updates. Labels come directly from
the authenticated native summary; hard graph construction/readout is detached
from optimization. Identical graph-array digests reuse their exact score.
Invalid endpoints remain labelled invalid; a model score does not sanitize them.
The existing `2e-5` primal parity tolerance applies to hard-reference replay,
and frozen model/context digests must match before/after.

Hard controls **29736511 submitted and RUN**, gpua10, one exclusive A10 GPU,
four slots/32GB total host request, 15-minute bound, dependency `done(29736126)`.
Run `/work3/s234818/mosaic/runs/20261016-known-ligand-hard-controls-01`.
Input summary SHA `2a159de33e26f050a8dcb4e8bee8d0059af2a141aa1f3da787cc4219e3ce2fa1`.
Source 06: 229 files, archive 38,888,106 bytes, SHA-256
`5937111ebd0f63cd3035e916237103af00b21941bed67c599b37cfa543edcc38`,
manifest `1a9ae1fd0f4437571b2d2ac4fbb37fef0fce424330267f4a1fd6a24621269ef0`,
Git reference `b3f57ccb413bf03853482762202e2156ba078815`.
Local release `results/releases/known-ligand-1000-20261016-06/`; read-only remote
source `/work3/s234818/mosaic/releases/20261016-known-ligand-1000-06/source`.
Upload hashes and launcher syntax passed. Exact operator
`results/hpc/20261016-known-ligand-1000/submit-hard-controls-01.sh`.
No control result is claimed at submission.

Base trajectory report **29736412 DONE**: all 3003 states rendered, ten output
files including offline HTML, native-derived plots and comparison analysis.
Full evidence **29736472 EXIT** before archive construction because readiness
pytest temporary directories contain `*current` symlink aliases. Preserve the
failed packaging run/logs. The corrected packager records only bounded pytest
current aliases in the manifest, never follows or extracts them, retains their
real directories/files, and still rejects any other symlink. This is an I/O
fix; numerical work and the completed report are unchanged. Local downloader
`download-02/` correctly stopped on evidence `EXIT`, without claiming success.

Native job final scheduler receipt was saved locally as `native-job-final.txt`:
started 11:57:27 and completed 12:42:46 Europe/Paris, 3325 CPU seconds. Some
older jobs had already aged out of `bjobs`; `bhist` is permission-restricted on
this host and was not bypassed. Their saved LSF completion logs, prior state
receipts and quota/native artifacts remain the outcome evidence; no successful
historical scheduler query is claimed for unavailable entries.

Hard controls **29736511 DONE**. All five declared controls were covered by
three distinct hard-graph evaluations; both 15%/30% endpoints have exactly the
hard-chain array digest and reuse its score. Same-context hard losses:
benzene **2.097959280014038**, initial chain and 15%/30% endpoints
**2.01330828666687**, invalid 50% endpoint **1.8704317808151245**.
Replayed benzene passed existing primal parity; model/context digests matched
the main study before/after. The invalid fluorine graph remained invalid and
was not repaired. This matched hard comparison itself ranks hexane and the
invalid graph below benzene; the contact objective is not an affinity ranking
or chemistry constraint. Control summary SHA-256
`7dc78247be4be56442ea4569c944d942dab1a8e31daf3b02e7cb36bb085335b9`.

Source 07 preserves the numerical controls and adds only safe inventory of
bounded pytest current-directory aliases. 229 files, archive 38,889,473 bytes,
SHA-256 `c179e4839fc80e258b18551a746188ec0f961fdae47b18e4362019b994b5e0c6`,
manifest `05cd383db826477b59389c38d99870ae55f3a6acd46667bc8c0ee7b9301050c0`,
Git reference `b3f57ccb413bf03853482762202e2156ba078815`.
Local `results/releases/known-ligand-1000-20261016-07/`; read-only remote
`/work3/s234818/mosaic/releases/20261016-known-ligand-1000-07/source`.
Upload SHA and launcher syntax passed. The model/gradient experiment was not
changed, and no failed scientific gate was waived.

Report augmentation **29736521 submitted**, hpc one slot/2GB, ten-minute bound,
dependency `done(29736511) && done(29736412)`; run
`/work3/s234818/mosaic/runs/20261016-known-ligand-controls-report-01`.
It authenticates the control summary SHA and matching model/context/native
summary identities, copies the completed report and adds the hard-score table
without recomputing trajectory plots/states. New evidence **29736523 submitted**,
hpc one slot/2GB, 15-minute bound, dependency `ended(29736521)`; helper
`/work3/s234818/mosaic/runs/20261016-known-ligand-1000-evidence-02`, archive
`/work3/s234818/mosaic/runs/20261016-known-ligand-1000-evidence-02.tar.gz`.
It also retains the failed first collector, separate hard controls and enriched
report. The source-07 packager never follows or extracts symlinks; it records
pytest current aliases while preserving every real file, and rejects other
links. Measured archive-size quota plus the 4 GiB reserve remains required.
Exact local operators `submit-controls-report-01.sh` and `submit-evidence-02.sh`.
Completed augmentation, retry packaging and verified download remain pending.


### Known-ligand completion download stopped

Evidence job 29736472; RuntimeError: Evidence job exited; inspect HPC logs, no success claimed. See `C:\Users\nikol\OneDrive\Dokumenter\GitHub\mosaic\results\hpc\20261016-known-ligand-1000\download-02\completion.json`. No successful download or native outcome inferred.


### Automatic completion receipt for known-ligand job 29736523

Recorded 2026-10-16T11:16:06.500352+00:00. Evidence job **29736523 DONE**; native study status **passed**, model job **29736126**. Archive SHA-256 `cfe36ec2176432a26b9923cda0526db1cf5a3c657219599e419633995574b96f`, 795141544 bytes; 15775 declared files verified locally using stdlib byte checks only. Local completion receipt `C:\Users\nikol\OneDrive\Dokumenter\GitHub\mosaic\results\hpc\20261016-known-ligand-1000\download-03\completion.json`. Offline report: `C:\Users\nikol\OneDrive\Dokumenter\GitHub\mosaic\results\hpc\20261016-known-ligand-1000\download-03\report\report.html`. No local model/chemistry evaluation was performed; failed native outcomes are retained and do not establish a 1000-update success.

### Final known-ligand report and raw-output delivery

Report augmentation **29736521 DONE** and evidence retry **29736523 DONE**
were confirmed by the saved scheduler receipt
`results/hpc/20261016-known-ligand-1000/final-postprocessing-jobs.txt`.
The archive includes 15,775 regular files, every real pytest temporary file,
and an inventory of 26 omitted duplicate pytest current-directory aliases;
no symlink was followed or extracted. The measured packaging gate required
4.948134486 GiB free, including the unchanged 4 GiB reserve, and passed.
The final captured quota was **94.65/100.00 GiB**, 283,729/20,000,000 files,
about **5.35 GiB free** (`post-package-quota.txt`).

The actual local transfer/integrity command was:

```powershell
py -3.13 scripts/download_hpc_when_complete.py --job 29736523 --archive /work3/s234818/mosaic/runs/20261016-known-ligand-1000-evidence-02.tar.gz --output results/hpc/20261016-known-ligand-1000/download-03 --study-run 20261016-known-ligand-1000-01 --report-run 20261016-known-ligand-controls-report-01 --ledger docs/experiment_ledger.md
```

It exited 0 on Windows/Python 3.13 after the allocated evidence job was DONE.
Only SSH/SCP, stdlib extraction, metadata reads, byte hashes and file copying
ran locally; no model, molecular conversion or numerical evaluation ran.
Archive SHA and all declared file sizes/hashes passed. The copied enriched
HTML is 38,875,386 bytes, SHA-256
`2603881fb382b9a17f85b848d47525d6828321680937a12d28fefa5b7bee78d5`.
The native summary reports **passed**, with 1000 accepted updates and 1001
saved states per mixture. Failed storage/download/packaging attempts above
remain failures; the separate completed native run establishes the result.

Final browser QA at `http://127.0.0.1:8768/report/report.html` confirmed all
three 1000-update cases, real target and reference, and hard-control scores.
Selecting the 50% case and update 1000 displayed C1–C2–C3–C4–F5–C6,
P(F5)=0.43394 and relaxed primal loss 1.8166388. The raw F5 chain and all five
chain bonds remain visible; fixed readable labels and probability-dependent
sizes render correctly. The browser logged no JavaScript errors. This is
visual/interaction QA of HPC-generated artifacts, not local numerical work.

The user explicitly accepts chemically invalid outputs for this phase and
wants exactly the produced graph plotted. The depiction uses raw argmax
labels/topology plus saved probabilities; sanitization is annotation only.
No state filtering, fragment removal, bond repair or identity substitution
was introduced. All raw arrays, failed gates and source releases remain
preserved. This scoped result does not close the full Stage 0 A–G,
both-mode/restart or M3 acceptance requirements.

## 2026-10-16 — requested fresh BRD4-BD1, 5000 updates, 3x initial rate

The user requested a new run with three times the learning rate and 5000
updates, using BRD4-BD from the project description. Exact protocol and
pivotal decisions are in [brd4_5000.md](brd4_5000.md). Retain six-carbon
starts and 15%/30%/50% mixtures, seed 17; set initial step 0.03 and 5000
total accepted updates per case. Backtracking may reduce the accepted step.
Preserve every derivative/null/parity/descent/frozen check and raw output,
including invalid categorical molecules. This is a fresh start, not a
continuation of T4's 1000-update endpoints.

The read-only project PDF extraction established section 4.1.2: whole
BRD4-BD1 target from 3MXF, JQ1 removed, no selected binding pocket. Use the
already implemented `NativeContactLoss`, negative mean ligand best-target
contact probability, rather than the old pocket/anchor NLL. These are
different objectives and target contexts; their losses are not comparable.
This scoped experiment does not close the pending PAE/canonical 25-slot
initialization/Stage 0–1 acceptance gates. JQ1 is excluded rather than
neutralized or claimed as a supported same-occupancy reference.

Input: authenticated RCSB 3MXF PDB, SHA-256
`534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f`.
Keep all 127 deposited residues, author 42..168, including deposited M43
and its SEQADV discrepancy. Sequence-only model context; no MSA, structural
template, coordinates, JQ1, solvents or ions. Allocated preprocessing alone
generates the fixture. Local work so far is PDF extraction, byte transfer,
static Ruff/format review and immutable packaging; no numerical test ran locally.

Source 01: 240 files, archive 38,947,610 bytes, SHA-256
`b3011e5be4502c383567c1f6253ae3aa774cf05fa841234c09912f428112a14f`,
manifest `0d020949dc2430c294197bae4fe4a65e53dcde7f8acb1adc387057dfcc73bd89`,
Git reference `12472e2677d70c100f0ed3eb2d6f66c20f7f65db` (dirty bytes included).
Local `results/releases/brd4-5000-20261016-01/`; immutable remote
`/work3/s234818/mosaic/releases/20261016-brd4-5000-01/source`.
Archive hash and launcher shell syntax passed before submission.

CPU readiness **29736568 DONE**, hpc two slots/4GB per slot, 15-minute bound,
node n-62-30-25. Run `/work3/s234818/mosaic/runs/20261016-brd4-readiness-01`.
**150 passed, three model items deselected, two existing warnings, 49.68s**.
Includes complete/missing BRD4 mapping, whole-target selection and existing
pocket regression checks, plus prior selected small-molecule unit coverage.
Exact operator `scripts/hpc/receipts/20261016-brd4-5000/submit-readiness-01.sh`.
No GPU pilot or 5000-update success is claimed by these CPU results.

Storage decision: work3 reported 94.79/100.00 GiB used while other agents'
jobs were active. Home reported 4.97/30.00 GiB, with a four-hour quota-report
lag. Place only the new BRD4 small/medium output artifacts under
`/zhome/95/e/206121/mosaic-runs/`; large model caches and immutable sources
stay on work3. A new home gate rounds used GiB up/capacity down and keeps
4 GiB plus the measured conservative run/report/archive bounds. No existing
results or other agents' jobs are moved or deleted. Duplicate Python
bytecode caches are disabled for this campaign. A combined source-02 CPU
check and native pilot remain required before long-run submission.

Allocated source-01 preparation produced fixture SHA-256
`a84e0d6d0a0643ec2a85c0daa7a67cb44b6f3a318ffaf749bb75cd0bcc07dfb1`;
the exact bytes were copied to `tests/data/brd4_3mxf/fixture.json`.
Source 02 adds this fixture, conservative home-quota gating/tests and generic
assessment resource accounting; the graph/model/gradient implementation and
pilot configuration remain identical to source 01. It includes 245 files,
archive 38,951,709 bytes, SHA-256
`cd14cc6377e8422a8abb12b639f8c9a34edf989fe097332e958bf60402b5e2c5`,
manifest `80bb2413904b955cc4ebc25a0c01e1940fd185b3bddfb6cbd3afed76ad75a4f2`.
Local `results/releases/brd4-5000-20261016-02/`, immutable remote
`/work3/s234818/mosaic/releases/20261016-brd4-5000-02/source`;
Git reference remains `12472e2677d70c100f0ed3eb2d6f66c20f7f65db`.
Upload checksum and both launcher syntax checks passed.

Combined readiness **29736575 submitted**, hpc two slots/4GB per slot,
15-minute bound, run `/zhome/95/e/206121/mosaic-runs/20261016-brd4-readiness-02`.
GPU pilot **29736576 submitted**, gpua10 one exclusive A10/four slots/32GB
total host request, one-hour bound, dependency `done(29736575)`; run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-pilot-01`.
Exact operator `scripts/hpc/receipts/20261016-brd4-5000/submit-readiness-and-pilot-02.sh`.
Submission is not a passed pilot or a submitted/completed 5000-update run.

Source-02 readiness **29736575 DONE**, node n-62-30-25:
**154 passed, three model items deselected, two existing warnings, 47.92s**.
This combined selection includes the new home-quota parser rejection and
conservative rounding cases. Local evidence `readiness-02-unit.log` and
`readiness-02-quota.json` under `results/hpc/20261016-brd4-5000/`.
The home preflight passed with 25 GiB conservatively reported available;
capacity/usage rounded to 30/5 GiB. The stated quota lag remains a limitation.
GPU pilot **29736576 RUN** on one A10, n-62-18-5.

Assessment **29736579 submitted**, hpc one slot/2GB, ten-minute bound,
dependency `done(29736576)`; run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-assessment-01`.
It authenticates source-02 bytes and the passed three-case pilot, permits
only steps 3→5000 in the matched configuration, and projects all ten trial
pairs per update, snapshots, JSON, a **512 MiB report reserve**, and a
conservative evidence archive, plus the unchanged 4 GiB reserve on home.
Exact operator `scripts/hpc/receipts/20261016-brd4-5000/submit-assessment-01.sh`.
Actual assessment/pilot outcomes remain pending; no long run is submitted yet.

GPU pilot **29736576 DONE**, 13:26:50–13:32:03 Europe/Paris, n-62-18-5,
932 CPU seconds, maximum host memory 4 GiB. All three cases passed native
derivative/null/parity/descent and frozen-identity gates, with four saved
states each. Raw primal initial→final losses:
15% -0.3350009322166443→-0.336289644241333;
30% -0.3623838424682617→-0.3641449809074402;
50% -0.2778552174568176→-0.2807735800743103.
Pilot summary SHA-256
`7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4`.
Frozen model digest before=after
`8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2`;
context digest before=after
`6747ba18833013da02dbf407d9faefb1adbc8314353b4882a34f99557b96d4f6`.

Assessment **29736579 DONE**, node n-62-31-4. It passed the unchanged
4 GiB reserve plus measured artifact bounds: **7.324157069437206 GiB**
required, 25 GiB conservatively available. Pilot 209,961,849 bytes; projected
new run 1,129,216,849; report reserve 536,870,912; archive bound 1,903,198,714.
Mean pilot timing intervals 3.21897/2.21838/2.25206 seconds per step project
38,447.02 seconds across 15,000 updates. These intervals include probe work
between snapshot zero and step three; this is a conservative budget, not an
ETA. Assessment SHA `cfc44f1e6619a0d0e97bdb761c006be76b4bbad0547dd3cd74656a504ac93cbd`;
bound final configuration SHA
`e893ad0416894409cfef7a3fb6ae61f17b53ae141858c3f519f7c5954f542e09`.
Local `pilot-summary.json`/`assessment.json` under the campaign directory.

LSF attached an external `GPU 1: Detected Power error` message to the passed
pilot on n-62-18-5; the initial JAX hwloc CPU-affinity warning is also retained.
This is not treated as a failed numerical gate, nor is the hardware message
discarded. First long-run submission using quoted `hname` exclusion was
**rejected before creating a job** (`Error in select section ... missing value`);
preserve operator `submit-5000-01.sh` and empty run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-5000-01`.
Queue/group inspection established gpua10→ra10→ra10e1→n-62-18-5, so a fresh
node cannot be requested within that queue. Do not retry on the warning-bearing
node or change other users' allocations. Use gpua100 for a new Ampere GPU;
its declared maximum is 72h, above this bounded request. Configuration,
model and prepared context remain unchanged, and numerical gates rerun on
the actual allocation rather than assuming cross-GPU parity.

Long run **29736600 submitted**, gpua100, one exclusive GPU, four slots with
8GB per slot (32GB total host-memory request), **16-hour bound**, dependency
`done(29736579)`. Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-5000-02`.
Exact operator `scripts/hpc/receipts/20261016-brd4-5000/submit-5000-02.sh`.
Both pilot summary and assessment hashes were verified before submission;
the launcher rechecks immutable source, final config/pilot/assessment bindings
and actual home quota at start. Source 02 is unchanged. This requests fresh
5000-update trajectories, not a restart from pilot endpoints. Submitted is
not running or completed; final numerical results remain pending.

Report source 03: 249 files, archive 38,961,260 bytes, SHA-256
`13bf30466bc786338c88df4b615f0acf44ebb7a897b8dcece84a610b452c9f6d`,
manifest `d6d4bbc17545df536e8dcea70f50d863f966b67ec25fb0a3b3f04f86739a9f9a`,
Git reference `4008e3c52673c1c9787c794cef2c72fe82130273` plus dirty bytes.
Local `results/releases/brd4-5000-20261016-03/`; immutable remote
`/work3/s234818/mosaic/releases/20261016-brd4-5000-03/source`.
Local manifest byte comparison confirmed all `src/`, the native experiment
driver and both BRD4 configuration files match tested source 02 exactly.
Only presentation/receipts/docs were added or changed. Template method text
now takes step size/trial count from the saved config and states the actual
whole-target objective; the old hardcoded 0.01 caption is removed. Bundled
Node `--check results/hpc/20261016-brd4-5000/template-script-check.js` passed
as a local static syntax check only. No local model/rendering/chemistry test.
Uploaded archive SHA passed before use.

Report **29736612 submitted**, hpc one slot/4GB, 90-minute bound, dependency
`ended(29736600)`; run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-5000-report-01`.
Exact operator `scripts/hpc/receipts/20261016-brd4-5000/submit-report-01.sh`.
It rechecks immutable source and at least 5 GiB home headroom, then invokes
`render_protenix_trajectory.py` without rerunning the model. The renderer
requires the passed complete study and preserves raw argmax outputs. This
submission is not a finished report or a verified download.

Final submission snapshot: **29736600 PEND**, waiting for slots/resources;
**29736612 PEND**, waiting for its model-job dependency. Saved local receipt
`results/hpc/20261016-brd4-5000/submission-state-and-quota.txt`.
Work3 now reports **90.25/100.00 GiB**, 286,480/20,000,000 files, reflecting
external storage changes; home still reports 4.97/30.00 GiB with its stated
lag. This campaign deleted/moved no existing data and changed no other
agent's job. New native completion, report rendering, evidence transfer and
final browser QA remain pending.

### 2026-10-16 — user-requested lower-queue GPU review and A10 replacement

The user asked whether A100 was necessary and whether a GPU with less queue
could be used. The workload does not require A100: source-02 BRD4 pilot
29736576 already passed on A10. Current eligible general queues included
A10/A40/L40S/A100/V100. A10 reported no pending work; queue counts alone
do not establish start-time ordering or an ETA. Restricted H100/Biosustain
queues were inspected, not submitted to, and no new access was requested.

Reading the full pilot notice using `bread -i 0 -w 29736576` resolved the
truncated earlier message: a clocks event due to power violation, with the
explicit statement **"This GPU can still perform workload."** The earlier
A100-only allocation choice was too cautious and is superseded. The full
notice remains at the cancelled original run's `pilot-power-notice-full.txt`.
No separate hardware-health test is claimed. The passed native A10 pilot
and unchanged per-run numerical checks support ordinary A10 use.

`bswitch gpua10 29736600` was rejected by DTU's `eswitch` policy; job 29736600
remained PEND in gpua100. Local receipt `a10-queue-switch.txt` under
`results/hpc/20261016-brd4-5000/`. No override or repeated switching was tried.
Normal owner-authorized cancellation/resubmission was then used for the
already eligible A10 queue. Only this campaign's old pending jobs were touched:
report **29736612** first, then native **29736600**, each checked PEND before
cancellation. Both were subsequently confirmed **EXIT**, with no native
allocation started. Preserve their paths, before/after scheduler receipts
and `cancellation.txt`. These intentional cancellations are not scientific
failures or evidence of numerical instability.

The first operator `resubmit-a10-01.sh` requested both cancellations but
stopped at an immediate EXIT assertion while LSF still acknowledged
termination asynchronously. It created no replacement jobs or new run path.
Its nonzero outcome is retained in local `resubmit-a10-01.txt`. The second
operator `resubmit-a10-02.sh` required both actual EXIT states and preserved
final cancellation receipts before submitting replacements; it exited 0.
No duplicate experiment ran concurrently.

Replacement native **29736629 submitted**, gpua10, one exclusive A10, four
slots/8GB per slot (32GB total), 16-hour bound, dependency `done(29736579)`.
New run `/zhome/95/e/206121/mosaic-runs/20261016-brd4-5000-03`.
Source 02/manifest `80bb2413904b955cc4ebc25a0c01e1940fd185b3bddfb6cbd3afed76ad75a4f2`,
final config `e893ad0416894409cfef7a3fb6ae61f17b53ae141858c3f519f7c5954f542e09`,
pilot summary `7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4`,
assessment `cfc44f1e6619a0d0e97bdb761c006be76b4bbad0547dd3cd74656a504ac93cbd`
are unchanged; pilot/assessment SHA checks passed again before submission.
All three 5000-update requests, initial step 0.03, backtracking, raw-graph,
frozen identity and numerical gates remain unchanged. No local numerical
test or new pilot was needed for this return to the already tested A10.

Replacement report **29736630 submitted**, hpc one slot/4GB, 90-minute bound,
dependency `ended(29736629)`; source 03 and the same complete/passed-study
and home-quota gates. Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-5000-report-02`.
Local receipt `resubmit-a10-02.txt`; actual scheduler states are captured in
`a10-replacement-state.txt`. Submitted does not establish completed updates,
a rendered report or a downloaded result.

## 2026-10-16 — requested exact-uniform BRD4 initialization control

The user requested a separate identical 0.03/5000-update study, changing
initialization to equal element probabilities and pair existence about
`3/n_atoms`. See [the exact-uniform protocol](brd4_uniform_5000.md).
Keep the original native/report jobs **29736629/29736630** unchanged.
Use one six-slot control with C/N/O/S/F/Cl each 1/6, every unordered pair
P(exists)=3/6=0.5, NO_BOND=0.5 and the four present-bond categories each
0.125; no self-bonds, atom/bond noise or hidden jitter. Expected degree is
2.5. Repeating this same graph/context/key three times would be redundant.

Reuse Mosaic's `initialize_graph` with equal element weights, the resolved
expected degree and both noise amplitudes zero. Explicit new replay policy
permits initialization replacement and the already requested longer update
budget, while rejecting target/fixture/objective/precision/seed/rate changes.
The original context/model/key is authenticated from passed BRD4 pilot
summary `7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4`.
The hard chain remains a native preparation scaffold only; all optimized
X/E entries are replaced. Native fixed slot/name metadata is unchanged,
so fully uniform probabilities do not assert full model permutation symmetry.

Initial raw argmax selects carbon on equal-element ties and NO_BOND because
it has the largest categorical probability. Positive bond-order states are
tied and native Protenix sees existence only. The report must show the
uniform probabilities, disconnected raw start and subsequent outputs without
repair/filtering, and must not label this run a 100% hard-chain mixture.
There is no affinity, chemistry or additional learned loss.

The graph/model `src/` implementation is unchanged from passed source 02.
CPU readiness must test exact equality/no hidden seed jitter, old-mixture
regression and explicit replay permissions. Resource assessment authenticates
the old pilot/assessment/config and unchanged native-library byte hashes,
then reserves the previous full THREE-case run/report/archive bound for
this ONE control, plus the same bound again for the outstanding baseline.
Required free home space is about **10.6483 GiB**, including the same 4 GiB
reserve; an early CPU gate requests 11 GiB. Actual uniform native gradient,
null, parity and descent checks run at GPU entry before accepting updates.
No failed gate is waived or favorable alternative start substituted.

Local work so far: source/docs editing, static Ruff review and byte-only
packaging. All numerical tests remain allocated DTU work. No new numerical
result or successful submission is claimed by this preparation entry.

Uniform source 01: 257 files, archive 38,973,485 bytes, SHA-256
`058c9dd851e33657897ed3f8c442e94cde6f3c4697c38518a98eab4810dd098c`,
manifest `cefff19ac9b68d4b546427565b1036fc625c0cde42f22a5157116c930d8252fd`,
Git reference `90f2fa0dd9af7e99afa982657428c93f2ac262ff` with dirty bytes.
Local `results/releases/brd4-uniform-5000-20261016-01/`; immutable remote
`/work3/s234818/mosaic/releases/20261016-brd4-uniform-5000-01/source`.
Uploaded archive SHA and both new/reused launcher shell syntax checks passed.
Local static Ruff checks passed; bundled Node `--check` on the extracted
template script passed. These are source/syntax checks, not numerical work.

CPU readiness **29736713 submitted**, hpc two slots/4GB per slot, 15-minute
bound; run `/zhome/95/e/206121/mosaic-runs/20261016-brd4-uniform-readiness-01`.
Exact operator `scripts/hpc/receipts/20261016-brd4-uniform-5000/submit-readiness-01.sh`;
local receipt `results/hpc/20261016-brd4-uniform-5000/submit-readiness-01.txt`.
It runs the selected small-molecule unit suite and the allocated uniform
resource/source assessment. Submission alone establishes no passed test,
native uniform gradient or completed optimization.

Uniform CPU readiness **29736713 DONE**, node n-62-31-16:
**158 passed, three native model items deselected, two existing warnings,
48.91s**. Exact float32 equal atom rows and bond rows, identical graphs at
different preparation seeds with zero noise, old mixture regression, and
explicit initialization-only/longer-budget restrictions passed. Resource
assessment also passed, authenticated old pilot/assessment/config and exact
unchanged `src/` byte map. Required aggregate headroom:
**10.648314138874412 GiB**; home gate conservatively reported 25 GiB.
Final uniform config SHA
`898d4bda519ebe53ce40881a5bf0ff66c53f81788ab1679111f990a201161836`;
new assessment SHA
`58aa2f68a88a13fa5d21d6d504808850a51e864651f4bff2e82cea793fca2fa1`.
Local `readiness-unit.log` and `assessment.json` under
`results/hpc/20261016-brd4-uniform-5000/`. These are CPU/metadata results;
native uniform derivative and trajectory outcomes remain pending.

Uniform native **29736735 submitted**, gpua10 one exclusive A10, four slots
with 8GB per slot (32GB total host request), 16-hour bound, dependency
`done(29736713)`. Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-uniform-5000-01`.
Source is the exact tested uniform source 01. Both old pilot and new readiness
assessment SHA checks passed before submission; the launcher checks all
identities and aggregate quota again at startup. The explicit
`--allow-initialization-change` and `--allow-update-count-change` policies
authorize only the requested initialization replacement and steps 3→5000,
with every other frozen target/model/objective/seed/precision/rate setting
unchanged. Native initial checks run before accepting optimization updates.

Uniform report **29736736 submitted**, hpc one slot/4GB, 90-minute bound,
dependency `ended(29736735)`; run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-uniform-5000-report-01`.
The same immutable release supplies report labels for the single uniform
control and distinguishes the context scaffold from the true initial graph.
It retains fixed readable element labels, probability-dependent sizes and
every raw invalid/disconnected readout. The renderer requires a complete
passed study and performs no model forward. Full trajectory, report,
download and final browser QA remain pending.

Exact operator `scripts/hpc/receipts/20261016-brd4-uniform-5000/submit-native-and-report-01.sh`;
local submission receipt `submit-native-and-report-01.txt` and scheduler
snapshot `submission-state.txt` under the uniform campaign directory.
The original native/report **29736629/29736630** were inspected only and
remain unchanged. No old job was cancelled/resubmitted or source modified
on HPC for this control. Submission is not a completed result or evidence
of hard-molecule recovery.

## 2026-10-16 — requested noisy-uniform BRD4 control (preparation)

User requested a third identical 0.03/5000-update study with a half-uniform,
half-standard-Gaussian start. See [the protocol](brd4_noisy_uniform_5000.md).
Use `0.5 U + 0.5 simplex(N(0,1))` for X and unordered E, seed 17,
retaining fixed occupancy/charge and absent self-bonds. Project the random
component first: static review found that projecting the combined proposal
can introduce zeros incompatible with unchanged central derivative/null
probes. No numerical experiment or failed native job used that draft formula.
Projected distributions need not retain Gaussian moments or baseline degree.
This probability-space interpretation is a recorded pivotal decision.
The shared `src/` library, native target/context/objective/rate/precision and
optimization gates are unchanged. Keep original jobs 29736629/29736630 and
exact-uniform jobs 29736735/29736736 untouched; all four were PEND in the
read-only snapshot at the start of this request.

Reuse the authenticated prior passed BRD4 native pilot for frozen shape/model
context. Require new allocated CPU tests and initialization-only replay/source
gates, reserving the previous full three-case artifact budget for each of
three outstanding/new studies (about 13.9725 GiB including 4 GiB reserve).
Native checks on the noisy graph remain required at GPU entry before updates.
Local source edits/static review/byte packaging do not establish numerical
validation. No new submitted or completed experiment is claimed here.

Noisy-uniform source 01: 261 files, archive 38,980,076 bytes, SHA
`4d36c3d8d0bb945553eca3e9adf5c53fce754bdf832c896c09866030e51cde21`,
manifest `ea4ceab78cb7605fdd715cb79bebacefc03ec8c8a5947834fa58540d5028c02e`.
Git reference 43fdcb17e06127ef6b5eec2a4c378d36aeec739a plus manifested dirty
bytes. Local `results/releases/brd4-noisy-uniform-5000-20261016-01/`;
remote `/work3/s234818/mosaic/releases/20261016-brd4-noisy-uniform-5000-01/source`.
Archive upload SHA and launcher Bash syntax checks passed. Static Ruff checks
passed; extracted report JavaScript passed bundled Node `--check`. A first
local static-extraction command used Windows' default text encoding and failed
before extraction; explicit UTF-8 fixed it. No numerical test was run locally.

CPU readiness **29736756 submitted**, hpc two slots/4GB per slot (8GB total),
15-minute bound. Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-noisy-uniform-readiness-01`.
Exact operator
`scripts/hpc/receipts/20261016-brd4-noisy-uniform-5000/submit-readiness-01.sh`;
local receipt under `results/hpc/20261016-brd4-noisy-uniform-5000/`.
The selected CPU suite includes exact-uniform/old-mixture regressions and
new noisy-formula, probability-floor, reproducibility and policy checks.
New aggregate resource assessment reserves each of three study budgets.
Latest read-only storage snapshot: home 4.97/30 GiB (four-hour-lag report),
work3 pool 6 89.58/100 GiB, 287351/20000000 files; no cleanup or alteration
of earlier jobs/files. Numerical and resource-gate outcomes remain pending.

CPU readiness **29736756 DONE**, node n-62-30-25: **167 passed, three native
model items deselected, two existing dtype warnings, 50.02s**. All 13 uniform
and noisy-control tests passed, including the independent formula/floor check.
Resource/source/config/fixture/reference assessment passed; all shared `src/`
bytes match passed source BRD4-02. Required headroom for three study budgets
is **13.972471208311617 GiB**; conservative home availability was 25 GiB.
Config SHA `c74107f4720b148ac3b19968fa32bd5dd4f093824cae5527345ec24c93625c71`;
assessment SHA `4e61063a54e64f671a9c4ce97b009b0fa5f22e659401235c3d4eb4d14a441185`.
Downloaded CPU metadata/log evidence: `unit.log`, `assessment.json`,
`quota_preflight.json`, `job-final.txt` in the local campaign directory.
This establishes CPU engineering/resource readiness, not native noisy-model
derivative acceptance or successful optimization.

Noisy native **29736761 submitted**, gpua10 one exclusive A10, four slots
with 8GB per slot (32GB total host memory), 16-hour bound, dependency
`done(29736756)`. Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-noisy-uniform-5000-01`.
Exact tested source 01/manifest above; native launcher authenticates the
pilot, assessment, config and aggregate quota again. Explicit policies allow
only initialization replacement and the already requested 3→5000 update
budget; context/model/seed/rate/loss/precision/check tolerances remain fixed.
The noise is projected before the 50/50 mix, keeping initial interior floors
for the unchanged central probes. The random categorical bond component
changes realized bond density; it is not a density-matched symmetry ablation.

Noisy report **29736762 submitted**, hpc one slot/4GB, 90-minute bound,
dependency `ended(29736761)`; run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-noisy-uniform-5000-report-01`.
It labels the exact recipe and retains every raw graph, readable element
label and probability-dependent size. It requires the completed passed study
and performs no native model rerun. Exact operator
`scripts/hpc/receipts/20261016-brd4-noisy-uniform-5000/submit-native-and-report-01.sh`.
Local receipt `submit-native-and-report-01.txt`, final scheduler snapshot
`submission-state.txt`. Both new jobs and the original/exact-uniform pairs
were PEND. All four earlier jobs remain unchanged. Native checks, accepted
updates, final report/download and browser verification remain pending.


## 2026-10-16 — combined report for all three BRD4 studies (preparation)

The user requested one report after all three studies complete. See
[the combined-report protocol](brd4_combined_report.md). Compare five saved
trajectories: three chain-mixture cases from native 29736629, exact uniform
29736735 and noisy uniform 29736761. All earlier native and individual-report
jobs remain unchanged. No model rerun, repaired chemistry or omitted failed
study is authorized by this report request.

Renderer changes allow explicit initialization-only comparisons and retain
per-input summary/config hashes, job IDs and five case colors. Target,
fixture/scaffold, frozen identities, numerical profile, seed, rate, objective
and update budget must match; each complete study must have passed. Synthetic
engineering render fixtures do not establish a native/model/chemistry result.
Numerical/rendering tests run on allocated CPU. Local static Ruff and Node
syntax/byte checks and immutable packaging are permitted. Readiness requires
17 GiB free home space to include the existing three-study budgets plus new
combined report/archive headroom. New report launch and archive packaging
recheck actual quota; no cleanup or existing-run alteration is performed.

Queue a separate allocated CPU report/evidence job after all three native
jobs end and new readiness passes. Completion follow-up should stay quiet
while pending, download and verify the finished evidence archive, inspect the
self-contained report and deliver it locally, then stop. Any failed native
or report job remains failed evidence and must be surfaced, not repaired or
silently substituted. Preparation alone claims no new job or completed report.


Combined-report source 01: 266 files, archive 38,981,862 bytes, SHA
`e5b047d437da7d1bfb9c23d55d5b20be6c12f147289188132aca62daa9b6b45c`,
manifest `40f9864f21f9c03c811492a15e5a7e2b3dee9e2f6b597ec3909f343dbeaf2075`.
Git reference fe66025fc6a0a598a2b90688cfd9add372763e71 plus manifested dirty
bytes. Remote immutable source
`/work3/s234818/mosaic/releases/20261016-brd4-combined-report-01/source`;
local `results/releases/brd4-combined-report-20261016-01/`.
Uploaded archive SHA and both Bash launcher syntax checks passed. Local
static Ruff and bundled Node `--check` passed; no local numerical work.

CPU report readiness **29736903 submitted**, hpc two CPU slots/4GB per slot,
15-minute bound, run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-comparison-readiness-01`.
Exact operator `scripts/hpc/receipts/20261016-brd4-combined-report/submit-readiness-01.sh`;
local receipt `results/hpc/20261016-brd4-combined-report/submit-readiness-01.txt`.
It checks the sealed source, at least 17 GiB free home space and all 13 report
loader/guard/synthetic-render tests. Submitted does not establish a passed
readiness check or actual combined BRD4 report.


CPU comparison readiness **29736903 DONE**, worker n-62-30-25: **13 passed
in 38.87s**. This includes a full five-case synthetic HTML/plot render;
its fabricated losses/graphs are engineering fixtures, not native outcomes.
The 17 GiB home gate passed with 25 GiB conservatively available.
Downloaded logs/quota/final scheduler receipt under
`results/hpc/20261016-brd4-combined-report/`.

Combined report/evidence **29736911 submitted**, hpc two CPU slots/4GB per
slot (8GB total), 150-minute bound. Dependency:
`done(29736903) && ended(29736629) && ended(29736735) && ended(29736761)`.
Run `/zhome/95/e/206121/mosaic-runs/20261016-brd4-combined-report-01`.
Uses exactly tested immutable combined-report source 01 and the manifest
above. Requires all three complete passed studies; overlays five cases and
retains every saved state without model reruns. Rechecks free home space,
then archives all three native run directories and the combined report payload
with per-file hashes and scheduler receipts. Archive:
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-combined-report-01.tar.gz`;
receipt has suffix `.tar.receipt.json`. Local exact operator
`scripts/hpc/receipts/20261016-brd4-combined-report/submit-combined-report-01.sh`.
Submission receipt and final compact state snapshot are in the local campaign
directory. All three native jobs and combined report were PEND; earlier
individual report jobs were not modified. Actual BRD4 comparison rendering,
archive, download verification and browser QA remain pending.

Created active hourly in-chat completion heartbeat
`finish-combined-brd4-trajectory-report`, named "Finish combined BRD4 trajectory report".
It makes a compact scheduler check, stays quiet on unchanged pending/running
state, downloads and verifies the completed archive locally, checks all three
source jobs/five 5001-state cases and actual report presentation, records and
pushes evidence and delivers the report. No GPU experiment changes or local
numerical work are permitted. Failed native/identity gates remain blockers.
The follow-up pauses after delivery or an actionable blocker. This scheduling
receipt is not completed report/download evidence.


## 2026-10-16 — section 4.2 canonical diffuse start, six slots (preparation)

Read complete PDF section 4.2 across pages 3-4, including visual inspection.
The user explicitly corrected the initial proposed N=25 scope: keep N=6 and
change initialization only. No 25-slot fixture/model preparation, numerical
run or job was created. See [the sealed protocol](brd4_canonical_diffuse_5000.md).
PDF SHA c9785f45dcd34fbc3feaebcc0bc57905dc606387f76308bb6364eadb40cbeda0.

Pivotal resolved values: atom prior C=.75, others .05; expected degree 2.25;
pair existence 2.25/(6-1)=.45, NO_BOND=.55, four present states .1125 each;
independent Gaussian sigmaX=sigmaE=.005 added BEFORE simplex projection.
Use one case/seed17 and the existing shared initializer. All upper-pair
variables are initialized independently of hard scaffold topology; fixed
native slot metadata remains. Prior mass and realized projected mass are
distinct. The 25-slot .08-.10 range is not copied to N=6. No hard repair,
chemistry constraint, learned scorer or affinity head is introduced.

Retain exact six-slot fixture, full BRD4 target, frozen weights/context, seed,
precision, native contact objective, .03 rate, 5000 updates and all checks.
The shared src/ library is unchanged; explicit config/example/report wiring
and tests are new. Local work is edits/static review, PDF reading and immutable
byte packaging only; all numerical tests remain allocated DTU work. New source,
CPU and aggregate quota gates precede submission, with actual native gates at
GPU entry. Reserve four full prior three-case artifact budgets plus 3 GiB for
comparison output (about 20.2966 GiB with 4 GiB reserve); early CPU gate 21 GiB.
A pending report-only replacement will add this fourth study/sixth trajectory
to the final comparison. Preserve every existing GPU and individual-report
job, and record requested/actual report cancellation separately. Preparation
alone establishes no new submission, native gradient or completed trajectory.


Canonical diffuse source 01: 271 files, archive 38,989,767 bytes, SHA
`f6efb3c2c8889e6f792482cd1d666c38631afe9b2bf78bcda5ba942e81f5e4c9`,
manifest `421213580f84a038cde5e47bb4048a659bfa4daee46645deccbb2e7568d7ead4`.
Git reference 48a287cb2743879fa00bff32ab5d339730b46b08 plus manifested dirty
bytes; remote immutable
`/work3/s234818/mosaic/releases/20261016-brd4-canonical-diffuse-01/source`.
Local `results/releases/brd4-canonical-diffuse-20261016-01/`.
Uploaded archive SHA and native/readiness/combined-report Bash syntax checks
passed. Local static Ruff and Node `--check` passed. PDF text extraction and
page-image inspection are document reading, not local numerical validation.

CPU readiness **29736979 submitted**, hpc two slots/4GB per slot (8GB total),
15-minute bound, run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-canonical-diffuse-readiness-01`.
Exact operator `scripts/hpc/receipts/20261016-brd4-canonical-diffuse/submit-readiness-01.sh`;
local campaign `results/hpc/20261016-brd4-canonical-diffuse/`.
It requires the selected small-molecule suite, the unchanged native-library
byte map, initialization-only config/fixture/reference gates, four study
budgets and 3 GiB comparison reserve. Early quota asks 21 GiB. Numerical and
native submission outcomes are pending; this is a submission receipt only.


CPU readiness **29736979 DONE**, worker n-62-30-25: **186 passed, three
native-model items deselected, two existing dtype warnings, 76.67s**. This
includes all 10 canonical recipe/guard tests, old initialization regressions,
report rendering and four-study/six-case recipe mapping. Resource/source,
config/fixture/reference gates passed. Aggregate required free home space
**20.296628277748823 GiB**, conservative availability **25 GiB**. Config SHA
`49e01e601f77e21770d19b4a40df365315e4521318245d14ccec6e7cdbe6e359`;
assessment SHA `4ad7e7f04c740815d248ee405f3979f4ba648f2cdaca872d48d290a387c53821`.
Local unit/assessment/quota/final scheduler receipts were downloaded to the
canonical campaign directory. This establishes CPU readiness only; native
canonical gradient and 5000-update acceptance remain pending.

Canonical native **29736988 submitted**, gpua10, one exclusive A10, four
slots/8GB per slot (32GB total host memory), 16-hour bound, dependency
`done(29736979)`. Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-canonical-diffuse-5000-01`.
Exact tested source 01; source/pilot/assessment/config/quota are authenticated
again at launch. Reuse the exact previous six-slot frozen context, replacing
only initialization and the previous three-update pilot budget with 5000.
No 25-slot context or new model/target/loss was introduced. Actual native
FD/null/parity/frozen/descent checks run before accepting updates.

Final combined report/evidence **29736989 submitted**, hpc two slots/4GB per
slot (8GB total), 180-minute bound. Dependencies: CPU readiness done and all
four native jobs ended (29736629, 29736735, 29736761, 29736988). Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-combined-report-02`;
final output `payload/report/`; archive is its sibling
`20261016-brd4-combined-report-02.tar.gz` with `.tar.receipt.json` receipt.
It requires four complete passed studies, renders all six 5001-state cases
with six distinct colors, and archives all four native runs plus the report.
Exact operator
`scripts/hpc/receipts/20261016-brd4-canonical-diffuse/submit-native-and-comparison-01.sh`.

Superseded combined report **29736911 cancellation requested while PEND**
after the four-study replacement was successfully submitted. Normal owner
`bkill` acknowledged termination; polling confirmed actual **EXIT** before
completion of the operator. No allocation had started and no model work was
cancelled. This intentional report replacement is not a numerical failure.
Receipt `superseded-report-29736911.txt` and final compact scheduler snapshot
are in the canonical local campaign directory. Original three native and
individual-report pairs remain unchanged and PEND; new native/report are
also PEND. No accepted canonical updates or final report/download are claimed.

Updated the existing active hourly heartbeat
`finish-combined-brd4-trajectory-report` (same name/schedule, no duplicate).
It now checks current native/report IDs, downloads/verifies report-02,
checks four source studies and six 5001-state six-slot cases, and ignores the
intentional superseded report cancellation. Browser QA and delivery remain
future work after actual completion; failed native gates are not waived.


## 2026-10-16 15:52 Europe/Paris — report follow-up authentication unavailable

Scheduled combined-report heartbeat made one compact SSH scheduler query for
native jobs 29736629/29736735/29736761/29736988 and report 29736989. The server
returned `Permission denied (password)` before a scheduler result. No job
state, accepted update, native failure, report completion or download can be
inferred. No jobs/files were modified on HPC and no retry was made in this
follow-up. Local receipt:
`results/hpc/20261016-brd4-combined-report/heartbeat-20261016T135224Z.json`.
The existing hourly follow-up remains active for a later connection attempt;
repeat alerts for the same unchanged authentication outage are suppressed.
This is an access failure, not a failed scientific experiment. Documentation
push is deferred until subsequent report work; no completed report is claimed.


## User-requested status retry — 2026-10-16T14:15:11.456899+00:00

One SSH attempt reached the DTU banner, then timed out on port 22 before returning any scheduler status. Native jobs 29736629/29736735/29736761/29736988 and combined report 29736989 remain unverified at this check. No HPC job was altered and no report was downloaded or completion inferred. Local receipt: `results/hpc/20261016-brd4-combined-report/connection-20261016T141511Z.json`. Existing hourly completion follow-up remains active; no repeated connection attempts were made in this turn.


## 2026-10-17 00:40 Europe/Paris — SSH restored; native runs still queued

After the user corrected VPN routing, the compact scheduler query succeeded.
All four native jobs 29736629/29736735/29736761/29736988 and combined report
29736989 were PEND. None had an execution host assigned. No model jobs or
report settings were changed and no completed trajectory/report was inferred.
No report was available to download. Local receipt:
`results/hpc/20261016-brd4-combined-report/connection-20261016T224007Z.json`.
The existing hourly completion follow-up remains active.


## 2026-10-17 00:41 Europe/Paris — requested GPU/resource submission audit

Read-only LSF audit confirms all four native jobs request one exclusive-process
GPU in real queue gpua10, four colocated CPU slots, 8GB per slot/32GB total host
memory and a 960-minute walltime. Queue maximum is 1440 minutes. Host group
ra10 -> ra10e1 -> n-62-18-5 exists, with two NVIDIA A10 GPUs, 32 CPU slots,
1007G configured host memory and host status ok. The same A10 supported the
previous passed native pilot. No impossible GPU/model/mode/slot/memory/walltime
request was identified in these submissions.

All four native pending reasons explicitly say **The queue is inactivated by
the administrator**. gpua10 reports **Open:Inact_Adm**, zero running slots;
gpua100/gpul40s/gpua40/gpuv100 and hpc also report Open:Inact. This is an
administrator dispatch pause across the inspected queues, not evidence of an
invalid request or an ordinary resource-contention-only wait. No reopening
ETA was provided by these scheduler queries. Combined report 29736989 is
correctly waiting for its native dependencies. Historical completed readiness
IDs have aged out of current bjobs lookup; their previous DONE/pass receipts
remain preserved and this does not establish a new failed check.

No job settings, queue, dependency or source were modified. Preserve all four
GPU studies and reports and wait for administrator reactivation. Local audit:
`results/hpc/20261016-brd4-combined-report/resource-audit-20261016T224134Z.json`.
This is scheduler/resource metadata only, not numerical model validation or
completed optimization evidence.


## 2026-10-17 12:44 Europe/Paris — BRD4 native outcomes; combined report blocked

The user requested another connection/status/download attempt. SSH succeeded.
The first compact scheduler view was stale around job termination; a detailed
read authenticated canonical job **29736988 DONE**, A10 `n-62-18-5`,
11:01:10–12:42:08, and combined job **29736989 EXIT, code 1**, CPU
`n-62-31-15`, 12:42:09–12:42:14. The earlier three native IDs had aged out of
current `bjobs`; `bhist` execution is denied by the cluster, so their persisted
LSF completion footers and failed native summary/case reports were inspected
instead. Do not infer successful completion from an absent active-job entry.

| Native job / study | Saved accepted endpoint | Saved states | LSF/native outcome |
| --- | ---: | ---: | --- |
| 29736629 / chain mixtures | 4998 of 5000, 15% case | 4999 | EXIT 1 / failed; 30% and 50% cases not reached |
| 29736735 / uniform | 3208 of 5000 | 3209 | EXIT 1 / failed |
| 29736761 / half uniform plus projected noise | 3818 of 5000 | 3819 | EXIT 1 / failed |
| 29736988 / section 4.2 canonical diffuse | 5000 of 5000 | 5001 | DONE / passed |

Every study retains N=6 and its previously sealed initialization/configuration.
The first three native errors are exactly
`AssertionError: No measured loss-decreasing projected step`. Both autodiff
and standalone primal decrease must exceed 1e-7 within ten halving trials.
Failure of that requirement is retained as a failed/partial outcome, not
convergence, permission to relax thresholds or a full-budget certificate.
No final frozen-model/context digest check was completed for those failed
studies. Their two unreached chain cases must remain visible in any future
partial-outcome presentation. Raw invalid graphs remain saved; invalid
chemistry is not the reported exception and was not filtered or repaired.

Canonical saved autodiff loss is `-0.15956248342990875` initially and
`-0.5622491240501404` at update 5000. Its initial/final model digest matches
`8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2`;
initial/final context matches
`6747ba18833013da02dbf407d9faefb1adbc8314353b4882a34f99557b96d4f6`.
All four initial context/model identities match the preserved pilot. Native
summaries retain fixture, model assets, immutable per-file source maps and
config hashes. Execution/config SHA-256 pairs are:

| Native job | Execution content SHA-256 | Configuration SHA-256 |
| --- | --- | --- |
| 29736629 | `562a83c38c4de54e953262ccf7f0edea6fa313faf72cc4ae3ac63d0ef203589d` | `e893ad0416894409cfef7a3fb6ae61f17b53ae141858c3f519f7c5954f542e09` |
| 29736735 | `969cc5f20c1f74156f3f793c2529942397adeea92082aeb56fdff592d4658a1e` | `898d4bda519ebe53ce40881a5bf0ff66c53f81788ab1679111f990a201161836` |
| 29736761 | `65652b971351d5f75ec3349b8a98513de2a7b25dde514f612bdafa655c60409d` | `c74107f4720b148ac3b19968fa32bd5dd4f093824cae5527345ec24c93625c71` |
| 29736988 | `60ef9132b73499d3736b49166d4148f078c7020eb762051ac378c12f9a2bfec1` | `49e01e601f77e21770d19b4a40df365315e4521318245d14ccec6e7cdbe6e359` |

Combined report 29736989 authenticated its 271-file immutable source and
passed its 8 GiB home-space check (22 GiB conservatively available), then
correctly stopped at `ValueError: Render completed authenticated trajectories
only`. The failed native inputs prevent a complete comparison. Neither
`20261016-brd4-combined-report-02.tar.gz` nor its receipt exists. This is not
a report-only engineering failure; no rendering retry, native rerun, threshold
change, study omission or replacement artifact was submitted. Existing
individual report jobs and all native artifacts remain unchanged.

Downloaded status evidence:
`results/hpc/20261016-brd4-combined-report/status-20261017T104405Z/`.
Initial transfer: 19 stdout/stderr/native/render-log/summary files,
17,963,200 bytes. `remote-metadata.json` SHA-256:
`ea613d9a41aa405a81ff053baf7111baefaad188ecfda464e5f73cb0df48c4ab`.
`verification.json` records matching file sizes and SHA-256 values over the
authenticated SSH/SFTP transfer. Failed per-case reports are retained alongside
the study summaries: three further files, 33,296,867 bytes, all SHA-256/size
verified in `case-reports-verification.json` against
`case-reports-metadata.json`. Total downloaded evidence is **22 files,
51,260,067 bytes**. Each failed per-case report retains its saved state prefix
and all ten rejected final trials; its failed status is not replaced by the
last accepted endpoint. The original full
snapshot/trial directories remain on HPC. All local inspection/verification
used Python 3.13 standard-library JSON, subprocess, filesystem and hashing
operations; no model, chemistry evaluation, numerical test or rendering ran
locally. No complete report exists to open or browser-verify.

Paused existing heartbeat `finish-combined-brd4-trajectory-report` via the app
automation tool after confirming native failure, preserving its prompt/name/
schedule. The complete-report objective remains blocked by the failed native
studies; pausing is not delivery or successful comparison. Protocol status
documentation was updated without changing scientific code or sealed releases.


## 2026-10-17 — authorized best-primal continuation, first CPU readiness

The user explicitly authorized changing the exhausted-search behavior: after
ten halving trials fail the original dual-loss threshold, take the candidate
with lowest independently evaluated primal loss, including a loss increase.
Resume saved molecules where possible and target 5000 total graph updates per
case. No pretrained model is trained. Canonical 29736988 already reached 5000
and is reused unchanged; chain 30%/50% will start fresh. The exact decisions,
resume tolerances, native evidence inheritance and report policy are recorded
in `docs/brd4_best_primal_continuation.md`. Historical failures are preserved.

Implemented separate update-selection and authenticated-prefix continuation
modules, a continuation runner, CPU plan builder, allocated launchers and report
policy/lineage display. Nonfinite/native-output checks remain mandatory. The
original strict default and historical sealed releases remain unchanged.
Local validation was static Ruff/AST/Node syntax inspection and packaging
only; all numerical tests were submitted to allocated DTU HPC.

Immutable source 01: 283 files, archive 39,017,028 bytes,
SHA-256 `28606ac1ad01bb6332aa05b7ce50100d4e943c5b041b2ef048c2c6c49b5dbf69`,
manifest `c47d520f8cf0f7b41c86ced392baaa0a05236a401eeb334026e740fed389396e`.
Git reference da8aa8c055d3adeb0d8cc70d484a2e9a9707a42c plus manifested current
bytes. Local `results/releases/brd4-best-primal-20261017-01/`; remote
`/work3/s234818/mosaic/releases/20261017-brd4-best-primal-01/source`.
Uploaded archive SHA, 283-file verifier and all three Bash syntax checks passed.
Exact operator:
`scripts/hpc/receipts/20261017-brd4-best-primal/submit-readiness-01.sh`.

CPU readiness **29745455 EXIT 1**, hpc two slots/4GB per slot, 15-minute bound,
worker `n-62-11-50`, run
`/zhome/95/e/206121/mosaic-runs/20261017-brd4-best-primal-readiness-01`.
The 20 GiB early quota gate passed. Selected suite: **195 passed, one failed,
three native items deselected, two existing dtype warnings, 65.42s**.
The sole failure compared a restored JSON prefix (pair lists) to the
pre-serialization Python fixture (pair tuples). Its graph replay and appended
updates executed successfully; the assertion stopped before the uninterrupted
comparison. The corrected test compares the exact persisted JSON metadata;
it retains graph-byte equality and the resumed-versus-uninterrupted endpoint
assertion. No native GPU continuation was submitted from this failed readiness,
and parent-plan generation did not run. Retain this failed source/job evidence.

The next release also copies its complete authenticated continuation plan into
each new study and exposes per-state fallback/loss-increase labels in the
report. These are new engineering bytes requiring a new allocated CPU pass.

Source 02 was sealed locally (284 files; archive SHA
`bd1f23af24031a46981fe877e0817c00807eea404945465ad8cd3207dffc49af`)
and never uploaded/submitted. Static review then added explicit actual runtime
precision/x64/XLA-profile authentication against the parent, so execution uses
new **source 03**: 284 files, 39,018,163-byte archive,
SHA `fedfc43426cc40361fcdd9fa36eca10834f314bb4475341e22cd3dd1083e099e`,
manifest `cb3f850acd5f0810bf6eb318869dc33f5657eb6c22f3f0c37e9c95327c424903`.
Same Git reference da8aa8c plus manifested current bytes. Local
`results/releases/brd4-best-primal-20261017-03/`; remote immutable
`/work3/s234818/mosaic/releases/20261017-brd4-best-primal-03/source`.
Uploaded archive SHA, 284-file verifier and all Bash syntax checks passed.
CPU readiness **29745459 submitted**, hpc two slots/4GB per slot, 15 minutes,
run `/zhome/95/e/206121/mosaic-runs/20261017-brd4-best-primal-readiness-02`.
Exact operator `scripts/hpc/receipts/20261017-brd4-best-primal/submit-readiness-02.sh`.
Its required tests, source/parent/prefix/configuration checks and 20 GiB quota
assessment must actually pass before any new native job is submitted.

CPU **29745459 EXIT 1**, worker `n-62-11-50`: **196 passed, three deselected,
two existing warnings, 65.10s**. The subsequent parent-plan check rejected
`context_files` equality with the pilot. Inspection established that all four
long studies share one stored context container identity, while the pilot's
equivalent archive uses different internal array numbers. Only `base_features`
and `native_features` JSON array references differed: the original encoder
labels arrays in insertion order; decoding sorted JSON and re-saving renumbers
them. The recorded numerical context digest already matched for all studies.
No GPU was submitted from this failed plan gate. Seven local downloaded
readiness files were SHA-256/size verified under
`results/hpc/20261017-brd4-best-primal/readiness-02/`; source/job evidence remains.

Source 04 corrects this identity comparison without accepting changed inputs:
each pilot/parent/canonical container must first match its own pinned file
hashes; decoded JAX tree structure/static metadata, every feature value and
dtype must then match exactly. The same decoded-context check runs at GPU
entry in addition to the original frozen digest/model checks. Two added CPU
regressions cover re-serialization and rejected array/static-metadata changes.
Archive: 286 files, 39,018,334 bytes,
SHA `8377d9dc2f5996c2e81e4ef1fd24519e034905a9a7a299cd5e452e9bc326d8c7`,
manifest `ec278a14e0847939f667dc2a1965f735da627684b3e7f38e0e59e624473e9873`.
Local `results/releases/brd4-best-primal-20261017-04/`; remote
`/work3/s234818/mosaic/releases/20261017-brd4-best-primal-04/source`.
The previous source/plan-gate failure remains preserved rather than overwritten.

Uploaded source 04 SHA and 286-file verifier/Bash syntax checks passed.
CPU readiness **29745466 DONE**, `n-62-11-50`: **198 passed, three native
items deselected, two existing dtype warnings, 65.27s**. Parent/source/decoded
context/strict-prefix gates passed; plan SHA
`4837faf4a35e49927c8fe828b65dfb91ee871da9028ade9db27e6bfb9db1ade1`.
Pins 12,075 case files (12,027 snapshots plus report/RNG/probe evidence).
The 20 GiB aggregate home gate passed with 22 GiB conservatively available.
Each native entry rechecks 8 GiB free space. Downloaded and byte-verified nine
readiness files, including unit XML/log, final scheduler receipt, plan and
quota receipts, under `results/hpc/20261017-brd4-best-primal/readiness-03/`.
Failed readiness-01/-02 artifacts remain separately preserved. No local
numerical validation was run; local post-download checks were stdlib hashes
and saved JSON/scheduler metadata inspection.

After authenticating readiness DONE and the plan hash, submitted at 13:10
Europe/Paris:

| Study | New native job | Run directory under `/zhome/95/e/206121/mosaic-runs/` |
| --- | --- | --- |
| Chain, 15% resume 4998; 30%/50% fresh | **29745472** | `20261017-brd4-chain-best-primal-01` |
| Uniform resume 3208 | **29745473** | `20261017-brd4-uniform-best-primal-01` |
| Noisy uniform resume 3818 | **29745474** | `20261017-brd4-noisy-best-primal-01` |

Each uses immutable source 04 and the pinned plan, dependency
`done(29745466)`, gpua10 one exclusive-process A10, four colocated CPU slots,
8GB per slot/32GB total host RAM and 16-hour walltime. Baseline LR=0.03,
ten halving trials and 5000 TOTAL updates per case. Only the authorized
fallback rule changes; finite/native/entry-replay/frozen identity gates remain.
Every native job was PEND in the authenticated post-submission receipt.
New numerical/native replay and completion outcomes remain pending.

Combined report/evidence **29745475 submitted**, hpc two CPU slots/4GB per
slot (8GB total), three-hour bound. Dependency:
`done(29745466) && ended(29745472) && ended(29745473) && ended(29745474)`.
Run `20261017-brd4-best-primal-report-01`, output `payload/report/`, archive
`/zhome/95/e/206121/mosaic-runs/20261017-brd4-best-primal-report-01.tar.gz`;
receipt `.tar.receipt.json`. Requires the three complete passed exploratory
studies and unchanged canonical 29736988, retaining six 5001-state cases,
explicit mixed policy segments, dotted resume boundaries and raw invalid
graphs. Archives the three failed original parents as well as all new runs
and the canonical study. Does not claim the failed parents completed their
old strict budgets. Report was PEND; no rendered/downloaded report is claimed.

Exact successful operator:
`scripts/hpc/receipts/20261017-brd4-best-primal/submit-native-and-report-02.sh`;
the earlier native operator 01 targeted failed readiness and was never run.
Local detailed scheduler resource/dependency receipt:
`results/hpc/20261017-brd4-best-primal/submitted-jobs.txt`.
The existing hourly heartbeat `finish-combined-brd4-trajectory-report` was
updated/reactivated, preserving its schedule/name/target and using the new
native/report IDs, canonical/plan pins and permissive acceptance semantics.
It remains quiet on unchanged/non-actionable states, verifies actual final
outcomes/artifacts, and pauses after delivery or a current native blocker.
No historical native or individual-report job was modified, cancelled or
replaced. Final report delivery remains outstanding.


## 2026-10-17 — user-directed parallel chain scheduling

The user explicitly requested splitting the sequential chain 15%/30%/50%
allocation into independent GPU jobs. At 13:16:05 Europe/Paris, native
29745472 and report 29745475 were PEND with no execution host, so no new chain
progress existed. Temporarily paused the existing completion monitor while
replacing these IDs. A first stdin-shell cancellation attempt rejected a
CR-terminated job argument; no cancellation was accepted from that attempt.
The versioned LF operator
`scripts/hpc/receipts/20261017-brd4-best-primal/cancel-sequential-01.sh`
rechecked both PEND states, saved their detailed pre-cancellation records and
received `bkill` termination acknowledgements for both jobs. This is an
intentional queued scheduling replacement, not native numerical instability.
Uniform 29745473 and noisy 29745474 remain untouched, as does canonical 29736988.

Implemented `--case-index` in the continuation runner: preserve the pinned full
parent plan, select the original mixture/prefix, emit a singleton configuration
and record its source case index. No numerical library code was changed.
New allocated CPU readiness must pass case-selection tests, the full engineering
suite, exact `src/` byte identity with source 04, unchanged pinned parent files
and the same 20 GiB aggregate quota gate. The replacement report accepts six
native input studies with unchanged initialization/policy comparison guards.
Final EXIT states for the intentionally cancelled 29745472/29745475 were
authenticated before submitting the replacement readiness job. Their saved
pre/post-cancellation scheduler records remain in the original run directories
as `job-before-parallel-split.txt` and `job-cancelled-for-parallel.txt`.

Sealed parallel source 01: 293 files, archive 39,020,819 bytes,
SHA-256 `b0a35eed62c51b53285607d21a0c04c1f541c3b0bd7af888392da43c2ddcc76f`,
manifest `1cf635c222bf750c298c7b4bcac30ec36eeb47e40f74480466d813cd64041504`.
Local `results/releases/brd4-parallel-20261017-01/`; remote
`/work3/s234818/mosaic/releases/20261017-brd4-parallel-01/source`.
The source records commit 2e9d76d plus manifested current working bytes;
the entire numerical `src/` map is identical to passed source 04. Upload
archive/manifest checks, 293-file verification and Bash syntax checks passed.

Allocated CPU readiness **29745498 DONE**, worker `n-62-29-16`, hpc two slots,
4GB per slot/8GB total, 15-minute bound: **205 passed, three native items
deselected, two existing dtype warnings, 78.92s**. This includes seven new
selection cases and the previous 198 checks. Readiness reauthenticated exact
numerical-library identity, the unchanged plan SHA
`4837faf4a35e49927c8fe828b65dfb91ee871da9028ade9db27e6bfb9db1ade1`
and all 12,075 pinned parent case files. Assessment SHA:
`9a38ff0ea26e9ae3e03dc479696db5e05d1dc49c0377b732f983a8b58c7929b8`.
20 GiB aggregate storage gate passed with 22 GiB conservatively available;
home quota can lag four hours, and this does not reserve storage.

Nine readiness files were independently size/SHA-verified after download,
including final scheduler, assessment, unit XML/log and quota records, under
`results/hpc/20261017-brd4-parallel/readiness-01/`. Two administrative
cancellation records were also downloaded by authenticated SFTP. Local
`verification.json` records checked artifacts and submission IDs; detailed
new resource/dependency evidence is `results/hpc/20261017-brd4-parallel/submitted-jobs.txt`.
Local work was editing/static review, immutable packaging and stdlib byte/JSON
checks only; no local numerical tests, model runs or chemistry evaluation.

After authenticating readiness DONE, assessment and plan hashes, submitted:

| Initialization | New native job | Unique run under `/zhome/95/e/206121/mosaic-runs/` |
| --- | --- | --- |
| Chain 15%, resume at 4998 | **29745501** | `20261017-brd4-chain15-best-primal-01` |
| Chain 30%, fresh | **29745502** | `20261017-brd4-chain30-best-primal-01` |
| Chain 50%, fresh | **29745503** | `20261017-brd4-chain50-best-primal-01` |

Each requests gpua10 one exclusive-process A10, four colocated CPU slots,
8GB per slot/32GB total host RAM and dependency `done(29745498)` only.
There are no inter-chain dependencies. The 15% allocation has a one-hour
walltime for replay and two remaining updates; fresh 30%/50% retain 16 hours.
All target 5000 TOTAL updates, LR=0.03, the unchanged best-primal fallback and
mandatory finite/native/replay/frozen identity gates. Source case indices
0/1/2 are preserved. Uniform 29745473 and noisy 29745474 remain on source 04,
their original readiness dependency and their original unique run directories.
All five current native jobs were PEND in the downloaded scheduler snapshot.

Replacement combined report/evidence **29745504 submitted**, hpc two slots,
4GB per slot/8GB total host RAM, three-hour bound. Dependency:
`done(29745498) && ended(29745501) && ended(29745502) && ended(29745503) && ended(29745473) && ended(29745474)`.
Run `20261017-brd4-best-primal-report-02`, output `payload/report/`, archive
`/zhome/95/e/206121/mosaic-runs/20261017-brd4-best-primal-report-02.tar.gz`
and receipt `.tar.receipt.json`. The six singleton inputs include unchanged
passed canonical 29736988. Complete passed native trajectories, all six
5001-state curves, fixed N=6 and frozen model/context identities remain
mandatory. Failed original parents remain archived as evidence. Report was
PEND; submission is not a rendered report or numerical completion.

Exact successful operators:
`scripts/hpc/receipts/20261017-brd4-best-primal/submit-parallel-readiness-01.sh`
and `scripts/hpc/receipts/20261017-brd4-best-primal/submit-parallel-native-01.sh`.
The existing hourly completion heartbeat was updated/reactivated with all
five current native IDs, report 29745504, canonical/plan/source pins and
six-input verification requirements. It remains quiet on unchanged states
and pauses on delivery or an actionable current native blocker. Final native
outcomes, report download/verification and browser QA remain pending.

## 2026-10-17 — first completed independent continuations

User-requested scheduler/progress check, approximately 21:03 Europe/Paris:

| Current native job | Observed outcome / progress |
| --- | --- |
| Chain 15%, 29745501 | LSF DONE; native passed, 5001 states through update 5000 |
| Chain 30%, 29745502 | RUN; approximately 2520/5000 saved updates |
| Chain 50%, 29745503 | PEND; no new worker/output |
| Uniform, 29745473 | Saved LSF successful completion; native passed, 5001 states through update 5000 |
| Noisy uniform, 29745474 | RUN; approximately 4754/5000 saved updates |

Chain 15% ran on `n-62-18-5` from 20:15:50 to 20:21:10. Its independent
primal loss changed from the original -0.3350009620 to -0.4594642818; the new
job restored update 4998 and added the final two updates. Uniform ran on the
same worker from 17:44:12 to 19:13:53, restoring update 3208 and finishing
at 5000; original/final independent-primal losses are -0.1398562789 and
-0.5648135543. Its late log includes expected best-primal fallback steps and
small increases. Raw chemically invalid outputs remain preserved.

Remote persisted summaries for both completed jobs report passed and match
the expected frozen model before/after digest
`8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2`
and context before/after digest
`6747ba18833013da02dbf407d9faefb1adbc8314353b4882a34f99557b96d4f6`.
Chain source index is [0]; both retain plan SHA
`4837faf4a35e49927c8fe828b65dfb91ee871da9028ade9db27e6bfb9db1ade1`.
Inputs/source/run directories are unchanged from the submission entries above.

Uniform had aged out of `bjobs -a`; `bhist` execution was denied by the HPC
installation, so its authenticated saved LSF stdout ("Successfully completed")
and persisted native report establish this observed outcome. An initial
remote metadata helper lacked LSF on PATH; rerunning it through a login shell
resolved that engineering-only inspection issue. Neither issue changed a job.
This check read saved JSON, file names and logs only, with no numerical work
on the login node or locally. Full evidence download and per-file snapshot
verification have not yet been performed for these new completed jobs.

Combined report 29745504 remains PEND; all six trajectories are still required,
including unchanged completed canonical 29736988. No report archive/HTML is
claimed yet. The active monitor and all native jobs remain unchanged.

## 2026-10-17 — ETA check and noisy-uniform completion

At the user-requested ETA check (remote Unix time 1792265880, approximately
21:38 Europe/Paris), noisy uniform **29745474** is LSF DONE with persisted
native status passed, one passed case and 5001 saved states through update
5000. Source/plan/run identities remain as previously recorded. Full archive
integrity verification remains pending final combined report delivery.

Chain 30% **29745502** remains RUN at approximately 4418/5000. The most recent
120 saved updates took about 139 seconds (roughly 1.16 seconds/update), implying
about 11 minutes for the remaining 582 at that same rate. A provisional
10–20 minute estimate allows modest variation; more fallback/backtracking can
extend it. This is a progress estimate, not an execution guarantee.

Chain 50% **29745503** remains PEND with no allocated worker. Queue wait cannot
be inferred from these observations. Rough runtime after start is 1.5–3 hours
based on the current fresh chain run and possible extra backtracking; this is
not a benchmark of the unstarted case. Report **29745504** remains PEND and
cannot finish before this last native input completes and passes. No job,
native policy, source or resource request was changed during this check.

## 2026-10-17 — chain 30% completion authenticated

User-requested check at 21:00 UTC / 23:00 Europe/Paris: chain 30%
**29745502** has completed. It aged out of `bjobs -a`, so its authenticated
saved LSF stdout was inspected: successfully completed on `n-62-18-5`,
20:21:12–21:49:27 Europe/Paris (5295 seconds). Persisted native status is
passed, one passed case, 5001 states, final step 5000, source case indices [1].
Initial/final independent-primal losses are -0.3623839021 / -0.4677774906.
Before/after model and context digests equal the pinned identities above;
plan SHA remains
`4837faf4a35e49927c8fe828b65dfb91ee871da9028ade9db27e6bfb9db1ade1`.
Source remains parallel-01 and run directory remains
`/zhome/95/e/206121/mosaic-runs/20261017-brd4-chain30-best-primal-01`.
Full snapshot/archive byte verification remains pending final report delivery.

Chain 50% **29745503** remains PEND, with explicit scheduler reason
`Job slot limit reached: 1 host`. Its readiness dependency is unchanged and
passed; no allocation or numerical output is claimed. Combined report
**29745504** remains PEND. This is the only unfinished native case among
the six required trajectories. No job or source was changed, no numerical
tests were run, and no combined HTML/archive was downloaded or opened.

## 2026-10-17 — user-authorized five-case interim report

The user explicitly requested presenting the finished jobs now and adding the
50% case later. This authorizes a separately labelled five-of-six interim
comparison; it does not waive completeness or native gates for any included
trajectory. Chain 50% 29745503 was confirmed RUN while preparing the report.
Its GPU job and the all-six report 29745504 remain unchanged.

Added an optional authenticated display-scope manifest, identifying the five
included native IDs 29745501/29745502/29745473/29745474/29736988, planned six
inputs and pending 50% job 29745503. Pins the expected frozen model/context
digests and preserves the original six-case color assignment. The HTML notice
and analysis metadata explicitly disclose the missing case and mixed policy
exposure. Every included trajectory still requires 5001 contiguous saved
states and unchanged native configuration/model/context identities. No missing
probabilities or states are fabricated and raw invalid graphs remain visible.

Presentation-only SVG optimization directly writes the existing raw argmax
labels/topology, fixed zigzag slot positions, atom radius 0.18+0.16P(element),
fixed readable labels and bond width 0.8+3.2P(exists). PNG snapshots keep the
existing plotting renderer. This avoids creating 25,005 plotting figures for
playback; saved probabilities, all native sources and scientific outputs are
unchanged. Added interim-scope rejection and SVG topology/size regressions.
Local Ruff formatting/checks passed; no local numerical tests were run.

Immutable source `20261017-brd4-interim-01`: 300 files, 39,036,540 bytes;
archive SHA `c153c3ecb412443de0f8a7b4f8a5b0b502b2389b3b871119ce9db06a56c21266`,
manifest `d01c386aba5121af454a76868d49b840a99ac1cf3fea66ca9a2687e9699ed490`.
Git reference 5676173 plus manifested working bytes. Entire `src/` map is
identical to passed parallel-01. Local release:
`results/releases/brd4-interim-20261017-01/`; remote source:
`/work3/s234818/mosaic/releases/20261017-brd4-interim-01/source`.
Uploaded SHA, 300-file source verification and LSF Bash syntax passed; source
is read-only. Exact operator:
`scripts/hpc/receipts/20261017-brd4-best-primal/submit-interim-report-01.sh`.

CPU report/evidence **29747700 submitted** at 23:06 Europe/Paris; hpc, two
colocated slots, 4GB per slot/8GB total, three-hour bound, worker `n-62-28-33`.
Unique run `/zhome/95/e/206121/mosaic-runs/20261017-brd4-interim-report-01`;
report `payload/report/`, sibling `.tar.gz` and `.tar.receipt.json` expected.
The job first verifies source/quota, runs all 207 selected engineering tests
and pinned-parent readiness, then renders and packages only saved artifacts.
8 GiB entry quota gate passed with 21 GiB conservatively available (quota can
lag; not a reservation). Packaging will separately check actual byte reserve.
At this entry the CPU suite was still RUN; no pass, generated report or
download is claimed yet. Full six-case report and active monitor are preserved.

Interim source-01 CPU suite completed: **207 passed, three native items
deselected, two existing dtype warnings, 67.93 seconds**. All 12,075 parent
case files and the unchanged numerical library were authenticated. Five-case
rendering completed, including all 25,005 states, plot/heatmap assets and a
208,057,532-byte standalone HTML. Native and rendering checks passed.

Job **29747700 EXIT** at the later packaging step: the collector correctly
rejected a pytest `current` directory alias because the wrapper supplied
`--run <run>/payload`, while its safe pytest boundary expects a run root.
This is an engineering packaging failure, not a native failure; original
report, test logs and failed packaging log remain intact. Corrected the caller
to supply the run root. The collector's symlink rejection policy is unchanged.

Source-02 seals the corrected call, a packaging-only retry and a regression
that records internal pytest aliases without following or archiving them.
303 files, 39,039,160 bytes; archive SHA
`5d3ba07b6f937c3e18ae1a5acdac488a2eae6c62badcfeb5aa5f04587b76f732`,
manifest `7e1bfe4751758f258749fd2a5a3e07b0167ceb544618160f5c6d022fd9d58246`.
Local `results/releases/brd4-interim-20261017-02/`; remote source
`/work3/s234818/mosaic/releases/20261017-brd4-interim-02/source`.
Uploaded SHA/303-file source/Bash checks passed, source is read-only. Renderer,
already-rendered files and all numerical library/native experiments stay
unchanged. No scientific results are re-executed by this retry.

Evidence-only CPU job **29747728 submitted**: hpc two colocated slots, 4GB per
slot/8GB total, one hour. Unique run
`/zhome/95/e/206121/mosaic-runs/20261017-brd4-interim-evidence-02`.
It tests the run-root regression on allocated CPU, checks the existing five
case analysis/identities and seals per-report-file SHA receipts, then packages
the unchanged rendered report and all original/native/failed ancestry evidence.
The expected archive is the sibling `.tar.gz`, receipt `.tar.receipt.json`.
Exact operator:
`scripts/hpc/receipts/20261017-brd4-best-primal/submit-interim-evidence-02.sh`.
Actual retry completion, download and browser verification remain pending.

### Interim delivery and viewer lifecycle — 2026-10-18

Evidence retry **29747728 DONE**, saved LSF successful completion on
`n-62-28-33`, 17 October 23:13:09–23:20:00 Europe/Paris (411 seconds).
The run-root packaging regression passed on allocated CPU (one test, 0.14s).
Archive receipt: 668,542,256 bytes, 146,396 files, SHA-256
`1e9d3e8ae228eb5e468e951c7961075be1d67c284f84adddcac21712147620c9`;
path `/zhome/95/e/206121/mosaic-runs/20261017-brd4-interim-evidence-02.tar.gz`.
The archive remains on HPC; full local extraction/manifest verification is
not claimed. Original render/package failure 29747700 remains preserved.

Downloaded the 12 finished report files and allocated CPU
`rendered_report_receipt.json` by authenticated SCP. All per-file byte counts
and SHA-256 values verified locally using stdlib only. HTML: 208,057,532 bytes,
SHA `5b4503b91672724b790cd3e7f9b734349a294fd7e9ed35688001ba3c4c7a65b6`.
Local directory `results/hpc/20261017-brd4-parallel/interim-report-01/` contains
the receipt, download/schema checks and `report/`. Saved HTML schema retains
five included IDs, 5001 contiguous states per case, X with six occupied slots
and 15 unordered E rows, planned count six and missing job 29745503. This is
file/metadata verification, not new numerical model or chemistry evaluation.

The user reported ERR_CONNECTION_REFUSED after the original local server
ended. Replaced its command-session lifetime with a hidden detached Windows
process using CREATE_BREAKAWAY_FROM_JOB and a separate process group. A fresh
command after the launcher exited confirmed HTTP 200, exact Content-Length
208057532 and a live server process. Added `scripts/serve_protenix_report.py`
to reproduce/reuse this viewer safely on loopback, without startup services.
Report bytes and every HPC experiment remain unchanged.

Automatic browser inspection/reload was denied by the browser URL policy
when binding the error tab; no alternate browser/DOM workaround was attempted.
Visual/playback QA is therefore not certified. The user can reload the existing
page at `http://127.0.0.1:8770/report.html`. The separate all-six report and
completion monitor remain active; the last scheduler check still had 50% RUN.

## 2026-10-18 — final 50% case and six-case report completed

The user requested adding the last case and an updated link. Both jobs have
aged out of `bjobs -a`; authenticated persisted LSF stdout confirms successful
completion. Chain 50% **29745503** ran on `n-62-18-5` from 17 October 23:02:29
to 18 October 01:25:10 Europe/Paris (8561 seconds). Native summary/case status
is passed, source indices [2], 5001 saved states ending at update 5000.
Independent-primal loss changed from -0.2778552771 to -0.5394011736.
Model and context before/after digests match the previously pinned identities.
Source remains parallel-01; no native experiment was restarted or changed.

Full combined report/evidence **29745504** ran on `n-62-31-15`, 18 October
01:25:10–02:18:20 Europe/Paris (3190 seconds), successfully completed. All six
native input IDs are retained in its analysis, including 29745503. Canonical
summary SHA check passed. Entry quota required 8 GiB and passed with 20 GiB
conservatively available; packaging's actual-byte-plus-reserve gate required
5.6151 GiB and passed. Source/manifest remains the original sealed parallel-01;
the interim report tooling did not alter this job.

Archive `/zhome/95/e/206121/mosaic-runs/20261017-brd4-best-primal-report-02.tar.gz`:
786,077,737 bytes, 186,506 files, SHA-256
`4d5540823d5b43c8b2ca60900c52c1c28653f5a43287369d9d533df41c2b9b4d`.
Remote receipt `.tar.receipt.json` was read and independently downloaded.
The standalone full report HTML is 417,532,320 bytes, with six probability
heatmaps and loss/change/molecule snapshots. Download is in progress into a
fresh local `results/hpc/20261017-brd4-parallel/complete-report-01/` directory.
Final local archive/manifest/schema checks and viewer delivery are pending
at this entry. Existing five-case report and server remain preserved.

### Six-case download, identity checks and viewer

The archive and receipt are now downloaded to
`results/hpc/20261017-brd4-parallel/complete-report-01/`. Exact remote path,
786,077,737-byte archive length and SHA-256 match the remote receipt above.
`scripts/verify_hpc_evidence.py` authenticated the archive and safely extracted
its members. A desktop/tool reset interrupted its final extracted-file hash
loop; no final archive verification receipt was claimed at that point.
The retry performs stdlib byte checks only and reauthenticates the manifest
against the original archive, with no re-extraction or scientific execution:

```powershell
py -3.13 scripts/hpc/receipts/20261017-brd4-best-primal/verify-complete-report-download.py results/hpc/20261017-brd4-parallel/complete-report-01
```

Local `study-verification.json` independently records all six native input IDs
29745501/29745502/29745503/29745473/29745474/29736988, each passed with
5001 contiguous states, six atom slots and 15 unordered bond rows. Chain
source indices remain [0]/[1]/[2]. Expected fixed model/context digests,
seed 17, 5000 total updates, LR=0.03, copied continuation plan identity and
native-summary identities match. All three original strict parent summaries
retain failed status. The canonical study did not copy a configuration.json;
its saved native configuration matches the original source JSON and its SHA
`49e01e601f77e21770d19b4a40df365315e4521318245d14ccec6e7cdbe6e359`.
These are saved-file/schema checks, not new numerical model validation.

All 13 rendered report files match the authenticated evidence manifest's
lengths and SHA-256 values. HTML: 417,532,320 bytes, SHA
`94e3072394c2f2c48813edaab05bb9c0c0577afb439428a7ab838d6a1b16c9a9`.
`final-scheduler-records/` contains authenticated successful LSF completion
stdout for 29745503 and 29745504, with downloaded byte/hash receipts.

Restored the local viewer with the existing detached loopback helper on
port 8771. A separate command after the launcher exited confirmed HTTP 200
and Content-Length 417532320 at `http://127.0.0.1:8771/report.html`.
The prior process had disappeared between tool sessions. Browser inspection
of the new large report disconnected the automation transport without returning
page state; visual/playback QA is not certified. No alternate browser-policy
workaround or local scientific rendering was performed. User received the
updated six-case link. The completion heartbeat is already PAUSED and remains
paused. Existing five-case artifact and unrelated working changes are retained.

Inspected the actual saved `loss.png` and `molecule_endpoints.png` locally
without rendering them again. All six colored curves reach update 5000;
strict-prefix boundaries appear as dotted vertical lines. Initial/final raw
graphs retain readable element/slot labels and visible probability-dependent
atom/bond sizes. All six final hard readouts are explicitly marked INVALID,
including dense halogen-rich or disconnected output; those graphs are kept
as produced. This static-image inspection does not certify interactive
selection, playback or browser behavior.

The versioned post-extraction operator above completed successfully on local
Windows Python 3.13: **186,506/186,506 files matched**, with no missing files,
path/size differences or checksum mismatches (463 seconds for the per-file
pass, eight I/O workers). Archive path/bytes/SHA and the extracted manifest's
exact bytes were reauthenticated first. Manifest SHA-256:
`5d070e3b72d41bd8c8cb6ed343d78921c5fb1c76f6d7693f50f9c96e85d423d1`.
Final receipt:
`results/hpc/20261017-brd4-parallel/complete-report-01/20261017-brd4-best-primal-report-02-download.json`.
This completes local evidence integrity verification. Ruff check/format and
Git whitespace checks passed for the new byte-verification operator/docs;
these are static engineering checks. No local numerical tests, scientific
rendering, chemistry evaluation or model execution were performed. Interactive
browser QA remains explicitly unconfirmed as recorded above.

## 2026-10-19 — opt-in BRD4 failure-aware chemistry port; HPC access blocked

The user supplied DBMol instructions and authorized integration into a fresh
50% chain/uniform initialization. Read the attachment, DBMol docs 63/64 and
latest ledger, repair/loss/schema/conversion/training-integration sources, and
Mosaic's existing native/continuation contracts. The attachment ends mid-sentence
in section 5.C; complete referenced documents supply the remaining terms.

Implemented an isolated opt-in package, fresh driver/configuration, allocated
CPU/GPU/report launchers and saved-only interactive reporter. Existing six
trajectories, old report bytes and historical outcomes remain sealed. Decisions
and completed/pending items are in `docs/brd4_chemistry_5000.md`. Retain six
occupied neutral slots, actual C/N/O/S/F/Cl ordering, BRD4 whole-target contact,
fixed Tiny/context/seed, LR .03 and 5000 total updates. Use categorical logits
for unfloored NLL and contact + .5 chemistry + .1 empirical element-JS. Both
objective and optimization geometry change; the old 50% comparison is exploratory.

Detached helpers are verbatim source symbols/whole pure modules with original
and extracted hashes. The live loss is JAX. Full Torch source is isolated as
test-only oracle data; no Torch/learned model is added to the Tiny runtime.
The exact versioned rule table is retained; the older expected-valence-excess
objective is excluded from production. Original depth-three and separate
depth-six searches use 32 checks/depth, beam 4, regret weight 1 and atom edits
off. Both search counts/timings are retained, with timing outside scientific
preparation identity. Complete targets, invalid partial witnesses and actual
fallback rows stay distinct.

All ten halving trials share one preparation, context, key, weights and empirical
reference. Unknown search errors, nonfinite native values/gradients and failed
FD/parity/identity gates remain failures. Native order-null probes apply to
contact alone; chemistry sees bond order. Record no-op/fallback updates, changing
refreshed surrogates, within-state deltas, per-term losses and best-observed
fixed-contact state/identity. Raw invalid graphs are retained; repair targets
never replace scorer inputs.

One bounded SSH attempt with BatchMode and ConnectTimeout=15 reached DTU but
returned `s234818@login1.hpc.dtu.dk: Permission denied (password)`. The remote
metadata command did not execute. No fresh quota/queue observation, upload,
numerical test, pilot or submission occurred; no new job ID is claimed.
No repeated attempt was made during this unchanged outage. All new numerical
validation, empirical preprocessing, model execution and rendering remain
allocated DTU work. Local checks are static source/JSON/hash/JS/shell checks
only; final scope and source-package receipts are recorded below.

Local static review used Python 3.13 stdlib `ast.parse`/JSON/SHA-256 only,
without importing project modules: **35 Python files parsed**, all eight
production extractions and eleven original oracle-source files matched their
manifest hashes. Captured reference manifest SHA-256:
`295c44903fbb7cf27d5c3827b1cc1e2f4159d471c3d55e742d90c6b14203520e`.
User attachment (9050 bytes) SHA-256:
`3cfe8ccb137350db037160415579c3d7941174b49a6c5321af1eab92a438c0af`.
Receipt: `results/static/brd4-chemistry-20261019-01/static_receipt.json`.
Ruff check/format passed on the handwritten modules, driver/helpers and two
test files, excluding byte-pinned original/extracted modules. `node --check`
passed on the main script extracted from `scripts/chemistry_report_template.html`;
no DOM/browser or trajectory rendering executed. Git for Windows `bash -n`
passed the three chemistry LSF wrappers and two submission operators. Git
whitespace checks passed. These checks establish static engineering integrity,
not numerical correctness, native chemistry coverage, validity or runtime.

Added a required disposable native-input assay with one internal neutral F
and two single neighbors. It must show positive chemistry loss, finite nonzero
chemistry gradient in the shared logits and frozen-reference combined FD
agreement before the requested all-carbon 50% initialization is optimized.
Its probes/pilot are excluded from the 5000-update budget. All new allocated
checks, empirical reference counts, pilot timing and native results are pending.
No new report or molecule result is claimed from this implementation turn.

Sealed local source commit reference **933d45d**, unique directory
`results/releases/brd4-chemistry-20261019-01/`: **350 files**, archive
**39,159,288 bytes**, SHA-256
`9d10271ab846d150bee1d47a13ac9892fa699f91a459f593ecd4ec35d08eac3f`;
manifest `1a8ead22c031b6d525f93072b6752598d6cbc00b0d42fdfb1f2cd92804ffc2cb`.
Command: `py -3.13 scripts/package_stage0_release.py --output results/releases/brd4-chemistry-20261019-01`.
Independent stdlib checks authenticated the archive, manifest, safe regular
member names and every one of 350 member hashes, without extraction or imports.
Receipt: `local_archive_verification.json` in that directory. Git-index byte
checks also authenticated all 19 original/extracted reference files. Scoped
Git newline rules and Ruff exclusions preserve byte-pinned snapshots; generated
metadata/initializer files use explicit LF. Source snapshot remains numerically
unvalidated; no upload or job was submitted.

Planned release base `/work3/s234818/mosaic/releases/20261019-brd4-chemistry-01`,
campaign tag `20261019-brd4-chemistry-01`. Readiness bootstrap SHA-256
`af8be632de5e3383acfe231e33e771f62bbc8bd55e7f8b4389a4a16309d0f4d1`;
copy separately with the archive and authenticate before execution. Native
submission remains gated by actual CPU DONE and its source-bound assessment.
These receipt notes were added after sealing and are not represented as bytes
inside that immutable archive. All scientific source bytes remain unchanged.

## 2026-10-19 — restored access; chemistry readiness submitted

The user requested another connection attempt, allocated tests and submission
of the fresh 50% chain/uniform chemistry run after passing checks. SSH access
worked. Official quota reports showed work pool 6 at 89.77/100 GiB and home
at 10.71/30 G (cached about four hours). The A10 queue is Open:Active, has
real running jobs and permits 1440 minutes; the requested 23 hours is within
that limit. Existing DBMol jobs were observed but left unchanged.

Transferred the previously sealed 39,159,288-byte archive, manifest and
separate readiness bootstrap to the planned release base above. Remote
archive, manifest and bootstrap SHA-256 checks passed; all 350 extracted
source files authenticated. The source directory is read-only. An initial
operator invocation failed before submission because it lacked the login
shell scheduler environment; no job existed from that attempt. The exact
declared readiness command was then submitted from a login shell, without
changing the source or scientific settings.

CPU readiness **29753139** was submitted to `hpc` at 01:41:38 Europe/Paris,
with four slots, 4 GiB/slot and one hour. It started at 01:41:40 on
`n-62-30-25`. The allocated quota gate passed with 19 GiB conservatively
available, against an 8 GiB reserve. Run directory:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-01-readiness`.
Legacy tests, original Torch oracles, new JAX/report tests, empirical-reference
preprocessing and assessment are pending at this entry. No numerical pass,
native GPU submission or optimization start is claimed yet.

Readiness 29753139 exposed a launch-selection defect: the shell expands
`tests/test_small_molecule_*.py` into explicit pytest arguments, and pytest
`--ignore` does not remove those explicit files. New chemistry fixtures
therefore ran prematurely in FP32 without their original-source oracles,
with a recorded failure and skips. The required later FP64/oracle phase
is not waived. Corrected the wrapper to select legacy files into a Bash
array before invoking pytest. Bash syntax and Git whitespace checks passed;
all scientific source/test/config bytes remain unchanged. Release/run 01
is preserved; prepare a unique release/run 02 for allocated revalidation.

29753139 reached actual EXIT after 117 scheduler seconds; pytest recorded
216 passed, one failed, 35 skipped, three deselected in 110.23 seconds. Its
single failure was the prematurely selected initialization roundtrip fixture:
FP32 log/softmax error 1.1920929e-7 exceeded decimal atol 1e-7. Changed only
that replay bound to `2 * finfo(dtype).eps`, with FP32/FP64 test coverage;
no chemistry value/VJP, native FD/null/parity or identity tolerance changed.
Release 02 (manifest 6ed656d054902b440fe777b4bd1630ceb52b63bfdb706f842d5d097437aaa763,
archive 2ad6b888163ff3123da400066a5a053ac90029bcf22dcbe33ef0b93b5902aa55)
was transferred but never submitted; retain it as superseded packaging.
Prepare release/run 03 with both engineering fixes. No oracle, composition
preprocessing, native model or optimization ran in the failed attempt.

Corrected release 03: 350 files, 39,163,306-byte archive SHA
382d8dbe2f3edb5850cf810d760ac9f62ea77cd035b71cd40e57d90c623e0d79;
manifest 3a3f4710d3d7a8e61138e3464e18bb0eb29173cf5677432bf3b4dfb2022b2a00.
All production `src/` and configuration bytes match release 01. Remote
authentication passed. Readiness **29753140** ran on `n-62-30-25`, then
ended EXIT after 104 scheduler seconds. Legacy suite: **208 passed, three
deselected, zero skips/failures**, 93.45 seconds. The next oracle stage
raised PyTorch's constant-output `does not require grad` error: inactive
source terms return detached zero tensors. `allow_unused` handles unused
inputs, not an output lacking a grad function. New JAX tests/reference/native
optimization did not run.

Fixed only the generator to record mathematical zero VJPs for those constant
terms, explicitly listing `constant_components`; differentiable terms still
use original-source autograd. Source snapshot, JAX loss and all native/parity
gates remain unchanged. Release/run 04 will rerun the complete sequence.
Failed attempt 01's logs, XML, scheduler and quota proofs are downloaded to
`results/hpc/20261019-brd4-chemistry/readiness-01/` and retained remotely.

Sealed local release 04: 350 files, 39,163,735-byte archive SHA-256
`3280845270bc0f647d309b5671a2670e0d49a72c5c60f14a334427d479be5983`,
manifest `368f455800678c7aee020f09c1625fda67e6504e84a355f49b975516453531b6`.
Ruff check/format passed for the oracle-generator fix. The initial SSH stage
of the transfer operator timed out. A single bounded retry after 30 seconds
reached the banner but disconnected with `client_loop: send disconnect:
Connection reset`. Stop further attempts during this outage. The traceback
identifies failure in the initial metadata SSH, before archive transfer or
the later submission stage; no release-04 test/GPU/report job was submitted.
Its remote directory/bootstrap and release-03 final-record bundle may have
been partly created: inspect those paths before reuse. Release-03 scientific
and error evidence remains on HPC; its final-record download is pending.
No claim of chemistry parity, native validation, empirical preprocessing or
5000-update execution is made. Current blocker is unstable SSH access,
not storage capacity. These final notes postdate the local sealed archive.

### User-requested connection retry — 19 October

The user requested connecting again, running the corrected tests and submitting
the 50% chemistry run only after passing validation. One bounded attempt using
`ssh -o BatchMode=yes -o ConnectTimeout=15 dtu-hpc 'bash -l -s'` reached the DTU
banner but failed authentication: `s234818@login1.hpc.dtu.dk: Permission denied
(password)`. No remote metadata command executed, so no fresh quota/scheduler
state or release-04 directory state is asserted. No upload, corrected numerical
test, empirical preprocessing or GPU/report job was submitted. The corrected
sealed local release-04 archive/manifest above and commit ea15497 remain ready;
the 208-test legacy result belongs to failed overall readiness 29753140 and is
not chemistry parity or native validation. Stop further attempts during this
authentication outage. All historical jobs/evidence and unrelated files remain
unchanged; no local numerical checks were run.

### Restored access; corrected readiness 29753148

The next user-requested retry succeeded. Official work pool 6 quota was
90.22/100 GiB, with 301,662/20,000,000 files; cached home usage remained
10.71/30 G. Both `hpc` and `gpua10` were Open:Active. Release-04 was absent,
and no new chemistry job existed, so the unique base was created without
overwriting historical evidence. Transferred the already sealed release-04
archive and reused its separately hash-authenticated bootstrap. Archive SHA
`3280845270bc0f647d309b5671a2670e0d49a72c5c60f14a334427d479be5983`
and all 350 extracted files authenticated against manifest
`368f455800678c7aee020f09c1625fda67e6504e84a355f49b975516453531b6`.

Corrected CPU readiness **29753148** is submitted to `hpc`, four slots,
4 GiB/slot, one hour, and observed RUN on `n-62-30-25`. Run directory:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-04-readiness`.
The complete legacy/oracle/new-JAX/report/reference sequence is pending at
this entry. The gated native run remains unsubmitted until actual successful
scheduler termination and its source-bound assessment authenticate. Existing
DBMol jobs, historical Mosaic jobs and unrelated working files are unchanged.

### Corrected readiness passed; native 29753152 and report 29753153 submitted

Readiness **29753148** reached actual DONE on `n-62-30-25` after 170 scheduler
seconds; persisted LSF record says Done successfully. Legacy suite: **208
passed**, three native-model tests deselected, 64.97 s. Chemistry/report suite:
**45 passed**, 87.76 s. Neither suite has failures/errors/skips. Twelve pinned
original-source value/VJP fixtures generated; oracle SHA-256
`149360975e19e01a2145543404b3b790f54b41cb892d5cd377b539327fb0551e`.
New JAX parity/gradient/directional/identity/report fixtures all passed.
No native Protenix numerical pass is inferred from these CPU tests.

Empirical reference preprocessing passed on 1,688,539 authenticated TRAIN
molecules. Source occupied-atom counts total 36,521,162, including 54,376 Br
excluded when conditioning onto the six supported elements. Excluded P/I
counts are zero; full source charge/element counts and data identities are
retained. Reference SHA
`9135e9985dfd6c6d6afd96726ae1d4c1a5edc6a4966127475a1fec888c849bdc`.
Assessment SHA
`dd4afb5f403c664997f8c8f9058c0d46c48ea3870800ea9fba79faa948f41d22`;
it binds config `42de1ea55cced1d2c0577f65332c26a439d34b61cc620301c439372f88d0ae72`,
source manifest `368f455800678c7aee020f09c1625fda67e6504e84a355f49b975516453531b6`,
the XML proofs, original-source oracles and reference bytes. Legacy XML SHA
`943ad9b77afb50126fd488b9dc86cdebc85e79fb8b69ce1578cea0e55a91a3fe`;
chemistry XML `927adc72fbe86ae51268a1f2a281b48252c073ab8a24a42eacdd2794cfb92858`.

After authenticating successful CPU termination and its assessment, reverified
all 350 sealed source files and invoked the unchanged native submitter.
GPU **29753152** submitted at 01:55:49 Europe/Paris to `gpua10`: one exclusive
A10, four slots, 8 GiB/slot (32 GiB total), 23 h, `done(29753148)`. Its scheduler
record confirms the requested GPU/resources and initially reports PEND/New job
waiting for scheduling. Run directory:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-04-native`.
It must pass native contact FD/null/parity, invalid-F positive chemistry/VJP
and combined FD, frozen identities, ten disposable pilot updates and runtime
fit before the fresh 5000 updates. Keep six neutral occupied atoms, seed 17,
50% chain/uniform initialization, LR .03, contact + .5 chemistry + .1 element-JS.

CPU report/evidence **29753153** submitted to `hpc`, two slots, 4 GiB/slot,
one hour, `ended(29753152)`, initially PEND. Run directory:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-04-report`.
It retains raw invalid graphs and partial/failure evidence without weakening
native identity gates. Submission/queueing is not native validation, an
optimization start/completion or a rendered report. Previous failed attempts,
six historical studies and other campaigns remain unchanged.

Readiness proof bundle downloaded to
`results/hpc/20261019-brd4-chemistry/readiness-04/`: 36,663 bytes, archive SHA
`7bf1a097f04338ccbc2c11ce4c6c07e28ef2611481a3afadad604efb067f8869`.
Local Python 3.13 stdlib authenticated exact bytes, 24 unique safe regular
members, source/configuration/assessment identities, XML test counts with no
errors/skips, reference/oracle hashes, persisted CPU success and submission
receipts. Receipt: `verification.json`. This was saved-evidence verification
only; no local numerical tests, chemistry evaluation or scientific rendering.

Failed readiness 29753140 proof also downloaded to `readiness-03/`: eight
files, 8,438-byte archive SHA
`acc04e1b46ea0982078fd8760e2e92ba2aa4d743b2a5c3261bf5279a36fcc0e4`.
Stdlib checks/safe extraction confirm actual EXIT and the preserved constant-
output oracle error. Receipt: `readiness-03/verification.json`. Historical
failure is not retroactively a passed readiness or native scientific failure.

### 19 October, 08:36 Europe/Paris — native started; 4544/5000 updates

User requested start status and ETA. A compact scheduler/log inspection
confirmed **29753152 RUN**, started 07:12:54 Europe/Paris on `n-62-18-5`;
**29753153 PEND**, dependency not satisfied. Native entry reserve passed with
19 GiB available against 8 GiB. Persisted study summary reports native contact
checks passed and coupled invalid-F chemistry/combined-direction checks passed
(chemistry gradient norm 0.09259258210659027). The disposable ten-update pilot
passed, ten strict improvements and zero fallback/no-op updates, mean
0.9513378069 s/update. Frozen model/context and runtime-fit gates passed
before entry into the fresh requested trajectory; final identities are pending.

At 06:36:46 UTC (08:36:46 Europe/Paris), saved `case-0/updates.jsonl` had
4544 contiguous completed updates, 456 remaining. Last contact -0.3821232319.
Saved timing means: last 100 updates 0.9587433985 s/update; last 500
0.9924594264. These project 437–453 seconds remaining, approximately
08:44 Europe/Paris; user-facing ETA about 08:45 with overhead. CPU report
queueing/rendering/packaging is additional and has no measured ETA yet.
This is a partial RUN status and timing estimate, not native completion,
terminal identity verification or a rendered report. Metadata/log inspection
only; no job/source/settings changed and no new numerical experiment ran.

### 19 October — native complete; report rendered; packaging engineering failure

User asked whether it is done. At 06:55:59 UTC, scheduler confirms native
**29753152 DONE** (1:32:04 runtime) and report **29753153 EXIT** (34 s).
Persisted native/case statuses are passed, 5001 states through update 5000,
5000 applied/strict-improving updates and zero fallback/no-op updates.
Final model digest remains
`8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2`;
context remains
`6747ba18833013da02dbf407d9faefb1adbc8314353b4882a34f99557b96d4f6`.
Driver elapsed 5503.295348621 s. Final contact -0.3851259649; saved raw
argmax is connected sanitized `CCCCCC`. Chemistry penalty is zero at the
final state; its complete saved history will be inspected without reevaluation.
Native summary SHA
`c110b4e0bcd716679102022ad51d46590a658f8f3f9471b7b91bfbc1166d85fa`.

Original rendering passed (render_exit=0). Its receipt retains native 29753152
and historical 29745503, 5001 saved new states, 5001 historical contact points,
complete=true. HTML is 18,169,151 bytes, SHA
`01ff604c69b6a80374a3447c68f4976c6da061b69b8aae9ef72f9cc191aec3cf`.
The report job's EXIT comes from packaging: the evidence collector recognizes
only the exact `pytest-unit-tmp` name and rejected the readiness phase's
`pytest-unit-tmp-legacy/...current` alias before creating an archive.

Prepare unique report-only recovery source/run 05. Collector accepts only
the three explicitly declared pytest basetemps, records internal current
aliases without following/archiving links and retains strict rejection for
escapes, cross-basetemp/file/parent/unknown-root/non-current links. Add focused
allocated boundary regression tests. Recovery authenticates native terminal
gates/source/config and the existing rendered report, copies its exact bytes,
and preserves the failed report attempt in the full evidence package. No
native experiment, loss, model/context or original report bytes are changed.
Local Ruff and Bash syntax checks passed; numerical/fixture execution remains
allocated HPC only.

### Report-only recovery 29753409 passed; saved report opened

Sealed source 05 contains the collector boundary fix, ten packaging tests,
three existing report tests and `brd4_chemistry_repackage.lsf`, with unchanged
native `src/`, examples and configurations. Manifest
`14e59d968eb9f342a0da68b9fac31b44a2702bcf564780fa7db199123147b069`;
351 files, 39,168,493-byte source archive SHA
`8d8e2ebc7b8a34c1b14651763cc25a6725c55fc5375462223f6cad598a9d2d89`.
Source base `/work3/s234818/mosaic/releases/20261019-brd4-chemistry-05/source`.
Submitted CPU recovery **29753409** to `hpc`, two slots, 4 GiB/slot, one hour,
after both native/original report ended. Actual run `n-62-30-25`, 08:59:42–
09:01:13 Europe/Paris, DONE, 91 s. **13 tests passed**, no failures/errors/skips,
5.78 s. New pytest negative links are removed in fixture cleanup so their
deliberately invalid aliases do not contaminate the enclosing evidence bundle.

Recovery authenticated the terminal native summary/case, full 5000-update
budget, frozen model/context and native gradient/pilot gates, unchanged
scientific source/config and the existing complete report receipt/files.
It copied rendered bytes without new rendering or numerical model/chemistry
execution. Original render job remains 29753153. Copied receipt SHA
`586d007df0d5d193a10462296e76a7720572c0bdd60c19b2f945c914dd78280a`.

Full evidence archive:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-05-repackage.tar.gz`,
**249,546,618 bytes**, **15,995 files**, SHA
`c0309f186da134d48f69bc6fd8594535325934f4904def3aaad377b2452d0c89`.
Entry 8 GiB and packaging 4.41119754 GiB quota gates passed with 19 GiB
conservatively available. Original failed report packaging remains in the
archive; no scientific result was altered or retroactively relabeled.

Native LSF final stdout authenticates successful completion at 08:44:58;
original report failed at 08:45:34 and recovery succeeded at 09:01:13.
Final scheduler/LSF files downloaded separately to `complete-report-01/`,
authenticated by a 3,881-byte metadata archive SHA
`d5c9260c8dcecb27e50026acf48607031fc87762b6a6eb4e404f184f6b355ee1`.
Actual stdout success wording is `Successfully completed.`; scheduler entries
say `Done successfully.`. Receipt: `final-scheduler-verification.json`.

Downloaded the already rendered report independently to
`results/hpc/20261019-brd4-chemistry/report-view-01/report/`; all four artifact
hashes/lengths and source/input/state receipt checks passed. Served by the
existing detached helper on fresh loopback port **8772**, preserving prior
servers. Browser inspection succeeded: complete banner; updates 0/2500/5000;
play/pause advances saved states; element/slot labels C1–C6 are readable;
atom/bond probability sizes and exact probability tables are visible. Viewer
tab is marked deliverable at update 5000. Presentation only, no local
scientific rendering, repairs or model execution.

Recovery's saved-record analysis reports zero chemistry penalty in **all
5001 states**, all saved hard readouts sanitized and only SMILES `CCCCCC`.
Contact changed -0.2778552473 to -0.3851259649; probabilities changed while
the hard chain remained hexane. Along this trajectory the repair term was
inactive; its active native derivative evidence is the separate invalid-F
probe, not an observed repair-driven optimization effect here.

Full archive copy initially stalled through SCP; its 17,825,792-byte partial
is preserved with `.stalled` suffix and is not evidence of a complete download.
Switched to one authenticated SSH byte stream of the same immutable archive;
the exact expected 249,546,618 bytes arrived. Standard-library archive hash,
safe extraction and all per-file checks are pending at this entry. Receipt
path and byte length match the remote receipt. No local numerical validation.

### Final full-archive integrity pass and report delivery

The full SSH-stream download matched the exact receipt archive path, length
and SHA. `scripts/verify_hpc_evidence.py` authenticated the archive and completed
safe extraction, then spent several minutes in its serial Windows per-file
loop. Stopped only that verification process after extraction and adapted the
previously used bounded post-extraction byte operator to the new pinned archive:

```powershell
py -3.13 scripts/hpc/receipts/20261019-brd4-chemistry/verify-complete-report-download.py results/hpc/20261019-brd4-chemistry/complete-report-01
```

It reauthenticated archive path/bytes/SHA, safe unique member names, the exact
archived/extracted manifest and **15,995/15,995 file paths, lengths and hashes**.
Eight bounded I/O workers completed the per-file pass in **83 seconds**, no
missing files or differences. Manifest SHA
`0ecf15951cb5bf7c61ee6706445f73270a5431700afb9bd510ad61d7537797c1`.
Final receipt:
`complete-report-01/20261019-brd4-chemistry-05-repackage-download.json`.
The stopped serial pass is not claimed complete; the successful replacement
retains every original archive/manifest/per-file gate. Ruff check/format passed
for the new stdlib operator; no local scientific/numerical execution occurred.

Additional saved-record checks in `study-verification.json` authenticated the
native summary/case equality, 5001 contiguous states/5000 contiguous updates,
all state snapshot SHA receipts against the evidence manifest, fixed six slots,
mixing .5/LR .03/seed 17, config/composition reference pins, passed native
contact/active-chemistry/pilot gates and terminal frozen identities. The
successful 13-test report-only XML and original input IDs remain present.
Recovered HTML exactly matches the viewer copy SHA
`01ff604c69b6a80374a3447c68f4976c6da061b69b8aae9ef72f9cc191aec3cf`.

`report-view-01/view-state-verification.json` records all 5001 embedded saved
frames with X[6,6]/E[15,5], the zero repair-penalty/hexane readout history,
expected model/context/input identities, verified report hash and HTTP 200
with exact Content-Length 18169151. Browser playback, controls, labels, sizes
and heatmaps were checked as above. Deliverable viewer is open at
`http://127.0.0.1:8772/report.html`; self-contained offline HTML is in
`report-view-01/report/report.html`. The native study and report delivery are
complete. Historical failures and unrelated working changes remain preserved.


## 19 October 2026 — two fresh delayed-chemistry 10,000-update arms prepared

User requested 50% uniform mix and random initialization, tiny initial chemistry
weight growing from about update 5,000 to a large endpoint. Prepared a versioned
v2 campaign with lambda=.001 through applied update 5,000, geometric ramp to
50 at update 10,000. Fresh six-slot 50% chain/uniform and independent per-row
Dirichlet(1), NumPy PCG64 seed 20261019; native key remains 17. LR .03 and original
repair/data/model/context/internal-loss pins remain. Preserve prior studies.

The schedule is fixed across every state's ten best-primal trials; refresh only
after acceptance, recomputing chemistry gradients. Retain weights in state,
trial, snapshot and derivative evidence. Only byte-identical raw native graph/key
observations can be reused. Compare changing-surrogate deltas only within each
update; retain best fixed-contact identity, raw invalid outputs and fallback/
no-op counts. Prior 5k run had zero repair in every state; a larger coefficient
is not a new gradient when this term is zero. See docs/brd4_chemistry_ramp_10000.md.

Local AST, Ruff check/format, Bash -n and viewer Node --check passed; these are
static checks only. No local numerical/chemistry/model execution. Prepared
allocated tests for scheduling/freezing, random schema, high-weight derivatives,
byte-bound saved-only two-arm reports and tamper failures, plus existing legacy
and required original-source oracle parity tests. Native driver adds low/high-
coefficient invalid-F derivative assays and disposable early/high-weight pilots.
Runtime gate remains empirical, not a guarantee of all future backtracking cost.

SSH access succeeded. gpua10 Open:Active/max1440min; requests 1GPU exclusive,
4slots/8GiB per slot/23h fit the actual queue. Home cached11.19/30GiB (18GiB
conservative available), work3 pool6 92.45/100GiB. New source needs about40MB;
no model downloads. Numerical outputs stay in dedicated home Mosaic directories
with fresh per-job quota gates. No tests/jobs have passed or been submitted at
this preparation entry. Immutable release and actual receipts follow.


### Ramp source01 authenticated; readiness29753654 running

Source01: 362 files, manifest
`dc54acf72b906da9f5e2ceb0f6458314064e48408414496d1717a81117a91c80`,
archive39,187,476bytes SHA
`d1735e168c5c6396a58aea341107ee09c58b7daee510a79c8d22bba0f6dedd9c`.
Remote immutable base `/work3/s234818/mosaic/releases/20261019-brd4-chemistry-ramp-01/source`.
SCP transfer and exact archive/per-file manifest verification passed; source
made read-only. Actual CPU readiness **29753654** submitted09:35:10 and started
09:35:12 Europe/Paris on `n-62-30-25`, hpc4slots/4GiB-per-slot/1hour.
Unique run `/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-ramp-01-readiness`.
Legacy tests running at inspection. This is a submitted/running allocated test
job, not a readiness pass or native GPU submission. Local release receipts in
`results/releases/20261019-brd4-chemistry-ramp-01/`.


### Ramp readiness passed; both independent native jobs actually queued

**29753654 DONE**, persisted assessment passed. Actual allocated XML:
217 legacy +57 chemistry/schedule/report = **274 passed**, no failures/errors/
skips;3 intentionally deselected slow/model tests. Legacy66.00s, new102.95s.
Original-source oracle generator retained12 value/VJP fixtures; training-only
reference export and exact configuration/source/test-proof checks passed.
Assessment SHA `35dac42a903ee397e77ff06f2b7a35cde5a191a475ebde007f844cc44912a895`.
Reference SHA `37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb`.
New config SHAs: mix50 `1b9ea19c4405a2289f111db7dd8fc41f9df17f4ddffdfe1033de2554197c56ed`,
random `3f3bb1fd0b087417a462f7b4b5743ce1fc72ddbd8d099af0348380912ac0685f`.
Downloaded scheduler/LSF success, XML/oracle/reference/assessment/quota bytes to
`results/hpc/20261019-brd4-chemistry-ramp/readiness-01/`; stdlib verification
passed all receipt hashes, XML counts/no errors/skips and current config hashes.
No local numerical work. Entry home quota gate ready.

After actual CPU DONE/proof authentication and fresh quota inspection, submitted
**29753667** fresh50% chain/uniform and **29753668** fresh random Dirichlet at
09:39:23 Europe/Paris. Both actual bjobs records confirm gpua10, one exclusive
GPU/four slots/8GiB per slot/23hours, sealed source01 manifest and readiness
assessment SHA above, distinct correct config arguments. Their sole dependency
is done(29753654); they can schedule independently. Both initially **PEND**,
new job waiting for scheduling. No native scientific checks or trajectory
completion claimed by these submissions. Unique run parents are
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-ramp-01-{mix50,random}`.

Combined report/evidence **29753669 PEND**, submitted09:39:24, hpc2slots/
4GiB per slot/1hour after `ended(29753667) && ended(29753668)`. It renders saved
probabilities only, preserves partial/failed outcomes and requires unchanged
model/context even for partial presentation. Expected report directory
`.../20261019-brd4-chemistry-ramp-01-report/payload/report/` and sibling tar
receipt. Report completion, download/verification/browser opening remain
pending. Tracked campaign receipt in
`scripts/hpc/receipts/20261019-brd4-chemistry-ramp/campaign.json`. Old studies,
reports and unrelated working changes remain unchanged.


### 19 October13:13 Europe/Paris — both ramp arms running

One read-only scheduler/saved-metadata check at11:13:43UTC confirmed native
**29753667 RUN**, started10:50:34 Europe/Paris on n-62-18-5 GPU0;
**29753668 RUN**, started11:42:57 on the same node GPU1. They execute
independently, with original source/config/resources unchanged. Entry quota
gates passed with18GiB conservatively available against8GiB reserve each.
Both disposable early/high-weight pilots passed; full-budget trajectories
have started. Final frozen identities and terminal outcomes remain pending.

At inspection, mix50 had **7596/10000** saved completed updates, last100 mean
1.0193857698s/update, latest contact -0.4007746875 and applied repair coefficient
0.2752359928. Its remaining2404updates project about40.8minutes, roughly
**13:55 Europe/Paris**, before final checks/report work. Random had
**3350/10000** saved completed updates, last100 mean1.4496285082s/update,
latest contact -0.4100923240 and coefficient .001. Remaining6650updates project
about2h40m40s, roughly **15:55–16:00 Europe/Paris**, if this pace persists.
Random has not reached its coefficient ramp; later search/backtracking may
change its speed. These estimates use saved timing fields only and are not
guaranteed endpoint times or passed terminal checks.

Combined report **29753669 PEND**, waiting for both native jobs to end. Its
queue/render/package time is additional. Native summary statuses are running
without summary errors; raw invalid random graphs/valence diagnostics are
retained as authorized, not silently repaired for display. Inspection bytes:
`results/hpc/20261019-brd4-chemistry-ramp/latest-progress-inspection.txt`.
No scientific job/source/settings changed; no local numerical validation.


### 19 October14:21 Europe/Paris — mix50 complete, random60%

**29753667 DONE**, successful LSF/scheduler outcome14:00:15, elapsed
3h09m41s allocation/11362.206292487s driver. Remote stdlib metadata
authentication confirms summary/case passed and identical case fields,
10001 contiguous saved states and10000 contiguous applied updates,
10000 strict-improving, zero fallback/no-op. All native contact/low/high-
coefficient derivative/pilot gates passed; model/context before/after match
the campaign pins. Final repair coefficient50.0. Summary SHA
`0236e16f8da2aa333bf469ec5cf07b2fbe962b3a5e4d09a631d90ae60e0dfca8`;
case report SHA `db39013cd0e1667eec8e9894c4ac89e388ad35d8d6eb5dfda46e2d1e2c7f372d`.
Config/reference SHAs match the passed readiness. Terminal metadata receipt:
`results/hpc/20261019-brd4-chemistry-ramp/mix50-terminal-metadata.json`.
Full archive/per-snapshot byte verification remains pending combined report.

Saved-record inspection shows **all10001 repair losses zero**, all saved
argmax SMILES `CCCCCC`, initial contact -0.2778551579 to final -0.4092489183.
Even weight50 did not activate a zero repair term. This is the saved trajectory
result, not a new chemistry reevaluation or affinity claim.

At12:21:19UTC, **29753668 RUN**, **6009/10000** completed updates; last100
mean1.4364611799s, latest contact -0.4325866699, applied coefficient
0.008876712352. Remaining3991updates project1h35m33s, around**15:57–16:00
Europe/Paris**, conditional on future search/backtracking cost during the ramp.
Native summary still running with no summary error. **29753669 PEND**, awaiting
random arm ending. No new model run, source/config change or local numerical
validation. Existing scientific jobs remain unchanged.


## 19 October — separate completed50% delayed-chemistry report requested

User requested a new report for the completed50% mix, prepared for adding the
random result later, separate from prior reports. Added explicit completed-arm
preview mode to the saved-only ramp reporter: one included native input plus
the byte-bound second configuration/pending job29753668; no partial random
trajectory loaded or invented. Native gates remain required for the included
arm, and pending comparison is distinct from a failed/partial included run.
Default two-input mode remains strict. No changes to src/, examples/, any native
GPU job or existing pair report29753669.

Prepared a unique allocated CPU report-only job, with source01 native/config
byte-map parity, terminal mix50 summary/scheduler/readiness authentication,
19 expected report/collector fixtures, full saved-snapshot validation/rendering
and compact presentation evidence packaging. Actual counts/outcomes follow;
preparation is not an execution pass. New family plan:
`docs/brd4_delayed_chemistry_report.md`. Preserve old report servers and files.


### Standalone preview source01 submitted as CPU job29755226

Source manifest `34462df7a3ba5001bced1c6266fa1a2851d77f11022694339f194e6c38a88671`,
365files;39,195,593byte archive SHA
`7ced77a30b6613011d538b017eb7b720539a8a64dc14b43e79eacc7e1c3e638c`.
Remote release `/work3/s234818/mosaic/releases/20261019-brd4-chemistry-preview-01/source`;
transfer/hash/all365source-file checks passed, source made read-only.
**29755226 RUN**, hpc2slots/4GiB-per-slot/1hour, after done(29753667).
Unique run `/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-preview-01`.
Source wrapper authenticates unchanged src/examples map against native source01
and the pinned passed mix50 summary before rendering. 19 allocated presentation/
collector tests selected. Home cached12.41/30GiB; no model downloads.
Static Ruff/AST/Bash/viewerJS checks passed locally; no numerical execution.
Test/render/package and download outcomes remain pending at this entry.


### Standalone delayed-chemistry first edition verified and opened

Preview **29755226 DONE**,19 report/collector tests passed/no failures/errors/
skips in72.67s. Original scientific src/examples byte map unchanged, native
source01 and pinned passed mix50 summary/scheduler/readiness proofs authenticated.
Saved-only render validated every10001snapshot/coefficient, then generated a
complete included-arm/pending-comparison report. Pending random metadata has
no frames; no historical experiment included and no native rerun. Entry8GiB
and packaging4.123253824GiB quota gates passed with17GiB conservative free.

Archive15,963,546bytes/154files SHA
`8aba45395a1ce5ad6a67030bc42d89272db95742a4c749555975cfe38a03fb9b`,
remote `/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-preview-01.tar.gz`.
Downloaded to `results/hpc/20261019-brd4-delayed-chemistry-report/edition-01/`.
Authenticated final scheduler/LSF success separately, exact receipt path/byte
length/SHA, safe extraction and all154per-file hashes via
`scripts/verify_hpc_evidence.py`; stdlib preview verifier passed job/state/
coefficient/graph-dimension/frozen/status/XML checks. No local numerical or
chemistry/model execution or scientific rendering. This compact presentation
bundle retains saved summary/case/native-check/readiness/source proofs; full
scientific trial/snapshot data remain in the native run and future pair archive.

Rendered HTML35,486,566bytes SHA
`981dec0a28d9e7835e9ffdcefae664e3fa0078ae881257d29da87919f55bc014`.
New independent detached viewer **http://127.0.0.1:8773/report.html**, served from
`results/hpc/20261019-brd4-delayed-chemistry-report/viewer/`; HEAD200/exact
length/hash passed. Prior report files/servers untouched. Browser QA passed
completed50%/pending random semantics, disabled placeholder, playback0to22,
endpoint10000/weight50, readable C1–C6, probability-dependent radii/widths,
saved probability heatmaps, contact/repair/coefficient/fixed-delta plots.
Tab marked deliverable; saved screenshot `.../viewer-final.png`, publication
receipt `.../publication.json`. Saved findings remain all10001hexane/zero
repair states; contact initial -.2778551579 to final -.4092489183.

Prepared automatic addition after verified random completion by updating the
existing paused heartbeat `finish-combined-brd4-trajectory-report` to ACTIVE,
name Finish new BRD4 chemistry report, hourly. Scope now only new native
29753668 and pair report29753669, preserving completed29753667, all prior
reports and edition-01. It verifies actual outcomes/native/frozen/archive gates,
then publishes the verified two-arm report only to new viewer8773. Quiet on
unchanged PEND/RUN/outage; pause after delivery or blocker requiring user action.
No new molecular job or duplicate monitoring automation.


### 19 October15:47 Europe/Paris — requested final random addition

User asked whether random finished and requested its addition to the NEW
report if complete. Scheduler29753668 remainsRUN,29753669PEND on its ending.
Saved progress9489/10000, last100mean1.009615738s/update, coefficient
16.5476097393, latest contact -.4009699821, no fallback on latest update.
Remaining511updates project8m36s, approximately15:56, then report queue/render/
packaging. Preparing a stdlib saved-record verifier and bounded post-extraction
byte continuation for the two-arm evidence; no scientific source/model work.
Do not publish partial random data into the completed first-edition viewer.


### 19 October15:57 Europe/Paris — random native complete; pair report running

**29753668 DONE**, actual LSF scheduler success15:57:35, allocation started
11:42:57, driver15261.500353694s. Persisted summary/case passed,10001states,
10000applied/strict-improving updates, zero fallback/no-op. Before/after
frozen model/context pins retained. Summary SHA
`9a2ebae13049babdeaca670bb13e707927058b8f2cd273a706d12fbf2177bdb2`.
Saved final raw argmax is connected/sanitized `FCCC(F)F`, six slots with
elements F/C/F/F/C/C. Initial chemistry1.4735963345 to final0; fixed contact
-.2969741225 to -.4144608378. These are persisted readouts/scalars, not a
fresh chemistry evaluation, affinity result or causal validity guarantee.
All native bytes/records will be checked with the final report archive.

**29753669 RUN**, started15:57:37 Europe/Paris on n-62-30-24, allocated
hpc2slots. It renders only the two ten-thousand-update arms. No native rerun
or setting change. The new stdlib pair verifier was Ruff/AST checked and
its source-map interpretation cross-checked against preserved edition-01
metadata; this is engineering/saved-byte work, not a numerical model test.


### Pair report29753669 complete; authenticated archive download started

Actual report scheduler/LSF success authenticated, render complete=true with
input IDs29753667/29753668 and10001states each. Rendered HTML71,949,822bytes
SHA `8575ad053dd80d9b53f6f34594b1b6e25982303f361b51b950b73fc3399467d7`.
Archive `/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-ramp-01-report.tar.gz`,
739,279,500bytes/61,267files, SHA
`fea18fc354e3185dbb3b6b38058cd992a8713ac068bbeb5dfe2e8df4616d41e0`.
Packaging reserve5.214276031GiB passed against17GiB conservative home free.
No storage blocker. Local free229.88GiB before download.

Fresh local `results/hpc/20261019-brd4-delayed-chemistry-report/edition-02/`
retains receipt and actual final scheduler/LSF proofs. Current random/report
DONE authenticated; older mix50/readiness bjobs have aged out, so preserve and
use their earlier successful final records and LSF stdout. Not-found archived
scheduler snapshots are not scientific failures. Download of exact immutable
archive started through SSH byte stream; verification/publication remain pending.
New task-specific stdlib verification/publication operators preserve source,
configuration, initialization, model/context, complete budget, native gates,
saved frame identity and every-file archive integrity. No local numerical work.


### Complete pair evidence verified; random added to new standalone viewer

Report29753669 DONE16:11:22 Europe/Paris. Final scheduler/LSF success
authenticated,739,279,500byte archive matched exact remote receipt path/length/
SHA `fea18fc354e3185dbb3b6b38058cd992a8713ac068bbeb5dfe2e8df4616d41e0`.
Safe archive validation/extraction via scripts/verify_hpc_evidence.py completed.
Stopped only its slow serial Windows hash loop45s after completed extraction;
do not call that interrupted pass complete. Task-specific stdlib continuation
reauthenticated archive and exact archived/extracted manifest, unique safe
member sets, then all **61,267files** with eight bounded I/O workers in160s.
All paths/lengths/hashes passed. Metadata/native/source/config/reference/initial
state/FD-pilot/frozen/budget/snapshot-manifest bindings subsequently passed in
verify_pair_download.py. No local numerical/model/chemistry or scientific render.
Proofs: results/hpc/20261019-brd4-delayed-chemistry-report/edition-02/
20261019-brd4-chemistry-ramp-01-report-download.json and pair-verification.json.

Both native jobs passed10000updates/10001states,10000strict/zero fallback/no-op,
final weight50. Mix50 all10001saved readouts remain hexane with zero repair;
contact -.2778551579 to -.4092489183. Random starts repair1.4735963345 and
ends0, contact -.2969741225 to -.4144608378; final raw connected/sanitized
`FCCC(F)F`. First saved native/schema-valid state8959;1012such states total
(including nonmonotone later validity). Full raw invalid graphs, late changing-
reference repair spikes and fixed-contact increases remain in the report.
These saved observations do not establish affinity or causal/generic validity.

Published only the verified pair artifacts to the NEW family's viewer, HTML
last, preserving edition-01 and all prior report families/servers. Existing
localhost8773server reused; HTTP200/exact71,949,822bytes/full-body SHA
`8575ad053dd80d9b53f6f34594b1b6e25982303f361b51b950b73fc3399467d7` passed.
Viewer http://127.0.0.1:8773/report.html now includes only jobs29753667/29753668,
10001frames each. Browser screenshot/keyboard QA passed both selectors/colors,
random raw invalid initial graph/playback/final F/C labels, sizes, probability
heatmaps and all four loss/coefficient/delta panels. Accessibility snapshot
transport exceeded a frame limit on large HTML; actual page interaction worked,
so no report/scientific rerun or weakened verification was needed. Browser tab
marked deliverable; viewer-complete.png/viewer-overlays.png and publication.json
record presentation. Follow-up heartbeat PAUSED after verified delivery.
Unrelated user changes and all native runs remain unchanged.


## 19 October 2026 — Mosaic Boltz affinity implementation and gated campaign

User authorized a fixed-atom Mosaic Boltz2 backend, allocated HPC tests and then
fresh independent10000-update50%chain/uniform and random Dirichlet arms with the
unchanged delayed chemistry schedule. No VAE/decoder/latent/training/DeFoG was
imported. New native worker contains only Boltz integration mechanics; JAX owns
Mosaic's six neutral slots, probability graph, chemistry and best-primal updates.
Quantitative ensemble affinity is optimized with weight1; binder probability is
logged separately. No contact/confidence objective. Fixed native predicted pose,
zero recycles and no MW correction are explicit. See brd4_boltz_affinity_10000.md.

Local static AST/Ruff/bash parsing passed; no local numerical tests/model work.
Source01 archive39,229,862bytes SHA
fc24ad5db5afdde7599c0c41e72c1a4790f9ab1db23ca1d35cd4d72fe2a1851c;
manifest8a621a975cf98800ea3a2a7413f1e07b1f317b988c02d69b593832a7d1fd94af,
386files authenticated at
/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-01/source.
Reuse cached pinned native Boltz environment/checkpoints, no installs/downloads.
Observed home13.79/30GiB, work3pool6 89.07/100GiB, fresh quota gates mandatory.
Actual CPU29758624, native readiness29758626 and controller29758628 submitted
PEND. Native waits for CPU DONE; controller waits for native DONE and verifies
actual persisted gates/proofs before submitting either long arm. No long arm
is yet submitted, no numerical pass claimed. gpua10 is Open:Active and its
1440minute maximum admits the planned1380minute native arm requests. Four slots,
8GiB per slot, one exclusive GPU; same actual GPU type is benchmarked first.
All original Protenix runs and unrelated working changes are preserved.

CPU29758624 legacy prefix:220passed/3deliberately deselected,185.24s, two existing
JAX scatter future warnings. All3new bridge/gradient-scaling/affinity-artifact
fixtures passed. Native mapping and original-source chemistry/oracle phases
still pending; no full CPU/native approval or long submission is claimed.

CPU29758624 now DONE/passed283:220legacy/bridge +4native Torch mapping +59
source-equivalent chemistry/oracle tests. No failures/errors/skips;3deliberately
deselected model/slow tests. Actual downloaded XML hashes/counts and composition
reference verified locally with stdlib bytes/XML, not local numerical execution.
CPU assessment SHA 4d64197d56a2ba964ad1ae4ecf2b9e62e5e3ed2c4d14c247bae0a139cb79343b.
Proof results/hpc/20261019-brd4-boltz-affinity/readiness-01/verification.json.
Implementation pushed ecaf328. Native29758626 PEND on host slot limit;
controller29758628 PEND. Both10000-update arms remain unsubmitted until native
readiness and all persisted derivative/parity/chemistry/runtime/identity gates
pass. Quiet hourly follow-up reuses the paused, completed report heartbeat to
finish this new authorized gated-submission scope; pause after actual arm IDs
are authenticated/delivered, or a blocker requiring human action. It does not
resume or modify any completed report/experiment.

### 19 October 2026, approximately 19:08 Europe/Paris — requested L40S queue change

User requested a less congested GPU queue for native Boltz validation. Around
19:02, scheduler queue snapshots showed gpua10 two A10 GPUs/21 waiting jobs
and gpul40s twelve L40S GPUs/59 waiting jobs. Counts refer to jobs, not CPU-slot
PEND totals; the lower waiting-jobs/GPU ratio is not a promised start time.
H100 queues require special group membership without established account
authorization; no access attempt was made. The L40S has 48 GB GPU memory.

DTU eswitch rejected in-place bswitch with a permission denial, preserved as a
control failure rather than a native numerical outcome. Historical readiness
29758626/controller29758628 were authenticated unstarted PEND/PSUSP, with no
payload or long-arm submission records. Both were intentionally cancelled;
actual EXIT receipts confirmed before replacement. No molecular experiment
had started, and these EXIT outcomes do not invalidate the 283 passed CPU tests.

Actual replacement native **29758990 PEND/gpul40s**, one exclusive GPU,
four CPU slots,8 GiB per slot,two hours. Scientific source/configuration bytes
remain source01 at
`/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-01/source`, manifest
`8a621a975cf98800ea3a2a7413f1e07b1f317b988c02d69b593832a7d1fd94af`.
All native parity/FD, coupled chemistry, finite-value, pilot/runtime, model/
context identity and quota gates are unchanged. Output:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-queue-02-ready/payload/`.

External launcher03 at
`/work3/s234818/mosaic/releases/20261019-brd4-boltz-launch-03/source`
contains only controller/tests/two batch wrappers, manifest
`60924c49eb9cbe3c8741afb2f0b152e6ed2525f542f610310bbcaf59a3db5d85`,
archive5,100 bytes/SHA
`4453763533a0b323c2dc50f3930b3673d20a8b2e66a9fbd285989d66ee65cde5`.
No scientific code/configuration changed. First launcher-only CPU29758998
EXIT before tests from LSF splitting inline Python-c; failed logs preserved,
not a native failure. Wrapper retry **29759003 DONE**, eight unit tests passed,
zero failure/error/skip,19:07:12 Europe/Paris; assessment SHA
`f3e6c52ee9c73d6770c8c05179e377cce7f4e2112b0aa5f4ceaf9b40d000c1f9`.
Launcher CPU output:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-queue-03-cpu/`.

Actual controller **29759027 PEND/hpc**, dependency
`done(29759003)&&done(29758990)`, uses authenticated wrapper/launcher against
unchanged scientific source01. It requires actual native DONE, matching
validated gpul40s queue, every passed persisted proof and fresh quota before
either bsub. Future arms remain independent, each gpul40s/one exclusive GPU/
four slots/8 GiB per slot/23h;23h is a limit, not an ETA. No long arm has yet
been submitted. Controller output:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-queue-03-controller/`.
Unchanged future native root:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-01-native/`.
Compact local resource-change evidence:
`results/hpc/20261019-brd4-boltz-affinity/queue-change-02/`.
Download complete and stdlib byte verification passed all23files, exact
lengths and SHA-256 hashes. Downloaded manifest SHA
`e1f65ded65a82cbb6c77431a702b305a2d1f24a8bc8e7333f3561671c47d575d`;
verification.json status passed. At19:08:37 Europe/Paris, native29758990 still
PEND with `Job's requirements for resource reservation not satisfied
(Resource: ngpus_physical): 3 hosts`, an ordinary GPU reservation wait;
controller29759027 PEND on dependency. No native pass or long submission inferred.
All numerical tests were allocated HPC; local work was static editing,
packaging and byte verification only. Historical experiments and unrelated
project-description changes remain preserved.

Final 19:10 scheduler authentication: native29758990 PEND/gpul40s, four
CPU slots reserved on n-62-13-14, waiting for ngpus_physical; controller29759027
PEND dependency. Recursive host groups confirm six L40S nodes versus one A10
node. Quiet follow-up now targets the replacement IDs and same-queue launch
gates; all old jobs/evidence remain preserved.

### 19 October 2026, 19:22 Europe/Paris — native preparation TMPDIR error

Authoritative scheduler evidence records L40S readiness29758990 RUN from
19:09:51 on n-62-13-19; this supersedes the approximate19:10 PEND annotation
as current status. All386source files and fresh10GiB home quota passed. Native
prediction saw its actual GPU and finished preprocessing. Its redirected
`prepared/prediction.log` then recorded multiprocessing queues/resource_sharer
`OSError: AF_UNIX path too long`: the batch TMPDIR under the long home-run
path exceeded Unix socket path limits. Outer prepare.log was empty because
of child redirection. No prepared context.pt, native affinity finite
differences, coupled checks or pilot validations had been reached. LSF's
power/clock warning says the GPU can still perform the workload; that warning
is not this integration bug.

Failure-stage snapshot captured17:22:08UTC/19:22:08 Europe/Paris at
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-tmpdir-failure-01`,
manifestSHA
`ee6f23873d7f3026caa7504e3ed77c73cb0e33342776815302dad0ac873dea5f`.
Readiness29758990 is still RUN, left untouched under the retire-only-unstarted
constraint and its original two-hour allocation. This is an observed
preparation error/hung stage, not an authenticated final EXIT or a failed
native derivative result. Controller29759027 verifiedPEND; cancellation
requested, final EXIT receipt still to be authenticated. No long arm submitted.

Preparing new immutable source02 with a shared unique node-local/tmp helper
and safe owned-directory cleanup, used by readiness and long launchers.
Scientific src/examples/configuration bytes remain unchanged and will be
byte-map checked against01. Full allocated CPU coverage planned299: original
283+8controller+8TMPDIR/socket/cleanup contracts. No new pass or source02/job
identity is yet claimed. Native retry must wait for its new CPU pass and
ended(29758990), avoiding simultaneous duplicate GPU validation. Preserve
source01/launcher03, their actual283+8passes, all failures and historical
experiments; no scientific gate is waived and no local numerical test runs.

Source02 sealed locally,393files/archive39,241,289bytes SHA
`11ccbe5563bebb3184183eaab194f791ae717c906d8557a3acd11fc0c8c685dc`,
manifestSHA
`5eb6e17ea8d19e52c99a589243228710994dcf137f59a6b9a6659a4fe719a7f4`,
destination `/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-02/source`.
Upload in progress; no completed remote authentication or new job claimed.
Static byte proof authenticated all180files under src/examples, including
configurations, exactly equal to source01. Only batch TMPDIR handling and CPU
coverage plumbing changed. This is an honest external post-packaging receipt,
not an assertion that later receipts are in the already sealed source02.
Future unique run directories:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-02-{cpu,ready,controller,native}`.
CPU 299coverage remains planned/pending, not passed. Preserved failure evidence
download completed: all12files/lengths/SHA verified using local stdlib at
`results/hpc/20261019-brd4-boltz-affinity/tmpdir-failure-01/verification.json`;
its manifest is the recorded ee6f2387...dea5f. No local numerical validation.

### 19 October 2026, 19:25 Europe/Paris — source02 authenticated; gated retry submitted

Remote393source02files, exact archive/manifest and entire180src/examples map
against source01 authenticated. Actual submissions19:25:27.988 Europe/Paris:
**29759068 RUN/hpc CPU**,four slots/4GiB per slot/1h,299checks expected,
not passed yet; **29759069 PEND/gpul40s native**,one exclusiveL40S/four slots/
8GiB per slot/2h,dependency `done(29759068)&&ended(29758990)`;
**29759070 PEND/hpc controller**,dependency
`done(29759069)&&done(29759003)`. Controller reuses authenticated unchanged
launcher03/eight-test proof with new scientific source02 and its CPU proof.
All native, proof, identity, finite, runtime and quota gates remain mandatory.
No native affinity pass is claimed from submissions or earlier CPU success.

Source02 `/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-02/source`
uses recorded manifest5eb6e17e...9a7f4 and archive11ccbe55...685dc. New unique
run family is `/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-02-`:
cpu/ready/controller/native. Readiness payload is `...-02-ready/payload/`.
Remote compact receipt
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-02-campaign.json`;
new versioned local receipt
`scripts/hpc/receipts/20261019-brd4-boltz-affinity-02/campaign.json`.
The02-native long root remains virgin and no long arm is submitted.

Old controller29759027 intentionally cancelled while verifiedPEND, finalEXIT
now authenticated. Predecessor29758990 stillRUN with known preparation error,
left unchanged. Original2h allocation from19:09:51 reaches its approximate
limit21:09:51 Europe/Paris; this is a resource bound, not a completion/retry
ETA. New retry waits ended(old), so no duplicate concurrent GPU validation.
Source01, all earlier failures and actual283+8CPU passes remain preserved.

### 19 October 2026, 19:29 Europe/Paris — source02 CPU 299 passed; native retry still pending

CPU **29759068 DONE/passed299**, actual schedulerDONE and LSF Successfully
completed authenticated at19:29:12 Europe/Paris. Breakdown16controller/TMPDIR
contracts +220legacy/JAXbridge +4nativeTorchmapping +59unchangedchemistry/oracle;
zero failures/errors/skips, three slow/model legacy tests deliberately
deselected. AssessmentSHA
`fadc01aec4cafa54205e5d49ecd764da1aa48ddb841368c0845dac7f0ddb6c6e`.
This supersedes the earlier CPU RUN/pending note, preserving its history;
it is not a native affinity validation result.

Downloaded13CPUproof files safely extracted and exact every-file lengths/SHA,
XML counts and unchanged composition reference verified with local stdlib at
`results/hpc/20261019-brd4-boltz-affinity/retry-02/cpu-final/verification.json`.
Archive12,878bytes SHA
`1d47c5f7c3f577b583e9acde2ad8819ebeb042e185d53cb9c11c57250076d770`,
manifestSHA
`e570b0e37f98e229fcbe95256646c833dd04fa34b59499d7e7fecccd85532553`.
Earlier12control/submission receipts separately byte-verified at
`results/hpc/20261019-brd4-boltz-affinity/retry-02/controls-verification.json`;
archive4,828bytes SHA
`36836d2f0b83a2589fbc1c299a7bf48a2f6788f67eec02913bccc4b0fa80ae4f`,
manifestSHA
`fe2cd588caf1a6bbeb851973534a8751ce53a74c08ac0b05f0cf83e46ea57183`.
Independent static review found no blocking launcher/TMPDIR issue.

Native29759069 stillPEND awaiting ended(predecessor29758990), which remains
RUN and untouched with known preparation error. Controller29759070 PEND;
neither long arm submitted and no native finite-difference/parity/pilot gate
passed. Predecessor21:09:51 allocation bound remains a limit, not an ETA.
All numerical checks were allocatedHPC; local evidence work used stdlib only.

### 19 October 2026, 20:03 Europe/Paris — explicit predecessor cancellation; corrected native RUN

User explicitly requested cancelling errored original validation29758990 so
the corrected queued job need not wait for timeout. Root authenticated RUN,
known AF_UNIX prediction error and absent native assessment, then bkill request
at20:03:16.509435 Europe/Paris (18:03:16.509435UTC). Scheduler acknowledged
being terminated; immediateRUN was asynchronous, not a final outcome. Actual
**29758990 EXIT** confirmed20:03:43.123083 Europe/Paris (18:03:43.123083UTC).
This user authorization supersedes the earlier leave-running restriction
for this errored job only; historical preauthorization notes remain preserved.

Corrected **29759069 RUN/gpul40s**, actualstart20:03:27 Europe/Paris,
n-62-13-19,one exclusiveL40S/four slots/8GiB per slot/two hours. ended(old)
dependency fulfilled, without waiting for old21:09:51 wall limit.
Controller29759070 PEND on actual native success. CPU29759068 actual299pass
unchanged; no native parity/FD/pilot passed and neither long arm submitted.
Original source/failure evidence and all scientific gates remain preserved.

Remote cancellation evidence
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-tmpdir-cancel-01/`,
before/request/cancel.txt/completion.json, manifestSHA
`e5104969dc191a6d134a5141917bf51f9c7228b703d10a2cf333ef4c8a6d6f15`.
Local proof pending download/verification under
`results/hpc/20261019-brd4-boltz-affinity/tmpdir-cancel-01/`.

Cancellation proof subsequently downloaded and verified: all seven files,
exact lengths/SHA and recorded e5104969...d6f15 manifest passed local stdlib
checks at `.../tmpdir-cancel-01/verification.json`. Corrected29759069 startup
uses `/tmp/mosaic-boltz.29759069.szmVwn`; native CLI detected GPU, finished
preprocessing and is running structure prediction. This is preparation
progress only; no native gate passed. Quiet follow-up updated with confirmed
cancellation and actual RUN; no new numerical/model test or setting change.

### 19 October 2026, 20:09 Europe/Paris — source02 native random bond FD blocker

Actual **29759069 EXIT1 at20:09:29**, start20:03:27,362seconds. TMPDIR fix
worked; native structure/affinity prediction had zero failed examples and
prepared the context. This numerical validation failure is distinct from
preserved source01 AF_UNIX preparation failure. CPU 299 pass remains valid.

Mix50 summary status passed: hard/native integer parity exact for all three
outputs; rawX/E finite differences all three scales passed; nonzero invalid
chemistry assay; low/high coupled checks; two disposable ten-update pilots;
runtime fit and frozen model/context identity checks passed. SummarySHA
`9e8121a2c7e3de93ab886244372aadc303928c6ddb2402d797172c1fcd667cee`.

Random hard/native parity and all three atom finite differences passed.
Typed-bond analytic9.497333526611328; centralFD epsilon0.01:
7.599136233329773 failed;0.003:12.296060721079508 failed;0.001:
9.71454381942749 passed. Requirement remains two adjacent passing scales;
only one passed, so native random validation failed. RandomsummarySHA
`bc8c12ea3df2a7455da6cadd84b69f8c7b9bf97ed33947c467d8aba867c87ad0`.
Other random coupled/pilot gates are not certified. Independent static review
found no concrete mapping/transport/backward bug; unresolved numerical gate.
Do not loosen/ignore it, omit random or submit mix50 alone.

Prepared modeldigest
`395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16`,
contexttensor digest
`495002ffa2a52cc85b58ef34424622763dfc4ca556139a98297e8789ae287aa5`,
contextfileSHA
`254d5b7f11605fb41db6f5abf7015f9cc856a19dfe1d4b506ccaea9007200770`.
Overall native assessment absent; both long runs unsubmitted, root virgin.
Controller29759070 PEND on failed native success; verified-unstarted
retirement and full evidence archive receipts still pending. Preserve every
failure/passed arm. Resolution requires human-directed next steps; no native
gate waiver or new molecular submission is authorized by this partial pass.

Controller29759070 subsequently verified unstarted PEND, intentionally
cancelled and finalEXIT confirmed. No source02-native arm root exists.
Full readiness-parent failure archive produced:171files/14,985,443uncompressed
bytes, including prepared native context, raw initializations/reports,
two ten-update pilot trajectories/probes/worker logs and scheduler/controller records.
Archive506,106bytes SHA
`a456234c316b5e534508f972bdc1ab47d4ed5d0431988067df673ec36cebf9b4`;
manifestSHA
`f54279683af6b7641b3c1a8ac6ea4ac957b7490ce8045c43f87cd7446599282a`.
Remote `/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-native-fd-failure-01.tar.gz`
and exact receipt
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-native-fd-failure-01.receipt.json`;
local download/verification was in progress at
`results/hpc/20261019-brd4-boltz-affinity/native-fd-failure-01/`.
Independent FD review found no concrete implementation bug. Coarse-scale
nonlinearity/FP32 explanations remain hypotheses; no probe/gate changes or
pass inferred. Heartbeat being PAUSED for unresolved native blocker, preserving
all evidence and stopping before any long molecular submission.

Full failure archive verification subsequently completed: all171files passed
local stdlib exact archive path/506106byte length/SHA, safe unique member
paths, exact manifest set and every-file length/hash. Both summary hashes/
statuses, random rawFDflags[false,false,true], mix50 passed parity/coupled/
pilot evidence, same frozen model/context and context.pt SHA authenticated.
Proof `results/hpc/20261019-brd4-boltz-affinity/native-fd-failure-01/verification.json`.
This establishes integrity of failure evidence, not a scientific pass.
Automation API confirmed existing finish-combined-brd4-trajectory-report
actuallyPAUSED, no duplicate. Numerical blocker remains unresolved; no user
permission request pending, no gate waiver, CPU 299passed and both long arms
NOTsubmitted.

### 19 October 2026 — user authorizes passed mix50 separately from random diagnostics

After the recorded paired failure, the user explicitly requested starting
the passed 50% mix and investigating random gradients separately. This
supersedes the prior paired-submission hold for mix50 only. Eligibility is
now arm-specific: source02 mix50 passed every native gate, whereas random
remains blocked. Parent 29759069 stays EXIT1, its original overall assessment
remains absent, and all 171 verified failure-evidence files are preserved.
No random gate is ignored or represented as passed.

Preparing an immutable evidence-only tooling release that validates the
requested mix50 configuration and its existing complete native proof,
source02 and CPU 299 identity, then creates an arm-scoped assessment accepted
by the unchanged source02 scientific driver. New tooling tests must pass
on allocated CPU before adoption. Actual release hashes, test outcomes and
new jobs are pending; **no mix50 run is submitted yet**.

The authorized mix50 execution retains exactly 10,000 updates, six neutral
slots, fixed authenticated model/context, original initialization, LR 0.03,
ten-trial best-primal policy, affinity 1/element-JS 0.1 and chemistry 0.001
through 5000 followed by the same geometric ramp to 50 at 10000. One exclusive
L40S, four CPU slots, 8 GiB per slot, 23h; all per-run native/finite/identity/
runtime/terminal gates remain. This is adoption of independently passed
arm evidence, not a new paired pass or a changed scientific objective.

Separate allocated random diagnostic will replay exact failed
GraphLogits(saved_logits).graph bytes, not raw sampled X/E; fixed bond row 2
aromatic 4-to-NONE 0 contrast; original eps 0.01/0.003/0.001, 15% tolerance,
two-adjacent-pass gate unchanged. Repeat no-grad/AD/grad-enabled forward
and checkpoint-disabled diagnostic control. Extra 0.0003/0.0001 scales are
diagnostic only, not historical certification or production settings.
No production change selected. See brd4_boltz_random_gradient_diagnostic.md.
All testing/diagnostics stay allocated HPC; no local numerical execution.

Independent external tooling release01 authenticated all seven files at
`/work3/s234818/mosaic/releases/20261019-brd4-boltz-independent-01/source`,
manifest SHA
`37d40949589c8d2d5e823fa00443345f24b791fcb6ad5ae501724001c5cc50a1`,
archive 17,945 bytes, SHA
`ddcf71b863750aa92f90ba9ebfcb0871e1b9de5097beac71d3bd850828497ab0`.
Source02 scientific code/configuration remains unchanged. Independent static
reviews of evidence-adoption and random diagnostic tools found no blocker;
these are review findings, not a CPU/native pass.

Actual allocated CPU **29759210** submitted to hpc, one slot, 2 GiB, 15 minutes;
output `/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-independent-01-cpu/`.
Planned 19 contracts comprise 13 arm-adoption and six diagnostic contracts.
No passed test result or new GPU job is yet claimed. Both mix50 long-run
submission and random diagnostic GPU submission remain pending these checks.
This receipt follows immutable tooling packaging and is not claimed within it.

CPU29759210 subsequently EXIT from an engineering packaging failure:
external tools01 omitted scripts/check_boltz_cpu_proof.py. Thirteen
arm-adoption fixture tests raised FileNotFoundError; six diagnostic contracts
passed. Arm adoption did not execute and no GPU job was submitted. Preserve
this actual failed job, logs and sealed tools01 rather than replacing its
outcome. Preparing external tools02 with only the unchanged verifier added;
existing tests/tooling and source02 scientific code remain unchanged. New
tooling hashes, CPU retry and outcomes are pending. Historical CPU 299 and
mix50 native validation remain valid; random is still unapproved.

Corrected external tools02 remotely authenticated all eight files at
`/work3/s234818/mosaic/releases/20261019-brd4-boltz-independent-02/source`;
same seven prior file bytes plus unchanged CPU verifier dependency.
Manifest SHA
`848cb213d91be09513439f719ece8e43ec55991bd473f1bc233f3d80eb204c1b`,
archive 18,486 bytes SHA
`7d5516aae2fa6bb0a8fee2625258fe86d32049d207f2424698f7f7bba8adaa2d`.
Actual CPU 29759211 submitted, hpc/one slot/2GiB/15min, output
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-independent-02-cpu/`.
Outcome pending; no new pass/adoption/GPU submission claimed.

### 19 October 2026, 20:40 Europe/Paris — arm adoption passed; mix50 and diagnostic submitted

Actual tools02 CPU 29759211 DONE/passed 19 contracts, zero failures/errors/skips,
authenticated scheduler/LSF success. Adoption rehashed all 147 original mix50
proofs. CPU assessment SHA
`bc76e33927104e88a9e019b706d0ed4c6e1fc8f3bd4460264acad13ca7594448`;
mix50 arm-scoped adoption SHA
`ca400a1477471790bcc5c93cd85a55455da77b331caa504f882b2020d954140e`.
Source02 scientific CPU 299 pass unchanged, original overall parent 29759069
stays EXIT1, random native failure remains uncertified.

Actual GPU submissions 20:40:18 Europe/Paris, initially verified PEND:
mix50 long **29759212**, output
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-mix50-independent-01/`,
unchanged source02/original mix50 configuration/10,000 updates, gpul40s one
exclusive L40S/four CPU slots/8GiB per slot/23h; random diagnostic
**29759213**, output
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-random-diagnostic-01/`,
same GPU/CPU/memory shape and 1h. Neither has a dependency on the other or
an initial dependency. Actual records saved locally at
`results/hpc/20261019-brd4-boltz-affinity/independent-02/gpu-submissions.json`.
No starts/completed updates/diagnostic success inferred. Random diagnostic
is not a long optimization; no long random run submitted.

At 20:41:42 Europe/Paris, actual scheduler **mix 29759212 RUN** from 20:41:13;
diagnostic 29759213 PEND on unavailable physical-GPU reservation,
ngpus_physical across three hosts, with no dependency on mix50. Random cause
remains unproven and its original gate remains failed.

Compact 30-file CPU/adoption/submission proof archive downloaded, safely
extracted and every-file lengths/SHA plus XML counts verified locally using
stdlib at `results/hpc/20261019-brd4-boltz-affinity/independent-02/proofs-01/verification.json`.
Archive 126,516 bytes SHA
`414dfa385545b8f7e3dc751fbbd4bd964f3e7926ad6feb48f23fafa9de164546`,
manifest SHA
`8e645c895fb51178ff96e25a732dbe077b52281bff91e4f4528644612f00462b`.
Proof integrity is not a long-run completion or diagnostic pass.

At 20:42:41 Europe/Paris, actual mix 29759212 RUN; its repeated initial source02
raw native FD report status passed. No completed trajectory/terminal check
claimed. Existing heartbeat finish-combined-brd4-trajectory-report updated by
automation API to actual ACTIVE, no duplicate, following unchanged mix plus
separate random investigation. Quiet on unchanged status/outages; concrete
bug fixes require authenticated allocated tests and every original gate.
Pause for a genuine unresolved curvature blocker; no tolerance/protocol
change selected. Latest diagnostic observation remains 20:41:42 GPU PEND.

### 2026-10-19 20:44 Europe/Paris — mix50 applied updates authenticated

Compact scheduler/artifact query at20:44:01 confirmed mix50 job29759212 RUN,
its repeated original native atom/bond FD report passed, case report `running`,
41 applied updates and42 saved states through state41. Terminal derivative,
10,000-update and frozen-after gates remain pending. Random diagnostic29759213
remained GPU-PEND for `ngpus_physical`; no random long run is submitted.
Saved query: `results/hpc/20261019-brd4-boltz-affinity/independent-02/latest-status.txt`.

### 2026-10-19 21:39 Europe/Paris — status follow-up inaccessible

One compact SSH scheduler/artifact query through `dtu-hpc` returned exit255:
`Permission denied (password)` for `s234818@login1.hpc.dtu.dk`. No scheduler
or new artifact outcome could be authenticated for mix29759212 or diagnostic
29759213. Their previous RUN/PEND observations remain last-known states only.
No jobs, scientific settings or gates were changed, and no second connection
was attempted. Existing quiet follow-up remains active. Local access receipt:
`results/hpc/20261019-brd4-boltz-affinity/independent-02/followups/20261019T1939Z-access.json`.
