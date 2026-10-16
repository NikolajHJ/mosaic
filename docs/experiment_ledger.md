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
