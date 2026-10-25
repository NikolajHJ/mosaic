# Experiment ledger

Retain submitted, completed, failed, and incomplete attempts with their source/config/input identities and artifact paths. A submission or successful dependency installation is not a completed model experiment. Never turn a skipped check into acceptance evidence.

## Required agent logbook workflow

This ledger is the **single repository-wide experiment logbook**. Follow the
[experiment folder policy](../experiments/README.md#agent-logbook-and-run-records)
and [root agent instructions](../AGENTS.md). Before a new attempt, create its
experiment README and `runs/<run-id>/README.md`. Append a dated ledger event
at preparation, submission, observed start, terminal outcome and handoff, linking
the experiment folder, run record, exact commands/source identities, job IDs
and saved evidence. Use explicit timezones and retain unsuccessful attempts.
Keep execution, scientific acceptance and report status separate. Record unknown
times/states honestly; documentation-only work must say no experiment executed.
Legacy entries and receipt paths remain authoritative history; migration must
link them rather than rewrite them into the new format.

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
the campaign pins. Final repair coefficient 50.0. Summary SHA
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

### 2026-10-19 21:41 Europe/Paris — access restored; mix ETA from saved timings

One compact SSH query at21:41:46 authenticated mix29759212 RUN and its case
`running`, with1,742 saved applied updates of10,000. Random diagnostic29759213
is still PEND on physical-GPU reservation across three hosts; it has not
started and no queue-start ETA is available. No jobs or scientific settings
changed. The last30 saved update times average 4.394seconds/update,
projecting about10hours remaining (approximately08:00 on20October Paris time,
rounded to allow final checks/record I/O). The complete-prefix mean is faster;
recent behavior is the basis of this operational ETA. Repair-search/trial
cost can change, especially after the weight ramp starts at5000; no completion
or terminal-native pass is implied. Local stdlib timing/schema receipt:
`results/hpc/20261019-brd4-boltz-affinity/independent-02/followups/20261019T1941Z-status-eta.json`.

### 2026-10-20 00:35 Europe/Paris — live mix progress and queued diagnostic

One successful compact SSH query at00:35:04 authenticated mix29759212 RUN,
case status `running` and3,970 applied updates of10,000 (last update step3969).
Random diagnostic29759213 remains PEND for physical-GPU reservation across
two hosts; it has not started. No jobs or scientific settings changed. The
last50 saved update times average 4.480seconds/update, projecting
about7.5hours remaining and a finish around08:00–08:30 Paris time on20October,
including a small allowance for terminal checks. This remains an estimate:
chemistry-ramp/trial/search behavior may change after5000. No completed
trajectory or terminal pass is inferred. Local operational timing receipt:
`results/hpc/20261019-brd4-boltz-affinity/independent-02/followups/20261020T0035-status-eta.json`.

### 20 October 2026, 08:59 Europe/Paris — mix scheduler completion; diagnostic still queued

One successful SSH status check at 08:59:37 authenticated mix 29759212 DONE
and actual LSF Successfully completed. Started 19 October 20:41:13, ended
20 October 08:35:34, elapsed 42,861 seconds. Persisted summary/case status passed,
10,000 updates indexed 0..9999, final chemistry coefficient 50, terminal raw
atom and typed-bond finite differences passed all three scales, and original
frozen model/context digests retained. The 10,001 contiguous saved states
and compact downloaded proof bundle are still being authenticated; no local
verification pass is claimed at this stage.

Random diagnostic 29759213 remains PEND for physical-GPU reservation on one
host and has not started. Its numerical failure remains unresolved; no long
random optimization submitted or settings/gates changed. Historical source02
CPU 299 and external tooling CPU 19 passes remain valid. Preserve all earlier
failed evidence and distinguish this actual remote completion from pending
saved-state/local proof verification.

### 20 October 2026 — mix50 completion metadata and compact evidence authenticated

Mix 29759212 completed exactly 10,000 contiguous updates 0..9999 and 10,001
contiguous saved states 0..10000; update JSONL, case report and summary case
records agree, N=6. Source02 file map/manifest match the sealed release.
Configuration SHA
`97faf3b192a7aca566ad86752c2f3efacad69e89b076c88e67e8d14b5435de0e`;
summary SHA
`7f49087f12efc762a46691581f80bc42c43395c00b0cc4778a9e04a6679aae7c`.
Initial/coupled/terminal checks passed; original frozen model/context digests
retained; final chemistry coefficient 50. All 10,000 applied updates recorded
strict improving, zero fallback/no-op, evaluated within each state's fixed
surrogate. Changing repair targets/coefficients preclude treating total loss
as one globally fixed objective.

Compact 21-file evidence downloaded and locally verified using
scripts/verify_hpc_evidence.py, safe extraction, every-byte/path/manifest and
schema checks. Includes snapshot bytes only at 0/5000/10000. Proof
`results/hpc/20261019-brd4-boltz-affinity/mix50-completion-02/verification.json`.
This verifies completion metadata and selected snapshot bytes, not all 10,001
snapshot files or a full rendered report. Remote
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-mix50-completion-02.tar.gz`,
21,533,349 bytes SHA
`0b79afbf9f1ecc788d9111b58cb1677bb6a75cb9adc3cf891ec81542c2cd8764`,
manifest SHA
`4d1d19a8a30a9cc485d5f94e84f49b1d1cfe7dd979e924d3ea3cc385857559aa`.
First completion01 packaging helper failed on unavailable hashlib.file_digest
in system Python; preserved its folder and all scientific artifacts unchanged.
Unique02 helper used standard streaming SHA. No experiment/model/tests rerun.
Latest random diagnostic remains 08:59:37 PEND, unstarted and uncertified.

Existing quiet heartbeat was updated through the automation API and remains
ACTIVE for the pending random diagnostic/investigation only; completed mix50
is preserved. The exact successful stdlib completion-audit helper is archived
with its hash in `scripts/hpc/receipts/20261020-brd4-boltz-mix50-completion-02/`.

### 20 October 2026 — new standalone Boltz mix50 report authorized

User requested a new interactive report for completed native 29759212,
10,000 updates/10,001 states, in a fresh report family. Preparing saved-only
allocated CPU tests/rendering and full snapshot authentication; no new
report job submitted or render pass claimed yet. Shared viewer gains an
affinity primary metric while preserving default contact behavior for old
callers. Binder probability stays separate. Existing reports/servers and
native experiments remain unchanged, no Protenix overlay or invented random
selector. Exact saved probabilities, heatmaps, readable labels/sizes and raw
invalid final graph (chlorine valence 2, canonical SMILES null) are required.
Changing repair-target totals are not one fixed objective; fixed native pose
affinity is a partial-gradient surrogate, not fresh-pose or experimental binding proof.
See brd4_boltz_affinity_report.md. Source/job/test/artifact/URL identities
remain pending actual execution and verified browser delivery.

### 20 October 2026, 09:14 Europe/Paris — report tooling authenticated; CPU report submitted

Authenticated all 12 report tools01 files at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-report-01/source`,
manifest SHA
`7cb5b82c04634f3b3b276943d8d7988bc29b698df2e315bf12ea7915ec82bf65`,
archive 34,412 bytes SHA
`f9b2bca789af2a455863c66b37bfbf1761dea94cf90ee3aeea647dddf35a2537`.
Scientific source02/configuration remains unchanged, 393-file source identity
required by entry gate. Saved-only tests/rendering remain allocated HPC;
no inference/chemistry/model execution or native rerun for this report.

Actual report CPU 29760530 submitted 09:14:27 Europe/Paris, hpc/two slots/
4GiB per slot/60minutes/zero GPU, initial verified PEND. Output
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-mix50-report-01/payload/`.
Expected 35 tests: 26 new fixtures+nine old contact/ramp; actual outcomes/counts,
render/archive/browser URL pending. Home 14.30/30GiB observed 09:04; 8 GiB
job reserve gate retained. Submission does not certify tests/render/quota.

### 20 October 2026, 09:15 Europe/Paris — report tests passed; render still running

Actual report 29760530 RUN from 09:14:30. Query 09:15:15 authenticated 35 passed
in 26.39 seconds: 26 new fixtures+three existing chemistry+six existing ramp
tests, zero failures/errors/skips. Entry native-proof/source/fresh quota
gates passed before pytest. Renderer is validating all 10,001 saved snapshots,
with no report completion receipt or packaging outcome yet. No model
inference or molecular experiment rerun; all tests/rendering allocated CPU.
Tools01 twelve-file source/manifest remains unchanged. Preserve distinction
between passed engineering tests and an uncompleted full report/delivery.

### 20 October 2026, 09:16 Europe/Paris — allocated report DONE; download verification pending

Actual report 29760530 DONE 09:16:29, started 09:14:30, 119 seconds; scheduler DONE
and LSF Successfully completed authenticated. All 35 tests passed, zero
failure/error/skip; assessment SHA
`ff68b20b5264f89e5a9c79fec921c11830cce3104c225cfcdb04c4cf030fb1a1`.
Actual renderer validated all 10,001 saved snapshots and 147 readiness proof
hashes. Complete receipt has primary metric affinity, input jobs [29759212]
and state counts [10001]. Scientific source/config/native artifacts unchanged,
no inference or model/chemistry experiment rerun.

Remote archive
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-mix50-report-01.tar.gz`,
sibling .tar.receipt.json, 24,432,529 bytes/451 files SHA
`7bb0849eff0daeaa2296f50b1c7516e751404c1cd5825f3d57ad54f3ed266a48`.
HTML 40,089,420 bytes SHA
`57759bd10fb79a5e142f2757c8d8df86c3dc7522c2f39408d9ed241f89a03974`.
Download/local archive verification in progress; no verified local report,
browser pass or working URL claimed yet. Existing reports/servers preserved.

### 20 October 2026 — report download verified and viewer published; browser QA partial

Report 29760530 download completed at
`results/hpc/20261020-brd4-boltz-affinity-report/edition-01/`.
scripts/verify_hpc_evidence.py plus stdlib checks passed exact archive path/
24,432,529-byte length/SHA 7bb0849e...66a48, safe extraction and all 451 files.
Manifest SHA
`b32f255b2937f3244d7fe1337fcfee63206e145e4e02fadf1e0e6479712accce`.
HTML 40,089,420 bytes matches SHA 57759bd1...03974. Published verified artifacts
HTML last into the fresh family's viewer and started detached loopback 8774.
Actual HTTP 200/exact served length/SHA passed at
**http://127.0.0.1:8774/report.html**; old report families/servers preserved.

Opened actual report in the in-app browser. States 0/10000 match; raw final
invalid chlorine-valence-2 graph remains visible with readable Cl1/C2/C3/C4/
C5/S6 labels and probability-dependent sizes. Heatmap/playback/coefficient-
ramp browser checks are underway, so complete browser QA is not yet claimed.
No local numerical/model/chemistry execution or native rerun.

### 20 October 2026 — full browser QA passed; standalone Boltz report delivered

Actual in-app browser report QA passed: exactly one correct mix50 selector;
state 0 `CCCCCC`; state 5000 coefficient 0.0010021663 with previous applied 0.001;
state 10000 weight 50/raw chlorine-valence-2 invalid graph/null canonical SMILES;
saved heatmaps six atom rows/15 unordered pairs; Play advanced 0 to 22 and Pause
worked. Cl1/C2/C3/C4/C5/S6 labels and probability-dependent sizes visually
readable. Report left paused at 10000 and marked deliverable.

Verified standalone report **http://127.0.0.1:8774/report.html** is published
from the new family, backed by authenticated report 29760530/35 tests and the
verified 451-file archive/HTML hash recorded above. Existing reports/servers,
native runs and every other experiment remain unchanged. Full browser QA
supersedes the earlier partial-QA note; no local numerical/model rerun.

Quantitative affinity, separate binder probability and raw/weighted repair
overlay plots were also visually checked and readable in the actual browser.

### 20 October 2026 — saved-only Boltz balance review authorized and allocated

User requested recommendations on LR 0.1, objective balance, update budget
and earlier/gentler chemistry. Saved-only analysis 29760613 is allocated in
hpc, one CPU/2GiB/10minutes/zero GPU, reading source02 run 29759212 metadata and
selected snapshots. Actual output/outcome/counts/minima are pending; no new
molecular experiment, model/chemistry rerun, local numerical test or scientific
configuration/report change is claimed.

Immutable source
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-balance-01/source`,
archive SHA
`d01c08b88c9f7d0abdd2b80069db9c95722ad24a4ed47922fe26ab168c05669b`,
analyzer SHA
`3a1edf8959b0bd7cdfa0dd142c1a6c51a55b300c0577feff1e9a7850d22594dc`,
wrapper SHA
`95c80c0abab4d3382acf79d380e4c585ef25e5f13079d8ad8cb10a29009226fd`.
See brd4_boltz_balance_review.md for the pending-evidence review.

Native source review identifies declared route 0 already_valid/skip_without_
self_target with chemistry loss/gradient zero, complete route 1 atom+bond NLL,
and local valence-risk guidance only on fallback 3/4. Saved states 9998/9999
valid then 10000 invalid are consistent with affinity-only accepted movement
across a hard chlorine-edge boundary; this is not a proven derivative bug.
Proposals only: retain N6/init/model/context/seeds and 10,000 updates;
earlier 0.001-to-5 ramp over first 8,000 then hold 2,000; LR 0.1 with matched 0.03
comparison desirable; prioritize explicitly new margin/hysteresis or
continuous feasibility guidance requiring allocated tests/native gates.
No coefficient/tolerance/model/optimizer change or new submission selected.

### 20 October 2026, 09:34 Europe/Paris — saved-only balance analysis completed and verified

Analysis 29760613 DONE/recorded; submitted 09:34:00, start 09:34:02/end 09:34:06
on n-62-11-61, 4 seconds. LSF Successfully completed authenticated. All five
proof files passed local stdlib bytes/schema verification at
`results/hpc/20261020-brd4-boltz-balance-01/verification.json`.
Analysis JSON SHA
`5e93dabedf857d359492e94d532564dbc4665774f72dd26a1612a6e7c9d2a750`.
No inference, chemistry evaluation, optimization or local numerical test.

Best relaxed affinity state 8294 = -0.1035926044, hard invalid. Best saved
connected/sanitized hard readout by associated relaxed score state 9639,
`SCCCCCl`, 0.5465328693; not a fresh score of that hard ligand. Late states
9500..10000 inclusive 501 states/218 valid/280 validity switches. States 9998/
9999 valid with chemistry 0, then 10000 invalid chlorine valence 2 with raw
chemistry 0.1123773083/weight 50/weighted 5.6188654155. All 10,000 updates strict
within-state improving, zero fallback/no-op. Accepted rates 0.03:2431,
0.015:1439,0.0075:4883,0.00375:1175,0.001875:72. Selected total-minus-native
logit gradient norm ratio 0.0284 at 5000, 1.1800 at 8294, 3.5324 at 10000 with
cosine -0.8692; at valid 9999 ratio 0.002499. This residual includes weighted
chemistry plus composition, not pure chemistry.

Source-reviewed declared zero-on-valid route explains absent chemistry
guidance at a valid boundary; no derivative bug proven. Recommendations only:
new declared validity-stability signal with native tests/gates; main schedule
proposal linear chemistry 0.01-to-5 over applied updates 1..8000, then 5 over
8001..10000; 0.001-to-5 geometric only an optional conservative comparator.
The linear candidate is stronger early despite its lower endpoint, and is
untested. Max LR 0.1/backtracking with matched
0.03 desirable; retain 10,000 updates/N6/init/model/context/seeds/affinity 1/JS 0.1.
No scientific config implemented or new optimization submitted. Full exact
evidence and caveats in brd4_boltz_balance_review.md; official Boltz docs
separate active-ligand affinity ranking from binder/decoy classification.

### 20 October 2026 — valid-state chemistry investigation prepared; HPC access unavailable

User withdrew the LR 0.1 proposal and requested useful chemistry gradients
on valid molecules while retaining LR 0.03; 0.01 is only a possible future
comparison. Draft design compares always-on hard-cap DP, atom-probability-
weighted local compatibility DP and valid-target bounded margins. No
production loss/configuration/default or molecular experiment changed.

Exactly one SSH status/quota attempt used BatchMode=yes/ConnectTimeout=20,
emitted the login banner and then failed Connection timed out to
192.38.95.132 port 22. No quota result, upload or allocated job occurred;
no retry. This is an access failure, not a failed scientific test. Local
work continues as static prototype/test authoring and immutable packaging
only; numerical comparison remains pending and release hashes are not
yet recorded. The prior saved analysis 29760613 remains verified.

Working design preference is soft atom-probability-weighted local
compatibility, tentatively because it supplies both atom/bond gradients
without a hard argmax-cap switch. No measured superiority or pass claimed;
allocated checks and sulfur/aromatic/connectivity limitations remain.
See brd4_continuous_chemistry.md; no new molecular run submitted.

### 20 October 2026 — continuous chemistry research source sealed locally; validation pending

Finished the isolated soft local compatibility prototype, hard-cap and
bounded valid-target margin comparators, saved-gradient probe driver and
11 planned engineering tests. Production `src/`, configurations, v1 loss
and LR 0.03 remain unchanged; optional 0.01 is a future control, not a
submitted setting. Ruff/AST/Bash syntax and independent static review passed.
No local numerical tests, new chemistry evaluation, model execution or
optimization occurred. HPC remains inaccessible after the single recorded
SSH attempt; no further connection, upload or submission was attempted.

Local four-file release
`results/releases/20261020-brd4-chemistry-compatibility-probe-01/source.tar.gz`,
8,592 bytes SHA
`edd16a3aafa5092e2decea16f13c9ba3d9575de37edfd6d54db7c4e61f0ff489`,
manifest SHA
`53fb385712f37412f204018e49dfb0f4c6a2a9adde14ff3e5b9e8760d70fea29`.
Campaign receipt has exact per-file pins and the access failure. Initial
local timestamp metadata generation lacked tzdata; the existing source and
archive were preserved and the receipt was completed with a stdlib UTC
timestamp. This was packaging metadata, not a scientific failure.

Planned saved-state probes explicitly distinguish unaccepted perturbations
using stored native gradients from actual reevaluated Boltz scores or native
readiness. The proposed separately weighted compatibility signal still
needs allocated enumeration/FD/invariance tests, saved-state measurements
and fresh native/coupled checks before any production adoption. Sulfur bias,
aromatic approximation and lack of connectivity guarantees remain explicit.

### 20 October 2026, 17:15 Europe/Paris — access restored; random diagnostic launcher EXIT1

Successful SSH at 17:15:21 authenticated random diagnostic 29759213's actual
run 09:52:19–09:52:22 on n-62-13-16, EXIT1 before model execution. A launcher
assertion queried old CPU 29759211 live after it aged out of the scheduler
window. Its actual 19-test pass and persisted final/LSF records remain
preserved. No new random native derivative/control result exists; original
source02 random FD failure remains unresolved and no long random run is
submitted. This is a launcher evidence-authentication failure, not a new
scientific gate outcome.

User explicitly authorized launcher-only repair using hashed persisted
actual scheduler-final/LSF-success records, not fabricated DONE. Preserving
the failed release/job/logs and preparing a new immutable tooling retry
with allocated tests. Source02/init/graph/context/original FD scales,
tolerances and gates unchanged. Exact downloaded failure hashes, new pins,
test outcomes and retry IDs remain pending. Separate compatibility-research
prototype/outage history is preserved; no compatibility numerical result
or production-loss change is implied by restored access.

### 20 October 2026 — launcher failure evidence verified; launcher03 CPU submitted

Original LSF EXIT1 footer/logs authenticate 29759213's failed launch.
bhist/bacct commands were permission-denied; no accounting history or DONE
fabricated. Capture01 stopped on bhist PermissionError and is preserved;
unique02 records unavailable accounting plus actual LSF outcome. Failure
archive `/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-diagnostic-launch-failure-02.tar.gz`,
5,604 bytes SHA
`c4b265ab600840b0b0a754e5ac10a8c8c5f37c647f82821a1523a281045b4323`,
manifest SHA
`14f8c555993de01165254d6b9bab2fc5b2f8a2384c5923ffdfa18e2dbf75121b`.
All 11 files downloaded/safely extracted/byte-verified at
`results/hpc/20261020-brd4-boltz-diagnostic-launch-failure-02/`.

Authenticated all seven launcher03 files at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-diagnostic-launch-03/source`,
manifest SHA
`dceadf75e4a0ae84e41cd325c5205c4f4d8d54ee43ba735f660bc4e1ca38f2c2`,
archive 13,945 bytes SHA
`8cf264153647df1149f79e6cf5cc05933b2c509c4ae8515dc3947392b9adfe17`.
Two independent static reviewers found no blocker in proof helper/wrappers;
original diagnostic code/scientific source02 unchanged. Actual CPU 29763673
submitted hpc/one slot/2GiB/15min, 19 expected contracts (13 new proof+six
unchanged diagnostic). Actual outcome pending; no GPU retry submitted.
Original random derivative failure/gates and compatibility research history
remain preserved; engineering proof checks are not native numerical passes.

### 20 October 2026, 17:20 Europe/Paris — launcher03 CPU rejected reserve before tests

Actual CPU 29763673 EXIT1 at 17:20:21 before any of the 19 planned tests or
native diagnostic/model. zhome_quota_preflight rejected requested 1 GiB
home-storage reserve because minimum 4 GiB is mandatory. This is a wrapper
configuration failure, not 19 test failures or measured storage exhaustion.
Preserve sealed launcher03/failed job/logs and the unchanged gate.

Preparing unique launcher04 changing only CPU wrapper reserve to 4 GiB,
followed by new allocated CPU checks. New hashes/jobs/outcomes pending;
original diagnosis/science02/native FD requirements unchanged. No pass,
GPU retry or long random optimization submitted from this failed stage.

### 20 October 2026 — launcher04 authenticated; CPU retry submitted

Authenticated all seven files at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-diagnostic-launch-04/source`.
Six files byte-identical 03; only a single 1-to-4 reserve character changes
in the unchanged-length 2,925-byte CPU wrapper. Manifest SHA
`2a35a34f8db8c5682139f341ec08e08da1a049d8da1cf0423ef9e5f3b4ec2754`,
archive 13,992 bytes SHA
`90483713ead73297b9481b1aced3978a84b1a20f1c7c78341c02704d13772457`.
Actual CPU 29763686 submitted hpc/one slot/2GiB/15min, unique output
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-diagnostic-launch-04-cpu/`.
Expected 19 contracts, actual outcome pending. Original CPU 29763673 actual
scheduler-final EXIT1 captured and preserved. No GPU retry or native result
from this submission; all diagnostic/scientific bytes and gates unchanged.

### 20 October 2026, 17:23 Europe/Paris — CPU04 passed 19; status-wrap fix prepared

Actual CPU 29763686 DONE 17:23:10, 19 passed in 1.13 seconds; fresh 4 GiB reserve
and LSF Successfully completed authenticated. Preserve this actual source04
engineering pass. Before GPU certification, inspection found bjobs wrapped
its real status as `Status<D\nONE>`; a literal DONE parser would reject it.
No GPU certification failure/model result is inferred.

Preparing unique source05 with whitespace normalization inside status and
queue fields, preserving exact canonical DONE/hpc requirements. Add one
contract, 20 expected tests, not yet executed. New pins/jobs/outcomes pending;
no GPU retry submitted and no change to science02/diagnostic/native gates.

### 20 October 2026 — launcher05 CPU 20 passed and proof bytes verified

Authenticated all seven source05 files at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-diagnostic-launch-05/source`,
manifest SHA
`e14c41ada91a0a1750776d57affbb33edb782f6d05cacd3519fa543b338960be`,
archive 14,124 bytes SHA
`bd70f72ed4814e2627371b202db5927c5b1034c4b879676f82fa3522bf5c721d`.
Actual CPU 29763691 DONE/passed 20 in 0.95 seconds, no failures/errors/skips;
scheduler/LSF/assessment/XML authenticated. Quota 15 GiB conservatively free.
Assessment SHA
`36857a6fc0dccef932ffe26aac9b742a5fe1291aec35884a7aa7bb6ca38a8702`.

All 13 CPU05 proof files verified at
`results/hpc/20261020-brd4-boltz-diagnostic-launch-05/cpu-proofs/verification.json`;
archive 6,510 bytes SHA
`e9c03d3200c97de550a89e3f8be97fc3ee0f1eb264d3e2aafd9c32d279088fd7`,
manifest SHA
`604c444d05e899296ecc95f7d766e90bbc760c394070150a1f99805369de0530`.
Earlier actual CPU04 pass also has 13 verified files, archive 6,465 bytes SHA
`a9296679b4d0fb703a127dfdab2cde3c77d8e1f56e7e838dd4138527b529cb1e`,
manifest SHA
`fe27760bafd9b965a0c00eb281bb14a9fe149903fc7caa634a41746e7ecab8f9`.
Original failures and scientific/diagnostic bytes/gates preserved. GPU02
preparation underway, not yet submitted; no native derivative pass inferred.

### 20 October 2026, 17:25 Europe/Paris — replacement random diagnostic actually submitted

Actual GPU diagnostic 29763693 submitted 17:25:41, initial bjobs PEND waiting
for scheduling. Verified resource arguments and scheduler state: gpul40s,
four CPU slots/8GiB per slot/one exclusive L40S GPU/60min, no inter-arm
dependency. Fresh quota 15 GiB conservatively free. Launcher05 CPU 29763691
actual 20-test pass comprises 14 new proof contracts and six unchanged
diagnostic contracts; independent static review found no blocker.

Output `/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-random-diagnostic-02/`.
Actual submission record
`results/hpc/20261020-brd4-boltz-diagnostic-launch-05/gpu-submission/submission.json`.
Original 29759213/CPU03 failure/CPU04 pass and original random FD failure
remain preserved. Science02/init/context/diagnostic bytes/native gates
unchanged. No new GPU start/result or native pass is inferred. No long
random optimization submitted; completed mix50 and verified report unchanged.

Final compact scheduler check 29763693 remained PEND, with four CPU slots
reserved on n-62-13-18 at 17:25:43 and waiting ngpus_physical on one host.
Valid queue/resources, no actual GPU start or start ETA inferred. Versioned
receipt `scripts/hpc/receipts/20261020-brd4-boltz-diagnostic-launch-05/campaign.json`.
Automation API updated existing finish-combined-brd4-trajectory-report to
actual ACTIVE following 29763693/current05 receipt; no duplicate. Separate
full mix50 report already verified 10,001 frames and browser QA at 8774 is
unchanged; no random long run or new native pass.

### 20 October 2026, 18:05 Europe/Paris — replacement diagnostic verifier EXIT1 before model

Actual 29763693 RUN 18:05:00 on n-62-13-18, EXIT1 at 18:05:05. Original CPU19
persisted-proof verification passed first. Source05 CPU XML declared-suite-
count versus test-case-record comparison then rejected the evidence; exact
format cause is being investigated. CPU 29763691 actual 20-test pass remains
preserved. No model loading, graph replay, checkpoint control or finite
differences reached. This is engineering proof verification, not a new
scientific gradient failure or pass.

Preserve all failure logs/source/job and original random native FD failure.
Failure archive/fix pins/new tests or retry outcomes pending. Science02/
initialization/context/native gates unchanged, random long unsubmitted,
completed mix50 and report unchanged. No new native result inferred.

### 20 October 2026 — subTest XML counter cause confirmed; strict source06 fix prepared

CPU05 raw XML has 20 passed test-case records but suite count 31, inflated by
11 successful unittest.subTest events under pytest 9. Preserve actual CPU05
outcome/cases and original inconsistent XML; strict verifier remains unchanged.
Source06 removes three subTest contexts while retaining all 11 variants/
assertions, adds inflated-counter rejection regression, 21 tests planned.
Producer will enforce strict XML counters before assessment. A separate
allocated post-DONE CPU preflight executes the exact GPU consumer on original
19/new 21 proofs with actual final scheduler/LSF pins before requesting GPU.
Pins/jobs/outcomes pending; no science/init/diagnostic/native gate changes.

All 12 failed-GPU evidence files downloaded/verified at
`results/hpc/20261020-brd4-boltz-diagnostic-launch-05/gpu-failure-01/verification.json`.
Remote `/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-diagnostic-launch-05-failure-01.tar.gz`,
8,437 bytes SHA
`afe2ab01bfdbdaf8e5d19d813a6b86a88a5bde272bfccdc796df8258ac93fd86`,
manifest SHA
`21df0ff070322c2281b901118e149331746f0a743e914679ae9584dd304992d1`.
Byte verification authenticates this engineering failure, not a native pass.

### 20 October 2026, 18:16 Europe/Paris — source06 CPU 21 passed; exact-consumer certificate submitted

Authenticated all eight source06 files at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-diagnostic-launch-06/source`,
manifest SHA
`ec45d67b99f8e1645bfca1b228a2ce7b706edb4d9312886a08d74549aa138a4a`,
archive 14,708 bytes SHA
`fdbd19455b342aeafff6e5d2b8bca9b9eaa86b7e9533efe0745a0d92a91e53f4`.
Actual CPU 29763944 DONE 18:15:38, 21 passed in 1.68 seconds (15 proof+six unchanged
diagnostic), raw XML suite21/cases21 zero errors. Independent review no blocker.
Assessment SHA
`8606e593748b626432498ba9ce7ee0a20b08f4d63092bf6f0ee27eaf76d4b076`.
All 13 proof files downloaded/verified at
`results/hpc/20261020-brd4-boltz-diagnostic-launch-06/cpu-proofs/verification.json`,
archive 6,598 bytes SHA
`b83ecdbc92ff116b2bef6d09d79e57ad7ec013b4d11658f7a3bec2bf66cba186`,
manifest SHA
`976383f3b2c5cc283ac81ebfb3ec085668bf2ff79d819ab0bfbd11b1cea0d0bc`.

Actual post-DONE certificate CPU 29763946 submitted 18:16:27, hpc/one slot/
2GiB/15min, output source06-certificate-cpu family. Must actually DONE and
pass exact GPU consumer against original 19/new 21 actual evidence before
GPU request. Outcome pending, no GPU retry03. Original strict native gates,
science/diagnostic/init/context unchanged; no new gradient pass inferred.

### 20 October 2026, 18:17 Europe/Paris — exact-consumer CPU passed; diagnostic03 submitted

Post-DONE CPU 29763946 actually DONE/passed original 19+new 21 exact GPU-consumer
certificates; scheduler/LSF/assessment/proofs authenticated. No additional
unit/model tests or native FD executed. Assessment SHA
`664e0ac8e95f67ac7766e46b3e99b0a0e45e512f27695e61ba3e8b000e51d47d`.
All 10 files verified at
`results/hpc/20261020-brd4-boltz-diagnostic-launch-06/certificate-proofs/verification.json`;
archive 5,997 bytes SHA
`2f5320d7c63999c1b1a5fc2558b568117a54fd042c9259c745e2c85c3664fed3`,
manifest SHA
`82779b348d8f6177e8e3c00a3c3dc3d41097cf84341577db862bb10722ac018c`.

After that actual authentication, GPU diagnostic 29763949 submitted 18:17:33,
initial PEND, gpul40s/one exclusive L40S/four slots/8 GiB per slot/60min/no
inter-arm dependency, fresh 15 GiB conservative free. Output
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-random-diagnostic-03/`.
Raw submission
`results/hpc/20261020-brd4-boltz-diagnostic-launch-06/gpu-submission/submission.json`.
Original source05 GPU failure/CPU03 failure/CPU04 and 06 passes preserved;
science02/diagnostic/init/gates unchanged. No GPU start/result, native FD
pass or long random optimization inferred. Mix50/report unchanged.

Final compact 29763949 check remained PEND: four CPU slots reserved on
n-62-13-14 at 18:17:35; reason Job slot limit reached on five hosts. No GPU
start/ETA inferred. Existing heartbeat automation API confirmed ACTIVE,
updated to 29763949/source06 receipt at
`scripts/hpc/receipts/20261020-brd4-boltz-diagnostic-launch-06/campaign.json`.
No duplicate automation or additional query in this follow-up.

### 20 October 2026, 18:22 Europe/Paris — GPU03 recorded; original native gate still fails

Diagnostic 29763949 actual DONE 18:22:34, started 18:20:07 n-62-13-18, 147 seconds.
Persisted status recorded, not readiness pass. Checkpoint/direct-block modes
with two repeats each all reproduce AD 9.497333526611328, original FD at 0.01
7.5991362333 failed; 0.003 12.2960607211 failed; 0.001 9.7145438194 passed.
All four groups fail unchanged two-adjacent-original-scale requirement.
Frozen model/context before/after match original pins, identity_status passed.

Preserving/downloading full raw NPZ/JSON/outcome evidence and preparing an
allocated CPU saved-record analysis. Exact archive/byte-verification and
interpretation pending; no curvature/checkpoint/other cause conclusion yet.
No long random run, tolerance/protocol/science change or omitted failure.
Completed mix50/report and all earlier launch evidence remain unchanged.

### 20 October 2026 — native replay evidence authenticated and downloaded

Actual 29763949 scheduler DONE/LSF success, recorded report, frozen digests,
source and replay bindings authenticated. All 152 files safely extracted/
every-file bytes verified at
`results/hpc/20261020-brd4-boltz-random-diagnostic-03/native-evidence-01/verification.json`.
Archive `/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-random-diagnostic-03-evidence-01.tar.gz`,
68,305 bytes SHA
`9b824dc06142a55e2fdb35a951859e35a5f40a5b2d7158599755cb276cd34a17`,
manifest SHA
`7696329cd526daf51ac05791bfc1d8da741e3c6ae1d2009f89d3fd64042b7fc6`.
Report SHA
`d5434e571adaf9581a8817db5ce88d387e1b31cc49a32dd13343880b98d1fda1`,
replay NPZ SHA
`10b0524519380e9483c98a99a257b018c46ce3d08f318774f64dcb4cb1732289`.
This verifies recorded native evidence, not original FD/readiness success.

Preparing a unique three-file allocated CPU saved-record analysis, with
source/job/outcome pending. No inference, optimization or gate waiver;
no strong curvature/other causal conclusion before its actual evidence.
No long random run; scientific data/history and mix50/report unchanged.

### 20 October 2026 — saved-record native gradient audit authenticated and submitted

Authenticated three-file source
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-gradient-analysis-01/source`,
manifest SHA
`b9306cb3e01ee7ab2d6610da04bb726c5a2da6f42ae2ae5290868158a55e585f`,
archive 6,516 bytes SHA
`5aeef2a40b623f12e4c354952d790a0034c8add25c23954308d455736b8b410c`.
Actual audit CPU 29764094 submitted hpc/one slot/2GiB/15min, source report
pinned to d5434e57...1fda1. Output
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-gradient-analysis-01/payload/analysis.json`.
Static review found no blocker; actual result pending. No inference,
optimization, native gate waiver or new long random run.

Audit comparisons distinguish values/per-head outputs across three modes/
repeats/variants from full AD gradients, available only in AD mode and
compared across its repeats/variants. Do not fabricate other-mode gradients
or claim cause/coverage before the actual analysis returns.

### 20 October 2026, 19:12 Europe/Paris — saved-record audit recorded; original random gate remains blocked

Actual audit 29764094 DONE 19:12:55, verified 132 forward/44 AD records. Exact
three outputs/per-head agreement across modes/repeats/variants; full GX/GE
exactly match only across AD repeats/variants where arrays exist. Original
selectors row 2/aromatic 4-to-NONE 0 match. Original FD F/F/T in all four groups.
Head reconciliation max 1.4901161193847656e-8; all probability mass positive,
max row-sum error 1.1920928955078125e-7.

rng_fixed false with changing RNG fingerprints drives generic classification
observable_execution_difference, while no recorded numeric output/gradient
difference occurs. Not a proven stochastic-output bug. Diagnostic FD 0.0003
9.5221400261/0.0001 9.5006823540 approach AD 9.4973335266; endpoint AD
at 0.003 is 2.8315377235/32.0355033875. This supports a steep scale-dependent
nonlinear/FP32 hypothesis, not proven cause. No concrete integration defect
identified. Original gate/settings remain unchanged, no random long run.

All nine audit files verified at
`results/hpc/20261020-brd4-boltz-gradient-analysis-01/evidence-01/verification.json`;
archive 13,308 bytes SHA
`70c71a2b9750bcbb20117dab87209478f73c35f5390eade2702be187a20c2b81`,
manifest SHA
`9b55f1e4e7b99dbd5c3337139a1ab69a222a404d13b38fa827839f44d822e65c`,
analysis SHA
`2c4b09423aaf53d581c409175bf7155d79ac4742c96cfa18f94ed38760a38fcb`.
Follow-up pause planned pending actual API confirmation/human scientific
direction; no new optimization or native readiness approval. Mix50/report
and every historical failure remain preserved.

Automation API subsequently confirmed existing heartbeat
finish-combined-brd4-trajectory-report actually PAUSED, no duplicate, pending
new human scientific direction. No automatic protocol/init change or new
long random run. Current result receipt:
`scripts/hpc/receipts/20261020-brd4-boltz-gradient-analysis-01/campaign.json`.
Source06 receipt retains actual native outcome/audit pointer/pause and every
historical failure. Both reviewers found no concrete integration defect;
RNG-state advancement is not observed numerical nondeterminism in the
exactly stable tested outputs and AD gradients. Original gate failure and
scientific scope remain unchanged; this actual pause supersedes the planned
pause note above.

## 21 October 2026 — six-milestone saved-results report delivered

The user requested one report spanning the most important early Protenix,
5,000-update, chemistry and latest Boltz-2 results; clarified DeFoG should be
included only if it actually ran in this repository/chat. No completed Mosaic
DeFoG experiment was found. The actual rule-based chemical-repair study is
labeled accurately; no DBMol VAE/latent/DeFoG experiment was imported.

Six selected completed trajectories: original carbon-chain 30% case of
29735673 (100 updates/101 states), BRD4 contact-only 50% 29745503
(5,000/5,001), constant-chemistry 50% 29753152 (5,000/5,001), delayed
Protenix mix29753667/random29753668 and Boltz mix29759212 (each
10,000/10,001). Total **40,106** saved states. This is an editorial chronology,
not the six lowest losses, a matched causal comparison or a new experiment.

Standard-library local presentation assembly authenticates the five existing
HTML reports/PNG figures against their preserved manifests/receipts. Only
saved records are selected/compressed for portable HTML. Existing scientific
plots remain byte-identical; no model, chemical conversion/evaluation, repair,
sampling, numerical tests or scientific plot generation ran locally or on HPC.
Legacy presentation field names are normalized without changing their values;
full probability/readout records, invalid graphs and captured components remain.
Exact roundtrip, compressed/unpacked SHA and contiguous schema checks passed.

New tooling `scripts/assemble_milestone_report.py` and
`scripts/templates/mosaic_milestones.html`; plan/command/scope details:
`docs/mosaic_six_milestones_report.md`. Fresh output
`results/hpc/20261021-mosaic-six-milestones/edition-01/`, HTML52,444,743bytes
SHA625c7d8a6b88dcbf98bad1b0d2bdb9182ee40a69ee8786a582b6c8c5506babe4.
Detached server started on fresh port8775 with the existing serving helper,
preserving prior servers. HTTP200/full body exact length/SHA authenticated at
**http://127.0.0.1:8775/report.html**. Output receipts:
`presentation_receipt.json`, `browser-qa.json`, parent `server.json`.

Actual browser QA passed all six selectors/endpoints, loaded original plots,
readable element/slot labels, probability sizing, six atom/fifteen bond heatmap
rows, playback/pause and valid Boltz9639 (`SCCCCCl`) versus invalid10000.
No console errors/warnings; tab marked deliverable; `report-preview.png` saved.
Node JavaScript syntax, Python AST and Git whitespace checks passed. Local
Python3.13 lacked Ruff; its attempted invocation returned No module named ruff,
not a lint pass or scientific-test outcome. No separate file-scheme browser test.

Targets, atom counts, contact/affinity scales, parameterization changes,
acceptance policies, changing repair references and fixed-pose affinity scope
are disclosed. No cross-model score overlay, hard-molecule affinity claim or
binding conclusion. The original random Boltz gate remains failed, no long
random Boltz run or new study submitted, and the existing heartbeat stays paused.
All native outputs, previous report families and unrelated PDF/OneNote changes
remain unchanged.

## 21 October 2026 — authorized fresh 25-atom three-model comparison prepared

User requested new10,000-update runs using Protenix, Boltz-2 and Boltz-2 plus
chemical validity with25atoms. Prepared three fresh independent arms, same
50% carbon-chain/uniform initialization, LR0.03, key17 and ten-trial best_primal.
Baselines optimize only native contact/affinity; the chemical arm adds existing
repair0.001 through5000 then geometric to50at10000. Element-JS is explicitly
zero in allthree, a declared difference from the historical chemistry run so
the Boltz comparison changes only repair. Six-slot studies remain untouched.

Native shape/correspondence mapping and JAX/worker boundary now explicitly
support6/25slots;25means300unordered pairs. Fresh Protenix context and shared
fresh Boltz25-carbon pose are required; no resized six-slot context or reused
six-slot native certificate. Native-only preparation executes no chemistry or
composition and does not fabricate absent chemical checks as passed. Added
allocated mapping/VJP/scope/pure-objective/controller tests; all old numerical
thresholds, parity/null/finite/identity/runtime gates preserved. New plan:
`docs/brd4_25_atom_comparison.md`.

SSH access worked. First non-login command lacked LSF/quota PATH; a quoting
mistake in the next read-only shell command is preserved in tool history.
Correct login-shell query observed actual public queues gpua10/gpul40s and
home14.52/30GiB at09:42Paris. No numerical work ran on login/local hosts.
Per-arm pilots/whole-campaign conservative storage projection plus4GiB reserve
will gate long submissions. Future Protenix uses gpua10; Boltz uses gpul40s,
each one exclusiveGPU/fourslots/8GiB per slot and23h for long runs.

Source/run candidate20261021-brd4-25-01 prepared; actual packaging, allocation
IDs, CPU outcomes and native readiness/submission remain pending at this entry.
User PDF/OneNote changes and all historical failure evidence preserved.

## 21 October 2026 — source01 CPU completion and user-directed chemistry hold

Immutable source01 authenticated450 members: manifest
4016584825436c9c597307ac51eddb242ac50885352e9c06841da2e1e201ee9c;
archive39,413,811bytes SHA
c625c367e34d2d589edc11629d1bf65aac2642da37b54ad2d4bee53622864d44.
Remote `/work3/s234818/mosaic/releases/20261021-brd4-25-01/source`, run
`/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-01/`.
CPU29768889 submitted09:56, actual RUN09:56:56/DONE10:00:54Paris, hpc4slots,
4GiB/slot/1hour. CPU assessment/XML authentication is recorded separately below;
scheduler DONE alone is not a native pass. No GPU work had been submitted.

The user asked to estimate chemistry runtime before starting its full25-slot
optimization. Original after-CPU29768890 was verified PEND/unstarted and bkill
accepted; actual EXIT authenticated09:59:16, no start time or dispatch artifacts.
This was an intentional user-directed cancellation, not numerical failure;
CPU29768889 and all historical experiments remained unchanged. Before/after/
cancellation text is preserved in source01 run/after-cpu/. This correction also
keeps the two baselines independent of the chemistry pilot's duration/outcome.

Preparing source02 launcher-only `chemistry-runtime-review-v1`: reject full
chemistry at both submit API and batch entry, allow bounded native checks and
two ten-update disposable pilots, expose saved runtime estimate before review.
Baselines can submit after their own passed gates/quota without waiting for the
third arm. Scientific src/config/driver bytes remain identical; no test gate or
runtime allocation changed. New allocated transition contracts cover this hold,
independent dispatch and unknown-outcome duplicate protection. All numerical
validation remains allocated DTU HPC; local AST/hash packaging only.


## 21 October 2026 — reference-mode fix sealed as source03; chemistry full run held

Source02 CPU29768926 actually DONE10:07:39Paris with341passes, no failures,
errors or skips; three historical slow/model tests deliberately deselected.
Actual scheduler/LSF/XML/source/reference proof authenticated and downloaded:
results/hpc/20261021-brd4-25/source02-cpu/evidence/, all20files byte verified.
Archive21,059bytes SHA 2e7de344bfaf09c87342230bac8ad2f0ca4c3cefe3ff30eb7cb647cd9077ef5b.
Callback29768927 was intentionally cancelled while verified PEND; actual EXIT
10:06:53Paris, no start/GPU dispatch, preserving that complete CPU result.

Static review found the new Protenix hard reference omitted the geometry-hook
mask; zero placeholder coordinates alone do not select geometry-off execution.
A new independent-hard-reference helper retains native hard element/bond fields
and explicitly applies that mask/placeholder scope without relaxed injection.
Added allocated regression for every feature, permuted maps, protein preservation
and no context mutation. This corrects reference construction, not the objective
or native threshold. No prior failed numerical gate has been relabelled passed.
All scientific configuration/remaining scientific files match source02; only
examples/brd4_25_atoms.py and src/mosaic/small_molecule/features.py change.

Unique source03 authenticated455files on HPC at
/work3/s234818/mosaic/releases/20261021-brd4-25-03/source. Manifest
1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b;
archive39,420,362bytes SHA
bfc6ad6f0af3268551258de55ca1bd196bb7ff35f6f690a305bf64a0d34fcff7.
New run /zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03/.
CPU29768990 actually RUN10:08:51Paris; after-CPU29768991 PEND on ended(CPU).
No source03 CPU/native pass or long submission inferred. The launcher preserves
chemistry-runtime-review-v1: native gates/two ten-update pilots permitted, full
chemistry rejected at submit and batch entry even if validation passes. Two
baselines can proceed independently after their own original gates and quota.

Local AST/source-map/packaging only; a documentation append first encountered
Windows default cp1252 decoding and stopped before touching the UTF-8 ledger.
It was repeated with explicit UTF-8; no numerical test/model/chemistry ran locally.
All historical files/jobs, report servers and unrelated PDF/OneNote changes remain.


## 21 October 2026 — source03 passed342 CPU checks; native stages submitted

CPU29768990 actualDONE10:12:22Paris,342passing cases, no failures/errors/skips;
34launcher/226legacyJAX/23mappingconfig/59chemistry, three historical slow/model
checks deliberately deselected. Actual scheduler/LSF/rawXML/source/reference and
all19evidence files byte verified locally at
results/hpc/20261021-brd4-25/source03-cpu/evidence/. Archive20077bytes SHA
4aba0625dbd004d73fc1b8faa185b104ef7ffc1c4c577a92ff2460223f5b0fa4.
AfterCPU29768991 actualDONE10:12:28, authenticates this exact proof and seals
cpu-certificate.json. Actual native validation29769026 gpua10 and Boltz fresh
preparation29769028 gpul40s submitted; callbacks29769027/29769029 wait on their
ended conditions. All four last PEND, no long optimization submitted. Actual
bsub command/resource receipts saved in initial-submissions.json and each stage.
One exclusive GPU/fourslots/8GiB per slot/twohours for native stages; genuine
readiness/runtime/quota required before23h independent baseline submissions.
Full chemistry blocked by both submit API and batch entry pending runtime review.

## 21 October 2026 — allocated25-slot repair component timing recorded

Supplementary immutable3file tools release
/work3/s234818/mosaic/releases/20261021-brd4-25-repair-timing-01/source,
manifest04d4fd4eaaa8e07c02c58a04b9f15c02e0a47fba9a9ec5dbb1e182084c641100,
archive3852bytes SHA1776307af4e202d8b805c8c8c272902863ecb7abee30f61132288ba10af313f6.
Numerical benchmark29769038 actualRUN10:14:00/DONE10:14:42Paris, hpc4slots,
4GiB per slot/20min request. Requires authenticated source03 CPU certificate
and exact scientific source/config bytes. No native model/optimizer/training;
FP32JAX CPU, exact6/25-slot chain-uniform and internal-F assay, one warmup plus
three measured repetitions of preparation and one chemistry value/gradient.

25-slot means0.1132757757s valid/0.4657968260s invalid, extrapolated10k calls
1132.7578s/4657.9683s (18.9/77.6min). Six-slot0.0550516450s/0.1966275177s,
about2–2.4times increase for the tested points versus20times pair count. All
raw timings/preparations/point NPZs preserved. Excludes native inference,
additional line-search evaluations/startup and later harder graphs; this is
component overhead, NOT end-to-end ETA, worst-case bound or native approval.
Native GPU pilots still required; chemistry full-run hold remains.

Actual scheduler/LSF plus recorded status/case schema/source/point hashes
authenticated. First evidence helper failed because non-login PATH lacked
bjobs, before capture; empty evidence01 preserved. Unique02 succeeded. All15
files safely downloaded and byte verified via scripts/verify_hpc_evidence.py at
results/hpc/20261021-brd4-25/repair-timing/evidence/. Archive16404bytes SHA
b7881b3dcf522e8ed60fa267b08e7464acde0b6430ba90ef51f7441cebacddf6,
manifestf553d6afdf7a5adc047171bb158af36a66ca462d885ba4170512341dc1349e50.
No numerical work ran locally/login. Existing heartbeat updated for NEW25-slot
native progress and measured total runtime review, hourly/quiet on unchanged
state; historical six-slot random blocker remains paused, no retry authorized.


## 21 October: source03 Boltz final-response contract failure retained

Fresh Boltz preparation29769028 actually DONE10:16:43Paris; afterprepare29769029
DONE10:16:49. Exact25-slot context SHA
099fdac537a08f65a14cb60463de24fa0c4ffd6541a142beaaee6e974977fa08,
tensor digestdf93ef76cc26d5888532dc2c36688dbbff4040375f5357a784b8b6a656ec2f34.
Actual validators Boltz-only29769078 and chemistry29769080, callbacks29769079/
29769081 submitted. Boltz-only29769078 ran10:17:15–10:19:03 and EXIT1 after
native hard parity, all three atom/bond FD scales, ten disposable updates and
runtime fit individually passed. Whole readiness remains FAILED, no assessment
or long run. Callback29769079 DONE retains that failed decision.

The actual model/context hashes before/after were exact. Driver incorrectly
compared the entire response dictionary (which additionally carries status,id,
elapsed_seconds) against a two-digest dictionary. This is a concrete response-
shape bug, not changed weights or a scientific threshold rejection. Preserve
raw failed summary/worker/native/pilot evidence. Source04 will require status
passed AND exact original model/context digest fields, retaining the raw response
and the transport's existing request-ID checks. Added allocated contracts accept
metadata only with exact hashes/success, and reject either mutation or absent/
failed success. No derivative/parity/frozen/finite/runtime gate is weakened.

Obsolete source03 Protenix29769026/callback29769027 and chemistry29769080/
callback29769081 were all verified PEND/unstarted, bkill accepted; actual EXIT
verified with no start times. Their original receipts and cancellation evidence
remain under source03 root. No long experiment has started or been cancelled.
Prepare unique source04 and rerun full allocated CPU/native tests, preserving
identical configuration/scientific settings and the full chemistry runtime hold.
Source03 passed subchecks/pilot timings are observations, never a readiness pass.


## Source04 allocated retry and preserved source03 failure evidence

Unique source04 staged/authenticated461members:
/work3/s234818/mosaic/releases/20261021-brd4-25-04/source,
manifest e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79,
archive39,425,961bytes SHA
b62ef75f6ede08fbed8854ff4efe167571be6eb050c1b222278e19e15b3dc879.
Source03-to04 src/examples bytes are identical except the driver now calls the
strict passed-status/two-digest verifier; all three configuration files unchanged.
CPU29769140 actually RUN10:21:56Paris, afterCPU29769141 PEND on ended(CPU).
New5response contracts join full regression/native gates; no pass or long job
inferred. Current run root /zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/,
receipt scripts/hpc/receipts/20261021-brd4-25-04/campaign.json. Full chemistry
remains held even on future readiness success, pending end-to-end runtime review.

Source03 final-response failure evidence, all85files including raw native FD,
hard parity, ten-update pilot, worker response, complete original prepared25-slot
context bytes and cancellation before/after records, safely downloaded/verified:
results/hpc/20261021-brd4-25/source03-response-failure/evidence/.
Archive664616bytes SHAe635ca196c54c672a9cbfbd9891c3ffdf127f5e81dfc78f53c52da9caab03c58,
manifest15c540fd57b54e0eddbab7f6f57a4d76b76900a504e20a5030d3940e52b777e3.
Actual frozen model/context fields before/after exact; metadata mismatch explains
engineering rejection, but the entire source03 readiness stays failed. None of
its individual passed subchecks or measured pilot times promotes source04.
All four obsolete unstarted source03 jobs actualEXIT10:20:21Paris; no long run
started/cancelled. Heartbeat updated to source04 current jobs and original gates;
old random scientific blocker still not retried, full chemistry still held.


## Source04 CPU347 passed; actual replacement native stages submitted

CPU29769140 actually DONE21October10:25:45Paris,347passes, no failures/errors/
skips; three historical slow/model checks deliberately deselected. Raw counts
34launcher/226legacyJAX/28mappingconfig/59chemistry include all five new response
contracts. Actual scheduler/LSF/source/reference/XML and all19proof file bytes
safely downloaded and authenticated at
results/hpc/20261021-brd4-25/source04-cpu/evidence/.
Archive20,154bytes SHA230d88e262bd92af319ba6d417141795e3a21df57a49456aa10a4107087fd99f.
AfterCPU29769141 actualDONE10:25:51 and authenticated actual native submissions:
Protenix validation29769176/callback29769177, fresh Boltz preparation29769178/
callback29769179. Same resource shape and original unchanged gates. These are
native validation/preparation, not submitted10,000-update arms. Current actual
receipts scripts/hpc/receipts/20261021-brd4-25-04/campaign.json; conditional later
IDs must be authenticated, not inferred. Full chemistry still held for measured
end-to-end estimate; supplemental19–78min/10k component calls is not total ETA.
Tooling/science/tests/docs pushed as784ab66; only user PDF/OneNote changes remain
unrelated. Runtime follow-up active/quiet on unchanged state; old six-slot random
scientific failure not retried. No local numerical work.


## 21 October 2026 — user approved chemistry runtime and released full-run hold

User explicitly accepted25-slot repair timing and requested full submission.
Diffusion gradients explicitly prohibited back to the molecule; existing fixed-
pose detached-coordinate affinity/trunk and repair gradients remain unchanged.
Temperature sharpening only brainstormed; T1 retained. Current execution plan:
docs/brd4_25_chemistry_release.md. Original native/runtime/quota gates unchanged.

Source04 scientific347-test pass and all461 scientific bytes reused unchanged.
New launcher07 five files manifest80274d8d5f066af3e364f0356182ab0aa124302613800aa8b68fe3ed3cdec964,
archive10105bytesSHA61d23c4d5f6c7e34b59cdfec60947c1a001c838994fd175d5dc07a5a44c8127b.
CPU29769499 actuallyDONE10:54:45Paris24passed/no failures/errors/skips;
bootstrap29769500 actuallyDONE10:55:02 authenticated existing prep29769178DONE.
Old callback29769179 had read stale schedulerRUN; original capture preserved,
new bounded retries require actual final status/LSF rather than assuming DONE.
Actual new native validators29769502Boltz/29769504chemistry and callbacks
29769503/29769505 submitted, latestPEND on physical L40S availability. No full
chemistry10,000-update ID yet; its submission remains conditional on real gates.

Protenix29769176 independently EXIT10:51:47 at original hard parity2e-5,
maxreported discrepancy~1.564e-4. No Protenix long run. Its obsolete callback
29769177 was intentionally retired while verified PEND/unstarted, actualEXIT
10:50:49; replacement29769501 records the actual failed native outcome without
blocking Boltz arms. Source05tools sealed/not staged after static callback-name
collision; source06staging rejectedCRLF before any test/job; source07 LF wrappers
passed Bash syntax/allocated contracts. All failed records preserved.

All56CPU/controller/preparation/XML/cancellation/Protenix-failure files byte-
verified under results/hpc/20261021-brd4-25/release07/evidence/ via standard-library
scripts/verify_hpc_evidence.py. Archive153209bytesSHA
055c6df3db215eed088da083f07c5ddfbf4473b1527ffc01892b062bed49aded.
No local numerical work, model change, temperature change, diffusion derivative
or native gate waiver. Exact actual receipts and source identities in
scripts/hpc/receipts/20261021-brd4-25-release-07/campaign.json. Follow-up active
for actual approved full chemistry submission, preserving old random blocker.


## 21 October 2026 — 25-slot native Boltz gates passed; independent long arms submitted; JQ1 pocket recommendation

Scientific source04 and launcher07 remain unchanged. Native validators29769502
and29769504 plus callbacks29769503/29769505 are actualDONE with LSF success and
persisted passed assessments. Original2e-5 hard parity, independent-primal raw
atom/typed-bond FD at.01/.003/.001, finite/frozen identities and required pilots
passed. Chemistry also passed nonzero invalid graph signal, low/high coupled
checks and both ten-step disposable pilots. No gate was relaxed. Protenix's
separate29769176 parity failure remains preserved and blocks its long arm.

Actual independent10,000-update submissions: Boltz-only29769525 submitted
11:02:11Paris, RUN since11:04:20; Boltz+repair29769540 submitted11:04:34, PEND
on physical GPU availability at capture. Both gpul40s/one exclusiveL40S/four CPU
slots/8GiB per slot/23h, no inter-arm dependency. Actual bsub receipts and bjobs
resource records authenticated. No completed trajectory is established.
New25-slot context tensor digestdf93ef76cc26d5888532dc2c36688dbbff4040375f5357a784b8b6a656ec2f34;
context.pt SHA099fdac537a08f65a14cb60463de24fa0c4ffd6541a142beaaee6e974977fa08;
initSHA90553f0e28762fe81fa0c40f840e1c2e4884a0f79a1bbf48553132e3c0e27391.
Model395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16
before/after exact for both readiness arms. Config hashes and actual assessment
hashes in scripts/hpc/receipts/20261021-brd4-25-release-07/campaign.json.

Saved pilot means1.74322s/update Boltz;2.02183/2.25575s chemistry early/high.
Original conservative projections27948.33s/35636.30s passed82800s. Fresh
submission quota checks passed with both ready arms plus4GiB reserve; not a
storage reservation or worst-case time guarantee.224proof files safely
extracted/byte-verified locally under results/hpc/20261021-brd4-25/submission02/;
archive1985802bytesSHA9f4061050e70e55160e61558b015c578dcf73e0da890084847808951eea38a8c,
manifest a21a5407415f169b251d3f86efb768e254881f851c20496b8893f276e678abdf.
Capture01 stopped on LSF wrapping a field name, without scientific changes;
partial directory preserved. Capture02 strictly normalizes whitespace, retains
raw output and authenticates exact job/status/queue and all native proof bytes.
Only stdlib byte/schema/receipt checks ran locally. The submission heartbeat's
endpoint is fulfilled and is paused; long studies remain unchanged.

User requested an experimentally established pocket for25-slot designs.
Recommended BRD4-BD1 acetyl-lysine/JQ1 pocket in3MXF, same127-residue construct.
Primary sources: https://www.rcsb.org/structure/3MXF (1.60A co-crystal),
https://www.thesgc.org/chemical-probes/jq1 (JQ1 BRD4(N) ITC Kd49.0+/-2.4nM),
https://pubmed.ncbi.nlm.nih.gov/24497639/ (experimental affinity mutational map).
Asn140/Tyr97/Trp81/Met149 map to supplied sequence99/56/40/108; WPF81/82/83
to40/41/42. Mapping is static fixture metadata, not native token/crop validation.
Proposal: derive spatial pocket around crystallographic JQ1 on allocatedHPC,
restrict Protenix contact objective, condition new Boltz pose then freeze it.
Current Boltz whole-complex affinity has no independent pocket selector; current
prepared pose is not yet geometrically audited against the JQ1 site. No
pocket-specific implementation/job was made and current runs were not changed.
No diffusion gradient, temperature change or AE/VAE/latent pipeline introduced.
Recommendation and versioning/validation requirements recorded in
 docs/brd4_25_chemistry_release.md. No binding/activity claim for generated graphs.


## 21 October 2026 — explicit stop and new JQ1-pocket 25-slot comparison

User authorized stop current25-slot runs, implement JQ1-pocket targeting for
fresh Protenix/Boltz2/Boltz2+chemistry10000-update arms, allocatedHPC testing and
conditional submission. No diffusion gradient back to molecule. Current old
jobs29769525/29769540 were actually cancelled; finalEXIT11:15:33/11:15:34Paris.
Preserve original readiness passes, partial artifacts and cancellation evidence;
these intentionally interrupted long runs are not completed trajectories or
spontaneous scientific failures.31stop/queue/raw-log files safely downloaded and
byte-verified under results/hpc/20261021-brd4-jq1-25/stop01/; archive462541bytes
SHAb50c8ac52abbd5322b9b72902ded3e62ad3bd126e009073826e2212d70ca9a29,
manifest009c73f1cc32e7975ed49c45a34eede10770171757fa7d9ddee9d5977c4dff91.

User requested lowest waiting GPU queue. Actual eligible queues and GPU host
state inspected; H100 is restricted/absent from user-eligible list. L40S had
smallest captured pending-slot backlog of eligible long queues. All3new arms
usegpul40s/one exclusive48GBL40S/4CPUslots/8GiBper slot;2hvalidation/prep and23hlong.
Snapshot is not an exact wait-time promise; public interactive queues are not
suitable for the10000-update allocation. Raw selection evidence preserved.

Implementation: docs/brd4_jq1_25_atom_comparison.md. Allocated extraction from
pinned3MXF defines6A heavy-atom JQ1 neighborhood with explicit author/sequence/
native mappings. Protenix contact partner set restricted; independent native
hard geometry-off reference compiled in matching full-trunk/head scope, all
input fields checked exactly, original2e-5 parity gate unchanged. Historical
source04 eager-vs-JIT parity failure retained, not declared passed. Boltz fresh
native pose constrained to sequence40/99(authorW81/N140), native potentials only
during pose preparation; actual anchor<=6A and>=50%ligand-near-pocket placement
gate precedes fixed detached coordinates. No AD through diffusion. Same25slots,
50%chain/uniform,LR.03,key17,T1,10000updates,ten-trialbest_primal,composition0,
chemistry.001through5000 thengeometricto50at10000. Full original native/FD/finite/
identity/pilot/runtime/quota gates retained; independent passed-arm submissions.

New source20261021-brd4-jq1-25-01 authenticated488files, manifest
2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c;
archive39462299bytesSHA5e4bf8601f1b09e244a7763dc6c6d8af5ddc76bfd3490830993d7cfd0b2c0906.
Actual CPU29769654 andafter-CPU29769655 submitted, CPUstarted11:26:36Paris;
initial93allocatedcontroller/pocketcontracts passed, remaining regression suite
still running at that snapshot. No CPU final pass, native result or long job
inferred. No numerical tests/model/chemistry executed locally; AST/JSON/LF and
byte/schema checks only. Source and receipts under scripts/hpc/receipts/
20261021-brd4-jq1-25-01/. Old studies/reports/userPDF/OneNote changes preserved.


## 21 October 2026 — JQ1 source01 full CPU419 authenticated; native stages queued

Actual CPU29769654 DONE11:30:22Paris and callback29769655 DONE11:30:38.
419passed/no failures/errors/skips,3deliberate legacy slow/model deselections:
93contracts,233legacyJAX,34Torch mapping/config/no-diffusion,59unchangedchemistry.
AssessmentSHAe342bda2bd510fff0aba33192f10dfe4d81ccca53e1b83562c83089ab54095e1.
RawXML/scheduler/LSF/all43proof files safely downloaded and byte/schema-verified
under results/hpc/20261021-brd4-jq1-25/cpu01/; archive29621bytesSHA
12ffe963f2f5e68989bd80eac6f90aee2e08f77746246d5b6a4631fca2c5a18a.
Allocated crystallographic extraction produced19pocket residues; exact definition
SHA9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43.
Actual Protenix validation29769668/callback29769669 and Boltz pocket preparation
29769670/callback29769671 submitted, initiallyPENDgpul40s/hpc as applicable.
No native pass or actual10000-update submission inferred from CPU/controller.
Independent conditional long submission remains authorized after original plus
pocket gates. New plan/receipt updated; old intentional stops and all failures
preserved, no local numerical work.


## 21 October 2026 — JQ1 Protenix checker plumbing failure; Boltz placement passed

Source01 Protenix native29769668 EXIT12:24:37Paris after hard native2e-5 parity,
pocket mapping and unchanged frozen identities. Missing primal_every_step flag
correctly triggered the existing independent-primal best_primal guard before
its one-step update. Preserve failed readiness/fullrawFD/null/context artifacts;
no full native pass or longProtenix submission. New Protenix-only retry supplies
explicit independent candidate forwards; no scientific threshold/feature/loss/
configuration change. All model and test work still allocatedHPC.

Boltz pocket preparation29769670 DONE12:26:24Paris with passed placement audit;
callback29769671 DONE and submitted two independent validators/callbacks. Exact
IDs and outcomes to be authenticated in subsequent entries. No long run inferred.

Interactive investigation: directgpua100i bsub20min rejected by site hook before
submission; officiala100sh job29770026 dispatched immediately n-62-12-19. Both
40GiBA100GPUs had existing processes (1295/33463MiB; latter100%utilization).
No numerical work ran; closed session. Interactive shell availability is not
freeGPU capacity. Preserve shared-node rules and runningL40S work.


## 21 October 2026 — independent Protenix source02 CPU validation submitted

Source20261021-brd4-jq1-25-protenix-02 authenticated492files, manifest
40ab2d4532c477b5074d6631587ae15188941d5167016b7421f993257a064438;
archive39465640bytesSHA52caceb014e66749d55ad85950e8cacdddd76e2ebeb5feaac6b975dec5f54b8a.
Actual CPU29770038 and after-CPU29770039 submitted. Contract100, legacy234
(with3deliberate slow/model deselections), native mapping34 observed passed;
remaining chemistry59 running. No finalCPU/native pass inferred.

Source01 failed Protenix evidence105files safely downloaded/verified under
results/hpc/20261021-brd4-jq1-25/protenix-failure01/. Archive80289400bytesSHA
bd9f8235d0c9fe4acb2fe80890d1906857db88695a45b0247a144d531a01af25;
manifestf136cf65ebbf5e1d036fa60bd908377ef52b69524b386fcbff958bad9cf798e7.
All3predeclared atom/bond FD scales agreed, null output/scalar differences0,
primal/AD scalar difference2.98e-8; full readiness still failed at omitted
independent-primal flag and is not relabeled passed. Exactrawrecords retained.

Source01 Boltz preparation29769670 DONE12:26:24 and callback29769671DONE;
actual native validators29770007Boltz/29770009chemistry and callbacks
29770008/29770010 queued. PreparedcontextSHA
a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093,
pocket9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43.
Saved audit: anchor distances3.70771694/3.46764302A,23of25atoms(92%) within6A.
Detachedpose true, not a binding/activity or ligandRMSD claim. No longpocket
submission yet. Interactive29770026 confirmedDONE after shell-only inspection.


## 21 October 2026 — corrected CPU427 passed; first pocket long job actually queued

Source02 CPU29770038/afterCPU29770039 actuallyDONE/passed427:100controller,
234legacy/JAX,34native mapping/no-diffusion,59unchanged chemistry; zero failures,
errors or skips,3deliberate legacy slow/model deselections. Actualscheduler/LSF,
rawXML and all38proof files locally byte/schema-verified under
results/hpc/20261021-brd4-jq1-25/cpu02/verification.json. Assessment
0bcf6532e8642f51e871f57ff537efaf0ba3504b94f5fcc42ffeaf70dbca9573;
archive28572bytesSHAbfd3762940b5eb3a1522a3a65df7b885ad4da988444155db26ee761a6584a407.
Protenix-only native29770090 and callback29770091 actually submitted.
Callback alone verifiedunstarted and bstop applied; actualPSUSP confirmed.
This reversible submission hold permits fresh whole-three-arm storage review
across BOTH source01/source02 roots before bresume; native remains unchanged.
Include all passed current projections, already-submitted arms and4GiB reserve,
subtract both roots' actual bytes conservatively. This is neither scientific
failure nor a user-requested pause. The existing monitor tracks release after
actual fullnative/identity/runtime/proof and aggregatequota gates.

Source01 Boltz native29770007 and callback29770008 actuallyDONE/passed.
Hard parity/rawatom+typedbondFD/primalAD/finite/frozen/pocket gates and10-update
pilot authenticated; mean1.761389260s/update, conservative28220.838900s<82800,
storageprojection1804441824bytes. Model395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16;
contexttensorfd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4.
Assessment41901b3ad903c5d14b0344a1876564ee2481ace6a3adc8be5a33245214b4f387,
52nativeproofs rehashed. Actual independent10000-update long29770093 submitted
12:35:37Paris, lastPENDgpul40s; bjobs and receipt match1exclusiveL40S48GB,
4slots,8GiB/slot,23h,nointerarmdependency. Submission is not trajectory completion.
Chemistry native29770009 nowRUN, callback29770010PEND; no chemistry long yet.

All168compactproof files downloaded/byte-verified under
results/hpc/20261021-brd4-jq1-25/boltz-ready02/verification.json; archive736630bytes
SHA18e1d6dcede92b79b43ccfc0a828ce1ec480d819f5eb976a2fb4955a470db072,
manifest70c1b16ef5515bfd1cc16aae7105e3ac839c1bb7e689f54588672080538f7e63.
Firstcapture stopped on aged-out liveCPU records; partialfolder preserved.
Correctedcapture authenticates exact pinned prior actualscheduler/LSF bytes and
labels persisted outcomes, retaining live-not-found responses; no fakeDONE or
forbiddenaccounting query. Historicalnativefailures remain failed.


## 21 October 2026 — both Boltz pocket native arms passed and submitted

Chemistry native29770009/callback29770010 actuallyDONE; all native/typedFD/
parity/pocket/finite/frozen plus nonzero invalidassay, low/highcoupledchecks
and both10steppilots passed. Assessment17ce0869f0a9888292e9017e03d407bf80d979a9cbfbed1367ae0785eb532394,
128nativeproofs rehashed. Pilot means2.048396136/2.270196313s/update, original
conservative projection35852.944692s<82800; projected artifacts5187761824bytes.
Independent10000-update chemistry long29770124 actuallysubmitted andPEND;
receipt and actualbjobs matchgpul40s/1exclusiveL40S/4CPU/8GiBslot/23h/nointerarm
dependency. Boltz-only29770093 nowRUN. CorrectedProtenix native29770090RUN;
its callback29770091 remains intentionallyPSUSP pendingfullnative andsharedquota.

All313proof files safely downloaded/byte/schema-verified under
results/hpc/20261021-brd4-jq1-25/boltz-both03/verification.json; archive2176935bytes
SHAb5d176a1f5396982309e5c7a11bdb352134e18a8190a20dc8e4d66a4aacc896a,
manifest1fe0710926a33b5b9d7d8e3c0c2bc1fc1f53bd12cd810eb512861c642dc51533.
Fresh stdlib-only shared-quota auditor authenticates all3actualnative/scheduler/
LSF/source/config/CPU/proofs, adds all3projections+4GiB and subtracts BOTH root
byte totals. It does not resumejobs. SHA
f84e64b64869fe4979cdb5f0e1c31de2a97eb265f12f6f0f1ee169fdc8f529e1;
preparedfor later execution only after ProtenixactuallyDONE/passed. LocalAST
review only; no local numericaltests or scientific execution.


## 21 October 2026 — all three JQ1-pocket long runs authenticated

Corrected Protenix native29770090 actuallyDONE/passed all original hard/rawFD/
null/parity/finite/frozen/pocket gates plus independent-primal update and10-step
pilot. Assessment70fa96f9009a0e7316fff341e8c091bfd5339f9c9e2f4456f39d35b6a2e13cf8,
74nativeproofs rehashed. Runtime mean0.2504813783s/update; conservative original
projection5557.220675s<82800; artifactprojection1439101824bytes. Model
8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2 and
contextb28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4
unchanged before/after. Original source01 failure remains failed historical
evidence; correctedsource02 is a new successful readiness.

Fresh shared quota audit authenticated all3actual nativeDONE/LSF/source/config/
CPU/proof identities. Required12726272768bytes(all3projections+4GiB), available
15820561335bytes after conservatively subtracting both roots' actual artifacts.
Cached quota remains no reservation; original callback repeats fresh ownquota.
Audit assessmentSHA52585026fd21d3c8baaa0f1214420090db9e8d9341e89a207e3cd8f1bbe209b3.
Only then exact heldcallback29770091 resumed; actualDONE/LSFsuccess authenticated.

Actual independent Protenix10000-update long29770179 submitted, lastPEND.
Boltz-only29770093 and Boltz+chemistry29770124 nowRUN. All3actual receipts and
scheduler resource vectors matchgpul40s/1exclusiveL40S48GB/4CPU/8GiBslot/23h,
with no inter-arm dependency. Fixed25neutraloccupiedslots,50%chain/uniform,
LR.03,10000updates,oldchemistryschedule and detached/no-diffusion scope retained.
This completes the authorized submission endpoint, not any10000-update result.

Protenix ready evidence127files downloaded/safely extracted/byte+schema-verified
under results/hpc/20261021-brd4-jq1-25/protenix-ready02/verification.json. Archive
155160236bytesSHA0fe698c72575d74f9acdc3c53b3c9ce719a0a36213fc96e468b5ca6c73b91f91,
manifest7b3f80654d5fd191d40e7419e9385b676bc304897a29054e7b2b004041e8489e.
Its scheduler snapshot precedes asynchronous resume completion; finalcontrol
proves actual callbackDONE and newsubmission rather than relabeling the oldPSUSP.
Finalcontrol42files verified under results/hpc/20261021-brd4-jq1-25/final03/
verification.json, archive44120bytesSHA
ae01f2098727e435588a9e16a401459f71816c081e52fe689ea24adb06dbc344,
manifestdecd27d60516955603019988894b6a50335a1fecbcea806ad0a0a7e3f4d0c0fb.
All tests/model/scientific work allocatedHPC; local stdlib byte/XML/schema only.
Relevant tooling/docs pushed, unrelated PDF/OneNote changes preserved. Existing
submission monitor can now pause at its authorized endpoint.


## 21 October 2026 — new three-arm 25-atom report authorized

User requested leaving current Protenix 29770179, Boltz-2
29770093 and Boltz-2 + chemistry 29770124 unchanged on HPC, then a new combined
report after all three complete. Plan: docs/brd4_jq1_25_report.md. Separate
saved-only25-slot renderer and lossless bounded FP32 assets are prepared;
no native experiment, optimizer, loss, diffusion scope or gate changed.
Local static AST/LF checks only; allocated report fixture tests and numerical
rendering are not yet established. Dedicated completion collectors will preserve
actual native scheduler/LSF outcomes before live retention expires. Existing
reports/servers and unrelated PDF/OneNote changes remain untouched.


### 21 October 2026 — JQ1 25-slot progress editions authorized

User requested downloading saved molecules near update 1500 and refreshing the
new report every 1000 further updates. All current native jobs remain unchanged:
Protenix 29770179, Boltz-2 29770093, Boltz-2 + chemistry 29770124. At 13:14:26
Paris all three were RUN, with latest saved states 1641/1099/850 respectively.
Report source01 CPU 29770499 and certificate 29770500 were actually DONE; 56
report fixtures passed. Live schema inspection found case.initial_checks is a
direction list, with passed status in its separate coupled report. New report-only
source02 corrects this equality contract and adds sparse incomplete editions;
no source02 pass/submission follows from this preparation. Final native/full-budget
gates remain unchanged. See docs/brd4_jq1_25_report.md for milestone/prefix rules.


Report source02 authenticated 511 files, manifest
cff9a2fd0a1ee7b2b1e3fb53d68ee4a46bee9d1fa453506684d7cc7bbfeb75e0,
archive 39,508,302 bytes SHA
f6608dd46584f8f7c0cbac21d41677555e13f98b74a86b94a93df11afce39ceb.
Allocated CPU 29770598 failed one aged-out-record fixture (68 passed, one failed,
34.65s): the missing-job message included its job ID and was incorrectly routed
to the live-status parser. Source03 routes only an actual Status field to that
parser and tests both completion-collector and legacy scheduler filenames.
Keep the failed source02 result. No GPU experiment was changed or retested.


### 21 October 2026 — First 25-slot interim report verified and published

Report-only source03 authenticated 513 files; manifest
3b1ba56f8ddd0dc163333d9dabccd1119e323553a160a0e7b6b41790e73d7252,
archive 39,510,050 bytes SHA3434ab4cabbac75441adfbf8be7d50fdbc5381d0740bf13c0ce92244076891e0.
CPU29770614/certificate29770615 actually DONE with70 passing report fixtures,
no failures/errors/skips. CPU assessment SHA
dadb07afdd95440aeb18ca2adaf133c8de8d3c6ccad711f3b305a1383f38d2d2.
Preserve source02's68-pass/one-failure aged-out-message fixture; it was corrected
without relaxing source, scheduler or native checks. Obsolete report-only pending
jobs were intentionally retired; actual EXIT receipts are in source03 campaign.
All three native GPU jobs remain unchanged.

Current completion collectors29770616/29770617/29770618 and finalreport29770619
wait on the same natives. First interim29770620 actually DONE/passed; included
updates2275/1133/996, with exact milestone/current snapshots and full scalar
prefixes. Archive197826bytes SHAe82677ec2d1135e450707ff5b8e47e1e47e6f77594865447142b7661748123e2,
all36files verified locally with safe paths/lengths/hashes. Report assessment SHA
bb51c887a27b8abc43af2df4cae0d122af95f050b7b8613bdf144b7c94e86179.
Actual final scheduler and LSF success, CPU XML/proofs and source/input identities
were checked. Local work was stdlib byte/XML/schema/publication/HTTP checks only;
all rendering/tests ran on allocated HPC. Browser QA passed selectors/endpoints,
1500 snapshot, playback, labels/colors, exact probability maps and loss overlays.
Displayed argmax molecules remain sanitized25-carbon chains at this capture.

Stable new link http://127.0.0.1:8776/report.html, preserving older editions/servers.
Existing heartbeat updated to ten-minute checks and per-arm1500,2500,...9500,10000
milestone publication; unchanged polls stay quiet. The complete report still
requires all native/full-budget/terminal/frozen gates and10001states per arm.
See docs/brd4_jq1_25_report.md and source03 campaign receipt for exact paths/IDs.


### 21 October 2026 — Report timeline correction requested

User could not scrub through progress. Edition0001 held only2–3 saved molecule
snapshots per arm. New report-only source04 includes every committed state in
progress editions using the existing lossless bounded chunk writer, with exact
update input/previous-next controls and playback pause on manual scrubbing.
No new source/pass/job follows from preparation. Preserve edition0001 and all
native settings/jobs. At13:29:12Paris all3nativesRUN; states3350/1201/1121.
Required allocated report tests and browser drag QA precede promotion. Full
final completion/native/source/identity gates remain hard requirements.


### 21 October 2026 — Timeline fix verified and published

Source04manifest5c3f123748f573dcb5e3d4532659441fc6f5666a1768461a459c94cf208ba31a
(518files;archive39518412bytes SHAd1f13f8c440948cb003e3185d3a8daaf3b957985941799d2248b2bce70e57601)
passed allocatedCPU29770704/certificate29770705:71fixtures, no failure/error/skip,
36.83s. Current collectors29770706/07/08 and finalreport29770709 preserve native
completion gates; old source03 report-only predecessors retired while verified
PEND after newCPU/certificatepass, actualEXITreceipts retained. Native jobs
29770179/29770093/29770124 remain unchanged.

Interim29770710 actualDONE/LSFsuccess/passed assessment; captured13:34:09Paris,
updates3634/1227/1148, every saved frame retained (3635/1228/1149;6012total).
Archive24677266bytes SHAaf2ae8c68c44329211bbf1ed676920cfe3dfa885103155f5690509f9d4a15831;
all154files safe-extracted and byte-verified locally. CPUassessmentSHA
 a2654b958374b19cb9f42113032ad3a3a4776c38a88fb387d6409da37ce872d0;
reportassessmentSHAf33a28fdfc86966262791fc7b5be4803c38ab2dddf223be87ee8fb1f47dc3d33.
Exact per-frame metadata contiguity/chunk lengths, identities, rawCPUXML/proofs
and final scheduler/LSF authenticated. Local work was staticAST/JSsyntax,
stdlibbyte/XML/schema/HTTP/publication and actual browser presentationQA only.
No local scientific evaluation/model/chemistry; tests/rendering allocatedHPC.

Published edition0002 at http://127.0.0.1:8776/report.html; oldedition0001intact.
Actual mouse drags selected1860then658, numeric1500 worked, Boltz600 and chemistry
777→778 worked, and manualscrub paused playback at500. Saved scalars updated;
hard molecule still a carbonchain at checked points. Screenshot andverification
are under results/hpc/20261021-brd4-jq1-25-report/edition-0002/. Monitor continues
same milestone publication cadence with every committed frame in each edition.


### 21 October 2026, 13:59 Paris — Next progress capture submitted

Compact status authenticated all3nativejobsRUN and finalcollectors/reportPEND;
latest committed states5850/1395/1300. Protenix crossed4500 and5500 since the
published3500 milestone, so report-only interim-0002 **29770891** was actually
submitted at13:59:44Paris on unchanged report source04. Exact queue hpc,2CPUslots,
4GiB/slot,30min, no native dependency/mutation; actual receipt/resources saved.
GPU jobs29770179/29770093/29770124 unchanged. This submission is not a completed
report. Current root/jobs.json and campaign receipt record the actual new ID.
New local edition0003 reserved; publish only after final scheduler/LSFsuccess,
passed saved-data report assessment, safe every-file byte/schema verification
and browserQA. Preserve edition0002 until verified promotion.

### 21 October 2026, 14:08 Paris — Saved-gradient audit answers apparent stasis

User asked whether the three JQ1 runs receive gradients and whether effective LR
is small. Read-only allocatedCPU **29770931** actually DONE/LSFsuccess; script
SHA `b595c73acde2ef448111c4564b64a0536190d4ab7841e36c04446477701e9c7a`,
7410bytes, hpc/2slots/4GiBslot/20min. Fixed report capture5900/1409/1314 was
authenticated against original jobs/source/config/model/context and saved
JSONL-prefix/snapshot hashes. No model, repair, new gradients or molecular
updates were executed by this audit. All numerical analysis ran on allocatedCPU.

Nonzero atom/bond gradients and actual probability changes exist in all3arms.
Last100 median accepted rates0.03/0.001875/0.0009375, combined logit-gradient
L2 norms0.006517/0.302833/0.389493, zero fallbacks/noops. Maximum since-initial
atom probability shifts0.03075/0.20975/0.20948 and bond shifts0.05823/0.25488
/0.25473 coexist with zero changed hard argmax labels. Saved-step replay residual
about1.2e-7 supports actual FP32 logit updates. This is descriptive evidence, not
a new native pass. Contact/affinity metrics remain distinct. Audited chemistry
loss0 at0/1000/1314, weight0.001; ramp is still future. No LR/objective change.

Archive11347bytes SHA
`f84014400a1c3f380a891d2c69068f81d73cf6fc9c5fffe701792c6c6e72cf3c`,
all11files safely extracted/byte verified. Audit JSON SHA
`c09198bde08ddba6b8b23a64ecb675b58ccf9de9a00a66d0bf6e53a5119e9708`.
Actual final scheduler SHA
`dfda43683ee8ad46079b94f235d90273175b001a4806df1a25e885bfc6d363f9`
and LSF stdoutSHA
`3b713648173804998c9496ff1db2aa21b489114d69eebb8220311776e11ff390`
authenticated separately from in-job packaging. Proofs/verification are at
`results/hpc/20261021-brd4-jq1-gradient-audit-01/`; campaign/script receipt in
`scripts/hpc/receipts/20261021-brd4-jq1-gradient-audit-01/`.
All3native GPU jobs remain unchanged and no completed trajectory is claimed.

### 21 October 2026 — Edition0003 delivered from saved prefixes

Interim **29770891** actually DONE/LSFsuccess/passed, captured14:02:25Paris:
updates5900/1409/1314, states5901/1410/1315, all8626frames retained. Source04 and
all native inputs/gates unchanged. Exact receipt archive33764779bytes SHA
`4e744970c5b095b9a2dfdf4c606abf266328b98535e78c546138744f2dbffd07`;
208files safely extracted/byte verified. Report assessment SHA
`933716fc8f6f40dcc61eecfd0b0ea6ba2381d7a00968c07c80ee9e2315e6d697`.
CPU71-pass rawXML/proofs, source/input pins, every-frame metadata/chunk bytes and
actual final scheduler/LSF success authenticated separately. No local numerical
work; all scientific rendering was allocatedHPC.

Published immutable edition0003 at http://127.0.0.1:8776/report.html, preserving
earlier editions. All187served files HTTP200/exactbytes. Actual browserQA:
all3selectors/latest endpoints, mouse drag3020, numeric1500, chemistry777 plus
next/playback to781, manualscrubpause500, heatmaps/labels/probabilitysizes/rawgraph.
Screenshot and verification at
`results/hpc/20261021-brd4-jq1-25-report/edition-0003/`.
Milestone progression5500/0/0; nextremoteinterim-0003/localedition0004. Report is
still INPROGRESS with terminal/native full-budget checks pending. GPU jobs remain
29770179/29770093/29770124 without mutation. Monitoring continues quietly until
new milestones or actual authenticated final completion/failure.

### 21 October 2026, 14:18 Paris — Milestone capture queued

One compact SSH status/artifact query showed all three native jobs RUN with
committed states7140/1557/1420 and no terminal summaries. Completion collectors
and final report remain PEND. Protenix crossed6500 and Boltz crossed1500 since
the latest published milestone5500/0/0; no interim was active and the next
unique parent did not exist. Submitted report-only **29771001** at14:18:57Paris:
unchanged source04, hpc/2slots/4GiBslot/30min, no native dependency or mutation.
Exact actual submission arguments and scheduler receipt were downloaded into
local edition0004 and copied to the source04 campaign receipts. Scheduler SHA
`4e30a59afa3931b233c8806e33e740be12ace4a639cf17c21b0e1b054a004400`.
Remote parent is source04root/interim-0003. Submission is not a passed report;
completion authentication, full download/verification and browserQA remain
pending. Published edition0003 remains intact at the stable8776link.

### 21 October 2026, 14:28 Paris — Next interim completed; download in progress

One compact scheduler/artifact query showed native jobs29770179/29770093/29770124
still RUN at committed states7725/1623/1511; no terminal summaries, final
collectors/report still PEND. Interim **29771001** is actually DONE/LSFsuccess
with a passed remote assessment, captured14:21:34Paris at7175/1570/1439 updates
and7176/1571/1440states. AssessmentSHA
`eff9ea20cb86ceaebd643dd40907a444aeec8f1a8d498de70c3e455789417964`.
Actual final scheduler SHA
`b9fffb7350d09dfd6d92cddfa18ea21279dc082a4439136d909973b27db832b5`,
LSFstdout SHA
`fbd4bad39632876641848d1b4b7c90c32a65af6ac8eeb73be2f03a4266e99091`
were downloaded and locally byte/status authenticated.

Exact archive receipt39386991bytes SHA
`dd54d783a35a52e571edd731abea859af19f66075393f2b6985bcd3664a8973c`,
236files. Archive transfer is in progress into reserved edition0004 (exec
session24976, followed automatically by path/length/SHA/safe extraction checks).
No full local byte/schema verification, publication or browserQA is established
yet. Do not duplicate transfer/submission while this active delivery is pending;
current immutable edition0003 remains published. Native jobs are unchanged.

### 21 October 2026 — Edition0004 delivered after transfer and browser QA

The existing archive transfer session24976 finished successfully; no duplicate
transfer/submission was made. Interim **29771001** actually DONE/LSFsuccess/passed,
captured14:21:34Paris, updates7175/1570/1439 and states7176/1571/1440. All10,187
exact frames retained. Archive39386991bytes SHA
`dd54d783a35a52e571edd731abea859af19f66075393f2b6985bcd3664a8973c`,
236files safely extracted and byte verified. Reportassessment SHA
`eff9ea20cb86ceaebd643dd40907a444aeec8f1a8d498de70c3e455789417964`.
RawCPU71-pass XML/proofs, native/source/config/frozen/readiness pins and exact
every-frame metadata/chunk bytes authenticated separately. All numerical work
and scientific rendering remained allocatedHPC; local stdlib byte/XML/schema only.

Published immutable edition0004 at http://127.0.0.1:8776/report.html, preserving
all earlier editions/servers. All215served files HTTP200/exactbytes. BrowserQA
all3selectors/latest endpoints, drag3673then1299, exact1500/next1501, playback
777to780/manualpause500, heatmaps/labels/probabilitysizes/rawgraph passed.
Screenshot/verification under localedition0004. Still INPROGRESS with original
terminal checks/full budgets pending. One compact statusquery earlier14:38Paris
showed all3nativejobsRUN at8300/1700/1571, no terminal summaries, finalcollectors
and finalreportPEND. Those newer states are not included in edition0004 and no
terminal completion is inferred. Published milestones6500/1500/0. Jobs unchanged.

After verified edition0004 delivery, the already checked newer native states
8300/1700/1571 crossed additional per-arm milestones7500/1500/1500. Inspected
root/jobs and verified the next unique parent absent, then submitted report-only
**29771969** at14:59:03Paris, remote interim-0004 / localedition0005. Exact actual
arguments hpc/2slots/4GiBslot/30min and standard source04 wrapper are saved;
no native dependency or mutation. Downloaded submission/scheduler receipts,
schedulerSHA `6ba1c1093ee6c4972af97dca6ca7d48001a85ec08e21ebf70b2b44f4a481ce7c`.
This next submission is not a passed or delivered report. Edition0004 remains
published until another fully authenticated/verified edition passes browserQA.

### 21 October 2026 — Matched gradient-scale analysis requested by user

Read-only allocated saved-data analysis, no scorer/new derivative/repair/native
update, using fixed report capture7175/1570/1439 and all original input pins.
Attempt02 CPU **29772023** EXIT1 after its analysis-only exact initial cross-model
array equality assertion failed. Failed script SHA
`4057c70b1e9714a5e757968409a3e8b9c94a1f8f0c98cc34cfef4466526ad1d8`
and logs/actual terminal scheduler preserved under audit02 receipts/results.
This is not a molecular run or native gate failure. New immutable attempt03
measures/discloses initial differences while keeping every scientific binding:
atoms exactly equal, bond max5.9604645e-8/RMS2.7476384e-8 (FP32-scale difference).

Corrected CPU **29772035** actually DONE/LSFsuccess, descriptive outcome recorded.
Two CPU slots/4GiBslot/20min in hpc; actual submission receipt/arguments saved.
Python SHA `82f1be70f73ac7e86c8d0449d688c2a43c8ced19f29cd795ad649ec62364bef7`,
wrapper SHA `feb66e7906d33471f2366054c8c9f3ead20c4af20ccecb5e604df37886340c4b`.
At updates1300–1399, median logit-gradient norms0.005920/0.303586/0.303586;
accepted LR0.03/0.001875/0.001875; measured logit-step norms0.00017753/0.00053153
/0.00053153; probability-step RMS1.292e-6/3.977e-6/3.977e-6. Boltz gradient51.28x
larger but LR16x smaller; actual step2.99x and probability movement3.08x larger.
First100 updates both LR0.03, Boltz gradients16.22x and probability movement
15.11x larger. Raw gradient size difference already present before softmax;
pullback attenuation comparable, so no disconnected Protenix gradient inferred.
Different objective units/reductions preclude interpreting norms as model quality.
No LR change/new native optimization was submitted; future LR0.1 pilot is only
a recommendation. Detailed matched windows and figure in report plan.

Archive585931bytes SHA
`19501ab905634cafb6acddcd9560d3cf8aa81a5cec9051c0ab91811c549a24f7`,
all14files safely extracted/byte verified. AnalysisSHA
`7f95f1a3692fc5380f39909d6dde975a368fd40b4172eacab1709dfff5a9454c`;
figureSHA `6bb952970750a5ac3f6470721df4efac4081c14b5eee903ff0d28a7f789443aa`.
Final schedulerSHA `9ed4aa147bbe5054652e3c4c5040fccb7297fbdb47a0d6a0f5045cbd929a9153`
and LSFstdoutSHA `6dde2c7b1fdd8d67d0ac1dc8215e76d2c977a18fc5a92862288319da8627b4b9`
authenticated separately. Verification/figure/JSON under
`results/hpc/20261021-brd4-jq1-gradient-audit-03/`. All numerical analysis/plotting
allocated HPC; local AST/bytes/schema only. Native jobs29770179/29770093/29770124
and report pipeline/earlier editions remain unchanged.

### 21 October 2026 — Edition0005 delivered

One compact15:09Paris status/artifact query authenticated all3nativejobsRUN,
committed9946/1918/1776, no summaries; finalcollectors/reportPEND. Interim
**29771969** actuallyDONE/LSFsuccess/passed, captured15:02:18Paris through
9425/1859/1726, states9426/1860/1727, all13,013frames. Exactreceiptarchive
49698929bytes SHA `ef0f74b678f8ded9caa333cfa28d0a313076eafc99cdc7bcee94c2b1bce473a3`;
294files safely extracted/byte verified. BinarySSHstreaming to exclusive local
file retained every exact path/length/hash/archive-member gate. AssessmentSHA
`4e9886989e697b1ae66e0b1c08f97f254170fb2d1c054206a973211c741e3632`;
finalschedulerSHA `aa0fe227dcb97cdce8689656fb2e20f49b33984cd6f41eb326f99a9140ef2bca`,
LSFstdoutSHA `33eca8af1b4731b21ebe290cb762acd9550466f55b1c6509bed210361a4b210b`.
RawCPU71passes/noerrors/failures/skips, source/input/frozen/readiness pins and
every-frame metadata/chunk bytes authenticated separately. Local stdlib only;
scientific rendering remained allocatedHPC.

Published immutable edition0005 at http://127.0.0.1:8776/report.html. All273served
files HTTP200/exactbytes. Actual browserQA all3selectors/endpoints, drag4825/1707,
numeric1500/next1501, chemistryplayback777to779/manualpause500 and exact heatmaps/
labels/probabilitysizes/rawgraphs passed. Screenshot/verification localedition5.
Native jobs29770179/29770093/29770124 unchanged; reportstillINPROGRESS. No terminal
completion inferred from Protenix9946. Published milestones8500/1500/1500;
nextunusedremoteinterim-0005/localedition0006. Older editions/servers preserved.

After edition0005 delivery, the checked Protenix9946 state crossed the next9500
milestone beyond published8500. Verified the unique parent absent and no other
interim pending, then submitted report-only **29772558** at15:14:05Paris:
remoteinterim-0005/localedition0006, unchanged source04, hpc/2slots/4GiBslot/30min,
no native dependency/mutation. Actual arguments/receipt/scheduler saved, SHA
`095f9f847c54393f8f0804ca6437f30d5cbd2bc4f76469b83a272c462294ac43`.
This is not a passed/delivered edition or native completion. Published0005 remains
intact until another fully authenticated/verified/browser-checked report replaces
the latest pointer. Nextunused after this is interim-0006/localedition0007.

### 21 October 2026, 15:19 Paris — Protenix terminal collector passed; edition6 transfer pending

One compact query showed Protenix native29770179 actuallyDONE, casepassed with
10000updates/10001states and summary present; allocated completion collector
29770706 actuallyDONE with persisted passed assessment. It authenticated native
DONE/LSFsuccess and full original scientific/source/frozen/readiness/terminal
gates under the sealed source04 contract. SummarySHA
`942b260ea5b1b60271c818e67dc4169c646cb1216d10baeab0d76c20bb3ee0ee`;
caseSHA `77c9712b204ac7a429d5a5477109337416ea48c1e19b97435508cffdcd676d54`.
Local completion-proof transfer is active in execsession51862 into
`results/hpc/20261021-brd4-jq1-25-report/protenix-completion-01/`; full local
verification remains pending. No completion is inferred from schedulerDONE alone.
Boltz native29770093 and chemistry29770124 remainRUN at1995/1839; their
collectors and strict finalreport29770709 remainPEND. No experiment changed.

Interim **29772558** actuallyDONE/LSFsuccess/passed, captured15:15:42Paris at
10000/1960/1812updates and10001/1961/1813states. AssessmentSHA
`ca440b1dc46f87dad88fee291605f9dfe3fba50cfe0a5c642ba4d9bf8562c486`.
Actual final schedulerSHA
`ebae16f5f4923a4812fc3b0b8f24edac9d1a24c745bfb91325bd5ed1cb96928b`,
LSFstdoutSHA `855fe998d336f6741262a60635ab1f2b69be650fef278f2a9702567291028769`
downloaded and byte/status authenticated. Exactreceiptarchive52544644bytes SHA
`f283da0ffb883987fa5385b16a215abca4d4c0918b8dfbee99f7e3ab0cc30518`,
310files. BinarySSH transfer is active in execsession8126 into reserved
edition0006, followed automatically by receipt/path/length/SHA/safe extraction.
Do not duplicate active transfers or submit another capture before this edition
is verified/published. Current edition0005 stays immutable/published. No edition6
local full byte/schema pass, publication or browserQA is established yet.

### 21 October 2026 — Edition0006 verified and published

Report-only interim **29772558** actually DONE/LSF success/passed, captured
21 October 15:15:42 Paris through **10000 / 1960 / 1812 updates** and
**10001 / 1961 / 1813 states**. All **13,775 saved probability frames** retained.
Exact archive52,544,644bytes SHA
`f283da0ffb883987fa5385b16a215abca4d4c0918b8dfbee99f7e3ab0cc30518`;
all310files safely extracted/byte verified. Transfer8126 completed successfully.
Actual final scheduler/LSF success, report assessment
`ca440b1dc46f87dad88fee291605f9dfe3fba50cfe0a5c642ba4d9bf8562c486`,
rawCPU71-pass XML/proofs, native/source/config/frozen/readiness identities and
all contiguous frame/chunk metadata, lengths and hashes authenticated.
All scientific rendering remained allocated HPC; local stdlib byte/XML/schema only.

Published immutable edition0006 at http://127.0.0.1:8776/report.html; all289served
files HTTP200/exactbytes. Actual browserQA covered all3selectors/endpoints,
mouse drags5119then1811, numeric1500/next1501, playback777to779/manualpause500,
exact atom/bond probability tables, overlays and readable raw graph labels/sizes.
Screenshot and verification under
`results/hpc/20261021-brd4-jq1-25-report/edition-0006/`.
Captured endpoint contact loss-0.08846936, Boltz affinity scalars0.22692089 and
0.23105149; these remain different metrics. Checked endpoint argmax graphs remain
C25 chains. Chemistry loss0 at1812, coefficient0.001; delayed ramp not yet begun.

The combined report remains IN PROGRESS. Separately, native Protenix29770179
and terminal collector29770706 are actuallyDONE, with persisted passed collector
assessment/full native gates established on allocated HPC. Local full-summary/
compact-proof transfer51862 remains active into protenix-completion-01; full local
native proof verification is still pending, not inferred from schedulerDONE alone.
The15:39Paris compact query showed Boltz29770093 RUN at2152 and chemistry29770124
RUN at1970; their collectors/finalreport remainPEND. Native runs/settings unchanged.
Published milestones10000/1500/1500; no new milestone beyond this edition, so no
new interim submitted. Next unused remoteinterim-0006/localedition0007. Older
editions/servers and unrelated PDF/OneNote changes preserved.

### 21 October 2026 — independent Protenix LR0.3 authorized and prepared

User requested a fresh10000-update Protenix comparison with nominalLR0.3,
leaving liveBoltz29770093/29770124 unchanged. New version/config/controller
preserve all original scientific settings/gates and no diffusion gradients.
Static review corrected configuration routing: old driver hard-coded0.03; new
pilot/production arguments and actualattempt-rate verification use declared0.3.
Original sealed experiments remain unchanged. Separate work3 output/quota gates
avoid consuming liveBoltz home storage. Local AST/JSON/static checks only; no
new source/job/pass yet. See docs/brd4_jq1_25_protenix_lr03.md and its campaign.

### 21 October 2026,16:10 Paris — Protenix LR0.3 CPU pipeline submitted

Source01 authenticated552files,manifest
`d5c1bf07868a6ef8f3c1a96a24afee11b76c8fb2affe334fb924d0a38b7849a0`,
archive39555775bytes SHA
`8c50ef45cf1d60d90b380f21e39aa41cebd95fa6aefd01c0fe5e107fde68ba5c`.
All145 originalsrc/ files unchanged; onlynew nominalLR/routing/scope/storage
contracts added. Dedicatedwork3 finitequota passed9GiBavailable/5GiBrequired
plus100000inodeheadroom; repeatedstagegates remain mandatory. ActualCPU29776049
and afterCPU29776050 submitted; native/long not yet submitted or certified.
Both liveBoltz29770093/29770124 remain unchanged. Fullplan/receipts in
docs/brd4_jq1_25_protenix_lr03.md and correspondingcampaign.

### 21 October 2026 — fixed-step three-arm restart authorized; originalBoltz intentionally stopped

User explicitly selected bothBoltz arms for fresh10000-update restarts atfixedLR.01,
and Protenix atfixedLR.3 withnohalving. This supersedes earlier leave-running
instruction for29770093/29770124 only. Both were RUN atstoprequest; actualEXIT/
TERM_OWNER authenticated16:14Paris afterbkill. Preserve all savedpartialresults;
no scientific instability/fullcompletion claimed. Source01ProtenixCPU29776049
EXIT134contractpasses/one fixture-path failure; callback29776050EXIT; noGPU/long
was submitted. New fixed-step source corrects fixtureisolation and retains every
scientific/finite/native/parity/identity/runtime/quota gate. No newfixedsource/job/
pass yet. See docs/brd4_jq1_25_fixed_steps.md and newcampaign.

### 21October16:20Paris — fixed-step source01 CPU pipeline actually submitted

585members authenticated,manifestdc4ac68d1818038ba177241311aecf62072a065ef147356186f1f49eba2d20f3,
archive39583098bytesSHAad7492baed5193abaa632810694c86e2ed29455cb5c6bd136b276cd59eb63e68.
CPU29776083RUN with150contractpasses, broadregressionspending; callback29776084
PEND. Native/longnotyetsubmitted/certified. FixedratesPro.3/Bol.01/chem.01,
onecandidate,nostepsearch. Work3freshquota9GiBavailable/5GiBrequiredpassed;
fullnative/pilot/runtime/sumofthreeartifactprojectiongatesremainhardrequirements.
OldBoltz2407/2208committedprefixeshash-preserved; oldfinal29770709EXITafter
intentionalunstartedretirement. All historicaldata/reportspreserved.

### Fixedsource01 CPU stops before native; source02 fixture correction

29776083/29776084EXIT:150contracts passed;legacy237passed/two toyfixturefailures,
3deliberatelydeselected. Nonzero coupled derivative guard correctly rejected
zero-bond/constant toys. Newsource02 changes fixtures only and outputrootidentity;
production fixedrates/nohalving/nativeFD/finite/parity/frozen/runtime/quota unchanged.
No native orlong job was submitted. All evidence preserved; fullCPUchecks repeat.

### 21October16:30Paris — fixedsource02 fullCPU482pass, three native validations queued

CPU29776103/afterCPU29776104 actualDONE/LSFsuccess authenticated. RawXML482passes
(150contracts,239legacy,34mapping,59chemistry),0fail/error/skip,3deliberatelydeselected.
Twelveproofsbyte/schema/XMLverified;assessment0e821878a92575c3a02370fc22c2b66df3307accf9d7d1c17917af806eabbad9,
certificatea595c100b39816d0f87f5d7d44c7a94c35d9138e6f510a0d284a79f20d5317f2.
ActualnativeGPUjobs29776117Pro/29776119Bol/29776121chem,callbacks29776118/20/22;
allPENDat16:30:34,exactargv/resourcesauthenticated. No10000-update longrun submitted
yet; allnative/gate/quota requirementsremainmandatory. Fixedrates.3/.01/.01,
nominalrateappliedwithoutsearch; noDiff. See fixed02campaign/plan/proofverification.

### 21 October 19:42 Paris — all fixed-rate native validations passed; storage blocked launch

All three original GPU readiness assessments and allocated callback decisions now
report passed: Protenix29776117, Boltz-2 29776119 and chemistry29776121. Their
saved actual DONE/gpul40s/LSF-success proofs remain available even where live LSF
records have aged out. Native proof counts are 74/52/128; original fixed rates,
parity/derivative/pilot/runtime/frozen/pocket/no-diffusion gates are unchanged.
Full raw local proof download is pending after a connection reset; the captured
producer assessments, decisions and proof hashes are preserved in the campaign.

No 10,000-update job was submitted. Callback29776122 stopped with exit1 at19:11
after the whole-work3 storage check rejected 8,427,765,472 projected bytes plus
4,294,967,296 reserve against 8,419,460,428 conservative available bytes. This
failed storage receipt stays failed; it is not a native gradient failure.

The user subsequently requested downloading previous results and cleaning storage.
Three existing local historical archives were freshly byte/member verified
(21,093 files), together with noncurrent duplicate source-upload archives. Remote
checksums and dependency exclusions are still required before removing each copy.
Only exact backed-up prior Mosaic files may be pruned; keep current inputs,
proofs, release directories, model caches, environments and other projects.
See `scripts/hpc/receipts/20261021-mosaic-storage-cleanup-01/campaign.json`.
A fallback split-output launcher is prepared locally with AST checks only; no
launcher test, release, new job or automatic adoption is established.

### 21 October 2026, 20:03 Paris — Mosaic backup cleanup completed; native proofs downloaded

User-authorized cleanup attempt `attempt-01` completed at
`2026-10-21T18:03:02.766380+00:00` under plan SHA-256
`d428ab7ae11fe13311fcb939a2a8c4e4a4b809b5040baa9cf89bc2d302e80771`.
The actual receipt records **21,099 files removed, 5,459,183,284 logical bytes**,
two changed files retained and one candidate already absent. Only previous
Mosaic copies matching verified local backups were removed. Both saved
unfinished-job inspections returned no Mosaic jobs. Current scientific
dependencies, proof/release directories, caches, environments, home artifacts,
other projects and unknown data were retained.

Refreshed storage-pool 6 quota: **86.45 / 100 GiB**, 296,102 / 20,000,000 chunk
files. This is a historical post-cleanup measurement. Fresh whole-artifact
projection, byte/inode quota and unchanged reserve gates are still required
before long dispatch and at GPU entry. The original storage rejection remains
failed; successful cleanup does not rewrite it.

Audit archive: `results/hpc/20261021-mosaic-storage-cleanup-01/cleanup-audit-02.tar.gz`,
4,611,888 bytes, SHA-256
`6db36da0eb9418f475fef04ae3314876db43751fa743ecdd5b0b6225583521e6`.
All 10 audit manifest files were byte/hash verified using the local standard
library only. The exact `receipt.json` was copied into the tracked cleanup
receipt directory, SHA-256
`ee253fdda15647320b7d3856ead5b0f3450fc1c4c503a0499c642b19afa2f7d2`,
alongside the compact audit/download receipts and evidence manifest. Full
deletion/verification journals remain in the local extracted audit. The campaign
retains the original cleanup-input index binding and binds the updated backup
index; no cleanup script was rerun for this documentation update.

The fixed-source02 native proof download is complete and locally byte verified:
294 files, archive 157,243,652 bytes, SHA-256
`fa4fd76c68be6b310f6b8f7724e26e3c5c86ae0ef530905d2c94f676c00dd210`.
Receipt:
`results/hpc/20261021-brd4-jq1-25-fixed-02/native-evidence-01/native-evidence-download-02-download.json`.
The failed transfer's empty partial remains preserved. All three original native
gradient validations passed; this download is integrity evidence, not another
numerical run or a new long-job submission.

An all-work3 launcher tagged `20261021-brd4-jq1-25-fixed-storage-01` is prepared
locally with AST checks only. CPU validation and staging remain pending, and no
new long-run job IDs are established. Fixed rates, proof requirements and
scientific gates remain unchanged. This documentation/history update used no
SSH, model execution or numerical evaluation and preserved unrelated PDF and
OneNote changes.

### 21 October 2026 — storage-01 and storage-02 preflight failures; storage-03 pending

Storage-01 was actually staged read-only with 11 authenticated files, manifest
SHA-256 `9cc079f0b89256cd723b5c6ee87e76f8a93d1b2db3cfdc8d99fc29272f152e7d`.
Its preflight failed on original pytest `current` symlinks before any CPU
validation or job submission. Preserve this failed attempt.

The new storage-02 release also staged 11 files, manifest SHA-256
`c8115e2686b94d78ff1889b9dc8161fdf00356639524853c46d4100fd8638fd9`;
archive 18,560 bytes, SHA-256
`8ed85ddae3e116640513b84e4786e8891315541768bb069d8cbfd0546124accc`.
It stopped at preflight before CPU validation or submission. Its byte check
passed with 13,468,288,122 available versus 5,368,709,120 required, but an added
filesystem inode minimum rejected zero. Actual saved `fhgfs` counters were
`f_files = f_ffree = f_favail = 0`; independently authenticated finite user quota
reported 19,703,870 remaining files. The original mandatory 100,000-file
headroom guard used quota. This storage engineering rejection does not undo
the original three native scientific passes or establish a scientific failure.

Pending immutable storage-03 will treat only that exact all-zero triplet as an
unreported supplemental filesystem metric, retain the original finite-quota
and 100,000-file guard, require coherent positive filesystem counters and use
their minimum when reported, and fail invalid/inconsistent metrics. Byte
budgets, reserves, original source02 identities and all scientific gates remain
unchanged. No failed receipt is rewritten as passed.

Independent local verification of the raw original CPU/readiness proofs is
recorded under verification SHA-256
`76821f5448323ee08819a21f394dc78105ca4745f58cd3112db024ce49605d6c`.
This is byte/schema evidence for existing proofs, not another numerical check
or a CPU pass for a storage launcher. Storage-01/02 established no CPU pass or
job; storage-03 remains pending with no new job IDs established. This ledger
update performed no SSH or numerical work.

### 21 October 2026, 20:16 Paris — storage-03 CPU/storage passed; long jobs submitted and pending

Storage-03 sealed and staged 11 files with manifest SHA-256
`9ee7bf1d6bf412acdc37fc83415476dbefdf45db89533571d4a646ee16f33772`.
Source archive: 19,732 bytes, SHA-256
`0473d0908ff6139bc4cbaf98421eab96a33db1f5614db78638500d0bb9684b8e`.
CPU **29777704** and callback **29777705** actually reached DONE/LSF success at
20:15:26 / 20:15:44 Paris. All **65 allocated storage-contract cases passed**,
with no failures, errors or skips. The original 597-file science02 source,
482 CPU cases and 254 native proof results were adopted unchanged; no new
native model run is inferred from these launcher tests.

The dispatch gate at `2026-10-21T18:15:42.453744+00:00` passed for all three
new work3 outputs. Full projected artifacts 8,427,765,472 bytes plus unchanged
4,294,967,296-byte reserve required 12,722,732,768 bytes, versus conservative
available **13,468,221,517 bytes**. Authenticated finite quota left 19,703,830
files, retaining the 100,000-file guard. The exact all-zero `statvfs` inode
tuple is recorded as `unreported_all_zero`, not independent filesystem inode
evidence. Quota/headroom is not a reservation, and GPU entry must repeat its
storage gate. Actual receipt:
`results/hpc/20261021-brd4-jq1-25-fixed-storage-03/launcher-evidence-02/quota-dispatch/split-storage-gate.json`.

Fresh 10,000-update jobs actually submitted at 20:15:43–44 Paris:
**Protenix 29777709** at fixed LR 0.3, **Boltz-2 29777710** at fixed LR 0.01,
and **chemistry 29777711** at fixed LR 0.01. All were **PEND at 20:16:22 Paris**,
waiting for physical GPU availability; no started/completed trajectory is
claimed. Each requests `gpul40s`, one exclusive GPU, four CPU slots,
8 GiB per slot and 23 hours, with no dependencies on the other arms.
New output root:
`/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-03`.

Local launcher-evidence download: 33 files, archive 17,016 bytes, SHA-256
`bd62f30084ec05c0c0e63063f0090a3c9c4d9d05d43d9f0549317d057d2c4c90`.
Byte integrity checks completed; independent XML/schema verification is still
underway. All original storage and storage-01/02 failures remain immutable.
This documentation update performed no SSH, numerical tests or job actions.

Latest actual compact query, **21 October 20:35:30 Paris**: Protenix
**29777709 RUN**, entry quota passed, with **951 committed states / 950 updates**
(state index 950, last update index 949). No summary exists; the trajectory is
partial and not completed. Boltz-2 **29777710** and chemistry **29777711** remain
**PEND**, without study outputs or an established initial molecule. Independent
verification of the new 65-case XML and launcher proofs remains underway.

Independent standard-library byte/XML/schema verification of all 33 launcher
proofs is complete. Raw XML confirms 65 passes without errors, failures or skips;
actual CPU/callback DONE and LSF success, all original adoption pins, whole-campaign
storage and the three independent submission/resource vectors matched. Receipt:
`results/hpc/20261021-brd4-jq1-25-fixed-storage-03/verification.json`, SHA-256
`e7edc4a4fd07ec1968fba600ce7ef19f691404aefede8fbdd588d0dc198bd90f`.
This verifies launch readiness and submission; trajectories are still in progress.

### 21 October 2026, 20:39:34 Paris - Boltz entry quota failures; Protenix partial progress

The actual compact query shows **Protenix 29777709 RUN**, with a contiguous
**1,576-state / 1,575-update** committed prefix and no summary. It is not a
completed 10,000-update trajectory; the current job/results remain preserved.

**Boltz-2 29777710** and **chemistry 29777711** actually **EXIT before any study
or model execution**, stopped by the whole-campaign entry byte-quota guard.
Boltz required 12,502,649,138 bytes versus 12,173,988,090 available; chemistry
required 12,496,820,290 versus 12,140,892,177 available. Raw quota used was
87.29 / 87.41 GiB, compared with 86.70 GiB at Protenix entry, lowering the
rounded quota-derived available bound by 1 GiB. File quota retained about
19.7 million available files; the all-zero filesystem inode tuple was correctly
recorded as unreported. Original native gradient readiness still passed for
both arms. Neither failed Boltz job performed a scientific step; these are
storage failures, not scientific validation failures.

Failure evidence: 13 files, archive 6,088 bytes, SHA-256
`7cdbb679cc3acb4e38c8a1a28fa8727a2b934362a95c3104858c6a94da157739`.
The shared byte verifier passed; local artifacts are under
`results/hpc/20261021-brd4-jq1-25-fixed-storage-03/entry-failures-01/`.
Preserve these failures, all three original source/outcome records and the
running Protenix prefix.

Root is investigating a declared launcher-only split-storage recovery for the
**two Boltz arms only**. No new recovery source, CPU pass or submission is
established. The home quota snapshot was 18.62 / 30 GiB, last updated at
19:03:58; bounded Mosaic-only home inventory is underway, with no output fit
yet established. No gate waiver or Protenix restart follows. This documentation
update performed no SSH, numerical work or job actions.

### 21 October - local split-storage recovery candidate; access and capacity unresolved

The active source04 recovery candidate is packaged locally under
`20261021-brd4-jq1-25-fixed-storage-04`, package **source03**: **12 files**,
archive **25,613 bytes**, SHA-256
`0ae3e8359f5ec7e6b25684746fca90aecc080c1fb2e86c59b71822981fd784a0`,
manifest SHA-256
`75a76dacf8de25f4ebd2d3b3fe114b1af6092064a01edf141650295300670ab3`.
The campaign, source manifest and `local-static-verification.json` are saved in
`scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-storage-04/`.
AST and standard-library byte checks passed for all 12 files against the archive,
manifest, working files and Git staging bytes. Two earlier local packages are
preserved with separate receipts; only mixed line endings in the storage helper
and fixture were normalized, with identical Python ASTs. **No remote staging,
allocated CPU pass, observer or replacement Boltz job is established.**

The declared placement keeps existing Protenix 29777709 unchanged on work3,
places only a new Boltz-2 output on work3, and a new chemistry output in a
separate dedicated home Mosaic directory. Scientific source02, configurations,
readiness and fixed-step settings stay pinned. Each filesystem must independently
fit its combined full remaining assigned projections plus the unchanged 4 GiB
and 100,000-file headroom. The candidate includes a CPU observer for actual
existing-Protenix terminal evidence, gated by allocated contracts and unchanged
adoption identities. All failed storage decisions remain failed.

Home capacity remains **unproven**. The 55-second metadata walk and subsequent
bounded read-only `find` attempt did not complete; partial inventory is not a
resource pass. Saved positive home filesystem inode counters do not establish
byte capacity or a user inode quota. Later SSH attempts failed authentication,
including the bounded 22:15 Paris terminal-evidence attempt (return 255), so no
later Protenix outcome was observed. Its latest verified state remains RUN at
1,575 updates / 1,576 states at 20:39:34 Paris; elapsed time is not completion.
All failed attempts and logs are preserved. No further storage deletion,
scientific execution, local numerical test or report publication occurred.

### 21 October, 22:29 Paris - access restored; source04 CPU validation submitted

The actual read-only query at 22:27:56 Paris found **Protenix 29777709 RUN**,
with **9,656 contiguous committed states / 9,655 updates**, no unfinished JSONL
tail and no summary. It remains unchanged and is not yet complete. The two old
Boltz records have aged out of live scheduler history; their authenticated EXIT
and LSF failure proofs remain preserved, with no study or model execution.

Source04 package source03 was uploaded with exact length/SHA, authenticated for
all **12 files** alongside original source03/science02, checked for shell syntax,
and made read-only. The fresh CPU-only work3 preflight passed: **17,100,677,313
available bytes versus 5,368,709,120 required**, with 19,674,822-file quota
headroom. Raw work3 usage was 82.87 / 100 GiB. This is an updated quota snapshot,
not a reservation or additional cleanup; the complete home/campaign storage gate
is still required. The exact old read-only inventory process inspection found
no active matching Mosaic `find` process before submission.

Actual CPU **29778180** and callback **29778181** were submitted at 22:29:43
Paris and initially observed PEND. CPU requests two hpc slots, 4 GiB per slot,
20 minutes; callback requests one slot, 2 GiB, 20 minutes and only
`ended(29778180)`. Exact actual arguments, IDs and scheduler receipts are in the
source04 campaign and `initial-submissions.json`. Submission does not establish
an allocated CPU pass, a home storage pass, an observer or new Boltz long jobs.
All scientific identities and existing Protenix remain unchanged.

### 21 October, 22:30:44 Paris - allocated recovery contracts passed; completion observer queued

CPU **29778180** actually completed DONE/LSF success. The saved raw XML contains
**85 cases**, zero failures, errors or skips across both recovery modules. The
immutable original-science adoption was captured. Callback **29778181** is RUN
and observer **29778184** is actually submitted/PEND with only
`ended(29777709)`; it observes saved Protenix terminal metadata and does not run
or modify Protenix. No replacement Boltz GPU IDs or complete home/campaign
storage pass are established. The current controller is performing the bounded
complete inventory and fresh per-filesystem budget checks.

A 27-file metadata/CPU proof archive was downloaded and safely byte-verified:
47,750 bytes, SHA-256
`c17196252f310101fb79fa17b1e72a40188149673f2d23d7770635a615656f98`.
Independent local XML/schema verification is underway. Protenix remains RUN in
this actual scheduler snapshot; no terminal scientific completion is inferred.

Independent standard-library byte/XML/schema verification is complete. All
27 metadata files and source archives (12 / 11 / 597 files) matched; raw XML
confirms **37 storage + 48 controller = 85 passes**, no failures, errors or skips,
with raw test-log runtime 3.69 seconds. CPU 29778180 actual DONE/LSF success,
the unchanged original adoption/native proof chain, both old failed Boltz
admissions and Protenix's exact original invocation were authenticated. Receipt:
`results/hpc/20261021-brd4-jq1-25-fixed-storage-04/verification.json`, SHA-256
`dff06db043a51285a90be3d363acf796f51cc8b216d12cc130a5902f31b2919b`.
CPU certificate SHA-256
`b9f7e4a35289b5b6e7ab240b71fe119ed54ca91ef322736152195f83071408b1`;
adoption SHA-256
`58b6b9fc2c55d3f8ebd144f474ab0928145597f9e34ffaa6fa33a68dc08d1cfd`.
This is existing-proof integrity and allocated launcher-contract evidence; it
does not establish a complete storage pass, new Boltz long jobs or Protenix
completion. At 22:33:09 Paris, callback remained RUN and observer PEND.

### 21 October, 22:36:48 Paris - full dispatch storage passed; two replacement Boltz jobs submitted

The saved actual scheduler/artifact query is
`results/hpc/20261021-brd4-jq1-25-fixed-storage-04/user-status-20261021T203646Z/status.json`.
Callback **29778181** completed DONE/LSF success after the complete per-filesystem
dispatch budget passed: work3 **15,874,334,545 available / 6,827,669,155 required
bytes**, home **11,811,160,064 / 9,459,309,120 bytes**. Both required budgets retain
the unchanged 4 GiB reserve. The bounded no-follow home inventory completed with
return code 0 and no recent Mosaic file/link bytes; it was not a partial-inventory
pass. Work3 file headroom is authenticated finite user quota; home's positive
physical filesystem counters are not a reported per-user inode quota. These
checks are observations, not reservations, and each GPU entry must repeat them.

Both independent replacement jobs were actually submitted at **22:34:20-21 Paris**:
**Boltz-2 29778203** is scheduler **RUN**, but has no study, saved state/update or
completed entry gate in this snapshot; **Boltz-2 + chemistry 29778204** is **PEND**,
also with no study or saved steps. Thus both jobs are submitted, but scientific
optimization has not yet been verified to start for either. Their fixed LR remains
**0.01**, one candidate per update and no halving, with unchanged scientific
source02, configurations and passed native readiness. The old failed admission
jobs 29777710/29777711 and their original gates remain failed and preserved.

Existing **Protenix 29777709** is actual scheduler **DONE**, with its summary
present. Allocated observer **29778184** also completed DONE/LSF success and saved
`observed_success`, authenticating the actual same-job scheduler/LSF and terminal
metadata for **10,000 updates / 10,001 states**, including saved summary/case,
JSONL and derivative proof hashes. This does not substitute for the future full
report frame/snapshot completion certificate or imply that all original native
bytes have been downloaded locally. No Protenix restart or further cleanup was
performed. No fixed-step report is published yet; published milestone progress
remains 0/0/0 until the first verified fixed-report edition.

Independent byte/JSON/schema verification of the 28-file compact delivery
evidence passed: archive 21,152 bytes, SHA-256
`ce780f8ffdbeef0b6dacdb2a1dd701caa057537dccfee82b8f318a56ee2129c0`;
verification receipt SHA-256
`f175fa887bfe65aed6bf227b4dea712ecb7e966aa16585ba5982a6ee44746dd6`.
Exact dispatch budgets/reserves, completed inventory/fresh quota, both actual
new submission/resource vectors, callback/Protenix/observer scheduler and LSF
outcomes and compact derivative proof hashes were authenticated. GPU entry
checks and saved Boltz optimization steps remain unverified in this bundle;
the four large Protenix original trajectory files are remote hash references
only, not full local original-byte or report-frame verification.

### 21 October, 22:53-22:56 Paris - updateable fixed-report fork implemented; allocated fixture failures preserved

The user's saved-only updateable report is implemented as a separate fixed-step
contract, full-prefix/strict-final renderers, completion collector, interactive
template, immutable publisher and allocated CPU orchestration. It pins current
scientific jobs **29777709 / 29778203 / 29778204**, the independent per-arm output
roots, original scientific source02/readiness and launcher03/04 adoptions. All
saved probability frames use bounded 100-frame chunks; exact-update input,
previous/next, playback with manual pause, per-arm status/counts, empty-arm
selection and clearing remain explicit. The latest-viewer wrapper checks the
new pointer every minute. This is implemented behavior, not a published report
or an allocated validation pass. Historical port 8776/edition-0006 is unchanged.

Report source01's actual CPU **29778232** exited after **222 passes / 3 failed
synthetic FP32 simplex fixtures**, with zero errors/skips. Callback **29778233**
also exited and correctly prevented all collector/render submissions. The
failed gates, raw XML/logs and actual scheduler/LSF bytes remain preserved;
local standard-library byte/XML verification is
`results/hpc/20261021-brd4-jq1-25-fixed-report/source01-failure/verification.json`.
Native gradients and trajectories are unchanged by this report-only failure.

A fresh immutable **report source02** corrected only the synthetic fixture's
bounded exactly binary FP32 construction and declared PyTest collection identity;
no production probability tolerance, native gate or scientific setting changed.
Its **13-file**, **53,487-byte** archive has SHA-256
`9ed7eb09c942a58f2cb8c9df5880ca7196215abcaf300551a74ee231d5e064e9`,
manifest SHA-256
`3b78673801b3875fd040dff9d814a869e5f0172fa8dea164ffb85656e6b3ad5e`.
Staging reauthenticated all 13 report / 597 scientific / 11 launcher03 / 12
launcher04 files and passed its fresh report work3 preflight. Actual CPU
**29778238** and callback **29778239** were submitted at **22:56:42 Paris**,
with exact arguments in
`scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-report-02/initial-submissions.json`.
The allocated repeat then encountered a collection error because importlib mode
could not import a bare sibling helper used by an unchanged legacy fixture.
No source02 CPU pass, collector/render success or publication follows. Preserve
this failed attempt and require a new immutable tooling correction plus full
allocated checks; do not weaken scientific or delivery gates.

The latest independently observed native update for this report preparation is
**22:41:32 Paris**: Boltz-2 **29778203 RUN**, entry storage and driver invocation
authenticated, **44 committed states / 43 updates**; chemistry **29778204 PEND**,
no study or saved steps. Protenix **29777709** remains the completed unchanged
10,000-update run with passed observer terminal metadata. No native job was
restarted, cancelled or modified, and no additional cleanup was performed.
No new fixed report is published; published fixed milestone progress remains
**0/0/0** pending actual report CPU/render pass, byte verification and browser QA.

### 21 October - report source04 passed all allocated CPU gates; first saved report is running

Immutable report source03 removed the importlib collection mode while retaining
all required fixtures and original import behavior. Actual CPU **29778240**
completed DONE/LSF success with **225 passes**, zero failures/errors/skips.
Certification **29778241** nevertheless exited because **35 unchanged legacy
cases had empty XML class identities**; the 190 new cases were attributed.
That attribution gate correctly prevented collectors and rendering. Preserve
this rejected certification and its original raw evidence at
`results/hpc/20261021-brd4-jq1-25-fixed-report/source03-cpu-evidence/verification.json`;
a raw pass count did not become an authenticated report CPU certificate.

New immutable **report source04** selects the shared releases directory as
PyTest's common root, so the new and unchanged legacy modules all retain their
actual XML module identities. This changes collection bookkeeping only, with
no scientific, renderer, probability tolerance, required-test or gate change.
Its **13-file / 53,476-byte** archive has SHA-256
`e1886095f890058ed167f27ab648aca4561a8bf5851fa1bbf33f7ec0eeed53f5`,
manifest SHA-256
`7af7f5732842bdee8db930b71a6185039a1b77ec2c5939afccd8b2d9fc91800a`.
The report source is read-only under
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-report-04/source`;
its output root is
`/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-report-04`.

Actual CPU **29778243** and certification **29778244** both completed
**DONE/LSF success**. All **225 cases in all seven required modules** passed:
83 fixed contracts + 44 fixed rendering + 42 fixed publication + 21 fixed
orchestration + 26 Boltz report + 3 chemistry report + 6 chemistry-ramp report.
There were **zero failures, errors or skips**, and every case has its required
module attribution. Independent local standard-library byte/XML/schema and
actual scheduler/LSF verification passed at
`results/hpc/20261021-brd4-jq1-25-fixed-report/source04-cpu-evidence/verification.json`,
SHA-256 `22de220c19ca4737c715c29173d62f86197b619ba6bbb84712029289a0035ef1`.
CPU certificate SHA-256 is
`2a5bb4b09c3ded76c03e631e9929a36a0d358ff77a8fe2c53b8379d5d22d5992`.
Numerical fixtures ran only on allocated DTU CPU; local verification checked
saved bytes/XML/schema and performed no numerical rendering or model work.

Actual report-only downstream submissions are now recorded in the source04
`jobs.json`: Protenix collector **29778245 RUN**; Boltz collector
**29778246 PEND**; chemistry collector **29778247 PEND**; first full-prefix
interim **29778248 RUN**; strict final report **29778249 PEND**. Pending Boltz
collectors wait on their own scientific job endings, and the strict final waits
on successful completion collectors. Submission or RUN alone is not a collector
or rendering pass. No new fixed edition has yet passed rendering assessment,
archive/byte verification, publication and browser QA; fixed published
milestone progress remains **0/0/0**. Historical port **8776/edition-0006** and
all scientific trajectories, jobs, native gates and original failure records
remain unchanged. No additional cleanup or scientific rerun was performed.

### 21 October, 23:07 Paris - first fixed-step report published with every captured saved frame

The first fixed-step report is available at **http://127.0.0.1:8777/report.html**.
Its immutable **edition-0001** is published from the separate viewer at
`results/hpc/20261021-brd4-jq1-25-fixed-report/viewer`; the small stable wrapper
checks its latest pointer every **60 seconds**. The historical report at port
**8776** still serves its preserved edition-0006. No historical edition or
scientific trajectory was rewritten.

Report source04 interim **29778248** actually completed **DONE/LSF success** and
passed assessment. The allocated Protenix completion collector **29778245** also
completed DONE, authenticating all **10,001 original saved snapshots** in
addition to the terminal native checks. The first report captured
**2026-10-21T21:03:01.148625Z** (23:03 Paris):

| Current arm | Saved updates | Saved states | Authenticated captured status |
| --- | ---: | ---: | --- |
| Protenix 29777709, fixed LR 0.3 | 10,000 | 10,001 | Native completed; terminal and every-frame collector passed |
| Boltz-2 29778203, fixed LR 0.01 | 765 | 766 | RUN; own contiguous committed prefix |
| Boltz-2 + chemistry 29778204, fixed LR 0.01 | 99 | 100 | RUN; own contiguous committed prefix |

Every captured probability frame and scalar point is retained in bounded
100-frame chunks. There is no interpolation, new inference, resampling, repair
or fabricated molecule. Both Boltz runs keep one finite fixed-rate candidate per
update with no halving. The combined report remains **IN PROGRESS** while
Protenix is shown individually completed.

The original source04 report archive is **37,187,599 bytes**, SHA-256
`bc7bf1da47981a4638bb4d06bf7e321d3b101053e26e92737ce0aa187dc61ae5`.
All **256 archived files** passed safe extraction and exact byte verification.
The first local publication attempt was correctly blocked before viewer creation
because its CPU scheduler parser did not accept an actually line-wrapped LSF
queue field; the original report archive and scientific outcome remain valid.
A new immutable **publisher-only source05** compacts whitespace for parsing the
unique exact CPU job/status/queue, retaining original proof bytes/hashes and all
authentication gates. It uses manifest SHA-256
`72c3569e49676dc00c448137829fea6ff3abba342b9d512979920314a8845698`,
archive **53,591 bytes**, SHA-256
`ab6bc6b2b026818e8b00fb8e6408a741c030e62edb752bacf8c2c56e07de76ca`.
Its CPU-only validation **29778252** actually completed DONE/LSF success with
**225 passes across all seven required modules**, no failures, errors or skips;
independent verification SHA-256 is
`5fe35da486dcad6d7c8d7b3634c0eeb2de98967193f55bb65cfc30f69370db46`.
No source05 dispatch, collector, rerender or scientific execution was performed.
The allocated-passed source05 publisher copied the **unchanged verified source04
report bytes** into the new immutable edition.

Publication time is **2026-10-21T21:07:20.278149Z** (23:07 Paris).
Published HTML SHA-256 is
`a5aae7516495443924989e47b666bca6d1b838cec3f5253b32bf64c4c5039f3d`;
report receipt SHA-256 is
`c38ee0cf56db85f3fe0ed212aab13a33809f7889e5e06c03cd87964b21108901`.
All **230 served files** returned HTTP 200 with exact expected bytes; HTTP
verification SHA-256 is
`264e5a01c5249f0f1180660503da5e31fe13cc43df9ca07e4913d033a84796c3`.

Browser presentation QA exercised actual Protenix slider drags to **2397 / 7201**,
numeric **1500**, next **1501**, middle **5000** and latest **10000**. Boltz
playback reached **2**, manual numeric **321** paused playback, previous reached
**320**, and its own latest endpoint was **765**. Chemistry interior **37** and
latest **99**, all three selectors, exact saved atom/bond heatmaps (**25 atoms /
300 unordered pairs**), raw graph labels and the visibly invalid Protenix graph
were checked. No captured arm was empty, so empty-arm clearing is covered by the
allocated fixtures rather than claimed as an actual empty-arm browser test.

Published fixed milestones are now **10000 / 0 / 0**. The ten-minute monitor
continues using each arm's latest published milestone, submitting a new saved-only
allocated CPU edition at **1500, 2500, ..., 9500, 10000** per arm and a strict
final report after all three successful terminal completions. The viewer updates
after those verified editions are published. Contact remains distinct from
affinity; changing repair totals are not one fixed objective, and a valid graph
or fixed-pose score does not establish binding or activity. No additional cleanup,
scientific setting change, native rerun or local numerical work occurred.


## 21 October2026 — authorized learning-rate increase and confirmed stops

The user requested fresh10,000-update restarts with Protenix LR3.0 and both
normal Boltz-2 and Boltz-2+chemistry/valid-output-loss LR0.05. Old queued
report collectors29778246/29778247 and final29778249 were verifiedPEND and
retired first. Both native Boltz jobs29778203/29778204 actually EXITed with
TERM_OWNER and LSF130 at23:25:47 Paris; completed Protenix29777709 and all
artifacts/reports are preserved. No new high-rate job or pass is established
yet. See docs/brd4_jq1_25_fixed_lrboost.md and the new campaign stop receipts.


At 23:36:42–43 Paris the new higher-rate release authenticated 610 source files
(manifest `9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323`)
and actually submitted CPU **29778388** / after-CPU **29778391**. Initial CPU
storage admission passed (15,292,863,601 available / 5,368,709,120 required bytes,
4 GiB reserve). Full CPU/native passes and long submissions require actual
terminal evidence; they are not inferred from these submissions.


At 23:41:11 Paris both new CPU/callback jobs were actual DONE, with all **598**
cases passed (266 contracts + 239 legacy + 34 mapping + 59 chemistry), no
failures/errors/skips and the original three deliberate legacy deselections.
CPU certificate SHA-256 `dc76df74f1432f59e4c5f6fb5491a1258752b639df893cf75bb5a6968e481630`.
New GPU validations/callbacks are Pro **29778424/29778425**, normal Boltz
**29778426/29778427**, chemistry **29778428/29778429**. At 23:42:21 all six
were PEND: physical GPU availability for validations, own-ended dependencies
for callbacks. New long-run IDs/readiness passes are not established. All
56 downloaded CPU/launch evidence files passed safe extraction and byte hashes;
independent XML/schema proof verification follows. See the new high-rate plan
and campaign receipts. Monitoring was updated to the new rates and exact jobs.


Independent high-rate CPU/launch verification passed
(`225ce416f5aabb9d549d4ecd066f2e062e66d3b92e30f05723562c46dc50cbca`),
including all 56 downloaded files, exact 598 actual test rows/log counts,
original scheduler/LSF successes and all six new submission vectors. The
unchanged aggregate XML subtest-count distinction is documented, not inflated
into extra passes. Grouped native-validation storage also passed
(7,516,192,768 required / 15,292,576,076 available bytes, 4 GiB reserve).
All six new native/callback jobs remained PEND at 23:45:52 Paris; no readiness
pass or long-run ID was claimed. Code and receipts pushed as `046430e`.


## 22 October 2026 — higher-rate readiness passed; long outcomes authenticated

All three new native readiness jobs passed with original gates. Full per-FS
storage and each entry gate passed, then three fresh long jobs were actually
submitted at 01:23:09–10 Paris: Pro **29779177** at LR3.0, normal Boltz
**29779178** at LR0.05, chemistry **29779179** at LR0.05.

Pro and normal Boltz each saved **10,000 updates / 10,001 states** and their
optimizer case reports passed. Both overall scientific summaries nevertheless
**failed terminal raw-gradient validation**. Pro raised `Predeclared epsilon
leaves the simplex interior` (original LSF exit1 at04:00:35); Bol's E_upper FD
sweep failed the two-adjacent-scale requirement, with false/false/true at
0.01/0.003/0.001 (original LSF exit1 at08:17:00). Frozen identities stayed fixed.
These are failed terminal scientific gates, not successful certified completions
or resource-admission failures. No retries, LR/FD changes or gate waivers.

Chemistry was actual RUN at09:40:51 and kept saving around09:45. Separately read
state/update logs yield a conservative shared committed prefix **8,677 updates /
8,678 states**. Do not treat the asynchronous extra update row as missing output
or claim an atomic prefix capture. Continue that independent job unchanged.

Delivery-metadata-02 archive **313,614 bytes**, SHA-256
`6448daedff400ef8f66ea30675f9079063ab13b163d4b66b5e6214066c14786d`,
contains 140 safely extracted/byte-verified files, 254 remote readiness proof
rehashes, raw terminal JSON/LSF and exact source/config/entry/dispatch metadata.
Large summaries/case/JSONL/NPZ originals remain HPC references; no local numerical
or full-frame validation is claimed. Aged-out live history is not fabricated;
first packaging failed solely on denied bhist execution and is preserved. All
science source610 and active chemistry remain unchanged. The local read-only
query now follows actual case-0 JSONL paths. See the higher-rate plan/campaign.


Independent delivery proof verification passed, SHA-256
`d939121ba93e71fccc03413d04ce93e3496624b8e81a9ebb0e54015b211ec267`.
It checked 140 listed evidence files plus the archive manifest, source610/config
bytes, CPU598 proofs, all 254 readiness references (64 original proof files
present locally and190 remaining remote-only), original native DONE/LSF records,
three actual long submissions/driver bindings and full dispatch/entry accounting.
The two terminal failures and conservative running chemistry prefix are confirmed.
All full-original, active-file and unavailable-history limitations remain explicit;
this evidence pass does not make either failed scientific run pass. New report
work is specified in `docs/brd4_jq1_25_lrboost_report_plan.md`; no higher-rate
edition has yet been published and all historical editions remain unchanged.


## 22 October 2026 — separate endpoint diagnostics and higher-rate report

The user authorized separate read-only diagnostics of the failed Protenix and
normal Boltz terminal checks, plus a saved-only updating report. Original jobs
29779177/29779178 and all failed raw gates remain unchanged; chemistry29779179
continues independently (actual RUN at10:34Paris,9450 committedupdates/9451states).
No restart, optimizer update, repair, fresh diffusion or scientific gate waiver.

New immutable tooling release `20261022-brd4-jq1-lrboost-analysis-01` contains16
files, manifest `214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc`,
archive76750bytes SHA `9707bec755c9c116a03a56abceabbe1caeaf019922f4bd68cb2a99141546f3e1`.
All17 safe archive members/current/Git-index bytes and13 Python ASTs were
independently verified; no local tests or model execution. Original610scientific
source is external, authenticated and unchanged.

CPU **29780814** and certificate callback **29780815** actually DONE/hpc/LSFsuccess;
all **278 cases** passed across all8 modules, with no failures/errors/skips. Fresh
whole-analysis storage passed with full remaining native projections, 2GiB report
allowance,512MiB bounded diagnostic outputs,4GiB reserve and100000-file headroom.
Separate GPU diagnostics actually submitted: Pro **29780821**, Bol **29780823**,
with own-ended metadata captures **29780822/29780824**. Saved-frame collectors
**29780825/29780826/29780827**, first interim renderer **29780829**, and final
presentation **29780830** were also actually submitted. No diagnostic result or
new report publication is established yet. CPU/dispatch evidence24524bytes SHA
`b6d05c1eea559db93c650d356e5fffadcac7c68b49fe61dc0084a6fa7125f8f7` safely downloaded
and all56 member hashes checked; independent raw XML/proof/schema audit follows.
See the new endpoint diagnostics plan and exact campaign/submission receipts.

Independent analysis CPU/dispatch verification passed, SHA-256
`ac8d58f18e84cb15823e7e68077c0ac04cfa82e35be129a6c44ddf6da3a080e9`.
All56 downloaded files, actual CPU/callback DONE/LSF successes, exact278 raw XML
rows and all8 module counts,11 submitted resource/dependency vectors and6 quota
gates were checked. The original610source/compact provenance chain was rehashed.
Pro/Bol presentation collectors actually DONE after every10001 saved NPZ frame
passed its audit; their scientific_status remains failed. First rendering and
separate GPU diagnostics are still pending completion. Code/receipts pushed
as `d1b65f2`; older reports and unrelated PDF/OneNote files remain untouched.

The first higher-rate report is now published and browser-verified at
http://127.0.0.1:8778/report.html, separate immutable edition-0001. Actual render
29780829 DONE/hpc/LSFsuccess captured10000/10000/9600updates, all29603 saved
states, with original Pro/Bol failures visible. Published milestone floors are
10000/10000/9500. Presentation success does not certify the failed native runs.
Archive154498144bytes SHA
`f83155ecd2b83b4eff0d0c7693d8142b3d8eeeccfd4f6035784e9581711a170f`
and all636 files passed safe extraction/hashes. Independent delivery verification
`6571fa254b224a7f70ed1b97865b09dec8590423a5debd1718b28f8b5d04d588`;
HTTP608files and actual browser controls/heatmaps/raw graph checks passed. New
hidden localhost server PID41952; old8777/8776 reports/servers preserved.
Initial SCP timed out, its process was confirmed ended, and the existing prefix
was authenticated before a successful byte-stream resume; no active transfer.

At11:03:11Paris diagnostics29780821/29780823 and captures22/24 remained PEND;
chemistry29779179 RUN, completion collector29780827/final29780830 pending original
dependencies. No diagnostic finding/pass yet. Monitoring continues with the new
published milestone receipt and exact submitted jobs, without modifying saved runs.


## 22 October 2026 — chemistry full budget and second saved report submitted

At11:13:55Paris original chemistry29779179 actualDONE/gpul40s,10000 contiguous
updates/10001states; original summary reports passed,311775359bytes SHA
`aa96fa193252b2205a89f31fefc352b4303a4878fe04f314146cbe8521c05cd0`.
Independent collector29780827 was RUN, so full terminal/frame verification and
delivery were still pending. Original failed Pro/Bol trajectories stay failed.

New saved-only interim29781041 (`interim-0002`) actuallysubmitted09:12:04UTC,
thenRUN/hpc with unchanged source16/CPU278/originalsource610/input pins. Full
entry storage gate passed8306479405required/9534557045available,4GiBreserve and
100000-file headroom retained. Lock/prior publication/actualjobs/exclusive
markers/contiguouschemistry prefix guards authenticated before submission;
only one new report job. The first caller guard failed before submission on
archive directory names; failure and read-only inspection preserved, with a
narrower unsealed caller filter and no sealed tooling/scientific change.

Both separate diagnostics29780821/29780823 and captures22/24 remainPEND;
existing final29780830 remainsPEND. No diagnostic findings or new publication
are established. Published chemistry milestone remains9500 in edition0001.
All original runs, old reports/servers and unrelated PDF/OneNote changes preserved.

At 11:18:34 Paris chemistry collector **29780827** was actually DONE/hpc, with
its assessment passed, scientific_status passed, 10000 updates/10001 states and
all_probability_frames_verified=true. Its original scientific job **29779179**
was DONE/gpul40s. The collector assessment is bound by the raw status receipt;
independent local delivery-byte/LSF/proof audit is still pending. Interim29781041
remained RUN, both separate diagnostics/captures PEND, final29780830 PEND. No
new edition was published; original Pro/Bol failures and all immutable artifacts
remain unchanged.


At 11:26 Paris the second immutable report edition was published with all three
10000-update/10001-state trajectories, capture09:19:33.637806UTC. Original
chemistry29779179 and its allocated collector29780827 passed terminal/source/
frozen/frame validation. The independent local byte/XML/schema/LSF delivery audit
passed, SHA5f4fbb8d458af62ab76294f20f4228a80ebdbbdf1d790ae40d10f3d88ba993a6.
The 157404869-byte report archive SHAe92c477b8d173c1c0a9e17813e9cb0f8549cd24c9e97578d43e7591823cbb30a
has647 files; compact metadata1514864bytes SHA81a6d395be476cddf3cf16e5259565cd3e732acb51af18a819382aebf70fc75f
has41 files. Both fresh transfers ended successfully; large original scientific
files remain authenticated HPC references. All1221 served files matched HTTP.
Pro/Bol remain scientifically failed; all_terminal=true does not make this a
successful scientific campaign. Separate diagnostics are still pending.

BrowserQA of edition0002 FAILED the manual numeric-during-playback path: entering
321 selected46 because the playback draw overwrote the textbox before Enter.
Paused numeric navigation, sliders/arrows, allselectors, terminalgraphs and
25atom/300pair heatmaps worked. This is a viewer defect, not a scientific failure.
Do not relabel QA passed or edit the immutable edition/source. A new viewer-only
immutable correction20261022-brd4-jq1-lrboost-viewer-focus-01 is being implemented,
adding focus-to-pause only, with original assets/proofs unchanged and new allocated
CPU/adoption requirements. No new patch source/job/pass is yet established.

## 22 October 2026 — lower-rate fresh restart, chemistry endpoint100

The user requested all three higher-rate LRs divided by10, then clarified
chemistry0.001 to100. Fresh10000updates each: Pro0.3, normalBol0.005 and
Bol+chem_valid_loss0.005. Keep the original hold through5000/geometric endpoint
at10000, all other scientific settings, fixed_step/onecandidate/nohalving and
all native/FD/parity/primal/finite/frozen/runtime/quota gates. Original higher
Pro/Bol failures and passed chemistry remain unchanged, and the two endpoint
diagnostics remain separate. The old three scientific allocations are finished;
no cancellation/resume was necessary. See brd4_jq1_25_fixed_lrreset.md.

Immutable attempt01 source622 manifest47263fedea42e593455b43d4b75c1c874670bb53d934dd93a531aec439abf403,
CPU29781350/callback29781351 EXIT after388contractpasses/3outdated new fixtures.
No native/long submission. Preserve the failure. A fixture-only attempt02 keeps
all scientific config/driver/contract/library bytes identical, changes three
fixture expectations and declared root identities, and repeats the full CPU.
Actual attempt02 source622 manifest832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3,
archive39601171bytes SHA306bdf055770e7245bb37de291800cb7da0998af7345aabb7d1b9e0b97057e25.
Read-only source/control nested under original analysis01root keep existing
sealed diagnostic-entry accounting aware of every new work3 byte.

Actual CPU29781363 DONE/hpc/LSF passed723actualrawcases:391contracts+239legacy+
34mapping+59chemistry, no failures/errors/skips,3deliberate legacy deselections.
CPU certificate6e02392176f71238b1af36564c796615758ba82af7715ebe487bd48030a5a5d5.
Callback29781364 EXIT only at fresh native-dispatch byte admission before any
native or long job: required10200547328 versus available7836349422 bytes. Keep
4GiB reserve,100000files, full3GiB validation plus2GiB oldreport/512MiB olddiag
allowances. Independent61file/rawXML/LSF/schema audit passed
19fd207053ee8ab4974693261d5d3e7090abfba615a4bce3355d4d5808805e27;
evidence112789bytes SHAc681d57d812e5bdd7f58c7cedca92c1a81ba1ba9c873223170f1967a6513cfd9.
CPU pass is not a native pass or an optimization submission.

Under the user's existing verified-backup cleanup authorization, an additional
limited work3-only plan backs up older lower-rate Pro/Bol study copies, an older
fixed report payload and duplicate archives. All current higher-rate studies,
their diagnostics/report inputs, home files, source releases, caches/models/envs
and other projects are excluded. Backup1620686314bytes SHA
1f6b91446dba1866ff7c6e732a1ce2457e60c59661943fdbd2eb6cf870a305f4 contains34847
originalfiles/1540133784logicalbytes, all independently stream byte/hash verified.
Slow backup01 was explicitly identified/stopped and its partial preserved;
backup02 transfer completed. Slow full Windows extraction was stopped with its
partial tree preserved; the separate complete stream audit verifies every
member without relying on that partial extraction. No scientific calculation.

Exact removal plan03bbc4fa67a286b8f51b6c02f3af8b898e0ed4acd80255e9f88527f82787c8bb
uses the existing per-file hash/inode/mtime/active-job/durable-journal algorithm,
an explicit inactive target allowlist and stricter2GiB cap. First execution
failed before verification/deletion because a direct SSH shell lacked bjobs.
Same plan/helper attempt02 runs through the normal HPC login environment; at
10:21:30UTC journal showed7952 exact files/439201656bytes retired, not final
completion. A fresh full unchanged native storage gate and authenticated
CPU723-tested submit API are required afterward. Never rerun the failed
certificate callback or overwrite its failed gate. Exact receipts are under
20261022-brd4-jq1-25-fixed-lrreset-02. No new native/long IDs established yet.

At10:21:56UTC the additional work3-only cleanup completed:34847exact backed-up
copies/1540133784logicalbytes,0changed/absent, all34847local archive members
verified before removal. Receipt56d925421ac6225cf191cdb00a28b8f76d6e46d7dc74de3f39af3a8a30a467b4;
full journal68a1573237fa5dafd21c299e87c1b1a089857ac9dd66d55fc7ed83cd103eb9b1.
All current higher-rate studies/diagnostic inputs/home/releases/models/caches/envs
and other projects remain untouched. Prior lower-rate copies and named duplicate
archives are preserved in the verified local1.62GB archive, with exact origin map.

Fresh original full native-dispatch gate passed10200547328required against
10286294816available, retaining full3GiB native+2.5GiB analysis allowances,
4GiB reserve/100000files. Metadata-only resource adoption reauthenticated actual
CPU723/source622/pins/finished predecessors and used original tested submit APIs
under dispatch lock, without repeating the failed callback or changing any gate.
Adoption434a42093cb466c32eb7fe144eb952f462c72e75014c0a5347fd95b1370ce8d4.
Actual three native submissions: Pro29783703/callback29783705,
Bol29783707/callback29783708, chemistry29783710/callback29783711. All6PEND at
10:27:52UTC/12:27:52Paris; no new native pass/long ID yet. Existing separate
diagnostics/captures/final remainPEND. Native+full two-FS gates still precede
fresh scientific long jobs; the observations are not reservations.

All41new proof files/archive7332680bytes SHA
f6dadc17eb468f87a46cbb177f545682267b365c6963f6aec3a25210a8e8ecf6 verified.
Independent submission/source/config/CPU/wholequota/backup/fullcleanup-journal
audit passed4c7979a05a635d5f7f19feec407827eb8e8af43ecc3ccdaef0740918d69f8361.

## 22 October 2026 — experiment organization policy and agent logbook established

Actor: Codex documentation/organization agent. Scope: local repository inspection
and documentation only; **no experiment executed**, no HPC connection/submission,
and no existing execution file moved.

Added root [agent instructions](../AGENTS.md), a prominent README entry point,
the [experiment folder policy and discovery map](../experiments/README.md),
[copyable experiment/run templates](../experiments/_templates/README.md) and
[script navigation](../scripts/README.md). New experiment-specific work belongs
under `experiments/<family>/<experiment-id>/`, with a README in each directory
and all experiment-owned code/configs/tests/report assets kept together.
This existing ledger is the required global agent logbook, linked to per-attempt
run records with real timestamps, commands, identities, jobs and outcomes.

Inspection found hard-coded imports, launch/config paths and source-packaging
enumeration, so existing experiments require incremental, dependency-aware
migrations. Historical receipt directories, sealed releases, failed evidence,
scientific gates and all pre-existing ledger history are preserved. The inventory
is a proposed ownership map; physical migration and execution validation remain
future work. This policy does not authorize a new scientific run or retry.

Documentation checks resolved all 89 local file/directory link targets across
the eight touched documents, found no missing experiment-directory READMEs or
new-file formatting issues, and passed scoped `git diff --check`. A comparison
against pre-edit copies confirmed preservation of the original README content
and ledger history, apart from these additive policy sections. These are local
documentation checks, not numerical validation.


## 2026-10-22T10:29:42.939514+00:00 — existing scripts migrated into documented experiment homes

Actor: Codex organization agent. User scope: organize existing scripts, update imports/paths, fill READMEs and logbook, preserve current runs and historical receipts. This was local source/documentation work; **no scientific experiment executed**, no HPC connection, submission, cancellation or restart.

Assigned **237 files** to **24 experiment homes** and documented shared release/evidence/graph/chemistry/campaign25/Boltz/reporting/HPC tooling. [Experiment index](../experiments/INDEX.md), [complete old-to-new map](../experiments/MIGRATION.md), and [static audit](../experiments/migration-verification.json) are the review entry points. Added protocol/file/command/dependency READMEs in every created experiment directory and indexed **52 existing dated attempts** with links to unchanged original receipt locations. Existing protocol/history documents remain in docs/ to preserve earlier links. Quoted saved statuses are not live scheduler observations or new scientific acceptance claims.

Moved implementations have old-path Python/shell forwarding entry points; default tests/ discovery loads each canonical campaign test once. Canonical path resolution, controller imports, template/source lookups, shell launch commands and source packaging/authentication now include the new homes. All 41 old/canonical declarative config/template pairs retain identical bytes. Shared fixture/oracle data remain in tests/data/. No objective/optimizer/model code, scientific setting or gate was intentionally changed. One provenance-only tuple in src/mosaic/small_molecule/provenance.py adds experiments/ to execution hashing; reverting only that tuple exactly reproduces the old file hash, while the other 144 library files remain byte-identical.

Preserved all **736 pre-existing receipt paths**; **735 hashes unchanged**. The initially empty live operator console scripts/hpc/receipts/20261022-brd4-jq1-25-fixed-lrreset-02/older-work3-backup-01-console.txt was populated by its concurrent task; the audit records both hashes, not a false unchanged-byte claim. No migration tool wrote a receipt file. Current attempt02 externally sealed source remains bound to manifest 832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3; saved CPU29781363/callback29781364 IDs are references only, with no fresh scheduler-state claim. The Git staging index matched its pre-edit tree; other staged/uncommitted work and all existing results/report locations were retained.

Local validation passed: 122 canonical Python ASTs; 84 script module names resolved without imports; 237 working-path mappings and old-source fallback; 147 Bash syntax checks; 41 declarative byte pairs; synthetic stdlib module/source-identity/authentication checks; all checked README links/folder docs; static signatures/numeric literals/guard-message comparisons. The sole live-console difference is recorded separately.

Exact check: `py -3.13 scripts/layout/verify_layout.py --root . --baseline results/static/20261022-script-migration/before --live-receipt scripts/hpc/receipts/20261022-brd4-jq1-25-fixed-lrreset-02/older-work3-backup-01-console.txt --output results/static/20261022-script-migration/layout-final.json`. Local-only packaging: `py -3.13 scripts/releases/package_stage0_release.py --output results/releases/20261022-script-layout-smoke-01`. The disposable archive authenticated **1731 files**, archive SHA-256 76a4f2add447fb57e380bc0ab73824fd0f2c59c4fef88b8a2484b25e24d9c888, manifest SHA-256 0966e7c9fc9551c345c37b459192d4949a65a191a3a63431f3f98c8dc76dbcc0. All members were independently rehashed; the extracted old verifier CLI passed and rejected a deliberately unmanifested experiment file in the disposable copy. No upload or scientific release adoption. Final documentation postdates this smoke archive.

Pre-edit copies, intermediate/final static receipts and smoke generator/outputs are retained under ignored results/static/20261022-script-migration/. Allocated CPU/native/numerical regressions were **not run** for the new layout and remain required before future scientific execution. No old pass, failure, cancellation, partial run or skipped/unreached gate is reinterpreted by this migration.


## 2026-10-22T10:34:14.557841+00:00 — migration preservation follow-up: concurrent receipt summaries

A final read-only observation retained all 736 original receipt paths. Since the earlier static audit, the concurrent existing campaign/cleanup task updated six campaign.json summaries in addition to its initially empty console log: seven changed files total, 729 still byte-identical at this observation. The migration wrote none of those original files. All seven pre-migration byte states were preserved additionally under ignored results/static/20261022-script-migration/preserved-receipt-baseline/ using matching-hash members from the already verified local smoke archive, plus the known empty-file state for the console log. [The migration audit](../experiments/migration-verification.json) retains before/current hashes, exact paths, observation time and copy provenance. Current copies keep the newer operational information. Concurrent ledger insertions were also retained; line-by-line comparison found no deletion/replacement of any pre-migration history line. These observations do not alter any scientific outcome or authorize further execution.


## 2026-10-22T11:16:24.623415+00:00 — documentation accuracy and account-portability review completed

Actor: Codex documentation agent. User scope: inspect all docs against the repository, especially recipes assuming the operator's private work3 folder. This was local source/document review with selective official-document checks; **no experiment executed**, no HPC connection, job submission/cancellation, remote access change, model import or numerical test.

The initial inventory read 584 documentation files and supported attachments; the final per-file inventory contains 590. Added the [documentation index](README.md), [own-account HPC guide](HPC_GUIDE.md), [findings and limits](DOCUMENTATION_AUDIT.md), [per-file audit](documentation-audit.json) and [original-proposal scope map](project_reference_status.md). Root README/AGENTS now make current guidance and portability policy prominent. Rewrote the generic Codex guide, made its duplicated named-person version a compatibility pointer, and retained the original workstation recipe under dated history. Corrected Python 3.12 requirements, lockfile assumptions, repository layout, misleading fixed hardware/queue claims, risky generic cleanup recipes and Stage 0 command/verifier descriptions. The TeX runbook compiled successfully through the built-in LaTeX compiler.

Added access/scope notes to all 33 dated protocol documents and all 24 experiment-home READMEs. Historical paths remain exact attempt bindings, not shared installations. The audit found **104 canonical executable source files** with concrete account references; documentation variables do not override their pinned controller paths/proofs. Another account still needs an explicitly reviewed port, authorized assets and required fresh checks. This review changes no controller, scientific configuration, sealed release or acceptance gate. The shared quota helper has a docstring-only correction: executable AST unchanged.

Local checks passed: zero broken Markdown targets, zero unresolved literal repository paths and no concrete account defaults in maintained reusable guides. Source/config checks matched Python, Stage 0 modes/seed/steps/rates/geometry and fixture dimensions. The layout audit passed 122 canonical Python ASTs, 84 module resolutions without project imports, 237 working path mappings, 147 Bash syntax checks, 41 declarative byte pairs and synthetic compatibility/source-authentication checks. The only canonical content change since migration is the declared quota-helper docstring. Scoped Git whitespace checks passed.

Preservation checks matched all 357 checked original receipt documents/imported attachments/project-PDF hashes, retained every original body of the 33 scoped protocols and retained prior ledger lines in order. Original imported note bodies, failed outcomes and running-job receipts were not rewritten by this review. Raw prior guides, before/after inventories, edits, source/layout/preservation checks and extracted proposal text are retained under ignored results/static/20261022-documentation-audit/. PDF/Office text extraction is not visual or institutional-policy certification; historical scientific claims and live remote permissions were not independently revalidated. Audit status is passed_with_documented_historical_or_imported_findings, not a native/scientific pass.


## 2026-10-22T11:44:44.537187+00:00 — flat script and test roots replaced with documented nested folders

Actor: Codex organization agent. User scope: put the vast majority of files into labeled subfolders under scripts/ and tests/. No scientific experiment executed; no HPC connection, submission/cancellation/restart, numerical/model test or sealed-release modification.

Moved all 73 loose script compatibility files into scripts/compat/python/ or scripts/compat/assets/. Moved 38 experiment-test adapters into scripts/compat/tests/ and all 44 shared test implementations into tests/core/, tests/models/ and tests/small_molecule/. The 38 canonical experiment tests stay with their experiment. scripts/ now contains only README.md; tests/ contains README.md and the required collection hook conftest.py. Added dedicated READMEs and prominent root/agent policy prohibiting new loose implementations. [Script navigation](../scripts/README.md), [test navigation](../tests/README.md), [additive path map](../experiments/folder-layout.json), and [verification](../experiments/nested-layout-verification.json) are the entry points. The original 237-file migration JSON/hash history is unchanged.

Updated pytest's canonical discovery/helper paths, test root/fixture lookup and one report fixture's script resolution. All 44 moved shared test ASTs match their originals apart from import bootstrap/repository/fixture path resolution. All non-pytest pyproject settings are unchanged. Default static discovery reaches all 82 canonical test modules exactly once; all helper module specs resolve without imports. Compatibility adapters are excluded from default discovery. New source packaging emits 155 declared old flat names inside a fresh archive only and authenticates their bytes; it never recreates loose checkout files. Existing shell/explicit test selections retain their old cohort membership inside a release. Canonical commands or scripts/layout/experiment_layout.py resolve/run/test-paths translate working commands.

A real pytest regression using two synthetic stdlib-only modules reproduced the legacy import-file mismatch. An initial collector path rewrite failed CLI argument matching; the final tests/conftest.py preserves requested node paths while using pytest's canonical module loader. Final explicit-alias and default-discovery invocations each passed two synthetic tests, including cross-folder imports, monkeypatch globals and canonical source paths. No import-mismatch checks were disabled and no Mosaic/numerical tests were imported. Synthetic sources/logs and the first disposable archive remain retained.

Final local verification passed: 166 canonical Python ASTs, 281 path mappings, 84 script module resolutions, 82 unique canonical test discovery paths, 41 declarative byte pairs, 147 Bash syntax checks, source-identity/authentication fixtures and alias conflict/missing/escape guards. All 966 protected library/receipt hashes are unchanged, with no missing files or concurrent differences observed. The documentation audit found zero broken links and zero unresolved literal repository paths across 597 documents; the updated TeX guide compiled and Git whitespace checks passed.

Disposable smoke02 authenticated all 1,962 members, including 155 legacy aliases; the extracted old-path verifier and packager help worked. Tampered aliases and unmanifested test inputs were rejected. Archive SHA-256 208621b85390daaa4b7717d5cd8dbef1ede3bf54a8c18880e62ef4867d70b653; manifest SHA-256 c506cc72ea092869a4e0783e0b08667b1564a886b84060d2f0a6582cbc9cb2d7. No upload/adoption occurred. Commands: py -3.13 scripts/layout/verify_layout.py --root . --output results/static/20261022-nested-layout/layout-final.json; py -3.13 scripts/layout/verify_pytest_routing.py --output results/static/20261022-nested-layout/pytest-routing-final.json; py -3.13 scripts/releases/package_stage0_release.py --output results/releases/20261022-nested-layout-smoke-02. Detailed before copies/checks are under ignored results/static/20261022-nested-layout/. Final navigation/report/logbook postdate this smoke archive. Allocated CPU/native/numerical regressions remain pending before future scientific execution; current running sources and historical outcomes are unchanged.


## 2026-10-22T11:55:45.542691+00:00 — organization changes prepared for the user-requested Git publication

The user explicitly requested pushing the organization/documentation work. Fast-forwarded origin/main to 82abf8aa0132a2fc56549667704920735032f2f4, preserving its independent JQ1 multistart additions. Staged the experiment homes, nested script/test layout, policy, guides, audits and additive logbook history. The original project PDF is a byte-identical move into docs/. Only the OneNote reading guide is included; 198 pre-existing local notebook note/attachment files are excluded from publication. No ignored run artifacts, caches or disposable releases are included.

Pre-publication validation caught Git checkout line-ending conversion of declarative copies/shared launchers. Added targeted .gitattributes rules preserving the current authenticated bytes and LF shell sources, without rewriting working source/config bytes. The second fresh staged checkout passed the full layout checker (166 canonical ASTs, 281 mappings, 82 test discovery paths, 41 byte-identical declarative pairs and 147 Bash syntax checks). Its synthetic pytest explicit-alias/default invocations each passed two stdlib-only tests. All 565 checked execution/config/PDF index bytes match the working copies; no notebook export payload is staged. The original failed staged check is retained alongside the successful check under ignored results/static/20261022-publish-layout/. Scientific/numerical validation remains pending allocated HPC; no HPC job action occurred. This entry records preparation and checks, not a push outcome.


## 2026-10-22T17:03:45+00:00 — lower-rate entry admission failed; separate diagnostic outcomes delivered

Actor: Codex Mosaic status/evidence task. User scope: current news on the authorized lower-rate restart and separate endpoint diagnostics. Read-only SSH query plus a fresh in-memory metadata/archive transfer; local standard-library byte/path/hash/JSON checks only. No model/numerical execution, new job, cancellation, retry, cleanup, source/gate change or saved-study mutation.

Original source622 native jobs29783703/29783707/29783710 actually started19:01:22/19:01:52/19:02:23Paris and exited1 at19:01:37/19:02:06/19:02:37, before model/gradient invocation. Entry work3 gates report available10191922566/10191904301/10191886069 versus required10200529817/10200522040/10200514283bytes; shortfalls8607251/8617739/8628214bytes. More than19.6million finite user-quota files remain; physical inode tuple allzero is unreported, not independent proof. Failed gates/decisions/own-job LSF hashes authenticated and preserved. CPU723 remains passed; no native readiness pass or long-submissions file. [Exact run outcome](../experiments/brd4-jq1/fixed-lrreset/runs/20261022-brd4-jq1-25-fixed-lrreset-02/status-20261022-1903.md).

Separate original Boltz diagnostic29780823 completed16:50:23Paris, capture29780824 completed16:50:48 with assessmentpassed/scientific_passedfalse. All423 files in599008-byte archive SHA271246fe15f82e6d71f5c7ca01e96959561a43a7c3cf7ce4c8679700a2feb2b5 and411 capture proof byte identities passed local audit; two original summary/terminal-report proofs remain explicit HPC references. The capture archive contains its own intermediate RUN snapshot; later original same-job LSF proves successful completion separately. The terminal original-scale bond discrepancy is reproduced, with closer agreement on the diagnostic-only finer ladder and zero reported repeat spread; no unique cause or retroactive scientific pass. Protenix diagnostic29780821 and capture29780822 failed the exact saved-logit probability identity control atstep0; frozen before/after match, terminal diagnosis remains incomplete. Final presentation29780830 remains PEND with invalid/never-satisfied dependency. [Diagnostic outcome](../experiments/brd4-jq1/fixed-lrboost/runs/20261022-brd4-jq1-lrboost-analysis-01/status-20261022-1903.md).

Combined75-file evidence delivery705571bytes SHAcb5b63ec3c3db90a68fc5d21e34749660dd4ff752e991e550b076ff78357d307 safely extracted/every member verified under results/hpc/20261022-brd4-jq1-25-fixed-lrreset-02/news-20261022-1903/. Original scientific outcomes remain failed/failed/passed, existing edition0002/failed browser QA unchanged. No large original summary download is claimed. New tooling or resource recovery requires a separately recorded immutable attempt and applicable checks; no failed source is restarted or relabeled.


## 2026-10-22T17:16:17.379453+00:00 — authorized storage recovery, unchanged lower-rate retry and separate Protenix diagnostic correction prepared

User explicitly requested enough storage and a fresh resubmission of all three lower-rate arms with unchanged settings, plus a Protenix diagnostic fix without saved-result changes. New attempts are fixed-lrreset-03 and lrboost-analysis-02 in their canonical experiment run folders. Full original scientific sources/configuration/library bytes remain pinned. Preparation is local standard-library byte/AST/JSON/packaging only; no new job/pass yet. Cleanup plan uses only exact duplicated completed report frames/curves and two archives, with all local archive members verified before any remote removal. Original native studies/proofs/releases, home/models/caches/envs and other projects are excluded. New allocated CPU/native and fresh full quota gates remain mandatory; old failed jobs/source/gates remain failed.


## 2026-10-22T17:36:06+00:00 — verified storage recovery, unchanged lower-rate retry and separate Protenix correction queued

User scope: free sufficient storage, resubmit all3unchanged lower-rate runs and fix separate Prodiagnostic without modifying saved results. Completed exact verified-backup cleanup1212files/852317966logicalbytes, planbfe10342b9d8378f7c5842ca3a2eb87806a46cdb08501d9e0813f81776ca7e9f, journalc1e82fc70a1a80d71c0d8bc8ed0ac380f5a962a033c7c6c21763434fbd3ed756. All636/647members of existing local interim archives reverified first. Retired only duplicated report frame/curve files and2archives; all original native studies/proofs/configs/releases/home/models/caches/envs/other projects untouched. Initial systemPython3.9 helper failed before deletion; same bytes/plan succeeded with declaredPython3.12, both receipts retained.

[Lower-rate retry03](../experiments/brd4-jq1/fixed-lrreset/runs/20261022-brd4-jq1-25-fixed-lrreset-03/README.md): source622 manifestff82857ecd4a985d0ba6cc16a87f1bf7b7bb2c7ffdbde18a183bbd63670ff567, archive39601344bytes SHA31809b975f0387db4ad15b6530c443afc1220fd501b6b8ac62e147bbcf739b01. All145scientificlibraryfiles/config/driver/contract bytes unchanged from parent622; launcher/storage root identity and additional diagnostic stale accounting only, plus path fixtures. ActualCPU29792104/afterCPU29792105 DONE/LSF, raw392+239+34+59=724passes/noerrorsfailsskip/3unchangedlegacydeselections. CPUcert864f7b92d2118c28ef7c5064a96812e5b25217e4c232c6791f2e08ebaca151ed. Actual native/callbacks Pro29792127/28, Bol29792129/30, chemistry29792131/32 PEND19:36Paris. No native readiness pass or10000-update long ID yet; all original scientific and whole storage gates remain.

[Prodiagnostic correction](../experiments/brd4-jq1/fixed-lrboost/runs/20261022-brd4-jq1-lrboost-analysis-02/README.md): source16manifest6d547929ce3227302729b022628cc26612d9829e72c0a70861870023463dac3a/archive78295bytes SHA799869cbce1ea09b6862c30b3d3c39a258872d0948de7981cbf93790f418ab13. CPU29792111 DONE282passes; callback29792112 EXIT on separate resource gate before any GPU, preserved. Raw probability probes stay exact; original saved-logit coordinate point is explicitly separate if FP32 replay differs, with measured equality/differences and separate baseline. No original scientific gate is waived. CPUcert5d31b9a130f5a904c5da4b0f2e8707c610ff49654acddf1b89b8945aa05cba6d.

Separate three-file resource launcher99292b7eed1662f5975a704aa4adfca5a93c1754ff3717c3d75f24473f8ef219/archive5285bytes SHAf71d8d2cb75a4b2580dc3771bd2b80962a68b032e6eeb2b9b2c6cb61f9456f43 adopts actualCPU724-tested current combined gate instead of independently reserving old completed predecessor output. All completed predecessors/source/CPU282/724 reauthenticated; full current lower validation/actual-long projections, original2GiB report/512MiB diagnostic allowances,4GiB/100000files remain, with launcher bytes additionally charged. Actual launcherCPU29792154/callback29792155 DONE12pass, cert9facf84cdfc9436bc86ab1155ae9a7c2ad19fda143a6d8b62a1ad8c6209e1e07. Preflight required10200529759/available12027869322bytes. Actual Prodiagnostic29792158/capture29792159 PEND19:36Paris. No repeated Boltzdiagnostic or original scientific run submitted by launcher.

Fresh93-file evidence251284bytes SHAb5d7a6b6b4f228d8e560184f7d7a9b8f8a810151890cdcbdeb9e0064427f46a4 safelyextracted/everymemberhashed. Independent localstdlib XML/source/proof/scheduler/LSF/resource/cleanup journal audit passed; no local numerical/model execution. Saved original higher-rate results stayfailed/failed/passed; existing immutable viewers/failedQA unchanged, no new report publication or diagnostic-native pass claimed.


## 2026-10-22T18:10:41.512579+00:00 — lower-rate Boltz readiness passed; Pro initialization identity blocked dispatch

User scope: news on gradient checks, all3lower-rate starts and separate Prodiagnostic. One compact live scheduler/artifact query followed by targeted new-failure metadata/proof retrieval; no scientific execution, job action, cleanup, setting or sealed-source change. Original saved studies remain unchanged.

Actual Boltz29792129 DONE19:57:13Paris and chemistry29792131 DONE20:01:19, same-job LSFsuccess/passed summaries/readiness. Source622/config/CPU724 pins matched; normal52 and chemistry128 raw-proof hashes matched. Original rawFD/parity/frozen/runtime/tenpilots and chemistry low0.001/high100 coupled checks/bothpilots passed. Pro29792127 EXIT1 at19:49:04 before raw gradients: exact initialization-array identity against original baseline failed150/150 atom probabilities, reported max absolute difference5.9604645e-08. Hardparity had passed but no native_gradient_checks result was produced. This is not established AD/FD disagreement; no tolerance or waiver. Scheduler GPU power annotations are preserved without claiming causation.

Pro callback29792128EXIT; normal callback29792130DONE. Chemistry callback29792132 recorded its own passed readiness then EXIT because the global dispatcher rejected Pro's failed decision. Both passed Boltz decisions remain passed. No long-submissions.json or long-dispatch-started marker exists: zero fresh10000-update long jobs submitted/started. Corrected separate Prodiagnostic29792158 and capture29792159 remainPEND, no findings. Existing automation remains paused.

Fresh108-file local metadata/log delivery byteverified;188 declared proof hashes rehashed on HPC matched. Native NPZbytes remain explicit remote references, no local scientific calculations. Independent stdlib audit passed SHAca00e03022542b2e1e90f83832dc2c646a567252190517850c117042cc4f8fe1; initial local member-prefix lookup failure preserved in tool output before unique authenticated lookup correction. [Dated record](../experiments/brd4-jq1/fixed-lrreset/runs/20261022-brd4-jq1-25-fixed-lrreset-03/status-20261022-2010.md). No new report publication or unrelated PDF/OneNote changes.


## 2026-10-22T18:20:41.105371+00:00 — exact Protenix initialization recovery authorized and prepared

User requested an initialization fix and all3unchanged lower-rate runs started without changing saved results. New attempt04 first performs a bounded separate allocated CPU/GPU replay probe of the exact original initialization, preserving bitwise checks and parent03failed/pass outcomes. No tolerance, model/loss/rate/policy change or local numerical execution. Full allocated CPU/newnative/fullstorage gates precede any future longdispatch. Preparation only; exact packages/jobs/results remain to be recorded in the new run README/campaign.


## 2026-10-22T18:36:35.557629+00:00 — exact initialization correction sealed, allocatedCPU728 passed; separate Prodiagnostic verified

User authorization: fix Pro initialization and get all3lower-rate runs started, identical rates/settings/checks, saved results unchanged. Isolated probe CPU29792346/GPU29792347 actualDONE/LSFsuccess authenticated original baselineNPZ351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf. Logits/key exact on bothbackends; stockCPU atomsoftmax reproduces originalX, stockGPU bondsoftmax originalE. Neither entire backend alone reproduces bothfields. This is FP32 replay dependence, not a changedseed or proven unique historical compiler cause.

New Pro-only parameterization uses unchanged FP32jax.nn.softmax for CPU atoms and scorer-device bonds, then copies atom probabilities unchanged to GPU. Same placement applies every state/trial, no special cachedinitialpoint/tolerance/projection/clipping/newtemperature/loss. One shared proposal-constructor line preserves the current input class through updates. Other144srcfiles/allconfigs/contract/baseGraphLogits byte-identical parent03; original exactfivearraybaseline equality and every native/FD/parity/null/primal/finite/frozen/coupled/pilot/runtime/quota gate retained. Full freshCPU and all3native checks repeat.

Source625active localpackage source02 manifestc2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057, archive39604161B SHA4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e staged read-only. Localdraftsource01 was unexecuted, preserved before launcher-upload stale-accounting path/document correction. New CPU29792389 actualDONE with396+239+34+59=728actualcases/noerrorsfailsskips/3unchangedlegacydeselections; cert6a87093f3da0d1ca0694c7152de84fdeed308251f709c685b526d978b1455e23. Owncallback29792390 dispatch inprogress; no newnativepass/longID established at18:35:31UTC.

Separate corrected Prodiagnostic29792158/capture29792159 actualDONE/samejobLSFsuccess. Independent416filebyte/schema/LSF/proofchain audit passed SHAf660e02ffe67051c5f8adc3c79887791cbdf6d62277e1d83ec64d0bcc6f4a9f7;385declared references rehashed remotely. Five original snapshot/summary/terminalreport files remain explicitHPC references. Raw endpointreport2045d6e8537ff86c176e6619725d206b32029058df351ce639cc88f90da9c1db copiedverbatim. Frozenmodel/context unchanged atsteps0/9900/10000; original terminalXrow3 source1to4 has symmetricmargin0.0001437509199604392, declared0.01/.003/.001 perturbations infeasible. Diagnostic-only.0001/.00003 agree withAD atterminal; finer/coordinate controls are not uniformpass certificates. Originalfailed study remainsfailed/cannotrestart; no scientificgatewaiver. New original saved results untouched. No cleanup or automation schedule change thisturn.


## 2026-10-22T18:38:38.220149+00:00 — initialization recovery fullCPU independently verified; all3newnative checks queued

CPU29792389/afterCPU29792390 actualDONE/LSF,728actualcases396+239+34+59/noerrorsfailsskip/3unchangedlegacydeselections. RawJUnit inherited3subtest header surplus is not extra cases. Independent36file/source625/config/XML/proof/scheduler/LSF/resource audit passed; sourcec2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057 andCPUcert6a87093f3da0d1ca0694c7152de84fdeed308251f709c685b526d978b1455e23 authenticated. All3actualnewnative/owncallbackjobsPro29792413/14,Bol29792415/16,chem29792417/18 PEND physicalGPU/hostslots. No newnativepass/longsubmission. All settings/checks/savedresults unchanged; exact-init placement and proposal-type preservation are documented engineering changes, with fullrequiredallocatedchecks repeated. Currentuserautomation remains paused; no cleanup thisturn.


## 2026-10-22T18:58:10.627079+00:00 — exact Protenix initialization and full native readiness passed

Protenix validation29792413 and owncallback29792414 actualDONE/samejobLSFsuccess. Fresh initialization NPZ is1197bytes and exactly the originalSHA351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf; all5arrays/key exact, original strict assertion retained. Native/rawFD/null/independentprimal/hardparity/frozen/tenupdatepilot/runtime passed atLR0.3. Empiricalpilotmean2.1728954s/update, conservative34393.4310s (estimateonly) fits82800s. Original fullstorageprojection formula yields1493521824bytes. Pro-only sameFP32softmaxplacement applies eachstate/trial, GPUscoringunchanged.

Independent142filebyte/source625/XML/LSF/proof audit passed,74nativeProproofs plus CPU728chain; no local numerical arraysloaded. NormalBol29792415/owncallback16 readinesspassed; chemistry29792417 RUN/rawgradient andearly10pilotpassed, owncallback18PEND. No long IDs or fullfinalstoragegate established yet; no duplicatejobs/cancellation/cleanup/scientificsettingchange.


## 2026-10-22T19:09:36+00:00 - fullhomeinventory timedout beforelongsubmission; all-work3 recovery prepared

All3source625native validations29792413/15/17 passed. Callback29792418 EXIT only because GNUfind fullrecenthomeinventory exceeded600s; no long-dispatch marker/long IDs exist. Preserve failedgate/LSF unchanged. New launcher-only all-work3 placement budgets all3unchanged scientific625remainingprojections + original2GiBreport/512MiBdiagnostic +4GiB/100000files on the actual outputfilesystem. No home output/write or old data move/delete. Original science625/CPU728/all3readiness/source/config/context/model/pocket/frozen/runtime gates adopted exactly; new launcher requires allocatedCPUcontracts and actualfullwork3/entry admission before3independentlongjobs. Preparation only, no newlaunch/pass yet.


### 2026-10-22 21:21 Paris — exact initialization/native passes and all-work3 launcher gate

All three source625 validations29792413/15/17 passed independently (74/52/128proofs). Pro initialization whole NPZ equals original1197bytes/SHA351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf. Native evidence338files/276remote proof hashes passed stdlib audit. Original callback29792418 EXIT1 solely on mandatory home inventory600s timeout before ANYlongmarker/output/submission; original failed gate and LSF preserved. New immutable launcher-only all-work3 adoption01 authenticated4files, manifest99fcd95e150a671adb30a1cb910079474452319ddd67a36b042dc6a8e94940fa; CPU29792659/own-ended callback29792660 actually submitted. Full20metadata fixtures ran with no errors/fails/skips in rawXML, but final CPU/LSF/certificate/fullcampaign gate/longIDs not established in this observation. Each new output assigned work3 with original all3remaining projections+2GiBreport/512MiBdiag+4GiB/100000files; same source625/config/nativepins, no new home output or altered scientific setting. No further cleanup or saved-result mutation. Static layout03 passed166AST/82tests/155aliases; local work remains stdlib-only.


### 2026-10-22 21:25 Paris — full storage passed and three actual lower-rate long submissions

All-work3 launcher01 CPU29792659 passed20 cases, but callback29792660 failed its first
scheduler-proof check before any long submission. Preserve failure/source. The first
raw response was not saved; later allocated readonly probe29792663 actualDONE captured
same CPU actualDONE/hpc. Exact transient cause unproven. New immutable launcher02
records every raw observation and bounds RUN/PEND waiting without accepting EXIT,
unknown/malformed/wrong-job/queue; required actualDONE/hpc/LSF/full26cases passed on
CPU29792667, callback29792668 actualDONE. Scientific625 and all new native passes
unchanged. Own source4 manifest40c5055150c06e0aeb9989d33f869aecb6642c3823f6bdc9330e66f41a3bcbbe.

Full all3 work3 dispatch quota PASSED required15364887328 vs25726653739 available,
including full native projections+2GiBreport/512MiBdiagnostic+4GiBreserve/100000files.
File headroom19622802 from finite userquota; allzero physical inodes honestly
unreported. Original600s failed home gate remains failed. Three actual independent
longs submitted: Pro29792670, Bol29792671, chemistry29792672, allPEND at21:25Paris.
All outputs ROOT04/run-ARM onwork3, no new home output; no saved result moved/edited,
no new cleanup. Same rates0.3/0.005/0.005, chemistry0.001->100, exact unchanged
science625/config/CPU728/all3 native proofs. Entry gates/first updates still pending.


Independent all-work3 submission delivery passed415 files/303 remote proof hashes,
source625+4launcher members/raw728+26caseXML/all3newnativeproofs/fullquota/actual
Pro29792670,Bol29792671,chem29792672 vectors. Tracked
experiments/brd4-jq1/fixed-lrreset/runs/20261022-brd4-jq1-25-fixed-lrreset-04/launcher-submission-verification.json
explicitly records scientific_start_verified=false. At21:27Paris current3jobs PEND
on hostslot/physicalGPU capacity; no entry/optimization started yet. Automatic LSF
execution needs no new submit. Preserve exclusive markers; never duplicate.
Existing heartbeat automation stays paused; no schedule changed. No scientific
completion or terminal pass follows from successful submission/presentation.


### 2026-10-22 21:31:50 Paris — Protenix actual entry and exact production initialization

Pro29792670 actualRUN/gpul40s, own full all3remaining-projection entry storage gate
passed (SHAe6f331c0a5502a60c8abcbd91e36bd7bea45dd4b7ad42f46b1891bad66a7289f),
validated source625driver invoked. NEW production initialization1197bytes SHA
351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf equals original.
No committed optimizer update yet; native startup/checks continue. Bol29792671 and
chem29792672 stillPEND on GPUcapacity, not failures and no duplicate/resubmit.
Additional independent raw scheduler audit verified all3actual command/source/queue/
4CPU/8GiBslot/23h/oneexclusiveGPU and completed launcher callbackLSF/DONE;
launcher-resource-verification.json and entry-observation.json record scope.
No scientific completion/all3start claim; existing heartbeat remainspaused.
Allsavedresultbytes/source625/settings/checks preserved. Code/first submissions
and independent full evidence were pushed2327e73 before this later observation.


### 2026-10-23 10:10 Paris — lower-rate terminal progress and preserved Boltz failure

Protenix29792670 completed all10000updates/10001contiguousstates. Original summary,
case and final raw atom/bond derivatives passed; frozen identities unchanged and
exact original initialization retained. Same-job LSF success ended22October23:36:53
Paris. Its live scheduler record aged out; use SUCCESS_PERSISTED original proof,
not a fabricated live DONE. Summary298580732bytes SHA
b94d4bae58b7c82b92f4cf0b6b0e9804734afff0f54ea0b162b65d825ec2661d.

NormalBol29792671 started23October03:15:59 and ended09:32:59Paris actualEXIT1.
All10000updates/10001states and optimizer casepassed, but original overallsummary
failed mandatory terminal E_upper FD. X passed; E row239 0->1 AD4.448079109,
FD -7.524552941/-6.906365355/3.221578896 at.01/.003/.001, all original flagsfalse.
No cause or diagnostic fix is inferred. Model/context frozen digests unchanged.
Original failed summary301971481bytes SHA
06a68008a0dfb8b6a66668447bcda4f01c7b54427d9b68f0d84e5926db278cc1;
terminalreportSHA1a95448ea1d3fe373f7a3f8ad64017ef7ffd620f52efc3086cf2262ecf4868a4.

Chemistry29792672 remainsPEND, jobslotlimit6hosts; no driver/study/update yet.
No resubmission, setting change, stop, diagnostics or cleanup occurred in this
read-only status turn. Saved results remain unchanged. Same-job complete LSF
bytes rehashed locally and remote original summaries/terminal metadata hashed;
large original summaries/all NPZframes were not downloaded or recertified by a
new allocated completion collector. See terminal-status-20261023.json. Existing
heartbeat remains paused.


### 2026-10-23 — separate lower-rate Boltz endpoint analysis and saved-only report prepared

User authorized separate diagnostics of failed original29792671, without saved-run changes/restart, plus completed Pro29792670/Bol29792671 report. New run20261023-brd4-jq1-lrreset-analysis-01 owns source17 and independent output root. Local source01 was an unexecuted draft; active local source02 removes inherited Pro-only dispatch and adapts the CPU scope fixture to the new declared one-Boltz-diagnostic/three-collector/two-interim/final plan. Source manifest761d66642052fde3da2da2016736fdd91bdea477ba714787a5c393b2038f8b83, archive80297bytes SHA4c809367734cf6e529d1ae59e0b30cac603d9eae147914b518d992fb8818dc53. Exact source625/CPU728/all3native/launcher40 CPU26/adoption/inputpins remain independently authenticated. Full all3remaining work3 projections+2GiBreport/512MiBdiagnostics+4GiB/100000file reserves required before each phase. No oldviewer/source/result mutation, no optimizer/training/diffusion/repair or original job action. Diagnostic uses original saved0/9900/10000, original rawscales plus diagnostic-only finer controls and unchanged frozen identities; cannot certify original failedgate. New viewer includes numeric-focus pause, old editions/servers preserved. Local AST/JSON/byte preparation only; allocated CPU/models/scientific frame validation/rendering still pending.


### 2026-10-23 10:30 Paris — analysis01 fixture failure preserved; new immutable analysis02 prepared

Analysis01 CPU29796461 EXIT1 with282passes/one inherited synthetic schedule assertion failure/zeroerrorsskip; callback29796462 EXIT correctly blocked dispatch. Production expected_weight correctly ends100; old fixture still asserted50 at terminal. Eleven raw CPU/callback/submission/log/XML files downloaded and independently byte/XML verified, no analysis-dispatch marker. Sealed source17 remains unchanged. New analysis02 source17 manifest955872e4621b1e5a3d23a69a989aadcf16c8d0e2793e6db464782f58be15fa37/archive80420bytes SHA013b3003443b2b2b4da543caa82dafef0032b31ef0e4cfb204938172d432abeb changes only two fixture expectations/name and separate own analysis root identities. Full283allocated fixtures required again, no scientific setting/gate/run changes. Prepared only; original P/B saved results unchanged and chemistry29792672 retained.


### 2026-10-23 10:36 Paris — analysis02 all283passed but XML attribution rejected; immutable retry03

CPU29796495 actualDONE/samejobLSF/full283passes with zero failures/errors/skips. Callback29796496 EXIT before any dispatch:35 raw JUnit rows have emptyclass names because external legacy testpaths lie outside pytest rootdir;18diagrows are correctly attributed. Saved full raw XML/logs/LSF/submissions/source; no XML rewriting or gate waiver. New analysis03 changes pytest rootdir to exact common ancestor of the two pinned sources and adds one allocated collection-root coverage regression. New17file manifest8e3045ddfd44a701127deb7af7a163536dfbf699123ba883f98207d6d57b95d5/archive80735bytes SHA62915ff15d9a493eadd08ab27a64ae64aed0a6e331f7481c0ce1d105bd7aa751. Full284allocated cases in8modules required before any newdiagnostic/report dispatch. Original source625/scientific runs/config/readiness/inputs/resultbytes remain unchanged; no cleanup. Newretry03 preparation only.


### 2026-10-23 10:39 Paris — new separate analysis full284passed and actual diagnostic/report jobs

Analysis03 CPU29796532 and callback29796533 actualDONE/hpc/samejobLSFsuccess. Full284actualcases82contract+54render+60publish+35ops+18diagnostic+26Bolreport+3chemreport+6ramp, zeroerrors/failures/skips/deselections. Exact rawmoduleattribution nowpresent. CPUcertc1cfb3730419a48e82c45b45981a874275dbdec31a13fdffbf865139b9416cf3; independent40-file byte/XML/JSON/submission/source/reserve audit passed. One local auditor assertion incorrectly counted both raw scheduler observations instead of certificate-bound final observation; failedauditorrecord retained, all rawproofbytes reverified after metadata-only correction, no scientific gate waiver. Original625/CPU728/native/allwork3launcher26/settings/results unchanged.

Fresh fullcapacity passed with full3remaining native projections+2GiBreport/512MiBdiagnostics+4GiB/100000files. Actual separate Boltzdiagnostic29796551/capture29796552 PEND, P/Bsaved10001framecollectors29796553/54RUN, chemcollector29796555 waitsoriginal29792672. Interim29796556 dependsP/Bcollectorpasses; diagnosticinterim29796557 follows first+Bolcapture; final29796558 dependsallcollectors+capture+secondinterim. Original chemistry29792672 remainsPEND unchanged. No diagnosticfinding/renderpass/publication yet. No savedrun restart/mutation/cleanup or automationunpause.


### 2026-10-23 — additional random Protenix + delayed chemistry explicitly authorized/prepared

User requested one new Protenix optimization concurrently with the separate Boltz endpoint/report work. New experimentprotenix-random-chem retains original passed lower-rate ProLR0.3/10000fixed1candidate/key17/N25/T1/frozencontext/pocket/contact and all checks. Existing PCG64/Dirichlet1 initializationseed17 replaces chainuniform; existing delayed chemistry0.001through5000 then geometric100at10000. This new declared objective/init variant requires full freshCPU/native low-high coupled/nonzero/bothtenpilots/runtime/storage before exactly one long. Original chainbaseline comparison explicitly remains for all oldversions; newrandom uses exact reproducibility and5-array validation-to-production identity. All old scientific/results/analysis unchanged, no cleanup/automationunpause. Local preparation only, no newsource/job/pass.


### 2026-10-23 — random Protenix/chemistry candidate631 sealed locally

New versioned candidate from authenticated original625 preserves every145scientificlibrary member exactly; only driver/contract add explicitly authorized randomProchem branch and six new config/controller/wrapper/fixture/doc members. Local631member AST/archive/byte audit passed. Manifestbb3ac9cae836b59f9a62fb33d4775c0b4388f3b200a7422ce8a620d2d405950f, archive39609007bytes SHAc005429dc2581c929662698e80da154fabc0f3365615facfa7582641fa7fcddc. Newrandom recipe17/Dirichlet1, chemistry.001->100, LR.3/key17/fixed10000; originalchainidentityguard remains alloldversions, newbranch requires exact independent recipe replay and validation/production arrays. Uses exact original frozen context as input; all newrandom/raw/coupled/pilot checks fresh. No allocatedtest/model/pass yet; newCPU and own callback only after fullpreflight, own native/long conditional. Original currentreport and Boltzdiagnostic unchanged.


### 2026-10-23 11:09 Paris — completed lower-rate P/B report delivered

Report29796556 actual DONE/hpc/LSF and passed presentation assessment; collectors
29796553/54 audited all10001 original frames/10000 updates each. Pro passed/Bol
failed scientific outcomes preserved. Archive90878067 bytes SHA9551b2e0b0a821e606fa740ac09df432ec6575f562dfe51390c84629d6009ae7,
all441 files verified after authenticated partial/range recovery; all timeouts,
stopped exact byte-transfer processes and partial evidence retained. No scientific
process/result mutation. Independent delivery audit passed; local archive-prefix/
UTF8 audit failures preserved before correction. New edition0001 at8779/report.html
captures10001/10001/0 states and milestone floors10000/10000/0. First HTTP audit
covered410 files while publication finished; subsequent stable-set audit passed
ALL413. Actual browser QA passed drag7263/numeric1500/next1501/mid5000/terminal,
Bol play2/manual321 pause and stability/prev320/terminal/25atom300pair tables,
all selectors, raw graphs and actual empty-chemistry clearing/disabled controls.
Screenshot, QA, source and publication hashes recorded in analysis03; server53268.
At09:09:36UTC current chemistry29792672 and separate Boltz diagnostic29796551 remain
PEND. No root-cause finding or retroactive gradient pass exists. Existing interim
29796557/final29796558 unchanged; future started chemistry requires fresh immutable
entry pins. Current report is delivered, endpoint investigation remains pending.
Heartbeat stays paused. All older viewers/editions/results remain unchanged.


### 2026-10-23 11:09 Paris — random Protenix chemistry776CPU passed; fresh native queued

New source631 bb3ac9cae836b59f9a62fb33d4775c0b4388f3b200a7422ce8a620d2d405950f
authenticated/read-only. CPU29796731/callback29796732 actualDONE/LSF/full776 rows:
444 contracts(48new)+239legacy+34mapping+59chemistry, zero failures/errors/skips,
3 unchanged legacy deselections. Raw XML contract header surplus3 is inherited
unittest subtests; actual collected/pass444 matches444 rows. Independent35-file
byte/XML/source/config/CPU/proof/actualsubmission/quota audit passed, certificate
7adde8e09679637c597b72c6806aa4fc51cec590ba494da029b457e8d22e1a5d.
Initial full old3remaining+analysis+new1GiBvalidation+4GiB/100000files gate passed
14371611825 required/21144155255 available; not new full pilot/long admission.
Actual fresh native29796780/callback29796781 queued, no native pass or long ID.
Same declared LR0.3/randomDirichlet1 seed17/chemistry0.001-to100 and original
frozen context/model/pocket/key/checks. Old results, separate Boltz diagnostic
and report unchanged; no cleanup or automation unpause.


### Original chemistry started, 23 October 11:18–11:19 Paris

Readonly scheduler observation at09:18:48.777054UTC shows original chemistry
29792672 RUN/gpul40s. Its separate saved metadata receipt, locally collected at
09:19UTC, authenticates the same submission, passed entry quota, driver invocation,
62 newline-committed state rows and61 update rows. This is actual optimization
progress, not a terminal validation or a new numerical analysis. Exact path/length/
SHA bindings are preserved in chemistry-start-observation.json. Original run,
settings and saved results were not changed by the analysis.

Published edition0001 remains the truthful earlier08:48UTC capture with chemistry
unstarted; it is immutable. Future chemistry publication requires a new immutable
source containing its actual entry/driver pins. Existing diagnostic29796551 and
capture29796552 remain PEND, with no cause established. New random Pro native
29796780/callback29796781 remain PEND; no optimization long ID exists yet. No
duplicate submission, cleanup, original-run restart or heartbeat unpause occurred.


### 2026-10-23 — prepared Protenix50k extension and exact plain continuation
User explicitly requested random Protenix50000 and passed plain29792670 plus40000. User selected chemistry hold.001through5000, geometric100at50000. New experiment experiments/brd4-jq1/protenix-extended-50000, attempt20261023-protenix-extended-01. Preserve original625/results; retire only superseded queued random10k native29796780/callback29796781 under actual dispatch lock after status/no-long inspection. New immutable segmented execution requires exact logits/FP32/key/global-index boundaries, all native/chemistry/CPU/runtime/wholequota guards, no gate waiver. Prepared only; no new50k scientific job/pass yet. Existing Boltz diagnostics/report and running chemistry unchanged.


### Protenix50k actual immutable staging/submission, 23October11:36Paris

Superseded random10k native29796780/callback29796781 both actualPEND/unstarted,
then intentionally retired/actualEXIT at09:26:51UTC under original dispatch lock.
No old10k long or study existed; all saved source/CPU776 artifacts retained.
New641 source staged read-only, manifest a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00,
archive39627601 bytes SHA8ba1c3370fee1f09b96af139dbf1403bb3c187c52d447ec4b0b23ebe4a65f63d.
Local unsubmitted draft01 was preserved; active package uses localsource02 with
stronger real chemistry-offset and parent scientific/frame-adoption checks.
NewCPU29798115/afterCPU29798116 actually submitted09:36:00–01UTC. Preflight
15327331773required versus20937322046 conservative available; own allowance2GiB
plus all old remaining/report/diagnostic budgets and4GiB/100000file reserves.
Not the full50k storage gate, which awaits actual newpilot projections.
No new CPU/native/scientific segment pass yet. See campaign.json and exact
submission/retirement receipts. New random50k ramp .001through5000->100at50000;
plain originalstate10000 plus40000, bothLR.3. Current immutable newsource changes
only driver/contract plus optional global-offset trajectory support; remaining
144scientificlibrary files exact631. No original run/result, other job, cleanup
or heartbeat schedule changed.


### Protenix50k CPU/adoption verified; both fresh native validations queued, 23October11:46Paris

CPU29798115 and callback29798116 actualDONE/hpc/same-job LSFsuccess. Full832 raw
cases passed:500contracts(56new)+239legacy+34mapping+59chemistry, zero failures,
errors or skips;3 unchanged legacy deselections. Raw contract XML header503
versus500 rows is the inherited3 unittest-subtest surplus, not three extra tests.
New allocated checks include exact split/uninterrupted FP32 logits/graph/key/raw
gradients, altered checkpoint rejection and real chemistry preparation/schedule
global offsets across the5000 ramp boundary. Other144scientific library members
remain exact parent631. Local checks were stdlib AST/JSON/byte/layout only.

Callback rehashed/authenticated original29792670 full summary/case/committed
JSONL/terminal native success/frozen/raw/parity/attempt rates and full-frame
collector. Exact saved terminal checkpoint14226bytes,
SHAedf15f06ba1b6591d62cbc3e04e19ecd93b553a0627da38c370fdeed3c1b780f,
is the plain continuation's immutable input; original10000 results preserved.
CPUcertificatec0f784d523d358dfd09d7f57fcfe333fbb5c069d93074681ea35562a5f423171.
Independent45-file byte/XML/JSON/LSF/source/config/actualresource/gate audit passed;
all641 source members and642 safe archive members independently rehashed.

Actual fresh native submissions: random29798159/callback29798160 and plain
continuation29798161/callback29798162. AllfourPEND at09:46:30UTC. No new native
pass or optimization segment ID yet. Both initial readiness checks, empirical
pilot runtime and whole50k/40k byte/file projections must pass before either
independent chain begins. No full extended-output storage pass is claimed from
the smaller validation allowance. Each later segment requires authenticated
predecessor scientific completion, exact saved-state replay, fresh native and
remaining-capacity/runtime gates. Failed/uncompleted segments never authorize
successors. No gate waiver, additional cleanup or heartbeat unpause.

Random50k keepsLR.3/Dirichlet1 seed17, .001through5000 then geometric100at50000.
Plain29792670 keeps native-onlyLR.3 and global10000->50000, no new chemistry.
All old runs/report editions/Boltz endpoint diagnostics remain unchanged. A new
50k report must pin actual new segment IDs and join global saved frames with
duplicate boundary states authenticated exactly; old contracts/editions cannot
be retargeted. All rendering/array/gradient checks stay allocated DTU HPC only.

### 2026-10-23 — prepared Boltz LR0.0025 comparisons and independent DBMol pilot

User authorized two fresh native Boltz2 LR0.0025/30000-update arms, with and
without chemistry validity loss, plus an independent binding-only DBMol composite
LR0.0025/10000 pilot. User explicitly selected stretched chemistry timing:
0.001 through applied5000, geometric to100at30000. New experiment homes
`experiments/brd4-jq1/boltz-lr0025` and `experiments/brd4-jq1/boltz-dbmol`.
All saved studies/current jobs remain unchanged. Allocated raw component+total
FD/parity/null/nonzero/independent-primal/frozen/FP64-scalar-FP32-graph tests are
mandatory before the new pilot; no AD-only substitute or scientific gate waiver.
Serial exact-logit segments preserve global indices and the requested budgets.
Fresh admission must include both pending Protenix50k chains and all concurrent
Mosaic output projections, not just each new job in isolation. No new experiment
executed or submitted by this preparation event. Local work is stdlib/static only.

Read-only HPC metadata at2026-10-23T10:13:50.839983Z authenticated both existing
checkpoint files: affinity2062139170bytes/dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e
and structural2286561469bytes/090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1.
Protenix50k native29798159/61 and callbacks29798160/62 remainPEND, with no full
extension projection yet; existingchemistry29792672 RUN and diagnostic29796551 PEND.
Raw work3quota75.07/100GiB is an observation, not a reservation or full-new-runs pass.
Planned new roots/releases were absent. No model/array calculation, cleanup or
existing scientific mutation occurred.

### Boltz LR0.0025 pair actually staged/submitted,23October12:26Paris

New source633 authenticated read-only, manifest43767a8564c220821fa06fa68c3a3690870cda2e22baeb7bf956b7f849094e76, archive39617746B SHAd19a5d8671b12990702ee0502f78e247e36bde97d33373461a6de1de25dcc78f. CPU29798466/afterCPU29798470 actually submitted10:26:39/54UTC. Full guarded serial30k pair configuredLR.0025, chemistry.001through5000->100at30000. Initial validation storage gate passed18235499918required versus19462959470conservative available,4GiB/100000files retained; not a full30k/Pro50k long-storage pass. All existing runs/results unchanged. Native scientific passes and optimization starts are not yet established. See experiments/brd4-jq1/boltz-lr0025/runs/20261023-boltz-lr0025-01/campaign.json.

### Boltz30k fixture failure preserved and corrected; DBMol fresh admission,23October12:41Paris

Baseline source01 CPU29798466 EXIT1, callback29798470 EXIT1. Actual466contracts:
464passed/twofailed/noerrors/skips; latergroupsunreached. Independent21-file
byte/XML/LSF/scheduler/source633auditpassed as FAILED evidence (archive51322B
SHAd7f937585e481c094eb2df1f2ee7f43a585f717dce704d29065299f0061d8334).
Cause: inheritedProtenixunitfixture monkeypatchwascapturedbylazilyimported
chemistry.checks, thenassertedReplayLogitsduringnewGraphLogitsfixtures.
No scientific/native/GPU testornewoptimizationran. Newimmutableattempt02
preloadsrealchecksintheunitfixtureandaddsrestorationregression; all145src,
driver/contract/protocol/configbytesexact01. Source633manifest3689071a4e7dd135071ff5b420c026543c4d10887c951077dff63d498d9f812c,
archive39618948B SHA40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa.
CPU29798538/callback29798539submitted10:39:27UTC; fullrevalidationpending.
CPUownallowance1GiBtotal,nativegroup2GiB,allfullprojections/4GiB100000files
unchanged; independentlypreparednoisypilot1GiB/externalstalebytesincluded.

DBMolsource657manifest24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436,
archive39676321B SHAec333fd3e0b33d9e5b82e0cd5e4e12c14ffb4f91782c18cd820d4bfab53de1f8.
FirstpreflightFAILEDbeforeanyCPU:19256933986required/19236336523available;
failedgateSHA9822d97a999ba1232692a17e4e9308518eecf09e583988e93f8954be6d5d0c1f
andindependent6-file/source657auditpreserved. Recover118913015bytesONLYfrom
threeexactredundantuploadtarballscreatedduringTHISrequestafterlocalarchive
backupverificationandall633/657/633remotestagedmembersrehashed. Everyimmutable
releasedirectory/savedstudy/historicalfailure/cache/model/envremainsunchanged.
Freshmetadataadmission02 reauthenticatesoldfailedgate,nopriorCPU/unknownmarkers
andunchangedsource; newgatesiblingretainsoldfailure. ActualCPU29798549/
callback29798550submitted10:41:05UTC after19225068665required/19235502439available
validationadmissionpassedwithunchanged4GiB/100000filereserve. No fullnewlong
storage/readiness/gradientpass is claimed. Noheartbeatunpauseoroldjobchange.

### Boltz30k CPU799 passed/native queued; DBMol entry resource failure, 23October2026

[Boltz attempt02](../experiments/brd4-jq1/boltz-lr0025/runs/20261023-boltz-lr0025-02/README.md): actualCPU29798538/callback29798539 DONE/hpc/samejobLSFsuccess;799 actualcases passed (467+239+34+59), nofail/errors/skips,3 deliberatelydeselectedlegacy. Certificateb606f924365ece881fc7a348d4c677a4d70858ce92cc259c015a63d2b2dd45b9; all46 compactfiles/source633/rawXML/vector proofs independently audited. Newnative Boltz29798605/callback29798606, chemistry29798607/callback29798608 submitted/PEND. These are validators, not optimization starts; no native passyet.

[DBMol pilot01](../experiments/brd4-jq1/boltz-dbmol/runs/20261023-boltz-dbmol-pilot-01/README.md): CPU29798549/callback29798550 EXIT1 atentry bytequota beforepytest/model, required19,175,262,065/available18,158,117,863. Independent24file/source657/scheduler/LSF failure auditpassed; no CPUcases/native/optimization claim. Preserve failedoriginalpreflight/passingadmission02/failedCPUentry; new small immutable launcher-only completion-aware storage candidate prepared separately, retaining source657 and all active/prospective/reserve/stale-byte/scientific guards.

### 2026-10-23 — DBMol launcher02 prepared after CPU entry quota failure

[Storage adoption02](../experiments/brd4-jq1/boltz-dbmol/runs/20261023-boltz-dbmol-storage-02/README.md) is a new five-file immutable metadata launcher, manifest5f479f2e5342469efd8606c9457c946a51dbcf0dd90d837ee472babfdd820a4b, archive20,406bytes SHAa26dafcac4fbc342a275dedb1c93ed8880062f9e3dafe48d188937c3137dd750. Scientific source657/config/loss/driver/worker/checks remain unchanged. Exact original Pro/Bol terminal summary/source/submission/LSF proofs are required before zero future growth; every current byte stays stale-charged and Bol scientific failure remains failed. All active/prospective budgets plus4GiB/100000files remain mandatory. NewCPU02 parents repeat all six inherited/current groups plus72 metadata cases, expected1135 actual cases. Both source seals authenticate before imports; CPU reference paths and dotted XML attribution corrected in this new launcher only. Local stdlib AST/byte review passed; allocated checks and actual submission are pending.

### 2026-10-23 — DBMol launcher02 actually staged and CPU submitted

Actual11:01:04–05UTC CPU29798699/callback29798700 submitted, newcpu-02/after-cpu-02 parents in original pilot01 controlroot. Smallsource5 manifest5f479f2e5342469efd8606c9457c946a51dbcf0dd90d837ee472babfdd820a4b/archive20,406B SHAa26dafcac4fbc342a275dedb1c93ed8880062f9e3dafe48d188937c3137dd750 authenticatedread-only. Exactsource657 and every historical failedCPU/gate reauthenticated before import; originalsummary/source/submission/LSF pins established terminalfuturezero forPro/Bol while actualsavedbytes remain stalecharged. Freshvalidationgate passed17,934,470,372required versus34,151,538,760available, own4GiB/100000file reserve retained; quota61.34GiB observed, no newhistorical cleanup by this action. Expected1135rawCPUcases notyetpassed. Native/full10k admission and all newoptimization starts remain unestablished.

### 2026-10-23 — DBMol CPU02 transport fixtures failed; immutable fixture03 prepared

CPU29798699/callback29798700 actualEXIT1 after1,056 passes/seven failures, no errors/skips. Base832 and native-objective83 unit cases passed; transport141/148 passed; metadata72 unreached. Independent67-file/source657+launcher5/rawXML/LSF/scheduler failure audit passed as failure evidence, archive134,712bytes SHAdfd351cac57375696f671bc259b105e20040fcfd04c138b6b86f3c86d7ecc2fb. All seven are fixture defects: nonfinite derivatives correctly rejected before probe-row creation; candidate/mock-reference alias changed expected policy fields; positive peer fixture used incorrect job prefix. Preserve all original failures and sealed sources.

[New launcher03](../experiments/brd4-jq1/boltz-dbmol/runs/20261023-boltz-dbmol-storage-03/README.md) adds two corrected fixture overlays only, retaining exact scientific657/scorer/worker/driver/native-FD policy/configuration and original controlroot. Newcpu-03/after-cpu-03 parents must repeat all1135 raw cases, keep old49/50 and699/700 failedproofs, and authenticate bothsource seals/unchanged whole budgets. Prepared only; no newCPU/native pass or optimization start follows. Baseline799pass and native29798605/607 remain unchanged.

### 2026-10-23 — fixture-only DBMol launcher03 staged and submitted

Source7 manifest7b67e6221e4d227b6e501478802ab4ebc8536d74272ca3a7d305bb6f15e7ffee/archive31,721B SHAe8d60f4a03f0d6dcd05b9dbe66add8885fa7253d95537b08b3e170ed7ca763ca authenticatedread-only. ActualCPU29798769/callback29798770 submitted11:15:30–31UTC after fresh validation gate17,852,645,235required/34,069,120,249available; reserves4GiB/100000files unchanged. Source657 scientificcode/config/driver/worker/nativeFD unchanged; two fixture corrections are explicit additive source7 members, and both earlier failedCPU attempts/rawcases/LSF/proofs remain required. New full1135-case allocated repeat pending; no native pass or new optimization start claimed.

### 2026-10-23 14:14 UTC — new Boltz/DBMol outcome verification blocked by SSH authentication

One compact read-only query returned255/Permission denied(password), captured under fixture03 observations and latest-status-unavailable.json. No retry or scientific mutation. Latest successful11:17UTC metadata: baseline native29798605/607 +callbacks PEND, DBMol CPU29798769 RUN/callback29798770 PEND. Baseline CPU799 earlier independently passed; DBMol current full1135 suite/native/fulloptimization outcomes remain unverified. Both prior failed DBMol CPU attempts and source657/launcher02/03 remain preserved. Restore DTU VPN/SSH authentication, inspect actual current receipts/proofs before further action, and never duplicate existing jobs. No heartbeat unpause, new cleanup or false completion.

### 2026-10-23 — DBMol CPU03 passed1135; callback attribution corrected in new launcher04

At19:01:29UTC the connection succeeded. Both30k Boltz native validators29798605/29798607 and their own-ended callbacks remain PEND with job-slot limits on four hosts. No pretrained native pass or production segment is established. DBMol CPU29798769 actually completed all1,135 raw cases (500+239+34+59+83+148+72), zero failures/errors/skips. Callback29798770 exited1 because nine unchanged bridge tests outside the launcher pytest root had blank XML module names; strict attribution correctly blocked certification and GPU submission. This is a reporting/packaging defect, not a failed native gradient evaluation.

Independent109-file evidence audit authenticates source657+launcher7+priorlauncher5, every XML/log/proof/LSF and actual submissions. Archive201,096bytes SHA1659285fa9f14ecca3425529f873020b028d2c8eab129efa3c143055786bd097. CPU success and callback failure are separate preserved facts; aged-out live records remain unavailable and genuine persisted LSF remains authoritative.

[Launcher04](../experiments/brd4-jq1/boltz-dbmol/runs/20261023-boltz-dbmol-storage-04/README.md) adds an exact unchanged source657 bridge fixture inside the same launcher root as the other two transport fixtures. Source8 manifestdf89def61047469e0db8ee6f4b235c75344dd4ad9f35a0833b0839f23859e9fa, archive35,608bytes SHA40ee3c1ea0e9181528c82c5d99e7d6347033a99051ba7e55b5ffdde1edb7cefc. Independent static AST/member/byte review passed; scientific scorer/worker/driver/config/native-FD/storage/reserves remain unchanged. Newcpu-04/after-cpu-04 repeat all1,135 cases before native dispatch. All earlier decisions and saved studies remain unchanged; no numerical work ran locally and no monitor was unpaused.

Actual newCPU29802621/callback29802622 submitted19:08:22UTC after source8+657 and43 preserved history proofs authenticated. Fresh validation admission passed16,003,104,260required/31,147,201,201available, retaining4GiB/100000files. This is not full optimization capacity or a native gradient pass. No GPU/native/optimization start follows from submission alone.

### 2026-10-24 01:30 Paris — DBMol CPU1135 passed; all three new Boltz validations failed startup metadata

Fresh read-only connection succeeded. DBMol CPU29802621/callback29802622 passed all1,135 raw cases (500+239+34+59+83+148+72), zero failures/errors/skips. Independent168-file/source657+launcher8+historical7/5/rawXML/LSF/source/resource audit passed; CPU certificate14199ca0444d614385c5bda4863fd449e063a34540085513c3985080e02b5880, archive286,297bytes SHAcfe7d8e0b80c455b84474769c46b7551b66588eeeee9ad327500fe6ab8ce918b. Earlier CPU03 pass/callback attribution failure and CPU02 failures remain preserved.

All three actual native validators exited1 before model workers/gradient checks with KeyError git_revision_reference in unchanged provenance.source_identity: plain29798605 at23October22:05:42Paris, chemistry29798607 at22:06:34, DBMol29802648 at22:07:35. Their own callbacks29798606/608/29802649 correctly failed on failed producers. Baseline633 and DBMol657 manifests omit the required revision metadata. CPU unit passes do not certify this native startup path. No derivative disagreement, native readiness pass, production segment or optimization start was established. Live scheduler records aged out; exact persisted same-job LSF failure and source/driver/stderr proofs remain authoritative. No old saved result or setting changed.

Baseline native64-file archive90,570bytes SHAbadea0d29e26d40123c51e6efd8376567a5ed54e11f6d537daa8d9040dec6fc5; DBMol independent failure receipt is in [launcher04](../experiments/brd4-jq1/boltz-dbmol/runs/20261023-boltz-dbmol-storage-04/run-record.md). Correction requirements: new immutable source/run identity carrying authenticated inherited git_revision_reference, allocated real generated-command startup regression before model construction, then repeat required CPU/native/storage checks without altering scientific policy or sealed failures. This status/evidence turn submitted no replacement jobs or corrective release, ran no local numerical work, and did not unpause the monitor.

### 2026-10-24 — authorized provenance/startup correction and unchanged Boltz resubmission prepared

The user explicitly authorized correcting the missing provenance field, testing actual generated startup commands and resubmitting the two fresh30k Boltz arms and10k DBMol pilot at unchanged settings/checks, preserving all saved results. New immutable roots are boltz-lr0025-03 and boltz-dbmol-pilot-02 under the existing analysisBASE; no failed parent is replayed. Rates remain0.0025, chemistry0.001through5000->100at30000, DBMol binding-only/no chemistry, original contexts/pocket/key17/N25/T1/fixed single candidate and all nativeFD/parity/null/nonzero/frozen/runtime/storage gates.

DBMol scientific02 changes only manifest metadata:657 exact old member bytes, manifest600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed, archive39,676,528bytes SHA592a4b3d7d95909111013dacc95d4a1301b6e875fb124267ad08af7ca402381e. git_revision_referencec74c659344123937608077e3e4078f75f09cd743 comes from authenticated ancestor625manifestc2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057. It is not inferred from currentHEAD. Baseline03 and launcher05 add explicit startup proof wiring and fresh output/peer identities; all mathematical settings remain unchanged.

Shared standard-library startup helper and28 allocated fixtures are prepared under scripts/hpc/startup and tests/core. The actual allocated callback executes ops.produce admission and captures its generated driver argv, runs the real driver/provenance with disposable owned-node output, and stops only at a classified worker-constructor sentinel before model loading. The summary remains an intentional failed boundary; a separate startup_only_passed receipt never certifies gradients or optimization. Native validate startup is tested after actual CPU certification; production run startup is tested only after genuine native readiness, including correct offsets and saved boundary inputs. Native/production entry authenticates matching receipts. No numerical work runs locally.

An initial read-only SSH preflight aborted without completed target/quota/peer evidence; raw output/argv/error remain under results/hpc/20261024-boltz-startup-recovery-01. No source, job or pass follows from that query. Source staging and allocated CPU/native resubmission still require authenticated fresh exclusive targets and capacity. Missing full Protenix641 peer projections remain a full-production blocker; no forecast, gate waiver or Protenix job/settings change is authorized by this correction. The paused monitor remains paused.

### 2026-10-24 — provenance correction locally sealed; allocated testing/resubmission blocked by SSH

New baseline03 source635 is sealed locally: manifestc214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6, archive39,628,961bytes SHAd77d298b7cd2df41b4fb97fb22bf241a623c0d9825edab89226c54fa90b4b9ba. All145 scientific libraries, driver, contract, serial protocol, configs and existing fixtures are unchanged. Only metadata controller/output/peer/startup wiring, its wrapper/doc, and shared helper/28fixtures are additive. Required allocated CPU count827, not passed.

DBMol source657 manifest-only correction600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed is paired with new launcher05 source10 manifest865303de06fc0a0bdf9297857c5bbc78c6ccd9ee05177e372a51869adc599bc3, archive46,951bytes SHAd3690757d45422550aca1317491903a446a13ccedb3209621886933229b24586. All657 scientific members and all3 corrected transport fixtures remain byte-identical to their authenticated originals; existing72metadata cases preserve CPU04pass/native-startup failures through explicit old HISTORY_ROOT. Required1163raw cases and allactual startup/native checks are pending allocated execution.

The actual science02 staging attempt returned SSH255/Permission denied(password) before its remote command, with empty stdout and exact rawstderr/argv/result/marker preserved. Initial read-only preflight also aborted without completed evidence. No source staging, newCPU/native/optimization job or new numerical pass is established; no failed marker was deleted/replayed and no later submission was attempted. The source correction and startup fixture wiring received local stdlib byte/AST review only. Saved results/settings/old releases remain unchanged and monitor remains paused. Restore DTU SSH/VPN access, inspect exact targets/unknown outcomes, then perform a distinct authenticated staging recovery and all required allocated checks. Full production still requires unchanged native/runtime/all-peer quota proofs; no Protenix scope or projection waiver.

### 2026-10-24 — access restored; provenance-corrected Boltz/DBMol allocated rechecks submitted

The user explicitly renewed connection/startup-test/conditional resubmission authorization. One compact read-only inspection succeeded08:54:35UTC and found all eight exact new roots/releases/uploads absent. Old SSH255 and markers remain preserved. Distinct science02 recovery staged657 unchanged scientific member bytes read-only with corrected manifest600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed. Baseline03 source635 and DBMol launcher05 source10 then staged/authenticated read-only; no old run/release was replayed or changed.

Actual baselineCPU29805913/callback29805914 submitted08:56:23–24UTC; DBMolCPU29805915/callback29805916 followed. Fresh validation preflights passed16,103,826,540/27,569,509,027 and16,003,104,260/27,569,248,222 required/available bytes, with4GiB/100000-file reserves and all current/stale/prospective scopes retained. At08:57:59UTC bothCPU jobs RUN, own-ended callbacksPEND. These are not actual startup, native gradient or production passes:827/1163 raw cases and real driver checks are pending. Full production still requires authentic full Protenix641 peer projections; no quota waiver or Protenix scope/job change. Settings/saved results/paused monitor unchanged. Exact vectors/byte hashes/snapshot receipts live in the new03 and launcher05 recovery-02-submissions.json records.

### 2026-10-24 — separate Protenix extension provenance recovery explicitly authorized

Fresh read-only08:59:40UTC proof rehashed all641 original Protenix extension source members and authenticated same-job LSFEXIT1 for random29798159/continuation29798161 plus correctly failed callbacks29798160/162. Both drivers failed source_identity with missing git_revision_reference before model construction; projection.json and segment parents are absent. This is an engineering startup failure, not a measured native derivative disagreement. Exact captured stdout/stderr/source hashes are in baseline03/peer-projection-blocker-02.json. No original source/result/job was changed.

The unchanged full-storage guards for Boltz03 and DBMol05 require authentic full peer projections; no allowance is relabeled a full budget. In response to the explicit scope question the user authorized a separate immutable provenance/startup correction and revalidation for those Protenix50k random and additional40k continuation launches too. New Pro02 preparation preserves the full original scientific libraries/driver/config/protocol/initialization/snapshot/ramp/offsets, all gradient/runtime/quota checks and old saved results. Real Protenix startup keeps its actual GPU backend guard and must execute on allocated GPU before the unchanged driver; CPU argument/provenance structure alone will not be called a GPU startup pass. New launcher-only production adoption for the Boltz/DBMol outputs is being prepared to authenticate Pro02 proofs rather than manufacture old641 projections or edit sealed controllers. Current CPU/native rechecks continue independently; no new Pro pass/submission follows from preparation.

### 2026-10-24 — full Boltz827/DBMol1163 CPU suites passed; both Boltz actual startup commands passed

Independent standard-library byte/JSON/XML/schema/source/resource/same-job LSF verification of194 compact files at09:05:32UTC authenticated baselineCPU29805913 DONE/827 actual rows (467+239+34+59+28), certificate06efbb027f1008a0921494af0e226c3fc7a78c0a8f4ab01b93e8b50f2c042836, and DBMolCPU29805915 DONE/1163 actual rows (500+239+34+59+83+148+72+28), certificate387070e8fe56b17ddbd7c8e0b82d2d6a44d9aa99e7015d3fbe4757ce82c0819b. No failures/errors/skips; inherited three legacy deselections are explicit. XML hpc-contract headers470/503 include three unchanged unittest subtest header increments; actual467/500 rows/log counts are the scope, no count inflation.

Baselinecallback29805914 actualDONE/same-jobLSF; both real generated validation commands reached the intentional pre-model boundary with exactsource/config/CPU/reference/pocket/context/argv bindings. Startup-only receipts remain explicitly not gradient/readiness/optimization passes. Nativeplain29805950 RUN/callback29805951 PEND; nativechemistry29805958/callback29805959 PEND. DBMolcallback29805916 RUN/actualstartup notyetverified; no DBMol nativeID yet. Optimization0. Old huge summaries and prepared context are exact remote-hash metadata-only references, not a claim those original bytes were downloaded. Detailed independent receipts are cpu-startup-verification-06.json in new03 and launcher05; savedresults/settings/oldfailures unchanged.

### 2026-10-24 — all three real validation startup commands passed; plain Boltz native readiness passed

At09:08:48UTC both afterCPU29805914/29805916 are actualDONE/LSFsuccess. Independent270-file proof/source/schema/vector audit verified all real validation startup-only receipts, exact generated argv/inputsource/context pins and CPU827/1163 raw cases. Baseline verification7189483347d31f5fc090d395716e52c48679a7a547be5cc9f8fdac808a254e0f; DBMol f42274bfb920ba9e8cbe2f7a8bf2e5ebf4f16de4398ca87e22982fa70561cead. DBMol actual native29805980/callback29805981 PEND; chemistry29805958 RUN/callback29805959 PEND.

Plain native29805950/callback29805951 actualDONE/same-job LSF. Independent byte/schema audit verified readinessSHA75f9feecf5f9eb0b1850579238718f74c1133db5e8326cf8d66e5ea96d81fea3, all52 raw proof files, exact immutable source635/config/LR.0025/fixed-single candidate/model/context/pocket, raw X/E epsilon.01/.003/.001 reported adjacent agreement, hard parity, frozen before/after and ten-update/runtime gates. This is native readiness, not10k or30k production; modevalidate has only its disposable ten-update pilot. Original numerical decisions stay allocated-HPC authoritative, no local array parsing or derivative recalculation. Full production0 and all scientific/reserve guards remain intact.

### 2026-10-24 — chemistry and DBMol native readiness passed with exact proof audits

At09:14:55UTC all actual initial native/callback jobs DONE with same-job LSFsuccess: chemistry29805958/59 and DBMol29805980/81, alongside plain29805950/51. Independent local stdlib byte/schema audits rehashed475compactfiles, all635/657/10sealedsource members and CPU827/1163proofchains. Chemistry128rawproofs authenticated0.001/100coupled checks and BOTHten-update pilots, originalrawFD0.01/.003/.001/twoadjacent, hardparity/nonzero/frozen/runtime. ReadinessSHA927a20848aa65d30f2d22e3b8441a561ce1e6d33a0ce72ea30773c56d368349a. DBMol53proofs authenticated allfour declared component/total rawX/EFD, productionJAXtransport, independentprimal, live exact recomposition, nulls, frozen, hardparity and ten-update/runtime. Readiness04918662a1671a7089e5d8155a92ef8aec3d4a6d894f71c3a19ea487a6686992. No numerical calculation ran locally.

Production remains0. Full guard requires actual new Pro03 peer projections, allocated metadata adoption and fresh whole-campaign capacity; no old641 projection is invented. Native validation startup passes are distinct from actual production startup. Local Pro02 candidate remains unsubmitted/preserved after preventive code review found its long-lived startup process could retain GPU resources; no allocated failure or actual OOM is claimed. New Pro03 isolates genuine GPU startup in an owned short-lived same-LSF subprocess before unchanged model execution, without allocator/scientific changes. All145scientific libraries/driver/config/protocol/saved parent inputs remain exact; full866allocated CPU plus realGPUstartup/native gates pending. Unknown extra noisy capsule inventory fails closed; known01 cap unchanged and two metadataCONTROL growth budgets explicitly charged. Old results/settings/paused monitor preserved.

### 2026-10-24 — Pro03 source staged; unchanged shared storage blocked CPU and all production

Pro03 source643 manifestda9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562/archive39,644,558bytes SHAbde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef is staged read-only; fresh09:25:58UTC collector independently rehashed all643members. Original638protectedmembers/145scientificlibraries/driver/protocol/config/parent snapshot remain exact. Original failed641 validation proofs and unsubmittedlocal02candidate remain preserved. The first outside-seal upload operator failed on stale size==0 before any mutation; exact failed operator/receipts and09:21:30RC0 eight-absent-target inspection remain unchanged. Distinct authorized recovery02 changed only that unsealed expected size and reauthenticated history/targets before staging; no sealed runtime or scientific change.

Mandatory Pro03 preflight failed BEFOREjobs.json/CPU/native/modelwork:41,643,669,252required versus27,472,908,456conservativeavailable, shortfall14,170,760,796bytes. It retains fullactual Boltz30k pair20,549,890,944/DBMol10k4,997,967,296projections, existing remaining scopes/report/diagnostic/noisy, bothProvalidationallowances/twoexplicitmetadataCONTROLallowances,4GiBreserve/100000files and entirecurrent stale accounting. This is not a full Pro50k/40k production-capacity bound; actual future Pro pilots may increase the requirement. Rawwork3quota65.81/100GiB, finitefileheadroom19,564,018; allzeroFSinode tuple remains supplemental-unreported, not independentFSproof. GateSHA6d1d0fae516de98615b83ebc005ab2330dd2146c08ef631f1bd5c3d28593a951; independentfailed-gate verification6b7228c4b2b14203e961243fc1fde0b92ee8cfc8c9c3de7bd00245150e4854a5. No ProCPU866pass, GPUstartup/nativepass or any new production job follows from staging.

Readonly Mosaic run/release inventory completedRC0; no safe verified inactive duplicate subset sufficient for the unchanged gate was identified and no deletion occurred. Protected saved studies/sources/proofs/caches/models/envs and all other projects remain unchanged. Home09:25:40snapshot21.19/30GiB (cached11:04:12Paris) has atmost8GiB after mandatory rounding, already below even smallest known full DBMol arm+own4GiBreserve9,292,934,592bytes before stale/recent-write charges. No home inventory/admission pass is claimed. Human was asked for additional approved HPC capacity/directory or a specific inactive dataset for verified archival. All3Boltz/DBMol CPU/actualvalidationstartup/native gradient passes remain valid; production and Prorevalidation blocked on resource admission, with no weakened checks/settings or monitor unpause.

### 2026-10-24 — unsealed peer-adoption preparation retained under resource blocker

Shared standard-library metadata tooling under scripts/hpc/peer_adoption and two documented consumer controls is prepared only, unsealed/unsubmitted. It preserves original635/657 scientific sources, CPU/native results and output roots, separately requires73planned allocated metadata fixtures, authenticates actual Pro03 CPU/native/full projections, and claims exclusive original dispatch ownership only after all original callbacks have actual terminal success. Exact generated production startup receipts, additional boundary-native checks, nondecreasing empirical projections, exclusive unknown-outcome guards and unchanged full byte/inode/reserve/runtime/scientific checks remain mandatory. No source archive/manifest or allocated metadata pass exists. Local AST/byte review is not numerical testing or an adoption/optimization result.

The required repository stdlib layout check ran24October09:29UTC and retains the preexisting default-discovery mismatch85paths/82canonical testmodules. No new AST/link/shell/byte error was reported; broad layout status remainsfailed and is not relabeledpass. Scoped independent source/byte/XML/LSF/readiness audits are separate passed evidence. No local numerical/model test, new cleanup, other-project change or monitor unpause occurred.


### 2026-10-24 — authorized exact-backup inactive-artifact cleanup investigation

The user explicitly authorized selecting artifacts unnecessary for current experiments, downloading complete laptop backups and verifying every selected original byte before removing only identical remote copies. New canonical storage-offload-01 records source/input/proof protection reviews and no-follow metadata inventories. Current Mosaic native passes remain valid; all production and Pro03 revalidation remain blocked by the unchanged full shared-capacity guard. No saved result is modified or scientific gate weakened.

Account-wide inspection also found independent source138 VAE training RUN and embedding comparison PEND; their actual sources, datasets, original checkpoints, contexts, catalog, bridge and current proof dependencies are protected. Prior complete backups were rehashed, and actual existence checks establish most duplicate report files had already been removed by three earlier cleanup operations; those bytes are not counted again. No remote deletion has occurred in this new operation. The first shared dependency scanner exhausted its 1GiB text bound by traversing historical documentation/proof histories; its incomplete evidence is retained and cannot authorize deletion. A narrower actual-runtime dependency audit and allocated small-file archive/verification/removal contract tests are being prepared. All local work is standard-library source/AST/metadata/byte inspection only; numerical/model work remains allocated HPC, unrelated changes and the paused monitor remain unchanged.


### 2026-10-24 — cleanup metadata operators validated; no original deletion yet

Dedicated operator source01/CPU29806186 retained actual EXIT:18passed/one synthetic symlink USTAR-path construction failure, before verifier invocation. New immutable02 corrected only timestamp-shaped legacy-root admission and fixture coverage; CPU29806198 actualDONE/LSF/31rawcasespassed. A separate real read-only inventory then correctly blocked on unsupported `bjobs -w -o` combination; source02 fixture scope remains passed, not a successful actual scheduler query. New immutable03 removes only incompatible `-w` and adds an unmocked allocated actual-command regression. CPU29806228 actualDONE/hpc/same-jobLSF/32rawcasespassed with source11 manifest9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d, archive26,508bytes SHA7b2be2caadf109d2e94945d606511004bf54a781341c006344d9c93ce586f02b. Independent local standard-library source/archive/byte/XML/resource/LSF audits passed. Both metadata preflight and entry use disposable128MiB plus unchanged4GiB/100000-file headroom; neither admits scientific production or changes the failed whole scientific guard.

Actual independent VAE/embedding source seals577/582members, source plans, selected checkpoints/contexts/catalog/proof metadata and fourteen raw entrypoint/controller members were byte authenticated. An additive import audit expands152to176reachable members and actual controller-MRO review closes a bare-import scanner limitation; historical command histories are not treated as active runtime input paths. Only exact reviewed legacy-run candidates can narrow the earlier conservative blanket protection; all Mosaic/home/current sources/models/envs/data/proofs remain protected. Fresh whole runs/submissions inode-alias and exact-file hash inventory is in progress. No file is deleted before a complete laptop archive, every-member verification and fresh current-job/race guards. No current saved scientific result, setting, job or paused monitor has changed.


### 2026-10-24 — source04 metadata CPU44 passed; incomplete backup02 intentionally retired

Source04 eleven-member seal8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe and31,403-byte archivece1fb5252bbd4ba57669198187efec8d77c041b33b958ee352c10248e5d3fd7b preserve all original file-open/hash/alias/activity/race/unlink guards; only canonical membership lookup and gzip level1 differ from03. The original04 caller failed Windows CreateProcess error206 before an SSH child started. Its source and failure remain. A distinct framed-stdin caller carried the same20,605-byte reviewed remote program and exact archive successfully. ActualCPU29806439 DONE/hpc at24October12:54:58Paris, same-jobLSFsuccess/44rawpasses/no failures/errors/skips, independently audited via unchanged verifier and98rawproofrecords. AuditSHA d712220f553d0988b0e1e3ae9655d78369999665f71641e5a670c52e163d3ce1. This is allocated byte/metadata validation, not a scientific admission or model pass.

Operation02/backup02 used the fresh04 authority but network pipe waits made its full32,041,124,599-byte original-member scope unsuitable for the unchanged one-hour bound. Root authenticated and intentionally stopped only ownedPID177888/UID306121/hpclogin1/exactargv/plan5fc74e3eeb3eedb2df5dfec541bb62b31e2b03dc01b8d0dd20eb3ad277dff6ea. Separately fetched raw1132-byte retirement completion SHA3324ff5a10fe9e461ab1946c4072859a8a09d4ffce72eca9d09cebcf2a9a2da2 establishes producer no longer running, every remote original retained and original deletions0. The1,470,834,414-byte local partial remains; transfer-record SHA241ed738bdecc407f8c4bdfe77586b054d7fcf7282e1099463d77794c425c731, no full-member verification/deletion authority. Earlier operation01 partial, every source/helper/plan and all failed evidence remain unchanged. No active cleanup transfer is established after this retirement.

A distinct operation03 is proposed using only25,413 complete single-link files/15,725,430,139logicalbytes from the original exact inventory;324 linked aliases in123groups remain HPC-protected with all current source/input/proof protection. New local plan/protection/helper binding preparation is metadata-only and unexecuted; no operation03 stream, original unlink, quota gain or fresh scientific-capacity pass is claimed. Mandatory laptop full-member verification, separate actual receipts, fresh activity/protection/content/inode/race checks and unchanged full Pro03/other scientific admission gates remain. No scientific settings, saved current results, other project state or paused monitor changed. The prior campaign bytes are preserved as storage-offload-01/campaign-checkpoint-02.json before its separate actual-retirement update.


### 2026-10-24 — single-link archive03 approved and fresh backup started

The distinct metadata-only recovery03 authoring caller passed after preserving two earlier pre-write local failures: Windows fstat/stat creation-time precision differed, then the old inventory active-ID list lacked the independently authenticated newer cleanup CPU29806439. The new caller compares each full metadata channel before/after, preserves shared device/inode/size/mtime/mode/link identity, and requires exactly the old audited active set plus that genuine CPU44 ID. Neither earlier failure wrote any03 output or changed remote files.

Approved03 selects25,413 complete nlink1 files/15,725,430,139bytes from76 exact inactive roots; all324 linked aliases/123groups and20 unused prior roots are additionally protected. Plan15,558,449bytes SHA8abfb724eaa6629ac73d6985d6216e898788868aec70a6784bcce8ba975e57bb; protection86,675bytes SHAa87eec9f7e68e9174604e3900490ec1ed5cbc941fec968c8c413e6cb95afd3e9. Actual protection-only staging returned0 after rehashing source04/all11members, CPU44/rawLSF/9proofs/adoption and retirement02; actual account queue contains VAE29802716RUN and29796558/29802717/29802743PEND. No operation directory was created by staging.

Fresh generic validated source04 backup client started24October11:11:32UTC, local unified session78079, distinct legacy-offload-03/backup-03. Read-only startup inspection authenticated actual owned producerPID254377/exactplanargv/pipewait and its three initial receipt files; laptop bytes are arriving. Archive remains incomplete/unverified, originals deleted0 and actual quota savings unestablished. Complete actual receipt, full compressed SHA and every original member, separately guarded unchanged-file removal, durable post-audit and fresh original Pro03 whole storage admission are still required. All scientific settings/results, active VAE/embedding jobs and paused monitor remain unchanged.

Preventive local transport review also found the unexecuted Pro CPU recovery command would be33,824Windows characters. A distinct unexecuted framed-stdin caller carries the exact13,820-byte original remote program SHA97b891efb301fbe4b50a34c34bc63e09bcc7341327f334c7d6f578316c6cb696 and original argv through a3,593-character command, preserving every local guard AST and all scientific/source/admission/submission code. It will be used only after verified cleanup/fresh capacity. Source643 is unchanged; no Pro CPU866/GPU/native/production result follows from this preparation. New storage-run attributes retain exact authenticated CRLF/LF proof/operator bytes through Git.


### 2026-10-24 — verified laptop archive03, exact inactive cleanup completed, Pro03 CPU recovery submitted

The authorized inactive-artifact cleanup completed without a current scientific result or setting change. Fresh operation03 source04/CPU29806439 authority selected25,413 complete single-link files in76 reviewed inactive legacy roots,15,725,430,139logicalbytes. All324linked aliases/123inodegroups and current Mosaic/VAE/embedding inputs/checkpoints/sources/proofs stayed protected. Earlier incomplete01/02archives, intentional producer retirement receipts and every failure remain preserved. No partial archive authorized removal.

The laptop archive completed11:44:30UTC:8,769,931,895bytes SHAac75713c102dba31b0ea9934c2303ebd5cb529d5c98c422baccfc42afda29ff3. Independently collected actual stream receipt and standalone all25,414-member verification matched exact plan/original paths/lengths/hashes/gzipEOF before removal. Only exact still-identical originals were individually unlinked; actual deletionRC0 and durable journal independently reconcile25,413intents/results,15,725,430,139logicalbytes,retainedchanged0,alreadyabsent0,256actualactivity rechecks. No recursive deletion was used. A separate read-only post-audit checked every original absence and rehashed3,489protected pins/575,469,439bytes, including seven full Mosaic source seals and explicit current independent VAE/embedding/model/context/proof pins. Scope exclusions and non-exhaustive cache hashing are stated honestly. Completion receipt cleanup-completion-03.json SHAb3f877b20f11a5e76d8d2e27ee8d60e6cf1193f4ed430438855d5f7d97d6d11f and restore-notes-03.md bind laptop archive/membermap/journal/audit.

Independent post-cleanup quota observation51.22/100GiB is not an exact attributable physical-reclamation measurement: concurrent VAE writes/cached quota remain possible. Allzero supplemental work3filesystem inode counters are unreported, with authenticated finite-user-quota file minima still mandatory. Cleanup transfer/deleter/audit allendedRC0; no replay or additional cleanup needed for routine checks.

Fresh original Pro03 preflight then actually passed41,643,669,252required/42,505,285,132conservativeavailablebytes at raw51.09/100GiB,19,588,704finite-userfileheadroom,unchanged4GiB/100000-file reserves. Entire stale analysis-root/external noisy scopes, actual Boltz30k/DBMol10kprojections and both metadataCONTROLallowances remain charged. Original failed6d1d0fae...gate stays failed/unmodified. This new pass is validation admission only: both Pro50k/40k empirical pilot projections/full-long capacity remain unestablished.

Distinct reviewed streamed caller carried exact13,820-byte original remote program/argv through Windows stdin after preserving its unexecuted33,824-character command. ActualCPU29806660 and own-ended callback29806661 submitted11:53:11UTC, hpc4CPU/4GiBslot1h and hpc2CPU/4GiBslot30min respectively. Read-only11:56:41UTC capturesCPU RUN/callbackPEND and rehashes all643read-onlysource members. Required866actualCPUcases, actualGPUstartup/native passes and production are pending, not inferred. The three Boltz/DBMol original native passes remain valid but production awaits genuine Pro03pilot/adoption/fullstorage proofs. Local verifier preparation is stdlibAST/byte/schema only. Saved results/settings, independent jobs, unrelated work and explicitly paused monitor remain unchanged.


### 2026-10-24 — Pro03 full866 CPU and command metadata passed; real GPU validations submitted

Actual CPU29806660 and callback29806661 are DONE/hpc/same-jobLSF success. Fresh12:02:20UTC collection rehashed all643readonly source members and independently verified101compact original files plus exact authenticated external metadata references. All866actualrawcases passed:500contracts+239legacy+34mapping+59chemistry+34startup, zero failures/errors/skips and original3deliberatelegacy deselections. CPUcertificateSHAab3148ea6dd8d25361f85e9d8283578e0cf4429844d820a4dab6b03673ef577b; independent verificatione4563eb2035e48e9d552db97d8b2f4e358757ca21be13fbc3e225049fdc1eec8. Protected source643/641/625, all145scientific libraries, original parent/logits/key/config/model/context/reference/pocket and failed641 startup history match their original byte pins. Large parent summaries/JSONLs/NPZs remain authenticated remote references, not a download claim.

Both generated startup commands passed exact metadata capture only, with zero updates and no claimed GPU/model load. Actual fresh randomnative29806664/callback29806665 and continuationnative29806667/callback29806668 submitted14:01:12/14:01:58Paris, each2h gpul40s/4CPU/8GiBslot/oneexclusiveL40S with own-ended metadata callback. Allfour arePEND at12:02:20UTC. Full GPU startup, raw native/coupled/parity/frozen pilots, genuine50k/40k projections and fresh production capacity remain pending; no production segment or gradient pass is inferred from submission. Original failed preflight remains failed, all new validation admission gates retain reserves/scopes, scientific settings and saved results unchanged.


### 2026-10-24 — separate Boltz/DBMol peer-adoption75 source packages sealed locally

Local standard-library byte/AST/JSON packaging sealed two seven-member metadata launchers. Boltz04 manifest9c8b496f65f41ed7eb76263cd947064d40fa3821a692f7276b00b4ccf70c4ced/archive28,268bytes SHA5958373b3dc53a41cf6106264c611f334c6165d93f562ce167b221bf5cf6eac5; DBMol06 manifestcad7c8efa0c83a301e4c9e1183d3d279a648cf65f9c3bc3d9d69281c63d32f11/archive28,270bytes SHAf3604c518ec1e4637c23f68cbbf0bad9d635d657bdd33c52122812df662e17db. Original635/657 scientific sources, launcher05, original CPU827/1163/native/readiness/driver/config/gradient settings/outputroots are unchanged. No package staging or allocated metadata75case pass follows from local preparation.

A narrow pre-seal observer correction authenticates allfive actual native/callback mappings and persisted same-job terminal proofs, but waits only on genuinely unfinished actual IDs. Already terminal callbacks need no live scheduler-retention dependency; unknown IDs wait and actual failure blocks. Two additive fixture intents extend unchanged73 to75 planned cases, with no local pytest/numerical/model execution. All original scientific checks, full empirically observed projections, unchanged whole byte/inode/stale/noisy/4GiB/100000file reserves, exclusive ownership/startup/unknown-outcome/serial-boundary guards remain. Root-owned framed-stdin bootstrap is outside the runtime packages and may submit only a fresh allocated metadata CPU and its own-ended callback after all exact absent-target/source/native/admission checks; original saved results and paused monitor remain unchanged.


### 2026-10-24 — both new peer-adoption metadata CPUs actually submitted after unchanged admission

The separate framed-stdin bootstrap completedRC0 for both sealed seven-member controls. It authenticated original own scientific CPU/native/LSF/source/projection proofs and absence of production before any mutation, then rechecked after readonly staging. Every exact fresh release/upload/CONTROL/unknown-outcome target was absent; immutable original studies were untouched. Fresh whole metadata admission retained actual peer maxima, only explicit Pro validation allowances while full projections are missing, both1GiB CONTROLgrowth budgets, entire stale/noisy scopes and unchanged4GiB/100000file reserves. This is not a full-long storage pass or storage reservation.

Actual Boltz metadataCPU29806690/own-ended callback29806691 and DBMol metadataCPU29806697/own-ended callback29806698 were submitted, hpc2CPU/4GiBslot/30min. Their bootstrap-submissions.json records exact actual vectors and completion/admission/source pins.75rawcases/source-bound actual scheduler/LSF/CPU certificate still require independent audit. Only metadataCPU+callback were directly submitted; future observer uses actual unfinished Pro03 callbacks, and production must still wait all five actual native proof chains, genuine full Pro03 projections, startup and fresh whole-capacity gates. No settings/saved results/monitor change, extra cleanup, duplicate native or production submission occurred in this bootstrap.


### 2026-10-24 — both75-case adoption CPUs passed; actual observers wait only on genuine Pro03 checks

Both metadataCPU29806690/29806697 and callbacks29806691/29806698 actualDONE/hpc/same-jobLSF success. Independent byte/JSON/XML/schema/source/actual-resource/final-scheduler audits passed75rawcases each with no failures/errors/skips. Boltz independentverification3163434bb1d137de1e199508193450237d303306c051edad2e5cb451826ecbdc; DBMol cb7eaa6e4bc1741cc14a4316d55066110e5dbe9f2cf99703a6980cd25d4a278e. Original827/1163 scientificCPUcertificates, all native/readiness/source/config/rate/FD/parity/frozen/saved results remain unchanged. Inherited metadata-entry/storage gates plus additional two1GiB CONTROLallowances/4GiBreserve/100000finitequota-fileheadroom passed; these metadataCPU gates do not establish full Pro50k/40k production capacity.

Actual allocated callbacks queued Boltzobserver29806694 and DBMolobserver29806701. Independent raw submission/source/resource/dependency/scheduler/own-callbackLSF audit passed, with both observers PEND on only ended(29806665)&&ended(29806668). Allfive original native mappings and three older terminal same-jobLSF proofs are still authenticated; aged-out live Boltz/DBMol records are not new failures and not fabricatedDONE. Pro03native29806664/29806667 remainPEND at12:11:20UTC. No observer/adoption/nativePro/full-production-storage/optimization pass is inferred. Each existing observer will require all actual native successes, full empirical projections, exact generated startup and fresh unchanged complete capacity before any conditional production submission. No duplicate jobs, scientific changes, monitor unpause or additional cleanup occurred.

The separate read-only evidence helper was corrected before first execution to keep the exact2,549,313-byte prior-parent snapshots.json/ec8fa624... inventory freshly byte/hash verified while explicitly leaving its10,001childframe references outside this CPU metadata audit. Both unchanged32MiBdownload/2GiBhash/6000file bounds and every ownCPU/nativeproof check remain. It does not claim all parent frames were newly downloaded or rehashed; the independent Pro03CPU866 audit has its own larger parent proof scope. This is presentation/proof-collection tooling, outside both sealed seven-member runtime packages. All local work was standard-library byte/AST/JSON/XML and metadata; no local numerical/model/test execution.


### 2026-10-24 — Pro03 gradients passed but random runtime failed; all five production chains blocked

Fresh read-only22:49:35Paris collection independently rehashed610compactfiles,
source/input/proof maps and persisted same-job LSF; original live scheduler
records have aged out. Both genuine Pro GPU startup probes passed with short-lived
children fully exited before driver execution. Continuation29806667/callback29806668
passed full native readiness/certificate, raw X/E derivatives, exact parent replay,
frozen identities and empirical runtime31491.270122712012s versus82800s cap.

Random29806664/callback29806665 exited1 after native-gradient/parity/nonzero-chemistry/
exact-draw/frozen component passes. Its mandatory runtime-fit gate failed:
353948.120387632s conservative forecast per10000-update segment versus82800s allowed.
This is an empirical forecast, not an actual98-hour run or walltime timeout.
Raw native X/E .01/.003/.001 sweeps all agree; do not call this failed gradients.
The random low/high coupled and disposable-pilot JSONs were not downloaded by
this compact failed-job traversal; authenticated driver-flow inference and two
ten-update stdout traces are separately labeled, not independent raw-report passes
or retrospective overall readiness. Random has no native readiness certificate.

Only continuation full projection is certified. Boltzobserver29806694 and DBMol
observer29806701 correctly exited1/failed_peer on actual random callback failure;
both preserved exact failureSHA83149f37e1eaefe7484abf5e307621d299489b1720c87180ece120c0e0f6ef02.
Earlier Boltz/chemistry/DBMol native passes and metadataCPU75 passes remain valid.
Full production storage was not reached/passed, and no segment submission or
committed production prefix exists for either30kBoltz, DBMol10k, Pro50k or Pro+40k.
Historical failed storage proofs and successful validation-only admission gates
remain distinct. Shared independent audit4a6539621942378797d29d1ad9fddaaaabbf9ad46bbfd2e8bc8e48d0621e29c7
is retained with exact native/observer outcome proofs and chained campaign snapshots.
No job/settings/results, sealed source, independent project, cleanup or paused
monitor was changed. All local verification was standard-library bytes/metadata.


### 2026-10-24 — saved timing inspection attributes random Protenix slowdown to detached chemistry preparation

Fresh23:34:38Paris read-only collection rehashed26existing compact raw pilot,
reference and coupled reports/4,079,953bytes. Current Pro03/Boltz03 retain identical
18chemistry source members and loss/search/progress settings. The new Boltz30k
chemistry branch is delayed .001through5000→100at30000; randomPro50k uses the same
loss but .001through5000→100at50000. Earlier completed randomProchem29753668 was
6slots/10000updates/15261.50s driver, rather than current25slots.

Pro random low-weight pilot first update64.95s then18.70–19.18s; its detached
repair preparation repeatedly18.07–19.32s, exhausts96base+192extra checks and
retains censored/no_progress references. The smaller-model comparison therefore
does not explain measured runtime. Current Boltz starts from valid chain/uniform,
early prep about.007s/zero repair checks, invalid high assay about.25s/96basechecks
and a witness. High Pro pilot prep falls to1.16s bystep10. Delayed coefficient
still computes detached targets every update. First-update overhead contributes,
but cannot alone explain later19second updates. No fine-grained profiler/new
model evaluation ran;98.3h remains conservative extrapolation, not actual10kruntime.
Both new low/high raw coupled reports directly saypassed and bind original frozen
context/model; this resolves the previous compact-download limitation only.
Original runtime failure/all5production block remain unchanged, with no gate
waiver, source/setting/result/job change, extra cleanup or monitor unpause.
Exact local stdlib byte/JSON/source/timing audit is chemistry-timing-inspection-03.json
in the Pro03run. All scientific/model/numerical execution remains allocatedHPC.


### 2026-10-24 — selected non-chemistry launcher prepared; no new job yet

The user explicitly asked to start all requested runs without delayed chemistry.
A new shared launcher selects plainBoltz30k/DBMolbinding10k/Protenixnative+40k,
retaining original635/657/643science,CPU827/1163/866, successful selected native
proofs, exact settings/parent/results and all original generated-command/native/
terminal/boundary checks. Independent original642 noisyPro1k receives separate
CPU/native validation and cannot block the otherthree. DelayedBoltzchem30k and
randomProchem50k remain deferred with no production/unknown marker allowed;
all existing bytes stay charged. Failed prior observers/gates are preserved.
New10file metadata manifest e63f830f4f01f0740151fdd0b2baeb125d0faf9a927b5cd2e57b7400fea9d4c4,
archive21600bytes SHA0782f673d4abd5536cff7cc6302b87eb86bbab9cf3e4846638e169f7dba7214b.
Local stdlib AST/bytes/archive checks only;42allocated metadata cases, fresh full
selected-capacity gate and actual submission remain pending. Full projections,
oldremaining/report/diagnostic/metadata/noise/learnedprior allowances, stale-root
charges and4GiB/100000file reserves retained. No extra cleanup/parallelchemistry
implementation/numerical execution, scientific source/result change or monitor
unpause. Four dated run READMEs reference shared scripts/hpc/selected_launch and
results/hpc/20261024-nonchem-selected-01.



### 2026-10-25 — stride20 user policy prepared; original42 pass and dispatch-role failure preserved

Previous selected launcherCPU29809133/callback29809134 actualDONE/hpc/LSF and42
raw cases passed. New independent byte/XML/source/resource/reserve audit
7d8185c98e332d9ac30a19ef5b68c105a38fc627dd8aa1262c4c5369057195fc verified original
proofs and full preflight34413013534required/41990259569availablebytes. Allthree
dispatches29809135/36/37 actualEXIT1 beforeGPU/driver/study on a metadata role
status selected_owner passed to a status=passed verifier. Preserve failed
source/owners and all originals; this is tooling, not native gradient failure.
NoisyCPU29809138/callback29809139 completed before the user's output change;
GPU29809146 was verifiedPEND/unstarted, then suspended and intentionallyretired
(actualEXIT/TERM_OWNER) with no driver/model/study/scientificupdates. A first
caller used unavailablebhold (no mutation), correctedbstop succeeded, and bkill
exit appeared after the short initial confirmation window. Read-only recovery
authenticated actualEXIT/noStarted/nooutputs before finishing immutable
retirementSHAc75c65397c2b36378c8127be7da3d2dde803fa8d38ab51a594e2b1d3839896ab.
No second kill/replay occurred.

User requests every loss value and full matrices every20updates. New persistence
adapter keeps all scalar state/update/attempt/component rows, periodic full
snapshots/trials/references, initial/end/segment boundaries and boundedrolling
latest/best checkpoints. Full initial/terminal derivative and failedcurrenttrial
files remain. No interpolated/invented frames. All mathematical source/config/
loss/rate/optimizer/native checks unchanged; artifact-only callbacks verify the
new exact retained-frame index plus original complete scalar/JSONL/terminal gates.
New66 allocated cases include dense/sparse exact arrays/scalars/key/evaluation
counts under allthree original sources, actual persistenceCLI startup and fixed
owner-roundtrip regression. Local AST/byte/JSON packaging only, no numerical work.
New12file manifestaf73bea37be4620d1515052a805b43db0a892aa74cd2e783708cc2d66ce07502, archive31395bytes SHA8c414e059b98ebd57520314d9e807c256cecbf6365844db83451445d4309df8b.
Local draftsource01 retained, correctedREADME source02 not yetstaged/submitted.
Whole new empirical sparseprojections retain2xsafety/1GiBpersegment/fullremaining
scopes/stalecharge/4GiB/100000file reserves; freshcapacity and allocatedpass
remainpending. Four datedrun records link sharedtooling. No savedresultchange,
scientificrestart, additionalcleanup, independentprojectchange or monitorunpause.



### 2026-10-25 — stride20 allocated fixture failure preserved; immutable67-case correction prepared

New sourceaf73...CPU29809167 actualEXIT1:66rawcases,63passes/3failures, noerrors/
skips. Allthree real original-source toy trajectories passed exact loss/scalar/
gradient/retainedarray/key/evaluation-count assertions before failing JSONL vs
Python readout tuples on syntheticline42. Actual saved JSON uses lists. Callback
29809168 correctlyEXIT/blocked; no new owner/dispatch/GPU/scientificupdates.
Full sparsepreflight actuallypassed31204244536required/41989918047availablebytes,
unchanged4GiB/100000file reserve, notreservation. Failure stays failed.

New67case metadata source12files manifestf3be7c2f85d495ae042ee8092eecb5274c6298cc8f12907a7ba9ba6d8fb740db,
archive32375bytes SHA75a227c738f1ce0540263dd21a174d464509cba136ef83558bb46d59407268e9. Compare against
actual reportJSON plus explicit memoryJSON normalization, retaining all exact
checks. Remove unsaved readout pair/element arrays; retaineveryloss/periodicfull
matrices/latest/best/rawderivatives. Adopt allthree initialowners before any
dispatch to avoid metadataabsence/scheduling races, with source-isolated actual
readiness childchecks; an additional fixture covers ordering. No loss/rate/model/
optimizer/FD/parity/frozen/runtime/source/config/savedresult change. Original
failedsource01/oldCPU42/oldrolefailures/noisyretirement remain immutable. Local
AST/byte/JSON preparation only; allocated67pass and submission pending. Three
datedrun02 records plus noisy05phase update written before execution.



### 2026-10-25 — stride20 numerical persistence67 passed; scoped startup adapter fix prepared

CPU29809209 actualDONE/LSF/67rawcases: authoritative module counts44routing+23
persistence, zero failures/errors/skips. Independent verification
13559d86a21703c8c9a40a5139afa7978df689d2d72fa34fcbed85d939b616e0
validated all rawproofs/resources/source/capacity67. CPUcertificate
ddb6d4fa028bee151d644a6fbb09ae4c2b7e20a6e36e177c99110e82a4fecac6.
Wholecapacity32277781752required/41989578302availablebytes, reserves unchanged.
Callback29809210 actualDONE; dispatch29809221/22/23 actualEXIT during startup
capture because module-load subprocess proxy also wrapped the probe, escaping
its interception and attempting the original driver on allocatedCPU/backend
admission. No scientificGPU/core updates; old probe/startup failures preserved.
Independent noisyGPU29809228 actuallysubmitted under validated67source; keep it
unchanged.

New immutable12files source782087e70537f11a3a868b61e73a7354e9c38e592f7dc8a65f8e13ea3ff132c3, archive33546
SHA0141251bfe024c531f36aaf703639adb9b8f56c8abca228060f771b0a2cb893e. Scope only proxyactivation after unchangedstartup
and restoration afterward; new proof names preserve failed markers. Three
additional routing fixtures/full70cases planned allocatedrecheck. Original
math/config/gradients/FD/parity/frozen/rates/optimizer/boundaries/results retained.
Control nests in existing1GiB treecap countedonce, wholedata/stale/reserves retained.
No noisyresubmission, extra cleanup or monitorunpause. Three run records written
beforebootstrap. Local AST/bytes only; newCPU/GPU outcome not yet established.


### 2026-10-25 — startup03 read-only pre-stage filter failure and source02 correction

Initial startup03 caller stopped before any new remote source/control/upload
or job at a broad *-segment-* assertion: it matched failed startup-command JSON
files. Preserve caller failure/localdraft source01. Newsource02 explicitly
classifies exact production phase markers versus known regular startup JSON,
rejecting unknown/nonregular entries;8fixture cases added (78total). No stored
artifact removal or priorfailure relabel. Source12files704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab,
archive34175SHA763dd122080bfc1896b391d26dbf7bba0b382af0f0e34feddef48b848ed0048d. Full numerical persistence
checks unchanged; localAST/bytes only. Fresh bootstrap02 has unique markers and
will recheck every exact target/proof/fullcapacity. Existing noisyGPU29809228
remains unchanged. No other job/result/setting/monitor changes.


### 2026-10-25 — stride20 startup03 full78 and actual startup passed; three GPU chains submitted

Source12files704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab authenticatedreadonly, archive34175bytes
SHA763dd122080bfc1896b391d26dbf7bba0b382af0f0e34feddef48b848ed0048d.
ActualCPU29809262/callback29809263 DONE/hpc/LSFsuccess.78rawcases55routing+23
persistence allpassed, zeroerrors/fails/skips. CPUcertb5fd3fb230bc6479b8f6f081381f6663c3b823c8643c41e93dbaf21942837d16;
independent deliveryaudit0b9a31565c2195c99c3a1658865d3d4b6053e1ed68cb72561fb307ccef84ee5d. Original scientificCPU827/1163/866 and
selectednative52/53/Procontinuationproofs/config/source/parent/model/context
pins retained. Pro commandcapture and Bol/DBactualpremodel CPUstartup proofs
passed; ProrealGPUstartup/native/terminal productionchecks arestillfuture.
Wholefullstorage passed32277683804required/41989268945conservativeavailablebytes,
4GiB/100000fileheadroom retained; work3FSallzero onlysupplementalunreported.
Metadata is nestedinside existing1GiB controlcap countonce, with old/source
bytes/entirebase stalecharged and all future scopes funded.

Actualdispatch29809268/69/70 DONE/LSF and submitted three independent scientific
GPUchains: plainBoltz30k first29809297/callback29809298; DBMolbinding10k first
29809294/callback29809295; Protenixnative+40k first29809291/callback29809292.
At22:36:22UTC allsixPEND. FourCPU/8GiBslot/oneexclusiveL40S/23h, ownendedcallbacks
only. No corecommittedupdates or newterminalpass inferred fromsubmission. Noisy
originalGPU29809228 remainsRUN and unchanged; no duplicate/resume/restart.
Original savedresults and delayed-chemistry deferral preserved; monitornotunpaused.

Newpersistence retains every scalar loss/component/attempt/update and full
matrices every20/initial/end/boundary plus boundedlatest/best. Matched allocated
source trajectories verify exact gradients/retainedarrays/keys/scalars/evaluation
counts. Matrixhistory about95percentlower; three-chain projection16.32094GB to
12.03863GB (4.28231GB reduction), with fixedscopes/reserves notdividedby20.
RawCPU/XML/LSF/resource/startup/source proofs and canonicalcampaign/READMEupdates
retained in the three startup03runrecords; originalfailedattempts remainfailed.
No modelmath, learningrate/loss/FD/gatewaiver, historicalthinning, extra cleanup,
independentprojectchange, localnumerics or reportframe invention occurred.


### 2026-10-25 — separate chemistry search speed study prepared

The user authorized testing the most promising chemistry-preparation speedups,
with a substantial reduction in search time as the goal. New experiment
experiments/brd4-jq1/chemistry-search-speed isolates exact hard-graph native-fact
caching, invariant cost/validation reuse and ordered per-depth spawn-pool checks
(2/4workers). Existing sealed scientific sources/results/jobs stay unchanged;
no optimizer updates, approximated stale targets or reduced search budgets.
79allocated-only fixtures and six authenticated saved N25 pilot inputs are
planned before correctness/performance findings. Original full643source seal,
input hashes and whole selected-campaign capacity plus separate512MiB allowance
are rechecked before submission and entry, retaining4GiB/100000file reserves.
Local work so far is AST/bytes/source inspection only; no speed or numerical
pass/job exists yet. Existing monitor remains paused and no cleanup occurs.


### 2026-10-25 — chemistry speed01 CPU experiment actually submitted

Immutable9file sourcea30fd3937ea79935d0c449c600d2ea42f5f26da31906508a31e306f81d1c13b0 staged/authenticatedread-only,
archive17604SHA5ec552bcdd5be9a98cba26b00a9db9720098214af66786eee09b7e9da9ce6b31. Six saved N25 pilot snapshots and
original643source rehashed, original3x32+6x32 search unchanged. Full unchanged
selected-campaign storage plus512MiB passed32499679864required/40594731268
available with4GiB/100000file reserves. Actual CPU29809404 submitted00:49:47
Paris,8slots/4GiBslot/2h, observedPEND00:49:55.79rawfixturepass required before
real saved-input benchmark; no result/speed/nativepass yet. No current run/job/
scientificsettings/savedresults modified, monitorremains paused. Staticlayout
command retained preexisting testdiscovery/historicalREADME errors.


### 2026-10-25 — chemistry speed01 completed: exact reuse cuts repeated search sharply

Actual CPU29809404 DONE/hpc/same-job LSFsuccess01:00:56Paris,668sallocatedwall.
All79unique rawfixtures passed/no failures/errors/skips;108saved preparations
(6inputs×6modes×3repeats) retained exact receipts/targetarrays/identities and all
36loss/component/gradient hashes. Immutable9files/sourcea30fd393... and original
643source da9db682... plus sixinputNPZs rehashed before/after.35deliveryfiles,
archive17604SHA5ec552bc... and rawXML/LSF/source/wholecapacity/inputbyte proofs
independently verified. Final audit/hash and rawproofs in chemistry-search-speed
run20261025-chemistry-speed-01. No local numerical tests or currentrun changes.

Difficult randomN25 warmbaseline16.87–18.17s; exactcache0.453–0.461s, combined
cache+cost0.322–0.331s(52–55×warm). Cost-only18.16–18.49s gave no materialgain;
2workers8.74–8.80s,4workers4.97–5.12s(3.4–3.6×warm). Combinedcoldfirst12.73s;
first encounters of actualadjacent savedstates0.322/1.683s with persistentcache.
Caching does not reuse probability-dependent targets, reduce96+192logicalchecks,
change loss/rate/gradient/scientificgates or certify a future50ktrajectory.
First-time invalidBoltz was0.302s vsbaseline0.271s, but warm0.089s vs0.270s;
validBoltz was already~0.013s. Profiling identifies hardgraphdiagnosis asdominant.
No automatic productionadoption, fullnativepilot, savedresult change, extra
cleanup, independentproject/job modification or monitorunpause. Optionallazy
neighbor41-fixture localdraft remains unexecuted/excludedfrom79pass claim.


### 2026-10-25 — further chemistry hybrid/GPU matrix experiment prepared

New explicit userrequest authorizes further optimization/parallelism andmatrix
GPUconversion. New chemistry-search-speed02 isolatesparent-ownedboundedexact
nativefactcache+4/8spawnworkers, plusrealGPUbatchedintegergraphfeatures and
scan/log-semiringmatrix locallosskernels. NativeRDKit/roundtriporacle remains
CPU/mandatory; no approximatevalidity/pruning/staletarget or currentrun change.
Full143allocatedCPU cases andmatched6savedinputtarget/loss/gradientparity are
required beforeownendedcallback mayqueue19GPU-onlycases pluskernelbenchmarks.
GPUFP32reassociation retainsoriginal2e-5value/gradienttolerance andrawFD
.01/.003/.001/15percent/twoadjacent. CPU16slots/2GiBslot/2h, GPUoneexclusive
L40S/fourCPU/8GiBslot/2h, separatejoint1GiBpersistentbudget plusunchangedwhole
storage4GiB/100000filereserves. Allsources/old79proofs/inputs reauthenticated.
LocalAST/bytepreparationonlysofar; no newCPU/GPU pass/job/speed result.
Existing optimizationjobs/savedresults/source01failurehistory andpausedmonitor
unchanged. Protocol/runrecord writtenbeforeexecution.


### 2026-10-25 — chemistry hybrid/GPU02 CPUactually submitted

Immutable19file source09c0ab837bede0c5779eac0da06885cfbef6a98744f67af19e7b5a8b33ffa222 staged/authenticatedread-only; archive35875
SHAe079f5a30c4e062b2d8dee04311824425a593d16278c3c378d592dd46d127432. Original643science+source01ninefiles/raw79XML/LSFassessment
and6NPZ/frozenreferencepaths rehashed. Whole originalselectedcapacity plus1GiB
passed32836817040required/39325369650available retaining4GiB/100000files.
ActualCPU29809477,ownendedcallback29809478 submitted01:15:35Paris; CPUactualRUN
01:15:54,callbackPEND. No new143fixturepass, GPUsubmission/result orspeedclaim
yet. No existingrun/source/result/job settingsmodified, monitorremainspaused.


### 2026-10-25 — chemistry GPU admission scheduler-lag correction prepared

Source02 CPU29809477 actualDONE/LSF,143 fixtures and all24 saved-input rows
passed; independent byte/XML/schema audit observation11 passed. Original
callback29809478 EXIT1 because own-ended dependency fired while its raw bjobs
query still returned CPU RUN. Later actualDONE proof preserved separately;
no GPU submission occurred. New immutable metadata-only adoption03 is prepared
with bounded actual-terminal waiting and allocated regression tests, own CPU
DONE/LSF/raw XML authentication, original source/input/plan/143-pass adoption
and unchanged full storage gate before submitting original source02 GPU argv.
No failed record rewritten, saved result/current optimization changed, numerical
gate waived or monitor resumed. No new adoption job/pass yet.


### 2026-10-25 — chemistry hybrid CPU passed; original GPU callback failed closed

CPU29809477 actualDONE/hpc/LSFsuccess,143 raw fixtures and24 exact saved-input
rows/72 preparations passed. Independent observation11 byte/XML/schema audit
passed; source19/archive/initialplan/sixNPZ+frozenreferences and original643
source reauthenticated. Warm difficult early preparation16s to.278–.288s;
first cold16.833 to8.674/7.633s with4/8workers. Alltargets/receipts/losses/
components/gradienthashes exact. NativeDetectChemistryProblems dominates cold
profile14.848s; SanitizeMol.054s. These are preparation, not fullmodel timings.
Originalcallback29809478 EXIT1 on cachedactualCPU RUN afterended; lateractual
DONE/LSFproof separate. No GPU submitted. Newadmission03 source staged but
first preflight failed closed on a concurrently replaced currenttrial.partial,
before anynewjob. Preserve failure; a distinct fullfreshcapacity recheck is
being made after verifying no CPU/GPU/unknown markers exist. No gate relaxed,
existing run/source/result modified or monitor resumed.


### 2026-10-25 — chemistry GPU benchmark actually queued after validated metadata correction

New five-file immutable admission03 manifest00e223b9b453aa0e390f315951eee3406916ac8c8499284945967a72fbf38698,
archive9050bytes SHAae2031e920ec7d40058a04215fd4c3bb767e16832672538f12e11b9a5564348b.
First preflight failed closed on a concurrent active trial.partial replacement,
before any job. Exact failure preserved; a separate complete fresh unchanged
gate passed after checking all unknown markers absent, without restaging or
changing the sealed source/formula. Actual CPU29809552/callback29809553 DONE
hpc/same-job LSF,19 raw metadata fixtures passed; original143 numerical passes
and exact plan/source/input/LSF/benchmark proof pins adopted independently.
Actual GPU29809554 submitted under unchanged source02/19GPU-only checks. At
01:29:38Paris PEND on physical GPU queue limit; no execution/speed/pass yet.
Dispatch required32815757995 vs conservative39303877607 bytes with unchanged
full projections+1GiB benchmark+4GiB/100000file reserves. Independent byte/
XML/LSF/schema/dispatch audit passed. CPU search targets/losses/gradients remain
exact; GPU matrix/scans still await actual hardware evaluation. Existing
scientific jobs/results/settings and paused monitor unchanged.


### 2026-10-25 — chemistry GPU checks and matrix benchmark completed

Read-only collection at08:49:45UTC/09:49:45Paris authenticated GPU29809554
original same-job LSF success on L40S,05:57:33–06:03:48CET (375seconds).
Live records aged out, retained as not_found; no synthetic schedulerDONE.
All19 raw GPU-only cases passed/no errors/failures/skips; six saved-input
feature and12 loss-gradient/FD comparisons passed original2e-5 and raw
.01/.003/.001/15percent/twoadjacent gates with explicit original-zero N/A.
Independent38-file source/archive/byte/XML/schema/LSF/input/entry audit passed;
original643 source files unchanged. GPU assessment45d384bb0db8d8b407d2e79029027588ed7908f2db84d13aa067de6d64114023,
rawXML9bfe530b6463b7cb0a29b0d1899d7b0dce807bf911c2fcad819e674a9ad1d56b,
benchmark79e5daba5d1201dc3a9c112e843ba851f875a59b6323bbf65004213db535d0c9.
Matrix warm local-risk value+gradient1.602 to.260ms(batch1,6.16x),
1.435 to.380ms(batch32,3.77x),1.256 to1.554ms(batch256,24percent slower).
Scan warm slower throughout. Matrix compile+firstcall26.526 to1.985s(batch1).
Matrix FP32 association is tolerance-equivalent, not bitwise identity. These
are GPU-versus-GPU component timings;1.34ms saved is small beside~280ms cached
CPU preparation and native model work. RDKit/search remains CPU. No production
adoption, new jobs, settings/saved-results changes or monitor resumption.


## 2026-10-25: separate matched full-model chemistry acceleration pilots

The user authorized combining native-fact caching, refreshed probability-cost reuse,
ordered persistent four-worker search and GPU matrix local-risk arithmetic, with
full-model Protenix/Boltz native gradients and update timing checks. Preparation
retains exact original random50k Protenix and chemistry30k Boltz configurations,
original validation drivers and every scientific/runtime/quota gate. Baseline
runtime failure is preserved if encountered; candidate science/runtime must pass.
No existing saved result/job, monitor or DBMol VAE setting is changed. Allocated
CPU contracts and two independent full-model GPU pilots are being prepared; no
new full-model pass, job or integrated speedup is claimed at this point. Protocol:
experiments/brd4-jq1/chemistry-search-speed/runs/20261025-chemistry-full-model-01.

Actual25October10:18Paris/CET: new26-file immutable capsule
347e753d481ec3d584f3b0b05cc2801ee745036ee17638d7c3672fc2ca2cba16,
archive71952bytes/4f5ec7f81f39a67667fa254321e47795d215c26ee537cac5ed280bade320366a.
Whole original storage plus2GiB passed33643657204/40125878917bytes, retaining
4GiB/100000-file reserves. CPU29811875 actuallyRUN/afterCPU29811876PEND;
284planned raw cases, no CPU/full-model pass or new GPU job yet. Existing runs,
DBMol VAE and paused monitor unchanged. Unsubmitted local draft archive preserved.

At25October10:23Paris/CET CPU29811875/callback29811876 actualDONE/hpc/LSFsuccess,
284raw cases passed79+26+20+18+67+23+51 with nofail/errors/skips/deselections.
CPUcertificate947dd2b539b9cbd951dd50f164107188c7973b0e850131e0e914b63fd3b53b9b;
rawXMLcada1b91fc5039352e72e13df582fd2eb92cd2034883ea89a53587ba5db42d7a.
Independent33file byte/XML/schema/source/archive/actualargv/certificate audit
passed. Fresh unchanged fullstorage+2GiB passed33643657204required/
40125741934availablebytes with4GiB/100000file reserves. Actual independent
full-model GPU pilots Pro29811883 and Bol29811884 submitted10:22:13,
BOTH PENDphysicalGPU at10:23:09; no native/fullgradient result or integrated
speedup yet. Original sources/results/settings/DBMol VAE unchanged; monitor
stays paused. Exact proof records in full-model-01/campaign.json and verification.json.


## 2026-10-25: read-only terminal status of non-chemistry chains

Fresh compactSSH capture09:29:28UTC/10:29:28Paris/CET,171bytefiles independently
rehashpassed, original same-job LSF/queue/exit/errors and newline prefixes:
Pro29809291 applied+10000 toabsolute20000 then terminalpredeclaredepsilon
leaves simplexinterior; Bol29809297 applied10000/30000 then affinityE_upperFD
failed; DB29809294 passedfirst2500/native29809968passed, next29810971 reached
global5000/10000 then terminaldbmol_contact_lossE_upperFDfailed. Their exact
own-ended callbacks blocked successors. Live scheduler entries are agedout;
FAILED_PERSISTED uses actual original LSF, no fabricated currentEXIT/DONE.
Independent noisyPro29809228 completed1000/1000, summarypassed, initial+terminal
raw/logitgradientchecks and runtimepassed. It remains a single-seed contact
feasibility result, notbinding/activity evidence. All originals/savedresults,
newchemistryfull-model pilots and pausedmonitor unchanged. No resubmission.
Existing genericdeliveryverifier failed historical/current DBMolprojection
comparison; failurepreserved, separate statusonlyaudit checks each recorded
gate ownunit without newadmission/sciencewaiver. AuditSHA 57c2b30f59c2a046294c134c784bbf5b8ed0c92ab66016445072baf637f88ff8
under results/hpc/20261025-nonchem-stride20-startup-03/evidence-06; per-arm
status receipts and campaign.json updated in existing startup03 run records.


## 2026-10-25: updating non-chemistry report and endpoint-check investigation

User explicitly requested saved-only easy-updating reports, explanation and
investigation of recurrent terminal validation, plus checking intended-budget
continuation queues. New report owner experiments/reporting/jq1-live-progress
separates complete saved prefixes from failed scientific gates. No original
run/source/result or gate is rewritten, no failed chain resumed or duplicated.
Pro continuation20ktotal, plainBol10k, DBMol5k remain stopped; noisyPro1kpassed.
New report CPU fixtures/rendering/source/byte/publication/browser QA are being
prepared. Endpoint geometry/numerical investigation is separate; no scientific
pass or new continuation submission follows from preparation.


Report preparation/actual outcomes25October: source01 CPU29812027 passed121
raw fixtures but failed exact JUnit module attribution (tests. prefix missing);
callback29812029 blocked rendering. New immutable source02 corrected metadata,
CPU29812093/callback29812094 and render29812097 actualDONE/LSF success121cases,
but first published immutable edition0001 failed browser layout QA (inline
progress span). All failures, releases and edition0001 preserved.

New immutable source03:17files, manifest ee69381d752bf32f3ae55cf76f69ae1330acf5b20c52ddc7c9ee3cd7f711218c,
archive51,389bytes/51459284779b90baf8f22d3effb7d33708d7c96d8ae5e424c73ccd13c3eeebd9.
CPU29812102/callback29812103 and renderer29812120 actualDONE/hpc/LSFsuccess.
All123 raw cases passed58data+33viewer+13ops+19prospective diagnostic toys,
zero failures/errors/skips. Full existing projections plus separate report/
speed/diagnostic allowances retain4GiB/100,000-file reserves; gates preserved.
The19 toys distinguish infeasible-boundary, resolved mismatch and unresolved
precision; no native endpoint qualification or production gate replacement.

Delivered http://127.0.0.1:8780/report.html edition0002, original progress
Proabsolute20k/50k (+10k)/Bol10k30k/DB5k10k/noisyPro1k1k, failed/failed/failed/passed.
Fresh archive27,439,579bytes SHA1fc9887364dec9d07a61130b76b54c9241a0a97d80637e1ee0dc6a5e38f08c2e,
all26members and56servedHTTP files byteverified, actual browser drag/numeric/
arrows/play/manualpause/selectors/latest/heatmaps/rawrecord and pointer-update
QA passed. Stable shell adds responsive CSS only; immutable edition bytes and
20 scalar/frame assets remain exact. All1,304 retained frames and every scalar
validated on allocatedCPU. Original nativeNPZ/bigsummary inputs remainHPC refs,
not claimed downloaded. Hidden localhostPID34336; all older viewers preserved.

Future fresh-bound refresh helper is static preparation only, unexecuted; its
explicit generated launcher can submit CPU report phases only and preserves
unknown markers. Viewer polls latest.json every60s, notHPC. Monitor remains
paused; no optimization successor or original saved result/source/job changed.
PlainBol needs20k further updates, DB5k, Pro30k after validated new continuations;
delayed-chem production arms remain deferred behind separate full-model pilots.
Repository-wide layout check still fails pre-existing defaultdiscovery and old
missing-README paths; no new-owner README/AST error reported. Unrelated PDF/
OneNote and other dirty project work were excluded from this change.


## 25 October 2026 — Refine the saved JQ1 report presentation

User-authorized redesign: fewer explicit objective loss curves, larger bond-aware
raw molecule view, compact navigation and folded matrices/provenance. Preparing
new immutable report generation04/edition0003 against exactly the prior saved
inputs. Original scientific settings/results/failures and historical editions
remain unchanged. Monitor stays paused; no optimizer/model/cleanup jobs. New
display coordinates and fixtures/rendering will execute only on allocatedHPC.
Local work so far is static/byte preparation, not an allocated pass.

Generation04 staged17files read-only; manifest856f788c5959c6fe5239374d3141808ec011907f4e092deb78a65f28ef9b196d,
archive58,014bytes SHA518fef6d982e3ed166e1826ad97c96b385967f5e65dc0423e8190e771a240fde.
ActualCPU29813776 and own-endedcallback29813777 submitted25October10:28:10UTC;
first observation CPU RUN/callback PEND.141 declared cases require actual raw
JUnit/module/LSF/scheduler authentication. Full original storage guard passed
preflight; no model/optimizer work. Layout verifier still reports pre-existing
default-discovery and historical missing-README errors, no new-owner errors.

Allocated CPU29813776 actualDONE/hpc/LSFsuccess; all141 rawcases passed
(71data+38viewer+13ops+19diagnostic-policy toys), no failures/errors/skips.
Independent localstdlib XML/LSF/count audit passed. No native endpoint pass
is implied by the19toy cases. Callback29813777 running, render notyetverified.

Refined renderer29813792 and callback29813777 actualDONE/hpc/LSFsuccess.
CPU141 rawcases independently passed. Full unchanged storage gate passed with
required35,522,705,396/available37,946,191,594bytes at render entry, unchanged
4GiB/100,000-file reserves. New archive28,362,248bytes SHA
a6ebfc23f9febe46283d19e6832586ba025f478a40bbd161ae72b7938569ce5e; all26files
safely extracted/hashed. Four scalarassets byte-identical to edition0002;
all1,304frameJSONs exactly type-sensitive equal after removing only display
coordinates/layout. Original failed/failed/failed/passed scientific statuses
and all probability/scalar/readout bytes preserved. No native qualification.

Published edition0003 at http://127.0.0.1:8780/report.html; all83servedfiles
HTTP-byte verified. BrowserQA passed all4cards, main loss semantics, DBMol
weighted3component curves, no metricmenu, bond-aware rawgraph, actualdrag17360,
numeric/missing15001/saved15000, arrows,play/manualpause320,DBboundary2500,
latest,matrices and folded details. Stable shell conditional legacyCSS keeps
all immutable old editions exact. Future refresh helper pinned to refined04/141
fixtures is static-only, uninvoked; monitor stays paused and no optimizer jobs
or saved results changed. Exact proof/QA/publication hashes are in the new dated
run campaign; full original native NPZs/large summaries remain HPC references.
