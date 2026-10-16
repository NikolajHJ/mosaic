# Stage 0 usage and future HPC validation

Documentation update: 2026-10-16 (implementation began 2026-10-12). The 2026-10-12–13 implementation turn occurred during the declared HPC outage and performed no numerical work, connection, transfer or submission. Access was subsequently restored and the focused gradient study below completed. Full two-mode Stage 0 A-G/M2/M3 acceptance remains pending; the earlier implementation receipt is preserved in the ledger.

See [the plan](stage_0_implementation_plan.md), [native contract](stage_0_native_contract.md) and [ledger](experiment_ledger.md). The 2026-10-10 protein smoke is preserved but does not validate this graph branch.

Completed focused study, 2026-10-16: the user confirmed restored DTU HPC access and authorized allocated-HPC Protenix relaxed-molecule gradient tests. Source-04 job **`29735139` DONE**, with **126 unit checks, seven GPU feature checks and one native pytest item covering two relaxed graphs passed**. The [gradient-test protocol](protenix_relaxed_gradient_tests.md) describes the DBMol-inspired pocket-contact reduction and exact rerun command. Real frozen-Tiny gradients reached both atom and edge-existence probabilities; both graph cases passed all three finite-difference scales and three AD-evaluated projected updates, with an additional standalone-primal initial-to-final decline and unchanged model/context digests. Historical failures and the pivotal precision/control decisions remain in the ledger/plan. This focused result does not retroactively pass the full Stage 0 A-G suite.

| Relaxed graph mixing | Standalone-primal initial loss | Standalone-primal final loss |
| --- | --- | --- |
| `0.15` | `1.7413299084` | `1.7370867729` |
| `0.30` | `1.6747362614` | `1.6332819462` |

To rerun the focused study, use the authenticated source-04 CLI/configuration on an allocated DTU GPU, retain `matmul_precision="highest"`, and choose a fresh output directory. The protocol gives the matched `--prepared-study`/`--prepared-summary-sha256` arguments. Local downloaded results are under `results/hpc/20261016-relaxed-gradients/20261016-gradient-evidence-04/`. These are engineering tests on an arbitrary 20-residue protein context; no binding or chemical-validity claim follows.

## Implemented entry point and defaults

[The example](../examples/small_molecule_stage0.py) reads [the configuration](../examples/configs/small_molecule_stage0.json) and [shared fixture](../tests/data/small_molecule_stage0/fixture.json). Default modes are `ligand_only` and `protein_ligand`, using the same 10-heavy-atom acetanilide fixture and an interior mixture of 0.15. The protein is the 20-residue engineering sequence `ACDEFGHIKLMNPQRSTVWY`, with one query MSA row and no template; it is not BRD4 or a scientific binding target.

Resolved defaults: seed 17, float32 graph leaves, geometry off, one final recycle, five projected updates, `eta_X=eta_E=0.01`, fixed typed RNG, one hard readout per evaluated step. The loss is diagnostic-only in ligand mode and conservative whole-bin native contact reduction in protein-plus-ligand mode. Diffuse carbon-heavy initialization and split-RNG options are implemented but are not a performed Stage 1 study. Every resolved setting is recorded; changing source/config/fixture/model/environment/context bindings invalidates resume.

The resident model loader verifies pinned source and all five native assets offline. It does not install packages or download a model. `--cache-dir` points to the actual `.../mosaic/protenix` asset directory, not its parent. Prepared context serialization preserves native sampled conformer bytes/order/masks; resume loads them instead of preparing the molecule again.

## Environment preparation and later full-suite validation

The dedicated environment is `/work3/s234818/mosaic/envs/protenix-tiny`, Python 3.12.11, pinned native revision `97cf8172f61829681582f06de7052485bc94eae4`, shared CUDA 12.9.1/cuDNN 9.13 modules. It began as a reduced 57-package protein-inference environment. Allocated CPU setup job `29735006` completed the hash-locked extension with `pytest==9.0.3`, `iniconfig==2.3.1`, `pluggy==1.7.0`, `pygments==2.21.0`, retaining the original packages and installation/export receipts. No full unrelated-model `uv sync` or per-GPU-job installation is needed. Exact setup and source identities are in the ledger.

The historical Protenix `==3.12` metadata correction is documented in [the native protein receipt](hpc_protenix_tiny.md). Reuse/authenticate that dedicated environment rather than silently applying another patch. Authenticate the current runtime/native file hashes and asset receipt before validation; the package version alone is insufficient.

The old 2026-10-10 quota value `98.43/100 GiB` is historical, not current headroom. The future quota preflight requires the **actual storage pool** of the run/cache and at least 4 GiB/10,000 files free, considering both user quota and filesystem space. A missing/unverified/insufficient quota exits 75 and records `tests_not_run`; it is not a failed/passed model test and does not authorize cleanup. Shared filesystem free space does not override user quota.

The completed focused study used the conservative all-finite-work-pool bound documented in its protocol after a denied fresh path-mapping query; no authentication bypass occurred. Final inspected quota was `89.01/100 GiB` and `250,966/20,000,000` files, with no storage issue. These values are a dated receipt, not a reservation or substitute for a new run's preflight.

## Future allocated-HPC commands

After an immutable release and verified environment are ready, run from that release on the allocated node. `PYTHONPATH` must include `src:tests`; the loader uses existing pinned assets and no network fallback. Replace the parent run path below with a fresh unique destination. Atomically create its `payload` directory; an existing payload must refuse a new run so pytest temporary-directory cleanup cannot erase earlier evidence. The parent may already contain scheduler stdout/stderr. Use either these manual commands or the launcher to claim that payload.

```bash
export PYTHONPATH="$PWD/src:$PWD/tests"
export HF_HUB_OFFLINE=1
export MOSAIC_STAGE0_CACHE=/work3/s234818/mosaic/cache/mosaic/protenix
run=/work3/s234818/mosaic/runs/UNIQUE_STAGE0_RUN
mkdir -p "$run"
mkdir "$run/payload" || exit 1
payload="$run/payload"
export MOSAIC_STAGE0_RESULTS="$payload/native-tests"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
manifest_sha=REPLACE_WITH_RECORDED_MANIFEST_SHA256
storage_pool=REPLACE_WITH_VERIFIED_POOL_NUMBER
"$python" scripts/verify_stage0_release.py "$PWD" "$manifest_sha" || exit 1
"$python" scripts/hpc/stage0_quota_preflight.py --run-dir "$payload" --storage-pool "$storage_pool" || exit 75
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export TMPDIR="$payload/tmp" PYTHONPYCACHEPREFIX="$payload/pycache"
mkdir "$TMPDIR"

# Selected unit/boundary tests, still performed on allocated DTU HPC.
JAX_PLATFORMS=cpu "$python" -m pytest tests/test_small_molecule_*.py -o addopts= --strict-markers -m 'not slow and not small_molecule_model' -o cache_dir="$payload/pytest-cache" --basetemp "$payload/pytest-unit-tmp" --junitxml="$payload/unit.xml" > "$payload/unit.log" 2>&1

# Real native A-G checks in both modes, including fresh-process replay.
"$python" -m pytest tests/test_small_molecule_*.py -o addopts= --strict-markers -m small_molecule_model -o cache_dir="$payload/pytest-cache" --basetemp "$payload/pytest-native-tmp" --junitxml="$payload/native.xml" > "$payload/native.log" 2>&1
```

The real suite fails missing allocation/GPU/assets/source prerequisites instead of skipping them into apparent acceptance. Its child resume process shares the same allocation; it does not submit another job. Both context modes, exact source/environment/device/driver/RNG identities, persisted outputs and failures are part of the acceptance evidence. Do not claim M3 from mock-only or independently passing track tests.

Ordinary unit fixtures run on CPU within the allocated DTU host so synthetic parent/child restart processes do not compete for an exclusive GPU. The focused launcher separately reruns all seven feature checks on the A10 before real-native work; those actual GPU precision checks are part of its completed receipt. The full A-G commands here remain future validation, with their numerical profile and source release to be bound explicitly.

The allocation-guarded [launcher](../scripts/hpc/small_molecule_stage0.lsf) takes four arguments: immutable release, unique parent run directory, verified actual storage-pool number, and independently recorded SHA-256 of the release's `source_manifest.json`. It atomically claims `RUN/payload`, refuses an existing payload, and places numerical outputs, logs and pytest temporary/cache directories beneath it. It authenticates the source manifest before invoking the quota helper and rejects unmanifested execution files in `src`, `tests`, `examples` and `scripts`. Then it runs quota preflight, selected unit tests, real tests, and the example without duplicating `--validate`. It performs no installation or asset download. Its current planned resource directives are one exclusive GPU, four slots on one host, 8 GB memory per slot, and two hours; actual later submission/resources must be recorded, not inferred from this unsubmitted script.

If passing that script as a command through `bash -l`, supply resources on the `bsub` command line, as for the historical protein example. Create absolute stdout/stderr destinations before submission. Do not treat comments inside a command-passed script as parsed `#BSUB` input.

## Example, bounded continuation and validation option

These commands also belong on the allocated HPC node:

```bash
"$python" examples/small_molecule_stage0.py --cache-dir "$MOSAIC_STAGE0_CACHE" --output "$payload/example" --max-updates 2
"$python" examples/small_molecule_stage0.py --cache-dir "$MOSAIC_STAGE0_CACHE" --output "$payload/example" --resume

# Alternative standalone validation route, in a separate fresh output directory.
"$python" examples/small_molecule_stage0.py --cache-dir "$MOSAIC_STAGE0_CACHE" --output "$payload/standalone-validation" --validate
```

`--max-updates 2` pauses the authored five-update driver after two updates; `--resume` continues from the persisted commit pointer and next unused key with unchanged identities/settings. It is not a changed compute budget or a new initialization. `--validate` runs the full mode acceptance checks before that mode's trajectory, retains validation outputs/failures, and stops if the report does not pass. Use this as an alternative to the real-test validation route when appropriate; the launcher already selects real tests and does not run it twice.

Per-mode artifacts include sealed prepared context, graph/identity manifests, immutable step/evaluation NPZ checkpoints, checkpoint commit pointer, trace, raw readouts, final/best score-state identities, and retained failures. A failed conversion does not repair/substitute the optimized graph. Validation adds real outputs/derivatives, protein regression, source/environment/RNG receipts and uninterrupted-versus-fresh-process continuation evidence.

## Static and packaging receipts

Implementation is authored and independently reviewed; numerical correctness is unmeasured. The ledger records the completed combined AST, JSON/TOML, Ruff 0.15.12, whitespace, shell-syntax and native-file hash checks, including the final launcher/verifier recheck. The immutable bundle is sealed locally under ignored `results/releases/stage0-20261013/`; its external `release_receipt.json` records final archive and manifest hashes. No transfer or submission is part of packaging.
