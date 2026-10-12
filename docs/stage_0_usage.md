# Stage 0 usage and future HPC validation

Documentation update: 2026-10-13 (implementation began 2026-10-12). Code, configuration, fixtures, tests and launcher are authored. **None of the numerical commands below has run in this implementation turn.** The user declared an HPC outage; no connection, transfer, job submission, local numerical test or model evaluation is performed. Gates A-G and M1 test evidence/M2/M3 acceptance remain pending.

See [the plan](stage_0_implementation_plan.md), [native contract](stage_0_native_contract.md) and [ledger](experiment_ledger.md). The 2026-10-10 protein smoke is preserved but does not validate this graph branch.

## Implemented entry point and defaults

[The example](../examples/small_molecule_stage0.py) reads [the configuration](../examples/configs/small_molecule_stage0.json) and [shared fixture](../tests/data/small_molecule_stage0/fixture.json). Default modes are `ligand_only` and `protein_ligand`, using the same 10-heavy-atom acetanilide fixture and an interior mixture of 0.15. The protein is the 20-residue engineering sequence `ACDEFGHIKLMNPQRSTVWY`, with one query MSA row and no template; it is not BRD4 or a scientific binding target.

Resolved defaults: seed 17, float32 graph leaves, geometry off, one final recycle, five projected updates, `eta_X=eta_E=0.01`, fixed typed RNG, one hard readout per evaluated step. The loss is diagnostic-only in ligand mode and conservative whole-bin native contact reduction in protein-plus-ligand mode. Diffuse carbon-heavy initialization and split-RNG options are implemented but are not a performed Stage 1 study. Every resolved setting is recorded; changing source/config/fixture/model/environment/context bindings invalidates resume.

The resident model loader verifies pinned source and all five native assets offline. It does not install packages or download a model. `--cache-dir` points to the actual `.../mosaic/protenix` asset directory, not its parent. Prepared context serialization preserves native sampled conformer bytes/order/masks; resume loads them instead of preparing the molecule again.

## Future environment preparation

The historical dedicated environment is `/work3/s234818/mosaic/envs/protenix-tiny`, Python 3.12.11, pinned native revision `97cf8172f61829681582f06de7052485bc94eae4`, shared CUDA 12.9.1/cuDNN 9.13 modules. It was a reduced 57-package protein-inference environment and lacks pytest. When access returns, prepare the minimum extension on an allocated helper: add locked `pytest==9.0.3`, retain the new requirements lock/export and environment receipt, and verify compatibility. This setup is not performed here. Do not run a full unrelated-model `uv sync` or install dependencies in every GPU job.

The historical Protenix `==3.12` metadata correction is documented in [the native protein receipt](hpc_protenix_tiny.md). Reuse/authenticate that dedicated environment rather than silently applying another patch. Authenticate the current runtime/native file hashes and asset receipt before validation; the package version alone is insufficient.

The old 2026-10-10 quota value `98.43/100 GiB` is historical, not current headroom. The future quota preflight requires the **actual storage pool** of the run/cache and at least 4 GiB/10,000 files free, considering both user quota and filesystem space. A missing/unverified/insufficient quota exits 75 and records `tests_not_run`; it is not a failed/passed model test and does not authorize cleanup. Shared filesystem free space does not override user quota.

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
"$python" -m pytest tests/test_small_molecule_*.py -o addopts= --strict-markers -m 'not slow and not small_molecule_model' -o cache_dir="$payload/pytest-cache" --basetemp "$payload/pytest-unit-tmp" --junitxml="$payload/unit.xml" > "$payload/unit.log" 2>&1

# Real native A-G checks in both modes, including fresh-process replay.
"$python" -m pytest tests/test_small_molecule_*.py -o addopts= --strict-markers -m small_molecule_model -o cache_dir="$payload/pytest-cache" --basetemp "$payload/pytest-native-tmp" --junitxml="$payload/native.xml" > "$payload/native.log" 2>&1
```

The real suite fails missing allocation/GPU/assets/source prerequisites instead of skipping them into apparent acceptance. Its child resume process shares the same allocation; it does not submit another job. Both context modes, exact source/environment/device/driver/RNG identities, persisted outputs and failures are part of the acceptance evidence. Do not claim M3 from mock-only or independently passing track tests.

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
