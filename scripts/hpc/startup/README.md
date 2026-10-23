# Allocated pre-model command startup checks

`premodel_startup.py` is shared by the BRD4 Boltz LR0.0025 comparison and the
independent DBMol objective pilot. Each immutable release includes an exact
copy as `scripts/premodel_startup.py`, alongside the allocated regression fixture
`tests/test_premodel_startup.py` from `tests/core/fixture_premodel_startup.py`.

After the actual full CPU suite and certificate pass, the allocated callback
calls `run_startup_smoke(ops, manifest, node_tmp_directory, arm=...,
phase="native", index=0)`. Before each segment submission, it repeats the check
with `phase="segment"`, its actual index and actual passed readiness/predecessor
certificates. It invokes
the real controller's `produce` function, retaining its real certificate,
input, context and storage checks. Only the final driver subprocess is captured;
all other subprocesses execute normally. It then executes that exact command's
driver with its real argument parser, provenance and pre-model initialization.
The native worker constructor raises a classified sentinel before loading a
model. All disposable driver outputs stay under the owned node `$TMPDIR`.

The returned receipt records `startup_only_passed`, not a native, gradient,
readiness or optimization pass. The caller may persist this JSON and its hash;
temporary array files must not become scientific outputs. Missing metadata,
changed sealed bytes, malformed arguments or failure to reach the sentinel fail
the callback and prevent native dispatch. Existing studies are never inputs to
this check's output path and are never changed. Local work is standard-library
source/byte inspection only; regression and real driver execution require an
allocated DTU HPC CPU job.
