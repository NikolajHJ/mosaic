# V100 default and conditional continuation migration

Question: can the five queued exploratory continuations use the larger eligible
V100 pool with their original objectives, settings and saved starting points?

New default: 32 GB V100 (`gpuv100`, `gmem=30G`); L40S remains an explicit fallback.
See [common policy](../../../../../docs/GPU_POLICY.md). Running jobs stay untouched.

Actual device comparison **29855106** and own-ended migration callback **29855107**
are submitted/released, both PEND at 2026-10-30T13:15:45Z. No GPU result or completed
migration exists yet. All five original L40S optimizers are still PEND/unstarted.
The callback replaces an original only if its actual device smoke passes, sampled
GPU headroom is at least 1 GiB, and the measured whole remaining horizon fits
V100's 24-hour limit. Otherwise it retains L40S. A job that starts meanwhile stays.

Budget: one 32 GB V100, eight CPU slots / 4 GiB each, 1-hour comparison; three
unchanged fixed updates per arm in separate children, prioritizing both chemistry
arms. Each child is bounded to 540 seconds, group to 3,000 seconds; no smoke state
is adopted. Capture actual memory and update timings. Same saved logits/key,
objectives, precision, rates, chemistry ramp, composition off and stride20.
Cross-device FP32 graph replay uses a predeclared dtype-relative check (32 eps,
32 smallest-subnormal floor); raw saved inputs remain byte-bound and untouched.
This is not a finite-difference or scientific qualification campaign.

Device guards in the existing chemistry transport now accept an explicit declared
profile. Both original acceleration implementations were copied unchanged except
those hardware metadata guards; reversing those guards restores every original
function AST exactly. Full-model arithmetic, detached teacher preparation and
original hybrid/matrix acceleration remain unchanged.

68 portable metadata/AST/fake-process tests passed (5 existing exploratory +14
GPU/migration +49 existing adapter controls). All five actual metadata CLI startups
passed without numerical/model setup. An initial test invocation had four Windows
pytest-temp permission setup errors; using a fresh workspace temp directory gave
49/49 passes. This is no GPU claim. See campaign, source and submission proofs.

Source is immutable at the path in `source.json`. Never replay submissions or edit
that source. Existing comparisons/callbacks must be reconciled before recovery.
The callback records each retirement and replacement under `smoke/migration-ARM`,
new optimizer IDs under `production/jobs.json`, and actual per-arm plans alongside.
Retirement first holds the exact original job, checks ownership/history/no payload,
and verifies EXIT before releasing its replacement. An execution race restores the
original instead. Submission timeouts remain unknown outcomes; never blind retry.

Use the saved metadata observer program in
`results/hpc/20261030-v100-default/status-20261030T131543Z/program.py` with a fresh
output label; extend it to include current production IDs and saved prefixes after
migration. Report updates reuse the canonical reporting command only when new
production states exist; the device-smoke states are excluded.

The source01 graph replay field named `byte_equal` uses NumPy array equality;
it does not distinguish signed-zero representations. Treat it as numeric equality,
not byte-identity evidence. Saved NPZ file hashes remain the byte binding.

Exploratory layout metadata verification passed with zero errors and an advisory
warning about 86 canonical tests outside the historical migration registry.
