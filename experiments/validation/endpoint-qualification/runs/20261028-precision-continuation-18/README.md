# Full remaining budgets: resource-count correction and portable checks

This retry preserves the user's preference for one full remaining Protenix job
(20k→50k,30,000 updates, LR0.3) and one full remaining DBMol-objective job
(7,500→10k,2,500 updates, LR0.0025), subject to complete measured admission.
Runtime uses actual pilot speed and validation/setup costs plus documented
margins; it does not reserve the full23h or both reference timeout ceilings by
default. All optimization settings, scientific checks and saved results stay
unchanged. The monitor remains paused.

Attempt17 CPU29841248 failed one negative fixture: Infinity was correctly rejected
by the strict JSON reader, but the test expected a later duration error message.
Torch17 cases passed; the JAX environment had789 passes and that one fixture
failure. Its controller29841249 exited without a GPU or optimizer submission.
Preserve the failed source, logs and XML; do not relabel it as807 passes.

A separate static review caught an undersized per-job file allowance. Stride20
retains three streams (snapshot, trial, reference);30,000 updates already need
at least4,502 such files, above17's4,500 cap before diagnostics. New18 accounts
all three streams and measured complete validation-file overhead for both
boundary checks. The byte formula,4GiB reserve and minimum file headroom remain
unchanged. This changes resource accounting only; retention is not reduced.

The user asked whether tests can run locally. Pure runtime-reservation and
storage-ledger fixtures are now explicitly portable metadata tests. They use
fabricated JSON/accounting inputs, standard-library production helpers and an
isolated test environment, with no model/scientific imports and no fake LSF job.
The remaining integration guards and every numerical/model/gradient/FP64/
chemistry/GPU check remain on allocated DTU HPC. Local checks never substitute
for the complete allocated release suite or authorize optimization.

Canonical preparation/staging entry points remain
`scripts/manage_precision_continuation.py prepare` and
`scripts/stage_precision_continuation.py`. Exact local commands, source seals,
allocated jobs, proof hashes and outcomes will be recorded next to this README
and in the global ledger before claims of a pass or submission. Source/root are
the analysis BASE plus`nonchem-precision-release-18/source` and
`nonchem-precision-continuation-18`. At initial preparation there was no18 seal,
allocated pass or job. The actual submission record below supersedes that state.

## Actual submission and access limit

The42-file release was staged read-only, manifest
`15fb307c704a0f460f882c701df6eec0655d678554c56f3abb581cbacf6a4b0e`,
archive329,913 bytes SHA
`84a9828ccce289a424bb95abd77d4585e5cde3ac2e2ed31598dbf508c6287c5f`.
CPU29841471 and own-ended controller29841472 were actually submitted after
fresh storage passed:11,670,735,360 required versus44,955,207,680 conservatively
available bytes, retaining reserves. No additional cleanup occurred.

The single later status query failed SSH authentication at15:18:18 UTC. No18 CPU
pass, GPU readiness, optimizer submission or progress can be claimed from that
failed observation. Inspect actual remote jobs before recovery; do not repeat
the stage/controller or manufacture states. A separate
[unsealed19 candidate](../20261028-precision-continuation-19/README.md) improves
portable resource identity and complete atomic-inventory retries without editing
this sealed source. Later local metadata98-case passes belong to that candidate,
not a claim that18's allocated suite passed. Saved results remain unchanged.
