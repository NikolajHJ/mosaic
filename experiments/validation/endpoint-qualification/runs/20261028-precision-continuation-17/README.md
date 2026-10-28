# Full remaining work with measured reservations, 28 October 2026

The user clarified that arbitrary small chunks are not desired: run as many
remaining updates as safely fit in one allocation, requesting an appropriate
measured walltime and storage allowance. This supersedes attempt16's artificial
2,500/1,250-update split. Existing results, losses, rates, initialization, RNG,
retention and scientific checks remain unchanged.

The first choice is one Protenix continuation from20k to50k (30,000 updates,
LR0.3) and one DBMol-objective Boltz continuation from7,500 to10k (2,500 updates,
LR0.0025). Request these full horizons only after fresh complete native/pilot/
replay/reference/frozen/startup checks and measured runtime/storage admission.
Walltime uses actual pilot update speed and complete validation/check setup
measurements with conservative margins, rounded upward to15minutes. Do not
automatically request23h, reserve both reference timeout ceilings as measured
runtime, truncate checks, or silently execute a different update budget.

If the full remaining work cannot fit the real allocation limit, preserve the
failed horizon admission and its computed largest safe update count, then prepare
that largest justified continuation in a new immutable plan. The total targets
remain50k/10k; later work requires fresh admission. This is not permission to
change optimization settings, weaken gradient gates, or rewrite old studies.

Attempt16's785 CPU pass and old source/proofs remain immutable. Its metadata
dispatch callbacks must be retired before any small optimizer child can start.
Pending superseded native validations may be retired; running validators can
finish read-only and must not overlap an unaccounted new resource admission.
Inspect actual receipts/states before each action. Never cancel another project
or the independent Protenix+DeFoG run. No scientific run is resumed by this record.

Account-bound new source and root are respectively the original analysis BASE
plus`nonchem-precision-release-17/source` and`nonchem-precision-continuation-17`.
Canonical entry points remain `scripts/manage_precision_continuation.py prepare`
and `scripts/stage_precision_continuation.py`. All tests and model/CUDA checks
run on allocated DTU HPC only. Preparation, actual submissions, starts and outcomes
will be recorded adjacent to this README and in the global experiment ledger.

Initial status: preparation only; no17 source seal, pass, staging or job yet.
Plain Boltz20k and chemistry retain their separate unresolved qualification
records. Saved results stay unchanged and the monitor remains paused.

## Preserved outcome

The41-file source was subsequently staged, manifest
`4d29d85856e696d7fc096dc5f12bd54ed51d64fc2032946be2b51a8d5b4c480c`.
CPU29841248 and controller29841249 actually EXIT1. Untouched raw XML has806
passes and one negative-fixture message assertion failure: Infinity was correctly
rejected by strict JSON parsing, before the expected duration error. There were
zero native/GPU/optimizer submissions. Full failure/source/LSF/XML evidence is
preserved in18's failure-verification record. The source remains failed;
[18](../20261028-precision-continuation-18/README.md) fixes the fixture and
three-stream file accounting, with no changes to optimization settings/results.
