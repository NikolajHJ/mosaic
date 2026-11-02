# Protenix random initialization with delayed chemistry guidance

Experiment ID brd4-jq1/protenix-random-chem. Explicit user authorization23October:
one additional Protenix optimization, same settings as the latest passed
lower-rate Protenix, with random initialization and delayed chemistry validity
loss, while continuing separate Boltz endpoint diagnostics and saved-only report.
This is independent of every existing run; no saved result is edited/restarted.

Protocol: original frozen Protenix tiny/model/context/JQ1 pocket and key17,
25 occupied neutral slots, temperature1, geometry-off pocket contact plus the
existing failure-aware chemistry loss/search, zero composition penalty.
Fresh10000 fixed updates atLR0.3, exactly one finite candidate; no halving,
backtracking, clipping or gradient normalization. Retain honest uphill/equal/noop
bookkeeping. Random rows use the existing independent PCG64/Dirichlet(1) recipe
with initialization seed17, separate from fixed native model key17. Chemistry
coefficient0.001 through applied5000, geometric to100 at10000; reference/target/
coefficient held fixed within each update. No diffusion/VAE/new model/decoder.

New protocol version explicitly makes the old chain-initialization identity
comparison inapplicable to this random branch only. It retains exact5-array/key
validation-to-production identity and fresh random recipe reproducibility,
original rawFD0.01/0.003/0.001/15%/two-adjacent/nonzero/null/parity/primal/frozen
guards, invalid chemistry nonzero signal, low/high coupled gates and both10update
pilots, runtime/quota gates. New initial simplex infeasibility or other failure
is a real failure; no redraw, clipping, epsilon shrink or gate waiver.

Every test/model/chemistry/numerical run/render occurs on allocated DTU HPC.
Local standard-library AST/JSON/hash/byte packaging only. Start CPU contracts,
then fresh native validation, then one independent long only after actual
DONE/LSF/proof/readiness/runtime/storage passes. Allocations native2h/long23h,
one exclusive L40S48GB,4CPU/8GiBslot; CPU4slots/4GiBslot1h, callbacks1slot2GiB30min.
Whole capacity retains all original three-run remaining projections, existing
report/diagnostic allowances, this run's full pilot projection and4GiB/100000-file
headroom. No cleanup or borrowing reserves. Unknown submission markers prohibit
blind duplicates. Keep all failed attempts immutable.

Current run and owned files: [attempt01](runs/20261023-protenix-random-chem-01/README.md).
Canonical implementations/config/fixtures and launchers are owned there;
compatibility aliases exist only inside a fresh authenticated source archive.
Base scientific625 source145 library files are preserved byte for byte.
Parent [lower-rate experiment](../fixed-lrreset/README.md); current independent
[Boltz diagnostics/report](../fixed-lrreset/runs/20261023-brd4-jq1-lrreset-analysis-03/README.md).
Current status: declared preparation only; no new source/job/native pass yet.


Latest23October11:09Paris: source631 authenticated/read-only, CPU29796731 and
callback29796732 DONE/full776 cases and independent35-file audit passed. Fresh
native29796780/callback29796781 queued. No native pass or10000-update optimization
submission yet. Old results and independent endpoint/report work are unchanged.
These launchers are account-bound historical execution entries; their constants
are not portable account defaults.


23October2026: user superseded the queued10000-update trial with
[50000 updates and a stretched ramp](../protenix-extended-50000/README.md).
Actual native29796780/callback29796781 intentionally retired before execution;
source631/CPU776 pass remain historical. No10000-update optimization was submitted.


30October2026: a separate fresh50k matched arm with per-step binding-gradient-relative noise is queued as29857358 on32GBV100. Ratio0.4 per logit block, LR included, separate noise seed230017; same delayed chemistry ramp, composition off, model/input settings and frozen within-step targets. Existing jobs and saved results stay unchanged. [Current protocol/submission](runs/20261030-gradient-noise-01/README.md). This arm uses the current exploratory policy; historical strict campaign receipts retain their original meanings.


1November2026: [new chemistry strategy comparison](runs/20261031-chemistry-strategies-01/README.md) implements22 paired/random-start20k experiments under the current exploratory policy. This new comparison does not inherit the old strict validation prerequisites described above. Existing jobs/results are unchanged.

1November2026: [structural repair comparison](runs/20261101-structural-repair-01/README.md) adds an opt-in connected-backbone target builder ahead of the existing beam search. Following the user's reuse request, two fresh30k structural trajectories use archived constant1 seed17/23 controls for the common first20k; the two duplicate, never-started controls were retired. A100/A40 versus archivedV100 timings are hardware-confounded. Earlier runs, saved results, loss and fallbacks remain unchanged.

## Current recovery after exclusive GPU request

See [2November exclusive-mode recovery](runs/20261102-exclusive-gpu-recovery-01/README.md): ten unfinished Mosaic producers resubmitted, five already committing updates at00:29UTC; completed runs and all saved inputs/results preserved. Actual successors and source/plan bindings override historical RUN/PEND records.

2November2026: [stronger structural guidance and annealed noise](runs/20261102-structural-strong-noise-01/README.md) adds four fresh30k runs: constant10 structural repair with/without binding-gradient noise annealed0.4 to0 by20k, each from sparse seed17 or50% carbon-chain/uniform plus small initial noise. Actual jobs29873078/81/82/84 are queued on A100/V100/A40/L40S; CPU39 and four startups passed. Existing runs/results are unchanged.

2 November 2026: [sparse 50k cooling/chemistry schedule](runs/20261102-sparse-annealed-50k-01/README.md) is running on V100 as29873389 (442 verified updates at11:09UTC). Chemistry1 through20k then geometric30 by30k; temperature0.7 to0.4 over50k; noise0.4 to0 by30k. Conditional CPU29873393 waits for completion and submits one additional20k at chemistry100/T0.4/noise0 only if the saved endpoint is invalid. CPU79 and canonical report-adapter44 tests passed; existing runs/results unchanged.
