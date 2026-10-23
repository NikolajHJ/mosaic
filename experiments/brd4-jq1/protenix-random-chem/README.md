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
