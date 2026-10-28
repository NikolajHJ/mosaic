

### 2026-10-28T14:16:14.073813+00:00 — Full remaining work preferred over fixed small chunks

The user clarified that each allocation should run as much remaining work as
possible, with an appropriate measured reservation. New immutable precision17
therefore requests Protenix20k→50k (30,000 updates, LR0.3) and DBMol7,500→10k
(2,500 updates, LR0.0025) in one job each if complete measured admission fits.
Walltime projects the full update budget plus measured complete validation/setup
with conservative margins and upward15-minute rounding; reference timeout ceilings
are not automatically reserved as actual elapsed work. All scientific checks,
original settings, exact boundary/RNG and retention remain unchanged. Oversized
horizons must preserve failure/max-safe proposals before a new immutable plan;
no silent shortening, fallback or gate waiver.

All four old source16 GPU validators/dispatch callbacks were verified PEND and
never entered a study before user-directed bkill. Actual later scheduler EXIT
was authenticated for29841081/82/83/84, with zero optimizer children/studies;
source16 CPU29841063/controller29841064 remain DONE with785 cases passed.
Retirement assessment SHA 2cf56b547d388d102c85573d03305f61f8ea8694b5ba9f0970e6ca88e84a4207 preserved; no original
result or independent DeFoG job was changed. New17 preparation requires807
allocated cases, real MPS/native/reference/pilot/startup/runtime/storage checks.
The41-file local17 package is prepared; no stage/pass/job is established yet.
An initial independent metadata auditor tried to reserialize sorted transport
JSON and failed its byte assertion. Original raw bytes were then downloaded,
rehash-verified and compared semantically; the failed assertion is preserved.
No source, numerical gate or saved result changed. Monitor remains paused.
Owner: experiments/validation/endpoint-qualification/runs/20261028-precision-continuation-17/.
