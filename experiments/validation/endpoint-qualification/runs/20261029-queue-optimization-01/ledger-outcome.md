

### 2026-10-29T09:32:03Z — Queue audit; DTU rejected in-place estimate change

Current validators29846764/29846766/29846795 remain PEND with120min hard
limits, eligible physicalGPU waits/CPUslots reserved/noETA. All12L40S busy;
A100/A40/A10 also busy; someV100free but pinnedcuDNN9.13 excludesVolta.
L40S lightest observed pending-core count; counts are not reliable waittimes.
Successful component timing priors457sPro/692sDB/1012sBolchem do not measure
complete new validation. Prepared60minPro/DB and45minchem scheduling estimates
for BACKFILL while retaining hardchecks. First edit on29846764 rejected by
siteesub255: onlyadministrators may usebmod; jobnotmodified. Other requests
notattempted, failure/evidence preserved; no workaround, cancellation, requeue,
source or scientific changes. OriginalIDs/submit times/hardlimits retained.
Keep current L40S positions; future new capsules should declare conservative
scheduling estimates before submission and actual-command validation.
Optimizer walltime already rounds measured fullpilots/setup/endpoint/reserve
upward15min;23h is a ceiling. No numericalwork/tests/cleanup/newjob/monitorresume.
Owner: experiments/validation/endpoint-qualification/runs/20261029-queue-optimization-01/.
