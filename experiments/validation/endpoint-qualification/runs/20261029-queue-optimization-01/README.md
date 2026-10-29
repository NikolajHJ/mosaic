# Queue audit and conservative scheduler estimates

User authorized29October2026 shorter appropriate reservations and compatible
less-busy GPUs. This attempt changes scheduler estimate metadata only for the
three exact pending validators; no queue migration, restart, cancellation,
scientific setting/check/source change or monitor resumption.

Live10:24:45Paris: all12L40S GPUs occupied; all A100/A40/A10 GPUs occupied.
Some32GiB V100s were free, but pinned cuDNN9.13's hardware matrix excludes
Volta. Existing sources explicitly require L40S; selecting V100 without a
compatible runtime/native qualification would not preserve current checks.
L40S had88pendingCPU slots versus A100884/A40220/V1002212. These counts are
not wait estimates or comparable job counts. All three own GPU jobs were
eligible and waiting only for physical GPUs; CPU slots reserved, no start ETA.

Previous successful independent component checks measured457sPro/692sDB,
with successful Bol chemistry native component1012s. These are timing priors,
not measurements of the complete new validation. Set bmod-We scheduling
estimates to60min for29846764/29846766,45min for29846795; retain120min hard
limits and all internal/check/cleanup bounds. Existing queue BACKFILL uses
estimated runtime. Keep exact IDs, original submit time and scientific argv.
A runtime estimate is not a guarantee, hard reservation or successful pass.

Future optimizer reservations already derive from complete new validation plus
1.5x observed update time, complete startup/endpoint costs,30min reserve and
upward15min rounding. There is no blanket23h request:23h is the admission ceiling.
Full remaining30kPro/2.5kDB are preferred if actual measured runtime fits.
Do not replace scientific data or arbitrarily split horizons to advertise short jobs.

Evidence is ignored results/hpc/20261029-queue-optimization-01: snapshot01,
runtime-policy01, and the exclusive estimate-operation receipts. No local
numerical work or allocated tests are needed for this reversible metadata change.
Own manifests/submissions and actual pending states are authenticated before
mutating each exact job. Unknown outcomes stop and require inspection.

## Actual outcome

At10:30:16Paris the first runtime-estimate edit for29846764 was rejected by
DTU esub: only administrators may usebmod; job not modified. The other two
requests were not attempted. No workaround, cancellation, requeue or new
source was performed. Exact failed intent/outcome remain under funded22
control queue-estimates-20261029-01 and the local ignored operation receipt.
Read-only10:32:03Paris verification confirms all three originalIDs stillPEND,
each120minhardlimit; no estimates or scientific settings were changed.

Decision: retain current queue positions and qualified L40S runtime. All12
L40Ss were occupied, but own fair-share priority was100 and CPU slots reserved.
No reliable start time is available. A100/A40/A10 had no freeGPU in the snapshot;
V100 is unsupported by the pinnedcuDNN9.13 matrix. Replacing pending jobs
would lose their submit time and require fresh authenticated resource/startup
validation, with no demonstrated queue advantage. Future new releases should
include conservative scheduling estimates at initial submission with actual-
command tests and immutable metadata guards. Never patch queued sealed sources.
Optimizer reservations already derive from complete new validation/pilot
measurements, not failed startup durations or per-reference timeout ceilings.
This audit ran no experiment/test/model and left the general monitor paused.

References: [DTU GPU hardware](https://www.hpc.dtu.dk/?page_id=2759),
[cuDNN9.13 hardware matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/reference/support-matrix.html),
[LSF estimated runtime](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=lsbparams-estimated-runtime).
