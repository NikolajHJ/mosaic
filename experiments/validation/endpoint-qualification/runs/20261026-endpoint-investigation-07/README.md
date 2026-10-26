# Boltz and DBMol continuation failure investigation 07

The user authorized investigating source06 validation failures, correcting only
demonstrated tooling defects and submitting blocked continuations after all
checks pass. Preserve source06, same-job LSF failures and saved scientific runs.
Plain Boltz10k-to30k and DBMol5k-to10k keep LR0.0025, original objectives,
N25/key17/T1, one fixed finite candidate, losses1/matrices20 and all other
optimization settings. No altered saved state, model, diffusion or objective.

Boltz passed prospective endpoint admission then failed the unchanged pilot
combined frozen-reference bond derivative check. DBMol failed prospective
endpoint admission. Read raw evidence before selecting a correction; do not
assume either is a precision artifact or loosen/skip its mandatory checks.
A genuine unresolved numerical disagreement continues to block execution.

Source06 also emits obsolete v1 policy labels in driver result metadata while
its controller requires the actual v2 policy; future tooling must use the
loaded authenticated policy version consistently and cover this integration.
New immutable sources and allocated CPU/actual startup/native/pilot/runtime/
frozen/full-storage/terminal gates are required before child submissions.
Preparation only; no new evaluation/submission. Local stdlib AST/bytes only;
all numerical work allocated DTUHPC. Preserve active chemistry Protenix03 and
its successor jobs. Monitor remains paused. Evidence/manifests recorded here.

## Investigation progress and access blocker

Existing authenticated records show that DBMol affinity, contact and confidence
component admissions passed, while the combined total failed. This does not yet
identify a cause or certify the entire native validator. Boltz failed a separate
pilot check on bond logits after raw simplex endpoint admission passed. The
actual detailed FD observations are still required before selecting a remedy.
No tolerance, epsilon ladder, loss weight, update rule or numerical gate changed.

Corrected the independently demonstrated driver-label defect: result and
invocation metadata now obtain v2 from the authenticated loaded policy rather
than hardcoding v1. Four regression cases were added but not executed. Static
AST reversal proves that the rest of the driver is identical to the committed
source06 driver. A new nonchemistry numerical retry has not been selected,
sealed, validated or submitted. Source06 and all original failures stay intact.

The only read-only SSH attempt timed out before collecting the needed reports.
See `hpc-access.json` and `investigation-status.json`. The canonical metadata
collector is `scripts/collect_repair_inputs.py --label <fresh-label>`; it must
never overwrite the preserved failed inputs01 attempt. No local numerical work
or new job, no further cleanup and no monitor resumption.
