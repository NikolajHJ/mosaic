# BRD4: three fresh 25-slot, 10,000-update arms

21 October 2026. The user authorized trying Protenix, Boltz-2 and Boltz-2 plus
chemical-validity guidance with **25 occupied atom slots** instead of six.
This is a new campaign; every historical six-slot experiment remains unchanged.
Authorization, CPU/controller submission and native preparation are distinct
from a passed native validation or an actual submitted long optimization.

## Declared comparison

| Arm | Differentiable objective | Initialization |
| --- | --- | --- |
| `protenix` | Frozen Protenix-Tiny whole-target contact, coefficient 1 | 50% neutral 25-carbon chain / 50% uniform categorical law |
| `boltz2` | Frozen Boltz-2 `affinity_pred_value`, coefficient 1 | The same probability law and fixed native key |
| `boltz2_chemistry` | The same Boltz affinity plus the existing failure-aware repair loss | Identical to the Boltz-only arm |

All request 10,000 **total graph updates**, LR0.03, native key17, categorical
logit/softmax parameterization and the existing ten-trial best-primal policy.
There are 300 unordered pairs, six supported elements C/N/O/S/F/Cl and five
bond categories. Occupancy and neutral charge remain fixed. The baselines have
**no chemical or composition guidance**; element-JS coefficient is explicitly
zero in all three arms, so the two Boltz objectives differ only by repair.
This differs explicitly from historical six-slot chemistry runs' element-JS0.1.

The third arm retains the existing repair schedule:0.001 through applied
update5000, then geometric increase to50 at10000. The loss/search/progress
settings are copied unchanged from the previous implementation. No continuous
chemistry prototype, self-target margin, DeFoG, VAE, decoder or learned prior
is introduced. Repaired targets/coefficients stay fixed within each trial set;
raw invalid/disconnected output remains saved. Ordinary best-primal increases
and no-ops are recorded, not reclassified as native failures.

The complete127-residue BRD4-BD1 3MXF sequence is retained. Fresh 25-slot
contexts are required; six-slot poses/features/digests cannot be resized or
adopted as new context validation. Protenix uses geometry-off whole-target
contact without an explicit pocket. Both Boltz arms share one newly predicted
25-carbon-chain pose, empty MSA,25 sampling steps,zero recycles and no MW
correction. Its context stays frozen during optimization. This remains a
fixed-pose partial-gradient surrogate, not fresh-pose/experimental affinity.

## Implementation and hard gates

The Boltz native feature map, worker transport and explicit first-order JAX VJP
are generalized to the explicitly supported six/25-slot shapes. Full ligand
correspondence, hard native feature/forward parity, neutral support, typed bond
mapping and exact FP32 request/response binding remain required. Historical
six-slot callers/defaults stay supported. A separate versioned native-only
preparation bypasses repair/search and composition execution for the baselines;
it records chemical checks as not applicable, never as already-valid/passed.
The shared trajectory writer retains every state/raw gradient/trial/readout.

Allocated CPU readiness reruns the full existing bridge/legacy, native mapping,
original-source chemistry oracle suite and launcher/temp-directory contracts,
with new25-slot mapping, callback, pure-objective and stage-transition checks.
No numerical tests or chemistry/model work is permitted locally. Local work is
static editing, AST/schema review, immutable packaging and byte/XML transfers.

Each arm requires its actual native raw atom/bond finite differences at the
original0.01/0.003/0.001 scales and15%/two-adjacent criteria, scalar primal/AD
parity, finite outputs/derivatives, hard-seed native parity and frozen weights/
context. Protenix retains its bond-order null check. Boltz retains nonzero
typed-bond contrast. The chemistry arm additionally requires nonzero invalid-
assay chemistry derivatives, low/high-weight coupled checks and both disposable
pilots. Baselines omit chemical gates explicitly because they omit that term.
No old failed random-start result is reclassified as passing.

Runtime must satisfy the unchanged empirical fit gate:
`1.5 * slowest_pilot_mean * 10000 + 1800 < 82800` seconds. All allocations remain
at most23h for long optimization; no unauthenticated mid-arm resume or silent
budget reduction. The two-hour readiness jobs use the same GPU/CPU/memory shape
as their future arm. Protenix requests gpua10; both Boltz arms request gpul40s;
one exclusive GPU,four CPU slots,8GiB per slot. These actual public queue names
were observed on21October09:42Paris. Their differing devices make runtime
comparisons exploratory, not a controlled hardware benchmark.

Each pilot records its largest saved per-update footprint. Before long
submission, use twice that observed footprint times10,000 plus1GiB per arm;
sum eligible-arm estimates and preserve4GiB home reserve. Account additionally
for the new campaign's current bytes because quota reports can lag4h. Stop on
unverified/insufficient quota; do not delete historical data or waive the gate.
At initial access home was14.52/30GiB; this is not a storage reservation.

## Staging protocol and current status

New immutable source/run identity: `20261021-brd4-25-01`.
Intended source `/work3/s234818/mosaic/releases/20261021-brd4-25-01/source`.
Intended artifacts `/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-01/`.
Source includes current uncommitted scientific/tooling bytes, configurations,
new fixture and required tests; Git revision is only a reference.

An after-CPU callback authenticates actual DONE/LSF success and every passed
CPU proof, persists a hashed certificate independent of live scheduler retention,
then submits fresh Protenix validation and Boltz preparation. After actual
preparation completion, its callback seals the context and submits two separate
Boltz validators. Per-arm ended callbacks retain actual DONE/EXIT decisions.
After all decisions are recorded and campaign-wide quota passes, only genuinely
passed arms were initially intended to receive separate long jobs. The user then
requested a measured chemistry runtime estimate **before starting that arm**.
The revised `chemistry-runtime-review-v1` launcher prohibits all full chemistry
submissions, even after a native readiness pass. Only the bounded native tests
and early/high-weight ten-update disposable pilots are allowed for that arm.
It saves the measured mean seconds/update and the existing conservative fit
projection in `chemistry-runtime-estimate.json`. These are empirical estimates,
not a completed full run or a guarantee about later invalid-graph searches.

Protenix-only and Boltz-only may proceed independently after their own actual
native readiness and shared campaign quota pass; they do not wait for a chemistry
decision. Each attempt has its own exclusive marker and persisted submission
receipt. Failed/blocked arms and the chemistry hold remain explicitly listed.
Duplicate/unknown submissions fail closed; never infer submission from a callback.

Source01 authenticated450 files, manifest
`4016584825436c9c597307ac51eddb242ac50885352e9c06841da2e1e201ee9c`,
archive39,413,811bytes SHA
`c625c367e34d2d589edc11629d1bf65aac2642da37b54ad2d4bee53622864d44`.
CPU29768889 actually DONE at10:00:54Paris. Its pending callback29768890 was
intentionally cancelled for the new runtime-review instruction, actual EXIT at
09:59:16 with no execution start and no GPU/native/long submission. Source01
and its test evidence remain historical. A new immutable launcher revision is
required; every scientific `src/`, configuration and driver byte is preserved.
Its new CPU/native outcomes and actual resource/submission IDs belong in the
campaign receipts/ledger after authentication.


Source02 authenticated451 files (manifest
68c5ad0d2ee4ab9f95c30dbbee0cec489000a70aafea179ad7711744aecc3625),
all185 scientific src/examples bytes identical to source01. CPU29768926 and
callback29768927 were submitted. Static review then found the new Protenix hard
reference omitted the geometry-hook mask; base features alone select native
geometry despite zero placeholder coordinates. This is a reference-mode defect,
not a numerical observation or a reason to relax parity. Pending callback29768927
was intentionally retired, actual EXIT10:06:53Paris, no start/GPU submission.
CPU29768926 continues unchanged and its actual result will be retained.

Source03 corrects only the independent Protenix reference construction: retain
hard base element/bond fields, add the explicit same geometry-off mask and zero
ligand placeholders without relaxed graph injection. The attached objective and
all native thresholds are unchanged. Added an allocated regression comparing
all hard-reference fields with hard injection under permuted atom/token indices,
including protein preservation and no base-context mutation. It retains source02
chemistry hold/independent-baseline launcher and requires new full allocated CPU
and original native GPU gates. Fresh source03 identities/results follow in the
receipt and ledger; no source01/02 pass is adopted as source03 validation.


## Authenticated source03 CPU result and actual current submissions

CPU29768990 DONE21October10:12:22Paris, **342 passed**, no failures/errors/skips;
three historical slow/model tests deliberately deselected. Counts34launcher,
226legacy/JAX,23native mapping/config and59unchanged chemistry. Actual scheduler,
LSF success, all raw XML counts/source/reference and all19downloaded proof bytes
verified at `results/hpc/20261021-brd4-25/source03-cpu/evidence/`.
AfterCPU29768991 DONE10:12:28 and actually submitted:

| Job | Scope | Queue | Latest authenticated state |
| --- | --- | --- | --- |
| 29769026 | Protenix native validation | gpua10 | PEND |
| 29769027 | Ended validation callback | hpc | PEND |
| 29769028 | Fresh Boltz25-slot pose preparation | gpul40s | PEND |
| 29769029 | Ended preparation callback | hpc | PEND |

No10,000-update arm has yet been submitted. Boltz validation IDs will appear in
`boltz-readiness-submissions.json` after genuine preparation success. Baseline
long IDs belong in `long-submissions.json` and each `run-*/submission.json`;
the full chemistry arm remains prohibited. Latest receipt:
`scripts/hpc/receipts/20261021-brd4-25-03/campaign.json`.

## Measured repair component overhead; total runtime still pending

User-requested supplemental timing ran as allocated CPU29769038, actual DONE
10:14:42Paris,42seconds elapsed. Exact source03 chemistry/configuration, four
CPU slots, one warmup and three recorded repeats per fixed point. No model,
optimizer or molecular optimization executed. Every raw timing is retained.

| Slots | Fixed point | Mean repair preparation + one chemistry gradient | 10,000 such calls |
| --- | --- | --- | --- |
| 6 | Initial valid chain/uniform | 0.05505s | 550.52s (9.2min) |
| 6 | Internal-fluorine invalid assay | 0.19663s | 1966.28s (32.8min) |
| 25 | Initial valid chain/uniform | 0.11328s | 1132.76s (18.9min) |
| 25 | Internal-fluorine invalid assay | 0.46580s | 4657.97s (77.6min) |

These tested costs grow about2–2.4times, despite20times as many possible pairs.
Bounded search prevents a direct pair-count-to-runtime extrapolation. This is
an empirical **component** estimate: one fixed-point preparation and chemistry
value/gradient, warmed up. It excludes Boltz inference, multiple line-search
trials, startup and later more difficult invalid/disconnected graphs. It is
neither a worst-case bound nor the full10,000-update ETA. Do not interpret
component status `recorded` as native validation `passed`. Full early/high GPU
pilots remain necessary before recommending a total run budget or releasing
chemistry. These measurements alone do not justify discarding repair for cost.

Timing tools3files manifest
04d4fd4eaaa8e07c02c58a04b9f15c02e0a47fba9a9ec5dbb1e182084c641100;
archive3852bytes SHA1776307af4e202d8b805c8c8c272902863ecb7abee30f61132288ba10af313f6.
All15raw JSON/NPZ/preparation/scheduler/LSF files safely downloaded/byte verified
at `results/hpc/20261021-brd4-25/repair-timing/evidence/`. Evidence archive
16404bytes SHAb7881b3dcf522e8ed60fa267b08e7464acde0b6430ba90ef51f7441cebacddf6.
The first capture helper lacked login-shell LSF PATH before capture; empty01
is preserved, unique02 succeeded. No scientific artifact changed.


## 21 October: source03 Boltz final-response contract failure retained

Fresh Boltz preparation29769028 actually DONE10:16:43Paris; afterprepare29769029
DONE10:16:49. Exact25-slot context SHA
099fdac537a08f65a14cb60463de24fa0c4ffd6541a142beaaee6e974977fa08,
tensor digestdf93ef76cc26d5888532dc2c36688dbbff4040375f5357a784b8b6a656ec2f34.
Actual validators Boltz-only29769078 and chemistry29769080, callbacks29769079/
29769081 submitted. Boltz-only29769078 ran10:17:15–10:19:03 and EXIT1 after
native hard parity, all three atom/bond FD scales, ten disposable updates and
runtime fit individually passed. Whole readiness remains FAILED, no assessment
or long run. Callback29769079 DONE retains that failed decision.

The actual model/context hashes before/after were exact. Driver incorrectly
compared the entire response dictionary (which additionally carries status,id,
elapsed_seconds) against a two-digest dictionary. This is a concrete response-
shape bug, not changed weights or a scientific threshold rejection. Preserve
raw failed summary/worker/native/pilot evidence. Source04 will require status
passed AND exact original model/context digest fields, retaining the raw response
and the transport's existing request-ID checks. Added allocated contracts accept
metadata only with exact hashes/success, and reject either mutation or absent/
failed success. No derivative/parity/frozen/finite/runtime gate is weakened.

Obsolete source03 Protenix29769026/callback29769027 and chemistry29769080/
callback29769081 were all verified PEND/unstarted, bkill accepted; actual EXIT
verified with no start times. Their original receipts and cancellation evidence
remain under source03 root. No long experiment has started or been cancelled.
Prepare unique source04 and rerun full allocated CPU/native tests, preserving
identical configuration/scientific settings and the full chemistry runtime hold.
Source03 passed subchecks/pilot timings are observations, never a readiness pass.


## Source04 allocated retry and preserved source03 failure evidence

Unique source04 staged/authenticated461members:
/work3/s234818/mosaic/releases/20261021-brd4-25-04/source,
manifest e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79,
archive39,425,961bytes SHA
b62ef75f6ede08fbed8854ff4efe167571be6eb050c1b222278e19e15b3dc879.
Source03-to04 src/examples bytes are identical except the driver now calls the
strict passed-status/two-digest verifier; all three configuration files unchanged.
CPU29769140 actually RUN10:21:56Paris, afterCPU29769141 PEND on ended(CPU).
New5response contracts join full regression/native gates; no pass or long job
inferred. Current run root /zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04/,
receipt scripts/hpc/receipts/20261021-brd4-25-04/campaign.json. Full chemistry
remains held even on future readiness success, pending end-to-end runtime review.

Source03 final-response failure evidence, all85files including raw native FD,
hard parity, ten-update pilot, worker response, complete original prepared25-slot
context bytes and cancellation before/after records, safely downloaded/verified:
results/hpc/20261021-brd4-25/source03-response-failure/evidence/.
Archive664616bytes SHAe635ca196c54c672a9cbfbd9891c3ffdf127f5e81dfc78f53c52da9caab03c58,
manifest15c540fd57b54e0eddbab7f6f57a4d76b76900a504e20a5030d3940e52b777e3.
Actual frozen model/context fields before/after exact; metadata mismatch explains
engineering rejection, but the entire source03 readiness stays failed. None of
its individual passed subchecks or measured pilot times promotes source04.
All four obsolete unstarted source03 jobs actualEXIT10:20:21Paris; no long run
started/cancelled. Heartbeat updated to source04 current jobs and original gates;
old random scientific blocker still not retried, full chemistry still held.
