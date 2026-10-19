# BRD4 Boltz-2 affinity with delayed chemistry

19 October 2026. User authorized implementation, allocated HPC validation and,
only after passing, two independent fresh 10,000-update runs: 50% chain/uniform
and seeded random probability initialization. Implementation is being validated;
submission or scheduler success alone does not certify numerical readiness.

**Current status, 19 October 2026, approximately 19:08 Europe/Paris:** the
283 CPU tests passed. Native validation is now queued as **29758990** in
**gpul40s**, requesting one exclusive NVIDIA L40S (48 GB). Replacement
submission controller **29759027** waits for native validation and the passed
launcher CPU check **29759003**. Neither long optimization arm has been
submitted. The earlier A10 readiness/controller were intentionally cancelled
while unstarted; their EXIT outcomes are not native test failures. See the
resource amendment below for exact identities and preserved failures.

## Scope and pivotal choices

Mosaic retains its six occupied neutral heavy-atom slots, element order
C/N/O/S/F/Cl, five unordered bond categories, logit optimizer, best-primal policy,
chemistry search/loss and trajectory artifacts. No VAE, decoder, latent space,
training, learned bridge, DeFoG or DBMol composite objective is introduced.
Only the previously audited native Boltz integration method is reused.

**Objective:** native quantitative `affinity_pred_value` with coefficient 1,
plus the existing failure-aware chemistry penalty and 0.1 empirical element-JS.
Affinity is log10(IC50 in micromolar), lower is better. Native ensemble binder
probability and mean raw binder logit are logged separately, not optimized.
No contact or confidence term is added. This choice isolates the requested
replacement of contact guidance by affinity; coefficients across unlike model
scores are exploratory, not experimentally calibrated.

The chemistry schedule is unchanged: 0.001 for applied updates 1..5000, then
geometric growth to exactly 50 at update10000. Detached repair references and
the coefficient are fixed throughout each state's ten halving trials. Accept
the first strict AD/independent-primal decrease, otherwise the best finite trial
including equality/increase, earliest tie. Preserve fallback/no-op/increase
counts and raw invalid/disconnected graphs. Changing-reference totals do not
form one fixed objective. LR0.03, native key17, random PCG64 Dirichlet(1) seed
20261019 and the exact previous 50% categorical mixture are retained.

**Backend boundary:** JAX on allocated CPU owns Mosaic's graph, chemistry and
optimizer. A persistent PyTorch process on that allocation owns the native GPU
model. An explicit first-order custom VJP returns the native unmodified X/E
derivatives; independent primals do no backward. JSON transport is lossless for
FP32 inputs, serial/request-bound, and has no gradient/result cache. Stop/log
binary outputs explicitly; higher derivatives/JVP through this boundary are
unsupported. Never treat ordinary tensor conversion as an autodiff bridge.

**Native identity:** upstream Boltz commit b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc,
official cached affinity checkpoint SHA
dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e.
Strict loading, eval mode, frozen weights, FP32/TF32 off, standard kernels and
zero recycles (one trunk pass), no molecular-weight correction. Preserve the
native quantitative mean and mean(sigmoid(member logits)) probability.
Typed embedding expectation includes NONE and explicitly maps Mosaic categories
to native IDs0/2/3/4/5. Native elements use atomic number, unlike Protenix's
offset channel convention. Charge stays zero; no charge optimization is implied.

**Prepared context:** fresh native prediction for the existing complete127-residue
3MXF sequence and hexane (`CCCCCC`), six slots, seed17, empty MSA,
25 structure/affinity sampling steps and one sample. No external MSA traffic,
crystal ligand transplant or DBMol prepared pose. The deposited sequence matches
the old experiment, but the numerical Boltz context is new. Its predicted pose,
reference geometry, names, masks, crop and auxiliary fields remain fixed during
optimization. The resulting derivative is a fixed-pose chemistry surrogate,
not a fresh pose or experimental affinity prediction for every changed graph.
Hard/fresh-pose rescoring remains necessary before interpreting molecular utility.

## Readiness and execution

- [x] Implement separate native adapter/trunk/head and explicit JAX worker bridge.
- [x] Reuse chemistry optimizer with objective-specific affinity artifact labels;
  historical contact behavior remains its default.
- [x] Author mapping/gradient/transport/trajectory regression tests and configs.
- [x] Allocated CPU regressions, unchanged chemistry oracle parity and Torch mapping tests:283passed.
- [ ] Fresh allocated native preparation; hard-seed/native integer forward parity.
- [ ] Both initializations: independent-primal atom/bond finite differences,
  observed bond-order signal, nonzero invalid-assay chemistry gradients,
  low/high-weight coupled checks and disposable early/high-weight pilots.
- [ ] Authenticate fixed model/context before/after and empirical runtime fit:
  1.5 times slower pilot mean times10000 +1800s below23h. This is not a worst-case bound.
- [ ] Authenticate final native readiness, then submit two independent GPU arms.

Historical initial resources: CPU readiness hpc4slots/4GiB per slot/1h; native readiness
gpua10 one exclusive GPU/four slots/8GiB per slot/2h. The modest BRD4 context is
tested on that actual GPU before long submissions. Each full arm uses the same
GPU queue/shape and23h, no inter-arm dependency. Cache assets are reused read-only
from the existing pinned Boltz environment; no reinstall or model download.
Home/work quota and exact immutable source/config/context/proof identities are
gated before numerical work and before each submission. No prior data is deleted.

All numerical work, native preparation, molecule evaluation and tests are on
allocated DTU HPC. Local work is static review, editing, packaging and byte/XML
verification. Actual source/job IDs, test outcomes and blockers follow below
and in the experiment ledger; pending jobs are not passed tests.


## Actual source and initial submissions

Source01 was authenticated on HPC:386files, archive39,229,862bytes,
SHA fc24ad5db5afdde7599c0c41e72c1a4790f9ab1db23ca1d35cd4d72fe2a1851c;
manifest8a621a975cf98800ea3a2a7413f1e07b1f317b988c02d69b593832a7d1fd94af.
Immutable source /work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-01/source.
The new plan is an external post-packaging record; it is not falsely claimed
inside source01. Cached official affinity and confidence assets already exist.
Observed home13.79/30GiB and work3pool6 89.07/100GiB; per-job gates remain.

Actual submissions: CPU readiness29758624(hpc), native readiness29758626(gpua10,
dependency done(29758624)), and allocated submission controller29758628(hpc,
dependency done(29758626)). All initially PEND. Controller must authenticate
actual native DONE, passed persisted assessment and every proof before bsub.
The two long trajectories are NOT yet submitted. Their unique future root is
/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-01-native/.
Compact receipt: scripts/hpc/receipts/20261019-brd4-boltz-affinity-01/campaign.json.


CPU29758624 remains RUN: the complete legacy/JAX bridge phase passed220tests,
three deliberately deselected model/slow tests, no failures/errors. Two existing
JAX scatter dtype future warnings were retained. Native Torch mapping and
source-equivalent chemistry phases follow; this prefix is not full readiness.


CPU29758624 DONE/passed283checks:220legacy/bridge(185.24s),4native mapping
(5.75s),59chemistry/oracle regressions(107.02s). No failures/errors/skips;
three model/slow tests deliberately deselected in the legacy selection. All
three XML files and exact composition reference were downloaded and checked by
local stdlib byte/XML inspection only. CPU assessment SHA 4d64197d56a2ba964ad1ae4ecf2b9e62e5e3ed2c4d14c247bae0a139cb79343b.
Compact local proof results/hpc/20261019-brd4-boltz-affinity/readiness-01/verification.json.
Native29758626 remains PEND on a real host slot limit; controller29758628 PEND.
No long native arm has been submitted. Code pushed as ecaf328. A quiet hourly
follow-up checks actual native readiness/controller outcomes, fixes genuine
integration bugs with new sealed allocated tests, and records actual long IDs
once gates pass; it must not reinterpret failed native checks as passes.

## User-requested GPU queue amendment, 19 October 2026

The user asked to use a less congested GPU queue to start native tests. At
approximately 19:02 Europe/Paris, the scheduler exposed two A10 GPUs and 21
waiting jobs in `gpua10`, versus twelve L40S GPUs and 59 waiting jobs in
`gpul40s`. These are job counts, not comparable CPU-slot PEND counts or a
start-time forecast. L40S has fewer waiting jobs per GPU in that snapshot;
neither that ratio nor the larger pool guarantees an earlier start. H100
queues require special groups without established account authorization, so
no access attempt was made.

DTU `eswitch` rejected an in-place `bswitch` with a permission denial. This
transport/control outcome is preserved; it is not a scientific failure.
Scheduler state and absent payload/long submission records established that
old readiness **29758626** and controller **29758628** were still unstarted
(PEND/PSUSP). Both were intentionally cancelled and their EXIT outcomes
confirmed before replacement. No molecular experiment was interrupted.

Replacement native readiness **29758990** is PEND in `gpul40s`: one exclusive
L40S 48 GB GPU, four CPU slots, 8 GiB per slot, two hours. It uses the identical
source01 scientific code/configuration and manifest
`8a621a975cf98800ea3a2a7413f1e07b1f317b988c02d69b593832a7d1fd94af`.
All preparation, native parity, finite-difference, coupled chemistry, pilot,
runtime, finite-value, frozen-identity and quota gates remain mandatory.
New readiness output:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-queue-02-ready/payload/`.

Only external submission tooling changed. Immutable launcher03 is
`/work3/s234818/mosaic/releases/20261019-brd4-boltz-launch-03/source`, manifest
`60924c49eb9cbe3c8741afb2f0b152e6ed2525f542f610310bbcaf59a3db5d85`,
archive 5,100 bytes, SHA
`4453763533a0b323c2dc50f3930b3673d20a8b2e66a9fbd285989d66ee65cde5`.
It contains the controller, launcher tests and two batch wrappers; scientific
source01 remains unchanged and is still the code executed by native jobs.

Launcher-only CPU **29758998** exited before tests because LSF split an inline
`Python -c` command. Its failed logs remain preserved; no native test ran.
The wrapper-based retry **29759003** actually DONE/passed eight unit tests,
zero failures/errors/skips at 19:07:12 Europe/Paris. Assessment SHA
`f3e6c52ee9c73d6770c8c05179e377cce7f4e2112b0aa5f4ceaf9b40d000c1f9`.
The earlier 283-test scientific CPU result remains valid. Launcher output:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-queue-03-cpu/`.

Replacement controller **29759027** is PEND in `hpc`, waiting on
`done(29759003)&&done(29758990)`. It uses the authenticated batch wrapper and
requires actual native DONE, matching queue `gpul40s`, all passed persisted
proofs and fresh quota before submitting either long arm. Both future arms
will request the validated L40S queue, one exclusive GPU, four slots, 8 GiB per
slot and 23 hours, independently. The 23-hour request is an allocation limit,
not a measured completion estimate. Their unchanged future root is
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-01-native/`.
Controller output:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-queue-03-controller/`.
Compact local resource-change evidence belongs under
`results/hpc/20261019-brd4-boltz-affinity/queue-change-02/`.
Its download completed: stdlib verification checked all 23 files, exact byte
lengths and SHA-256 hashes. Downloaded manifest SHA
`e1f65ded65a82cbb6c77431a702b305a2d1f24a8bc8e7333f3561671c47d575d`;
`verification.json` records passed. At 19:08:37 Europe/Paris,
29758990 remained PEND with `Job's requirements for resource reservation not
satisfied (Resource: ngpus_physical): 3 hosts`; this is an ordinary GPU
reservation wait. Controller29759027 remained PEND on its dependency.

Final scheduler check at 19:10 Europe/Paris: native29758990 still PEND, with
four CPU slots already reserved on n-62-13-14; it is waiting for a physical
GPU. Recursive host-group inspection confirms six eligible L40S nodes
(n-62-13-14..19), two GPUs each, versus one A10 node. Controller29759027
remains PEND on native completion. No start or completion ETA is established.
