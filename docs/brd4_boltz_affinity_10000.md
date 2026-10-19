# BRD4 Boltz-2 affinity with delayed chemistry

19 October 2026. User authorized implementation, allocated HPC validation and,
only after passing, two independent fresh 10,000-update runs: 50% chain/uniform
and seeded random probability initialization. Implementation is being validated;
submission or scheduler success alone does not certify numerical readiness.

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

Initial resources: CPU readiness hpc4slots/4GiB per slot/1h; native readiness
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
