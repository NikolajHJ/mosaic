# Matched latent optimization: learned embeddings versus relaxed chemistry

## Attempt003 CPU readiness passed; native comparison GPU queued

The [allocated CPU terminal snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1792784006917505500.json)
records **29802735 DONE**, **23 October 2026 21:32:09–21:32:49 DTU-local**,
n-62-31-20. **All 212 fixtures passed**, zero failures/errors/skips, JUnit
25.020s; the complete CPU/model/start stage took **32.1028s** and returned
`passed`. Strict original36 loading, bitwise ordinary/capture decoder parity,
unchanged frozen state and actual shared-start preparation passed. This is
allocated CPU readiness; it does not pass native GPU derivatives or molecular
optimization.

The prescribed starting samples were **indices 0, 3 and 6**, the first audit
posterior-mean groups with distinct decoded identities in authenticated stored
order. Their occupied counts are **25, 19 and 19**. Each pair shares the same
saved latent and bound native context; no start was replaced according to a
score or gradient. The immutable `starts.pt` SHA256 is
`832592de7183b51ac22542ead558bc453006d3d349d942e2933dd1a52fb40ebd`;
CPU JUnit SHA256 is
`613c8e0b193e1e6ef7907615cd93f198a15417497f2fec3e30818f82606ec3b1`.

The [actual GPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt003/submit-gpu-1792784164515503800.json)
returned zero, **29802743 submitted at 21:36:07 DTU-local**,
`submitted_not_passed`: `gpua100`, **one exclusive 80GB A100/four CPU slots**,
`rusage[mem=16GB]`, **24-hour walltime and 23-hour shared internal cap**.
Fresh submission quota was **64.80/100 GiB used, 35.20 GiB free**.

The [saved 21:37:42 queue snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1792784262997389600.json)
confirms **29802743 PEND**, job-slot limits on **five hosts**, eligible pending
time **96s**, ineligible **0s**. It supersedes the initial
[21:36:20 new-job wait](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt003/status-1792784180809306300.json).
No GPU logs/result artifacts, native derivative
pass, trajectory start or fresh native outcome exists in this observation.
There is no reliable start estimate. Submission leaves the VAE continuation
unchanged; its similarly numbered independent jobs are a different campaign.

## Exact context bank restored; unchanged attempt003 CPU submitted

The [guarded remote restoration completion](../outputs/hpc-submissions/source138-context-restore-20261023/completion.json)
at **23 October 2026 21:30:43 DTU-local** records `exact_context_bank_restored`:
all **44 cold-offloaded source056 prepared contexts**, **1,451,429,736 raw
bytes**, were restored to their original paths with exact recorded sizes and
SHA256. The [transport/restoration receipt](../outputs/hpc-submissions/source138-context-restore-20261023/remote-restoration.json)
and durable intent/completion preserve the operation. No model was regenerated,
job or setting changed, checkpoint written or original cold archive modified.

The [independent read-only verification](../outputs/hpc-submissions/source138-context-restore-20261023/independent-verification.json)
at **21:31:52 DTU-local** rehashed **all 50 catalogued contexts**, including the
six contexts that had remained online, and passed. Original epoch36, trained
bridge and source136 immutable archive hashes matched before and after
restoration and in this independent check. Immediate post-restoration quota
was **64.41/100 GiB used, 35.59 GiB free**; the later submission snapshot is
different and includes concurrent storage activity.

New independent
`source138-epoch36-embedding-comparison-attempt003`
[seal](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt003/seal.json)
contains **582 members**, with **the exact same member-manifest and comparison
plan hashes as attempt002**. No source change was needed to restore the missing
inputs. Its upload/authenticated staging and
[actual CPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt003/submit-cpu-1792783925232386600.json)
returned zero, submitting **CPU 29802735**, four slots, `rusage[mem=8GB]`,
one host, 45-minute walltime, `submitted_not_passed`. Fresh submission quota was
**64.17/100 GiB used, 35.83 GiB free**. The submission alone establishes no
CPU pass, GPU job, derivative result or optimization start. Both failed earlier
attempts retain their sealed bytes and outcomes.

| Attempt003 binding | SHA256 |
| --- | --- |
| Source archive | `bccba5d6a762349b450baf03b24251e12994be598926ce8960f6e5869d55ce9b` |
| Unchanged release manifest | `4a598d1143f6afc180d18dea618cdc09a859f019a7fc042b9025fcb8fb8487dd` |
| Unchanged comparison plan | `60f555eb6193b804d8446dc864a57c52662e4c146be424b8f7cc118f6d6ecd6c` |

## Attempt002 fixtures passed; start preparation requires exact context restoration

The [attempt002 terminal snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt002/status-1792783426704147500.json)
records **CPU 29802683 EXIT, exit 1**, **23 October 2026 21:22:38–21:23:17
DTU-local**, n-62-31-16. Its numerical fixture gate **passed all 212 cases**,
zero failures/errors/skips, pytest 25.64s/JUnit 25.552s. Strict loading of
authenticated original epoch36, bitwise ordinary/capture decoder parity and
unchanged frozen tensor state also passed. JUnit SHA256 is
`3216abfe7fe65bbe69a2171afd4f88376da054941f2c930a6e7b1c9531fa3991`.

The **whole CPU stage did not pass**: actual start preparation then raised
`FileNotFoundError` for historical source056
`contexts/count-19/prepared.pt`. No shared-start artifact or GPU submission
followed. This is a missing cold-offloaded input, not a failing fixture, native
gradient or new chemical outcome. Preserve the complete failed attempt.

The [verified 21 October cold-offload restoration map](../outputs/hpc-results/storage-cleanup-20261021-001/cold2-restoration-map.json)
contains the exact missing context and the other 43 offloaded source056 count
contexts. Its original archive is preserved locally at
`outputs/hpc-results/storage-cleanup-20261021-001/cold2-payloads.tar.gz`, SHA256
`ec6797d064f5219b1279ea12b92a737a314bbfd02ebd8c3be25c690968ce866e`.
Restoration preparation uses standard-library byte extraction and verification
for all **44 source056 prepared contexts**, without regenerating models,
altering the prescribed starts or selecting only favorable counts. Exact
restoration evidence is recorded separately when complete; no restoration or
replacement allocation follows merely from locating that archive.

The later [local extraction proof](../outputs/hpc-submissions/source138-context-restore-20261023/local-extraction-proof.json)
now establishes **all44 exact source056 contexts** extracted/authenticated
against the cold-offload map, **1,451,429,736 raw bytes**. The original cold
archive hash matched before and after extraction and remains unchanged. The new
restore capsule has exactly **45 regular members**: those44 files under their
original paths relative to `/work3/s234818/latentmol`, plus its manifest. No
checkpoint or other cold payload is included; no model was imported and no
remote path or job changed during this preparation.

| Restoration preparation artifact | SHA256 |
| --- | --- |
| [18,476,907-byte restore capsule](../outputs/hpc-submissions/source138-context-restore-20261023/source056-prepared-contexts-exact-restore.tar.gz) | `ef82e950d0f1bb426d098592ba7d4b7a0584ef75fb8029babe93a57690c50d8b` |
| [Exact path/member manifest](../outputs/hpc-submissions/source138-context-restore-20261023/restore-manifest.json) | `60765caddc95f881ed2207b12ac8ed69398f9b47c3635d7c71eae08cd32938e1` |

This is a verified local restoration package. Guarded upload/remote restoration,
fresh quota and an independent allocated retry must be established by their own
receipts before any new readiness or optimization claim.

Attempt002's [seal](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt002/seal.json)
contains **582 members**; upload/staging and its
[CPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt002/submit-cpu-1792783353627483400.json)
returned zero. Fresh submission quota was **63.07/100 GiB used, 36.93 GiB free**.

| Attempt002 binding | SHA256 |
| --- | --- |
| Source archive | `b3041c26afa15f10983c9746813885e72b0e1bbc35326c7f85d5002b3ca07d06` |
| Release manifest | `4a598d1143f6afc180d18dea618cdc09a859f019a7fc042b9025fcb8fb8487dd` |
| Unchanged comparison plan | `60f555eb6193b804d8446dc864a57c52662e4c146be424b8f7cc118f6d6ecd6c` |

Release and campaign are the attempt001 paths below with terminal
`attempt002`. All frozen VAE/bridge/Boltz bindings and independent continuation
remain unchanged. A future independent attempt must complete CPU/model/start
preparation and all native derivative checks before matched trajectories.

## Attempt001 failed two fixture hash-path calls; evidence preserved

The [attempt001 terminal snapshot](../outputs/hpc-results/source138-epoch36-embedding-comparison-attempt001/status-1792783307992603600.json)
supersedes its initial submission-only state: **CPU 29802658 EXIT, exit 1**,
**21:20:59–21:21:33 DTU-local**, n-62-30-24. Of **212 cases**, **210 passed
and two failed**, zero errors/skips, pytest 24.49s/JUnit 24.387s. Both failures
were fixture calls in `tests/test_embedding_comparison.py` that passed strings
to the existing `file_sha256(Path)` helper while checking saved checkpoint
hashes. The gate stopped before original-model/start preparation; no attempt001
GPU was submitted. No failure is relabelled as passed.

The comparison-specific correction imports `Path` and wraps those two fixture
arguments. A [static sealed-manifest comparison](../outputs/implementation-checks/source138-embedding-attempt001-002-manifest-delta.json)
confirms all `src/` members and the comparison plan are identical, with no
added/removed members. The two changed archive members are the corrected test
and `scripts/hpc/check_chemistry_launch.py`, the independently edited VAE
startup-correction helper that this comparison runner/operator does not call.
The comparison's production code, scientific settings and gates are unchanged;
this byte evidence is separate from allocated numerical validation.

## Attempt001 sealed, authenticated and CPU submitted

On **23 October 2026**, the independent
`source138-epoch36-embedding-comparison-attempt001`
[seal](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt001/seal.json)
packaged **582 authenticated members**. The
[upload](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt001/upload.json)
and [staging](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt001/stage.json)
returned zero; staging reports `staged_authenticated`.

The [actual CPU submission](../outputs/hpc-submissions/source138-epoch36-embedding-comparison-attempt001/submit-cpu-1792783254961755100.json)
returned zero and submitted **29802658** to `hpc`: **four CPU slots**, declared
`rusage[mem=8GB]`, single host and **45-minute walltime**. Its saved state is
`submitted_not_passed`. Fresh submission quota was **62.55/100 GiB used,
37.45 GiB free**. No CPU result, GPU submission, derivative pass, trajectory
update or fresh native outcome is established by this submission receipt.

| Attempt001 binding | SHA256 |
| --- | --- |
| Source archive | `e4764c6c6fb3444fd5032285078f5d96f4daa8c8547a5aae84dd5fedf7b69da7` |
| Release manifest | `c239d5fecc97133ea5c78aa8a7e81ac7ebab63acf238adfb7c4dc8625f440303` |
| Comparison plan | `60f555eb6193b804d8446dc864a57c52662e4c146be424b8f7cc118f6d6ecd6c` |

Release:
`/work3/s234818/latentmol/releases/source-138-epoch36-embedding-comparison-attempt001`.
Campaign:
`/work3/s234818/latentmol/runs/source138-epoch36-embedding-comparison-attempt001`.
The exact operator command binds archive/plan hashes, release and campaign;
its allocated wrapper uses `env/boltz311` through `scripts/hpc/boltz_run.sh`.
Numerical CPU tests and actual original-36 starting-data reconstruction must
pass before any dependent GPU stage. The GPU request is prospectively bounded
to **one exclusive 80GB A100/four slots, 24-hour walltime/23-hour shared internal
cap**; it is not submitted by the CPU receipt.

The source137 mapping pilot, earlier attempts and source136 VAE continuation
remain unchanged. This embedding-comparison namespace is independent of any
separately authorized VAE startup-correction candidate with a similar source
number; full release/campaign identities distinguish the studies.

## Scope and current status

On **23 October 2026**, the user authorized a matched latent-optimization test
of the completed embedding mapping against the original relaxed-molecule path,
using the **same frozen original epoch-36 VAE and identical starting latents**.
The existing source136 VAE continuation, jobs, settings, checkpoint lineage and
immutable release remain unchanged. This is an independent experiment on
`codex/epoch36-boltz-embedding-bridge`.

The implementation, allocated CPU readiness and GPU submission are established
above. The rest of this record defines the prospective scientific protocol;
GPU submission does not establish derivative, trajectory, native-score or
comparative success. The sealed plan and allocated receipts are authoritative
about what actually executes.

The preceding [mapping pilot](76_epoch36_boltz_embedding_bridge.md) completed
1,000 supervised updates and its native exact-oracle injection check. That
result is the prerequisite artifact, rather than evidence that the learned
gradient improves decoded molecules. Audit examples are correlated and charged
categories were uncovered; this comparison must retain those limitations.

| Frozen artifact | SHA256 |
| --- | --- |
| Original epoch-36 VAE | `3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f` |
| Trained mapping checkpoint | `013834958d52f7aee999ca4c39496e14fc5333d69eb697d47433821f48290343` |
| Boltz affinity checkpoint | `dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e` |
| Boltz structure/confidence checkpoint | `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1` |
| Pilot's prepared 13-slot reference complex | `8b6e47fdf5a53a8bd92c5ebf0baf7471ca6a329bbc524e9957a3e94a9938f611` |

No VAE, bridge, projection bank or Boltz parameter is trainable during this
test. The mapping is loaded strictly and authenticated against the completed
pilot. It receives no additional reward fitting or supervised refinement.

## What the paired arms compare

Both arms minimize the same declared full DBMol scalar:

`affinity loss + 2.5 × contact loss + 2.5 × confidence loss`.

The [loss reference](77_dbmol_boltz_loss_implementation.md) defines the native
outputs, reductions, two checkpoint-specific trunks and float64 contact/loss
arithmetic. Within each case, identical pose, receptor/MSA context, pocket
residues, atom/token maps, temperature, occupancy, checkpoint precision and
objective weights are used for the paired arms. Cases can have different
occupied atom counts; each uses its own authenticated count-specific prepared
complex and the same fixed slot binding in both arms.

The **relaxed arm** follows the existing frozen decoder's atom/charge/bond
probabilities through the deterministic native-feature adapter. The **learned
arm** maps the same decoder's final hidden states into the corresponding
checkpoint-specific early chemical projections. The ordinary molecular heads
still define the learned arm's decoded hard identity. This is a comparison of
two continuous optimization surfaces associated with the same decoded graph
convention; the learned embedding need not equal an exact native molecule at
intermediate continuous points.

Occupancy and pose are fixed. No atom birth/death, latent inversion, molecule
repair, DeFoG guidance or pose gradient is part of this comparison. A fixed-slot
conditioned hard graph and its ordinary unconstrained hard decode must both be
recorded, so a changed natural occupancy cannot be hidden by the scorer's mask.
Unsupported, invalid, unchanged and charged outcomes remain in the denominator.

## Starts, update budget and step receipts

Select the **first three audit source groups with distinct decoded identities**
whose accepted saved sample has origin `posterior_mean` in the authenticated
source137 paired data, using deterministic stored order. Save their existing latent tensors once
before either arm runs, with their group/sample identities, source split,
hard labels and tensor hashes. Do not regenerate them, add perturbations,
choose a replacement according to a model score or screen for a favorable
gradient. The audit groups were held out from mapping fitting/development
selection; this does not make them independent of VAE training.

Both arms consume exact copies of these tensors, with the same dtype and
tensor hash, and request **50 updates per start**. The common settings are
learning rate **400**, maximum gradient norm **1**, trust radius **3**,
temperature **1** and `best_trial_v2` with **16 trials**. These are a bounded
paired pilot, not an estimate of general molecular-design performance. The
sealed configuration records the exact update/evaluation budgets.

A saturated or nearly flat relaxed path is a measured outcome. Starts cannot
be removed, replaced or selected according to which arm has a larger gradient
or lower Boltz score. Initial gradient norms and directions, latent movement,
hard identity changes and actual oracle cost make this outcome interpretable.

Both arms use the same declared learning rate, trust radius and
[`best_trial_v2`](62_full_budget_best_trial_optimization.md). Each state first
tries the full proposal and applies it on observed nonincrease, including
equality. Otherwise it halves through at most 16 sizes and takes the first
strict improvement, or the lowest-loss finite trial if no size improves. Such
a fallback can increase the loss. Ordinary failure to descend does not end the
requested budget. A finite zero/no-op proposal can consume an update and must
remain visible.

Save each applied state and all trial receipts. Count attempted/applied,
strictly improving, full-step equality/nonincrease, forced fallback and no-op
updates separately. Retain the terminal state and best observed state under
each arm's own scalar with their exact objective/context identities. An arm's
best learned score and another arm's best relaxed score are not interchangeable
ranking criteria.

## Derivatives and cross-scoring

Fresh allocated checks authenticate all frozen files and contexts before any
trajectory. The existing native exact-oracle injection result does not certify
the learned mapping's latent derivative. New checks must establish the learned
graph's attachment, finite supported gradients and declared multi-scale
finite-difference behavior, with observed native repeat precision retained.
Failed gates stop dependent trajectories and remain scientific outcomes.

At the predeclared checkpoints, save each continuous arm's own loss, all three
DBMol components and native outputs, latent displacement, conditioned/natural
hard atom/charge/bond tensors, validity and canonical identity. Saved `z` and
the authenticated frozen model reproduce decoder probabilities; those
probability tensors are not themselves stored by this audit. Evaluate
both continuous paths at the same saved latent when reporting gradient or loss
differences; paired starts alone do not make the later states identical.

Every valid selected hard graph receives a common fixed-context discrete
evaluation, independent of the learned embedding. Its score is shared across
arms when the slot-labelled graph and prepared context are identical. An
improved continuous objective with an unchanged or invalid hard graph is not
evidence of improved molecular design. Report molecules and starts, in addition
to correlated checkpoints and directed atom/pair counts.

## Fixed-context hard scores and fresh native confirmation

The implemented hard score adapts `hard.to_relaxed(...)` through `adapt_graph`
and the same frozen scorer at the exact held pose, without learned embedding
injection. The existing `FrozenBoltzDBMol.discrete_reference` helper supplies
the native integer trunk/head reference used in interface validation; the
comparison's hard audit does not call that helper. Both routes inherit the
prepared atom names, reference positions, crop, masks and other context. Label
the implemented result **fixed-context hard projection**, not independent
native re-featurization of a changed molecule.

The existing `fresh_trajectory.execute_request` offers a separate ordinary
native confirmation route. It writes a fresh input YAML from each candidate's
canonical SMILES, preserving the declared protein/MSA; calls the pinned Boltz
CLI with predeclared seeds/settings into a new directory; then executes
`processed_identity.audit_processed`. That audit compares heavy-atom identity,
per-atom charges and typed bonds, complete ligand coverage, native generated
atom names/features, target and MSA linkage, and the affinity artifact's record
association. Native standardization that changes the molecule is an ineligible
result, with its reason retained.

This native CLI route generates a **new pose**, reference conformer and
chemistry-dependent native fields. It is a fresh-pose affinity follow-up, not
the same fixed-pose full DBMol objective and not a derivative through diffusion.
Its native affinity schedule can differ from the optimization schedule only
when that distinction is declared and applied equally to all candidates.
Fresh structure/confidence predictions alone do not reconstruct the selected
fixed-pocket contact objective.

Freeze candidate selection before inspecting any native outcome. Include each
paired original start, both terminal states and both best-own-objective states:
**five memberships per start, 15 total**. Deduplicate identical strict-valid
canonical molecules while retaining every arm/start/state membership. Each
unique valid identity receives pose seeds **17 and 23**, for at most **30 native
requests**. Invalid states retain their failure reasons and receive no disguised
replacement or repair. This initial requested comparison has no random-control
arm; conclusions concern the two declared surfaces and their shared starts.

The native schedule is 100 structure and affinity sampling steps, one diffusion
sample each, zero structure recycles, native affinity defaults of five recycles
and `bf16-mixed`, 128 MSA rows, maximum MSA parse length 8192, and no molecular
weight correction. These are applied equally to every request. Save all
preprocessing, prediction, identity and timeout failures, hashes, costs and
incomplete pairs. Repeated identities reuse the same native request rather
than receiving extra favorable random draws.

The authenticated native template is
`/work3/s234818/latentmol/boltz-preparation/source054-active-set-attempt001/count-13/input/carbon_chain_13.yaml`,
SHA256 `ff3f32119c9f6a51b37169306de81e43e60327797e79ae7a53ef90ee5a445117`.
Its MSA is `seq1.a3m` in the same input directory, SHA256
`d98f3b09f429e4fb087d67ca3b89f00c3165553444ef221ead6ed2cc01488bd4`.
The protein sequence SHA256 is
`978a204bab1837aa3304a838a621e6ef24fb4e878395db76c9c9ed095ce25f48`.
A [read-only standard-library SSH check](../outputs/implementation-checks/embedding-comparison-native-source-readonly-20261023.json)
confirmed those available bytes while
preparing this comparison; no model or native preprocessing ran on the login
node. The original C13 ligand entry is replaced by each exact candidate SMILES
before ordinary native parsing.

The implicit native affinity recycle/precision/MW settings were checked against
the installed `boltz/main.py`, SHA256
`b5ded4dac1552af2e079c335d56e64ea580b4d5ca0202a9542385ce3e318c3b3`:
lines 1372–1375 set five affinity recycles; line 1262 selects `bf16-mixed` for
Boltz2; and the affinity MW flag defaults false and is passed into model loading.
The new follow-up checks these source bytes and records their provenance before
requests, so an unexposed default is bound to actual native code.

`embedding_comparison_rescore.freeze_selection` freezes the complete role
denominator and deduplicated request list. `load_target` checks the template
against the prepared target/MSA identities. `run_followup` executes serial
ordinary native requests only inside a sealed allocation and records partial
execution if insufficient shared time remains. An internally corrupt native
identity audit stops the stage; an ordinary native prediction failure remains
a terminal failed request. A paired gain is eligible only when start and
candidate both pass the actual processed-input identity audit at the same seed.

Success in this small study could establish a useful local optimization result
and whether it survives ordinary native rebuilding. It cannot establish general
bridge accuracy, chemical synthesizability, experimental affinity or superiority
across targets and latent starts.

## Execution and preservation

All numerical tests, native preprocessing, derivative evaluations, optimization
and native predictions run inside allocated DTU HPC jobs. Local preparation is
limited to source edits and standard-library syntax/byte/provenance work. New
immutable attempts retain exact source, manifest, plan, model, bridge, prepared
context, start tensors and result hashes. A queued allocation is not a passed
gate or an optimization start; partial runtime results are not completed paired
budgets.

Use serial bounded stages with no allocation beyond 24 hours and no internal
budget beyond 23 hours, preserving fresh quota checks and remaining-time
headroom before child processes. No historical sealed source, failed decision,
mapping-pilot record or independent source136 continuation artifact is rewritten.
