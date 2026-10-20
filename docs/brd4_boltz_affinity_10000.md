# BRD4 Boltz-2 affinity with delayed chemistry

19 October 2026. User authorized implementation, allocated HPC validation and,
only after passing, two independent fresh 10,000-update runs: 50% chain/uniform
and seeded random probability initialization. Implementation is being validated;
submission or scheduler success alone does not certify numerical readiness.

**Current status, 20 October 2026, 17:25 Europe/Paris:** SSH access restored
at 17:15:21. **Mix50 run 29759212 DONE**, with actual LSF successful
completion from 19 October 20:41:13 to 20 October 08:35:34 (42,861 seconds).
Completion metadata is now authenticated: **10,000 contiguous updates
0..9999 and 10,001 contiguous saved states 0..10000**, with matching update
JSONL, case report and summary case records. Initial/coupled/terminal checks
passed, final chemistry weight is 50 and original frozen model/context
digests are retained. Local compact evidence verification passed 21 files,
including snapshot bytes at 0/5000/10000. This does not verify all 10,001
snapshot files or a full rendered report.

**Replacement random diagnostic 29763693 PEND**, actually submitted at
17:25:41 with verified gpul40s/one exclusive L40S/four CPU slots/8 GiB per
slot/60min and no inter-arm dependency. Launcher05 CPU 29763691 passed 20
contracts and its actual outcome/proof bytes are verified. Original
29759213 remains EXIT1 before model execution from aged-out live CPU
lookup; earlier failures are preserved. Source02/init/FD gates unchanged.
No new native diagnostic result; no long random optimization submitted.
Tools02 CPU 29759211's 19-test pass, source02 CPU 29759068's 299-test pass and
the adopted mix50 proof remain valid. Parent 29759069 stays EXIT1. Historical
records below remain preserved; latest sealed scientific campaign receipt:
`scripts/hpc/receipts/20261019-brd4-boltz-affinity-02/campaign.json`.

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

## Native preparation error and source02 preparation, 19 October 2026

Authoritative later scheduler evidence records readiness29758990 starting
at **19:09:51 Europe/Paris**, on L40S host n-62-13-19. This supersedes the
approximate 19:10 PEND annotation above as a statement of current status.
All 386 source files and the 10 GiB fresh-home-quota gate passed. Native
prediction detected its actual GPU and finished preprocessing, then Python
multiprocessing `queues`/`resource_sharer` raised `OSError: AF_UNIX path too
long`. The batch script's home-run `TMPDIR` was too long for the Unix socket.
The diagnostic is in `prepared/prediction.log`; outer `prepare.log` was empty
because the child redirected its output. No `context.pt`, affinity finite
differences, coupled checks or disposable native pilots were reached. An LSF
power/clock warning explicitly states that the GPU can still perform the
workload; it is not the identified preparation bug.

Captured failure-stage evidence, 17:22:08 UTC / **19:22:08 Europe/Paris**:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-tmpdir-failure-01`,
manifest SHA
`ee6f23873d7f3026caa7504e3ed77c73cb0e33342776815302dad0ac873dea5f`.
Readiness29758990 remains RUN and is not being cancelled; its original
two-hour allocation bounds this errored preparation stage. Record its actual
terminal scheduler outcome later, rather than treating the observed error as
an already-final EXIT. Controller29759027 was verified unstarted PEND and its
cancellation requested; actual final cancellation receipt remains pending.

The source02 fix being prepared introduces a shared helper for a unique,
node-local `/tmp` directory and safe cleanup of that job's owned directory.
Both readiness and long-run launchers use it. Scientific `src/`, `examples/`
and configuration bytes are unchanged and must be authenticated against
source01. No model, objective, initialization or numerical gate is relaxed.
The full allocated CPU phase is planned to run **299 tests**: the original
283 plus eight controller tests and eight TMPDIR/socket/cleanup contracts.
That total is planned coverage, not an executed or passed result. At this
preparation stage, source02 hashes, new job IDs and outcomes were still pending;
the subsequent seal is recorded below. Its native retry will require
the new CPU pass **and `ended(29758990)`**, avoiding duplicate concurrent GPU
validation while preserving the already running allocation and all evidence.

Source02 is now sealed locally: **393 files**, archive **39,241,289 bytes**,
SHA `11ccbe5563bebb3184183eaab194f791ae717c906d8557a3acd11fc0c8c685dc`,
manifest SHA
`5eb6e17ea8d19e52c99a589243228710994dcf137f59a6b9a6659a4fe719a7f4`.
Its immutable destination is
`/work3/s234818/mosaic/releases/20261019-brd4-boltz-affinity-02/source`;
upload is in progress, not yet a completed remote authentication or submission.
Static comparison authenticated the entire 180-file map under `src/` and
`examples/`, including configurations, as exactly source01. Changes are batch
temporary-directory handling and CPU coverage plumbing only. This receipt is
an external post-packaging record; it is not claimed inside the sealed archive.
Future unique run directories are
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-02-{cpu,ready,controller,native}`.
The 299-test CPU rerun is still pending, with no new job IDs or pass claimed.
The preserved preparation-error download has also completed: all **12 files**
passed local stdlib length/SHA checks at
`results/hpc/20261019-brd4-boltz-affinity/tmpdir-failure-01/verification.json`.

## Source02 authenticated and actually submitted

Remote authentication subsequently passed all 393 source files, the exact
archive/manifest identities above and the unchanged entire 180-file scientific
map against source01. Actual submissions at **19:25:27.988 Europe/Paris**:

- CPU **29759068 RUN/hpc**, four slots, 4 GiB per slot, one hour; planned
  299 checks, not yet passed.
- Native retry **29759069 PEND/gpul40s**, one exclusive L40S GPU, four slots,
  8 GiB per slot, two hours; dependency
  `done(29759068)&&ended(29758990)`.
- Controller **29759070 PEND/hpc**, dependency
  `done(29759069)&&done(29759003)`. It uses unchanged passed launcher03 and
  its eight-test proof with scientific source02 and the new source02 CPU proof.

Run directories use the source02 family already recorded above. Current
native payload is
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-02-ready/payload/`.
Remote compact receipt:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-affinity-02-campaign.json`.
The source02 `-native` long-arm root is still virgin; neither arm submitted.
All native/proof/identity/finite/runtime/quota gates remain unchanged.

Old controller29759027's intentional cancellation while unstarted is now
authenticated EXIT. Predecessor29758990 remains RUN with the known
preparation error and is not cancelled. Its original two-hour allocation
would reach its limit at approximately **21:09:51 Europe/Paris**; that is an
allocation bound, not a promised terminal time or retry/completion ETA.
The ended dependency prevents simultaneous duplicate native validation.

## Source02 CPU completion and verified evidence

CPU **29759068 DONE**, actual LSF `Successfully completed` and scheduler
success authenticated at **19:29:12 Europe/Paris**. **299 tests passed**:
16 launcher/TMPDIR contracts, 220 legacy/JAX bridge, four native Torch mapping
and 59 unchanged chemistry/oracle checks. Zero failures/errors/skips; three
slow/model legacy tests were deliberately deselected. Assessment SHA
`fadc01aec4cafa54205e5d49ecd764da1aa48ddb841368c0845dac7f0ddb6c6e`.
This completes the latest CPU readiness and does not certify native affinity
parity, finite differences or pilots.

Local stdlib verification safely extracted the 13-file CPU proof archive and
checked every path/length/hash, XML counts and unchanged composition reference:
`results/hpc/20261019-brd4-boltz-affinity/retry-02/cpu-final/verification.json`.
Proof archive 12,878 bytes, SHA
`1d47c5f7c3f577b583e9acde2ad8819ebeb042e185d53cb9c11c57250076d770`;
manifest SHA
`e570b0e37f98e229fcbe95256646c833dd04fa34b59499d7e7fecccd85532553`.
The earlier 12-file control/submission proof was separately verified at
`results/hpc/20261019-brd4-boltz-affinity/retry-02/controls-verification.json`:
archive 4,828 bytes, SHA
`36836d2f0b83a2589fbc1c299a7bf48a2f6788f67eec02913bccc4b0fa80ae4f`,
manifest SHA
`fe2cd588caf1a6bbeb851973534a8751ce53a74c08ac0b05f0cf83e46ea57183`.

Latest native29759069 remains PEND until ended(29758990); controller29759070
remains PEND. Neither long arm is submitted and no native gate is claimed
passed. The predecessor's unchanged two-hour allocation bound remains
21:09:51 Europe/Paris, not a completion ETA. Independent static review found
no blocking issue in the temporary-directory or launcher changes.

## User-authorized cancellation and corrected native start

The user explicitly requested cancelling the errored original validation
29758990 so corrected29759069 could start without waiting for the old timeout.
After authenticating its RUN state, known AF_UNIX diagnostic and absent
assessment, `bkill` was requested at **20:03:16.509435 Europe/Paris**
(18:03:16.509435 UTC). The scheduler first acknowledged asynchronous
termination; its immediate RUN response was not falsely called final EXIT.
Actual **29758990 EXIT** was confirmed at **20:03:43.123083 Europe/Paris**
(18:03:43.123083 UTC). This explicit authorization applies only to this
errored predecessor; historical leave-untouched notes describe the policy
before that instruction and remain preserved.

Corrected **29759069 RUN/gpul40s**, scheduler start **20:03:27 Europe/Paris**,
on n-62-13-19, one exclusive L40S GPU, four slots, 8 GiB per slot, two hours.
The `ended(29758990)` dependency was fulfilled; the corrected job did not wait
for the prior allocation's 21:09:51 limit. Controller29759070 remains PEND
on actual native success. All original failed-preparation/source evidence and
the source02 299-test CPU pass remain unchanged. No native parity/FD/pilot
gate has yet passed, and neither long optimization arm has been submitted.

Remote cancellation evidence:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-tmpdir-cancel-01/`,
including before/request/cancel records and `completion.json`; manifest SHA
`e5104969dc191a6d134a5141917bf51f9c7228b703d10a2cf333ef4c8a6d6f15`.
Local cancellation proof download completed: all **seven files** passed
stdlib length/SHA verification at
`results/hpc/20261019-brd4-boltz-affinity/tmpdir-cancel-01/verification.json`,
with the exact manifest SHA above. Corrected29759069 startup uses the actual
short node-local directory `/tmp/mosaic-boltz.29759069.szmVwn`; native CLI
detected its GPU, completed preprocessing and is running structure prediction.
This is preparation progress, not a passed native affinity gate. Quiet
follow-up is active with the confirmed cancellation and current RUN state.

## Source02 native outcome: random typed-bond derivative gate failed

Readiness **29759069 EXIT1**, actual scheduler outcome **20:09:29
Europe/Paris**, from start20:03:27, elapsed362 seconds. The short node-local
TMPDIR fix worked: native structure and affinity prediction reported zero
failed examples and produced the prepared context. This is a new native
numerical-gate failure, distinct from the preserved source01 AF_UNIX error.

The **mix50** validation summary passed: exact hard/native integer parity
for all three outputs; raw atom and typed-bond finite differences at all
three scales; low/high-weight coupled chemistry checks; nonzero invalid
chemistry assay; two disposable ten-update pilots; runtime fit and frozen
identities. Summary SHA
`9e8121a2c7e3de93ab886244372aadc303928c6ddb2402d797172c1fcd667cee`.
This one passing validation arm does not certify the paired campaign or
authorize omitting the random arm.

The **random** validation passed hard/native parity and all three atom
finite-difference scales. Its typed-bond analytic derivative was
**9.497333526611328**. Independent central differences were:

| Epsilon | Finite difference | Gate |
| --- | ---: | --- |
| 0.01 | 7.599136233329773 | Failed |
| 0.003 | 12.296060721079508 | Failed |
| 0.001 | 9.71454381942749 | Passed |

The declared requirement is **two adjacent passing scales**; one of three
does not pass. Random summary SHA
`bc8c12ea3df2a7455da6cadd84b69f8c7b9bf97ed33947c467d8aba867c87ad0`.
Later random coupled/pilot gates are not certified by the passing mix50 arm.
Independent static review found no concrete mapping, transport or backward
implementation bug explaining the discrepancy. Do not weaken the gate,
reinterpret this as a pass, silently change settings or submit mix50 alone.

Prepared native identities:

- Model digest `395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16`.
- Context tensor digest `495002ffa2a52cc85b58ef34424622763dfc4ca556139a98297e8789ae287aa5`.
- Context file SHA `254d5b7f11605fb41db6f5abf7015f9cc856a19dfe1d4b506ccaea9007200770`.

The 299-test CPU pass remains valid. Overall native assessment is absent,
source02 long-arm root remains virgin, and both long runs remain unsubmitted.
Controller29759070 is still PEND and may be retired only after authenticating
its unstarted state and preserving evidence. Final controller cancellation
and full failed-readiness archive receipts are pending. Preserve all failed
and passed evidence; further resolution requires an explicit human-directed
decision, without waiving any native gate.

Controller29759070's final EXIT is now confirmed after intentional cancellation
of its verified-unstarted PEND state; the source02 long-arm root does not
exist. The complete readiness-parent failure archive preserves the prepared
native context, exact raw initialization, reports, two ten-update pilot trajectories,
derivative probes, worker logs and scheduler/controller records: **171 files,
14,985,443 uncompressed bytes**. Archive **506,106 bytes**, SHA
`a456234c316b5e534508f972bdc1ab47d4ed5d0431988067df673ec36cebf9b4`;
manifest SHA
`f54279683af6b7641b3c1a8ac6ea4ac957b7490ce8045c43f87cd7446599282a`.
Remote archive:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-native-fd-failure-01.tar.gz`,
with exact receipt
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-native-fd-failure-01.receipt.json`.
Local download and stdlib verification completed at
`results/hpc/20261019-brd4-boltz-affinity/native-fd-failure-01/verification.json`:
exact archive path/506,106 bytes/SHA, safe unique members, exact manifest set
and every file's length/hash passed for all 171 files. Both summary hashes/
statuses, raw random FD flags `[false, false, true]`, passed mix50 parity/
coupled/pilot records, unchanged frozen model/context identities and context.pt
SHA were authenticated. This verifies the failed scientific evidence's bytes;
it does not turn the failed random gate into a pass. Independent finite-difference review
established no concrete implementation bug. Coarse-scale nonlinearity and
FP32 effects are unproven hypotheses, not reasons to waive the gate or alter
probes until they pass. The automation API confirmed existing heartbeat
`finish-combined-brd4-trajectory-report` **PAUSED** for this unresolved native
blocker; no duplicate automation, scientific setting change or experiment.

## User-authorized arm separation after the paired failure

The user explicitly asked to start the independently passed 50% mix and
investigate random gradients separately. Submission eligibility is therefore
arm-specific. This supersedes the earlier hold on mix50 solely because the
random arm failed; it does not change either scientific result. Parent
29759069 remains EXIT1, its original overall assessment remains absent,
and the random initialization remains unapproved for long optimization.

The adopted mix50 evidence includes its complete source02 native parity,
atom/bond finite differences, chemistry checks, disposable pilots, runtime
fit and frozen identities, plus source02 CPU 29759068's 299-test pass. The
unchanged source02 driver consumes a mix50-only assessment. This assessment
covers the requested passed arm, not the failed paired campaign.

## Authenticated tooling and passed adoption

The first external tooling release omitted the unchanged
`scripts/check_boltz_cpu_proof.py` dependency. CPU 29759210 EXIT retained
13 FileNotFoundError arm-fixture failures and six passing diagnostic
contracts; adoption was not reached. The failed release, job and logs are
preserved in the ledger and receipts.

Corrected external tools02 added that verifier without changing the existing
seven files or source02 scientific code/configuration. All eight files were
authenticated at
`/work3/s234818/mosaic/releases/20261019-brd4-boltz-independent-02/source`.
Manifest SHA
`848cb213d91be09513439f719ece8e43ec55991bd473f1bc233f3d80eb204c1b`;
archive 18,486 bytes, SHA
`7d5516aae2fa6bb0a8fee2625258fe86d32049d207f2424698f7f7bba8adaa2d`.

Actual CPU 29759211 DONE/passed 19 contracts: 13 arm-adoption and six
diagnostic contracts, zero failures/errors/skips. Scheduler/LSF success was
authenticated, and adoption rehashed all 147 original mix50 proof files.
CPU assessment SHA
`bc76e33927104e88a9e019b706d0ed4c6e1fc8f3bd4460264acad13ca7594448`;
mix50 arm-scoped assessment SHA
`ca400a1477471790bcc5c93cd85a55455da77b331caa504f882b2020d954140e`.
Independent static review found no blocking issue in either tool. These
engineering/adoption checks do not certify the random native gradient.

## Current independent GPU jobs

Both jobs were actually submitted at 20:40:18 Europe/Paris on 19 October 2026,
without initial dependencies or dependencies on each other. Scheduler
verification at 20:41:42 established:

- **Mix50 long run 29759212 RUN**, started 20:41:13. One exclusive L40S in
  gpul40s, four CPU slots, 8 GiB per slot, 23h. Output:
  `/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-mix50-independent-01/`.
- **Random diagnostic 29759213 PEND**, waiting for an unavailable
  `ngpus_physical` reservation on three hosts. One exclusive L40S in gpul40s,
  four CPU slots, 8 GiB per slot, 1h. Output:
  `/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-random-diagnostic-01/`.

Mix50 retains six occupied neutral slots, seed 17, LR 0.03, ten-trial
best-primal policy, affinity coefficient 1, element-JS 0.1, exactly 10,000
updates and the same chemistry schedule: 0.001 through 5000, geometric ramp
to 50 at 10000. Prepared context, model and all native/finite/runtime/identity/
terminal gates are unchanged. A started job is not a completed trajectory.
No long random optimization has been submitted.

The diagnostic replays the exact failed `GraphLogits(saved_logits).graph()`
bytes at bond row 2, aromatic 4 to NONE 0. Original epsilon scales 0.01/0.003/
0.001, 15% tolerance and two-adjacent-pass requirement remain unchanged.
Repeated forward/backward modes, a checkpoint-disabled control and smaller
scales investigate the discrepancy without selecting a production change.
The smaller scales and control are diagnostic only. See
[the diagnostic protocol](brd4_boltz_random_gradient_diagnostic.md).

## Verified compact evidence

Actual submission records:
`results/hpc/20261019-brd4-boltz-affinity/independent-02/gpu-submissions.json`.
The 30-file CPU/adoption/submission bundle passed safe extraction, every-file
length/hash and XML checks using local stdlib at
`results/hpc/20261019-brd4-boltz-affinity/independent-02/proofs-01/verification.json`.
Archive 126,516 bytes, SHA
`414dfa385545b8f7e3dc751fbbd4bd964f3e7926ad6feb48f23fafa9de164546`;
manifest SHA
`8e645c895fb51178ff96e25a732dbe077b52281bff91e4f4528644612f00462b`.
This authenticates engineering/adoption/submission evidence, not a completed
long run or a resolved random derivative failure. Chronological failed and
intermediate receipts remain in the experiment ledger.

At 20:42:41 Europe/Paris, mix 29759212 remained RUN and its repeated initial
source02 raw native finite-difference report passed. This is a per-run native
gate result, not completed updates or terminal validation. The latest observed
diagnostic state remains its 20:41:42 GPU-resource PEND.

The automation API confirmed the existing heartbeat
`finish-combined-brd4-trajectory-report` ACTIVE, with no duplicate. It follows
the unchanged mix run and separate random investigation, stays quiet on
unchanged state or outages, permits only a concrete bug fix with authenticated
allocated retesting and all original gates, and pauses on a genuine unresolved
curvature blocker. No tolerance or production protocol change is authorized.

At **20:44:01 Europe/Paris**, job29759212 retained **41 applied updates and
42 saved states** (latest state41), with case status `running` and its native
entry-gradient report passed. This is authenticated partial progress, not a
completed10,000-update result. Diagnostic29759213 remained GPU-PEND.

## Mix50 completion metadata and compact local evidence, 20 October 2026

Mix50 29759212's completion records authenticate exactly 10,000 contiguous
updates 0..9999 and 10,001 contiguous states 0..10000. Update JSONL, case report
and summary case records agree; all graph records retain six slots. Source02
file map and manifest match the sealed release exactly. Configuration SHA
`97faf3b192a7aca566ad86752c2f3efacad69e89b076c88e67e8d14b5435de0e`;
summary SHA
`7f49087f12efc762a46691581f80bc42c43395c00b0cc4778a9e04a6679aae7c`.
Initial native, coupled chemistry and terminal checks passed; frozen model/
context identities retained their original digests; final coefficient 50.
All 10,000 applied updates are recorded as strict improving, with zero fallback
or no-op. These comparisons use each state's fixed surrogate; changing
repair references/coefficients do not make the entire run one globally
comparable objective.

Local compact bundle verification passed **21 files**, including only
selected snapshots 0/5000/10000, at
`results/hpc/20261019-brd4-boltz-affinity/mix50-completion-02/verification.json`.
`scripts/verify_hpc_evidence.py` plus exact every-file byte/path/manifest and
schema checks passed. This authenticates completion metadata and the selected
snapshot bytes; it does **not** authenticate every one of the 10,001 snapshot
files or a full rendered report. No model or numerical test was rerun.

Remote archive
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-mix50-completion-02.tar.gz`,
21,533,349 bytes, SHA
`0b79afbf9f1ecc788d9111b58cb1677bb6a75cb9adc3cf891ec81542c2cd8764`;
manifest SHA
`4d1d19a8a30a9cc485d5f94e84f49b1d1cfe7dd979e924d3ea3cc385857559aa`.
The first completion01 packaging helper failed because system Python lacked
`hashlib.file_digest`. Its folder remains preserved, no scientific artifact
changed, and unique completion02 used standard streaming SHA instead.
The latest random diagnostic observation remains 08:59:37 PEND, unstarted.

## New standalone report requested, 20 October 2026

The user requested a new interactive report for completed Boltz mix50
29759212. Preparing a separate saved-only report family containing its
10,000 updates/10,001 states, with affinity as the primary metric and binder
probability separate. Shared-viewer contact defaults remain compatible with
earlier reports. Preserve all old reports/servers, saved invalid graphs and
exact probabilities; do not add Protenix overlays or a fabricated random
selector. The saved final raw graph's chlorine valence 2/null canonical SMILES
must remain visible. Allocated CPU tests/full-input authentication/rendering
and verified download/browser QA precede delivery. Report tools01 is now
authenticated, and CPU report 29760530 DONE at 09:16:29 Europe/Paris after
119 seconds; scheduler/LSF success authenticated. All 35 tests passed and the
actual renderer authenticated all 10,001 snapshots/147 readiness proofs,
then produced the complete affinity-primary report for input 29759212 only.
The 451-file download passed safe extraction/every-file verification and is
published at **http://127.0.0.1:8774/report.html**, with actual HTTP 200/exact
length/SHA verified. Full browser QA passed endpoints, raw invalid final
graph, readable labels/probability sizes, heatmaps, playback and ramp.
Output is
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-mix50-report-01/payload/`.
See [the report plan](brd4_boltz_affinity_report.md) for scope and gates.

Local report edition:
`results/hpc/20261020-brd4-boltz-affinity-report/edition-01/`; all 451 archive
files passed scripts/verify_hpc_evidence.py plus stdlib checks, manifest SHA
`b32f255b2937f3244d7fe1337fcfee63206e145e4e02fadf1e0e6479712accce`.
The new family's viewer was published HTML last and served by detached 8774,
preserving old reports/servers. Full browser QA passed: one correct selector,
state 0 `CCCCCC`, state 5000 coefficient 0.0010021663/previous applied 0.001,
state 10000 coefficient 50/raw chlorine-valence-2 invalid/null SMILES, six-row/
15-pair heatmaps and working playback/pause. Report left paused at 10000 and
marked deliverable; no other experiment changed.

Browser review also confirmed readable affinity, separate binder-probability
and raw/weighted repair overlay plots.

## Random diagnostic launcher failure discovered after restored access

At 17:15:21 Europe/Paris on 20 October, preserved LSF/log evidence showed 29759213
EXIT1 from 09:52:19–09:52:22 before model execution. Its live query for old
CPU 29759211 no longer returned that completed job, triggering a launcher
assertion. The actual passed CPU/proof records remain authoritative. The
explicitly authorized fix is evidence authentication from hashed persisted
actual final scheduler/LSF success, with allocated tooling tests and a new
immutable release. Preserve this failure; no fabricated success, scientific
source02 change, FD-gate waiver or long random submission. Failure hashes
and the launcher CPU submission are recorded below; actual CPU outcome
and GPU retry remain pending. See
[the diagnostic record](brd4_boltz_random_gradient_diagnostic.md).

Failure capture02 downloaded/safely extracted/all 11 file bytes verified;
bhist/bacct history was permission-denied, so actual EXIT1 comes from the
original LSF footer/logs. Failed capture01 is preserved. Authenticated
seven-file launcher03 keeps diagnostic/scientific bytes unchanged and
requires 19 allocated CPU contracts. Actual CPU 29763673 submitted hpc/one
slot/2GiB/15min; no result or GPU retry yet. Exact archive/source pins and
failure download path are in the diagnostic record above.

The subsequent CPU 29763673 outcome was EXIT1 before tests at 17:20:21,
from requested 1 GiB zhome reserve versus mandatory 4 GiB minimum. This is a
launcher parameter failure, not measured storage exhaustion or a native
failure. Source03/job/logs preserved; unique launcher04 will change only
that CPU wrapper reserve and rerun allocated checks. No retry GPU submitted.

Launcher04 is now authenticated (seven files, six identical 03; only the
reserve character 1-to-4 changed in the 2,925-byte CPU wrapper). Actual new
CPU 29763686 submitted hpc/one slot/2GiB/15min, outcome pending. Exact
source/archive pins are in the diagnostic record; GPU retry remains gated
and unsubmitted.

CPU04 actual 29763686 DONE 17:23:10/19 passed in 1.13 seconds/fresh 4 GiB reserve/LSF
success is preserved. Subsequent pre-certification inspection found actual
bjobs `Status<D\nONE>` wrapping; source05 will normalize whitespace and
retain exact canonical DONE/hpc checks, adding one planned contract for 20.
No new failed native run, accepted fabricated status or GPU retry implied.

Launcher05 all seven files authenticated; actual CPU 29763691 DONE/passed 20
in 0.95 seconds, zero failures/errors/skips, with scheduler/LSF/assessment/XML
and 13 proof files verified. Conservative quota 15 GiB free. Exact source and
proof archive pins are in the diagnostic record. This is engineering proof
validation, not native derivative success. Replacement diagnostic 29763693
actually submitted 17:25:41, initial PEND; resource arguments and bjobs verified.
Output `/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-random-diagnostic-02/`.
Mix50/report unchanged and random long optimization remains unsubmitted.
