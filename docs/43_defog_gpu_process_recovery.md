# DeFoG GPU process recovery for the ramped random-start study

**Latest verified snapshot, 10 October 00:10 DTU-local:** startup and the
six-update native repair/gradient path worked, but all ten long cases failed:
four timed out at 6,000 seconds and six returned 1. Controller completion is
not cohort success. Source080 completed all three saved-state replay cases;
the original Boltz FD mismatch patterns reproduced and none of the original
gates was certified. Numerical cause remains unresolved, separate from the
confirmed validation-runtime bottleneck. The
[diagnostic report](47_defog_long_run_diagnostics.md) and
[ledger](05_experiment_ledger.md) retain verified evidence; no jobs, code or
scientific settings changed during this status-only turn.

**Historical investigation update, 9 October 2026:** the short native repair/gradient gate
worked. Verified long-run records show prior 1/2/4 each reached 60 accepted
updates, then hit the 6,000-second cap inside full gate 60. Repeated full Boltz
FD validation on every repair refresh dominates runtime; the budget was
undercalibrated. Separate contact/total Boltz FD mismatches remain numerically
unattributed, while their target-only repair checks pass. Source080 is now
sealed/uploaded, CPU **29661601 DONE** with 31 engineering passes; native
saved-state replay **29661602 PEND**, a four-hour A100/MPS allocation.
CPU dependency is fulfilled, but no real-model attribution outcome is
established. Production policy/tolerances remain unchanged. See
[diagnostic report](47_defog_long_run_diagnostics.md) for measured costs,
unchanged tolerances and attribution limits.

Historical status, **9 October 2026 10:19 DTU-local**: native **29656359 RUN** on
A100 n-62-18-10. Its real **six-update gate completed/passed**, with eight
eligible native repair targets from eight attempts, two selected refreshes,
six nonzero guided accepted updates and seven target-only finite-difference
passes. This validates the short repair/gradient path. **No long case has
completed**: prior 0–5 failed (1/2/4 hit their 6,000-second caps; 0/3/5 exited 1),
prior 6 is running, and 7–9 are unattempted. Follow-up logs identify Boltz
validation failures: **0/3 failed the `dbmol_contact_loss` component FD gate**,
**5 failed the conditional Boltz `local_v1` FD gate**, all mid-run. Their
numerical causes remain undetermined; they are not established DeFoG-gradient
failures. Short-gate success does not establish long-cohort stability,
validity of the raw shared graph or efficacy. The
[verified capsule](../outputs/hpc-results/status-20261009T081813Z.tar.gz),
[gate trajectory](../outputs/hpc-results/status-20261009T081813Z/defog/controller/20261008T221046Z_ramped-defog-random_156be604/native-gate-trajectory.json)
and [campaign](../outputs/hpc-results/status-20261009T081813Z/defog/controller/20261008T221046Z_ramped-defog-random_156be604/campaign.json)
preserve these outcomes. No code or scheduler changes were made in this recheck.

Historical status, **8 October 2026 23:30 DTU-local**: native **29656359 remains
PEND**, awaiting A100 queue resources. No controller directory or native logs
exist; the repaired real-model gate has not started. Earlier CPU/native-frame
proof remains passed, but there is no successful DeFoG repair/guided update or
efficacy result. The [verified read-only status capsule](../outputs/hpc-results/status-20261008T212940Z.tar.gz)
and [scheduler snapshot](../outputs/hpc-results/status-20261008T212940Z/scheduler_snapshot.json)
record this state. No code, configuration or job changed during this recheck.

Recovery background, 8 October 2026: the user authorized a fix and verified rerun.
Static inspection of downloaded, remotely hash-matched native source identified
two adapter-contract mismatches: Boltz's native frame feature has a singleton
ensemble axis, and DeFoG's sampled diagonal is one-hot `NONE`. Source079 recovery
work corrects these expectations while preserving native features, frozen model
weights, sampling policy and scientific acceptance criteria. Source079 is now
sealed, authenticated and uploaded; CPU **29656209 DONE**, all **195 fixtures
passed**, and actual 13-/22-atom saved native frame-helper parity passed.
Native **29656359 submitted/PEND** at **16:55 DTU-local**, one 80 GB A100/MPS
allocation for 23 hours. Its actual coexistence and six-update native repair/
gradient/accepted-update gate must pass before the ten long cases. No native
repair or efficacy result is established. Details and
native source identities follow below.

Historical failure, 8 October 2026 **10:30 DTU-local**: native **29634091 EXIT**, after
its repeated A100 CUDA coexistence probe **passed**. The scientific gate then
failed with `Unsupported native confidence frame shape or coordinates`.
All four prescribed DeFoG attempts separately failed `native_continuation`
with `Native transition changed count/shape/symmetry or emitted nonfinite data`.
**None of the ten long trajectories launched.** At that snapshot the detailed
causes were not established and no new fix/rerun was claimed. This authenticated
live observation is now backed by the downloaded/hash-verified
[source077 evidence capsule](../outputs/hpc-results/source077-dense-chemistry/evidence001.tar.gz),
including the successful A100 probe, confidence traceback and all four worker
responses; see the ledger for digest, export scope and verification command.
Source076's CPU **29634060 DONE**, **161 engineering tests**, no-MPS control
**29634064 DONE** and positive MPS probe **29634065 DONE** remain valid startup
evidence. Startup success does not imply native repair or gradient success.
All numerical diagnostics, fixtures,
repair calls and optimization runs must execute on DTU HPC. Local editing,
static review and immutable packaging remain permitted.

This recovery preserves the scientific protocol in
[docs38](38_ramped_random_defog_study.md) unless an explicit new receipt
declares a runtime-policy change. It retains the original source069 failure;
it does not mutate old source, overwrite failed run directories or change
the separate decoder-training experiments.

## Source079 implemented recovery and engineering preflight

[Source079 archive](../outputs/source-releases/source-079-defog-contract-recovery-attempt001.tar.gz)
SHA-256 is `744862c9a310f87ff078edb2ab3185fd2468eaa00339222d024c1ffe0ceebcce`;
scientific manifest is
`21025b7bdb84ceeaf515ebe885ec0941491acf797a6ada66ffa02d73f1a5f29f`.
Release: `/work3/s234818/latentmol/releases/source-079-defog-contract-recovery-attempt001`;
control: `/work3/s234818/latentmol/submissions/source079-defog-contract-attempt001`.
The candidate-repair fixtures now reproduce native diagonal `NONE` vectors,
while confidence fixtures cover the single ensemble frame axis and explicit
unsupported layouts. The new allocated
[native-context diagnostic](../scripts/diagnose_native_graph_contracts.py)
follows CPU fixtures/preparation, validating actual prepared counts 13 and 22
against the native confidence helper without rewriting its feature input.

The [separately hashed metadata launcher](../outputs/hpc-submissions/submit-source079-native.py)
authenticates old source076 MPS evidence and unchanged probe/bootstrap/native
wrapper identities before reusing that startup proof. The new GPU allocation
must still repeat coexistence, then pass its actual six-update native repair/
gradient/accepted-update gate before any long cohort. These steps are required
engineering/scientific gates, not completed native outcomes at this snapshot.
CPU **29656209 DONE**, two slots/4 GB per slot/30 minutes, has 195 fixture
passes and actual native-helper parity. The ledger retains exact commands,
environment and claim limits. Local
AST and remote shell syntax checks supply no native numerical proof.

### Passed CPU/native-layout proof and queued real-model run

[Evidence001](../outputs/hpc-results/source079-defog-contract-recovery/evidence001.tar.gz)
SHA `29829a057478d4c6ba9b965dd99db537df0d48a0f38edbc1cd2810f6928c144b`
contains **31 safe members / 30 verified files**, with a
[metadata-only verification receipt](../outputs/hpc-results/source079-defog-contract-recovery/evidence001.tar.gz.verification.json).
Numerical fixtures ran on HPC Python 3.11.9 / Torch 2.5.1+cu121 in the CPU queue;
local verification used Python 3.12.14 / Windows 11 without model imports.
The [actual frame report](../outputs/hpc-results/source079-defog-contract-recovery/extracted-evidence001/submissions/prepared/frame-contracts/report.json)
shows exact native frame/mask parity, 5,096/8,256 nonempty interface pairs,
the old layout clause failing, the corrected clause passing and unchanged
contexts. No learned model score/gradient or DeFoG transition was evaluated
by that diagnostic.

[Native receipt](../outputs/hpc-results/source079-defog-contract-recovery/extracted-evidence001/submissions/receipts/native-receipt.json)
confirms **29656359**, `gpua100`, four slots/16 GB per slot, 80 GB GPU, 23 hours,
MPS plus exclusive reservation, `done(29656209)` dependency. Protocol SHA is
`0ad8eb7e000d9e653ad880c17f308f0fe668878e6811626d7c3a2ed2c2b46f1a`.
[Scheduler snapshot](../outputs/hpc-results/source079-defog-contract-recovery/extracted-evidence001/scheduler_snapshot.json)
at 16:55 DTU-local records CPU DONE/native PEND. A queued job is not a passed
real-model gate. Original failures and frozen-model/gradient boundaries remain.

## Static native-contract diagnosis and correction scope

The pinned Boltz [featurizer](../outputs/hpc-results/source079-defog-contract-recovery/native-source/featurizerv2.py)
stacks frames at line 1469 as `[N_ENS, N_TOK, 3]`; collating the single ensemble
produces `[B, 1, T, 3]`. Our validator required `[B, T, 3]` and rejected this
native layout before the confidence helper. Native
[`compute_frame_pred`](../outputs/hpc-results/source079-defog-contract-recovery/native-source/confidence_utils.py)
explicitly reshapes frame indices to `[B, multiplicity, T, 3]`. The correction
uses a compatible validation view and leaves the original feature tensor
passed to native confidence unchanged. It does not invent coordinates or
change the scoring head, pocket, objective or gradient boundary.

The pinned DeFoG
[native transition](../outputs/hpc-results/source079-defog-contract-recovery/native-source/graph_discrete_flow_model.py)
at lines 749–764 converts sampled edge labels to one-hot categorical vectors,
then applies
[`PlaceHolder.mask`](../outputs/hpc-results/source079-defog-contract-recovery/native-source/defog_utils.py).
An occupied diagonal retains one-hot `NONE`; it is not an all-zero categorical
vector. Our validator incorrectly required zero diagonal vectors, and its
fixture modeled that incorrect layout. The correction must accept the native
`NONE` diagonal while preserving count, off-diagonal symmetry, finite-data,
categorical and exact representation checks. This changes adapter conformance,
not DeFoG sampling, molecule-validity acceptance or target selection.

These are **static diagnoses of concrete boundary mismatches**, based on native
source bytes whose SHA-256 digests were compared with HPC. They do not prove
that both fixes suffice for a successful real-model call; allocated fixtures
and the native six-update repair/gradient gate remain required. The six native
file digests are recorded in the ledger. Local work was source/metadata
transfer and reading only, with no numerical test or model import. Original
source069/source076 failure artifacts remain immutable.

## Failure evidence and the runtime change

Original native job **29621180** failed its six-update preflight on 7 October.
Its wrapper finished at **17:47:52 UTC** with return code 1. The native worker
reported `CUDA-capable device(s) is/are busy or unavailable`; the controller
explicitly records that the long cohort was not launched. The 8 October
09:45 DTU-local read-only recheck confirms those retained records. The sole
metric is a step 0 schedule event: weight 0.001, no target hash, zero recorded
optimization oracle calls and no optimization improvement. No accepted or
guided update and none of the ten long trajectories occurred. See the
[downloaded failed wrapper](../outputs/hpc-results/source073-budget-recovery/extracted-status002/defog-wrapper/run.json)
and [ledger](05_experiment_ledger.md).

The process-admission failure was reproduced with a live parent CUDA tensor
and the exact separate DeFoG interpreter under `exclusive_process` without
MPS. Enabling **LSF-managed NVIDIA MPS** admitted both CUDA processes with
matching GPU UUIDs while retaining exclusive GPU allocation and
exclusive-process compute mode. The two diagnostics used different V100
hosts; no model weights were loaded. They establish a launch remedy rather
than native repair, memory sufficiency or scientific usefulness.
DTU's official multiple-task guidance prescribes
`num=1:mode=exclusive_process:mps=yes` for this use case.
[DTU single-GPU guidance](https://www.hpc.dtu.dk/?page_id=5150)
IBM documents MPS support for multiple CUDA clients in EXCLUSIVE_PROCESS mode
and dedicated allocation through `j_exclusive`.
[IBM LSF GPU request documentation](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=o-gpu)

An initial DEFAULT-compute-mode idea was superseded before sealing or testing.
The implementation must use job-managed MPS, **without manually starting an
MPS daemon**, and retain the entire GPU for this allocation. Allowing both
cooperating model processes does not authorize parallel independent
experiments on the same device. Exact requests, effective MPS pipe, device
UUIDs and allocation behavior are retained in the
[negative control](../outputs/hpc-results/source076-defog-mps-recovery/extracted-evidence001/exclusive-control-probe.json)
and [positive probe](../outputs/hpc-results/source076-defog-mps-recovery/extracted-evidence001/mps-probe-probe.json).
The original failed worker model-load traceback remains preserved separately.

Both models still consume memory while the worker runs. Allowing a second
CUDA context does not establish sufficient GPU memory, a compatible native
checkpoint, correct identity mapping or a successful repair. Record actual
compute mode/device/allocation, parent and worker context checks, model-load
receipts, peak memory and failures. Attribute the original error to process
restriction only if the diagnostic distinguishes it from driver/device
availability, memory and interpreter failures. The diagnostics reproduced
the exact busy/unavailable error with a working parent and authenticated
worker, and eliminated it with MPS. The actual A100 launch probe also passed;
its subsequent native model/representation gate failed as recorded above.

## Frozen-model and latent-gradient boundaries

The current molecular latent vector is decoded by the existing frozen VAE.
Frozen Boltz affinity/confidence branches consume its **one shared current
relaxed graph** under the same DBMol composite objective and initial-pose
pocket as source069. The decoder, encoder and external models remain frozen;
**only molecular `z` is optimized**. This is separate from the decoder-only
fine-tuning comparisons in docs39–42.

At refresh boundaries, outside autograd, a prescribed hard draw is mapped
exactly into the pinned official DeFoG checkpoint's supported neutral,
fixed-occupancy representation. Reject unsupported labels/charges without
coercion, rescreened replacements or silent identity changes. The isolated
DeFoG Python environment and frozen checkpoint/native source identities must
remain authenticated. Candidate-start late-time continuation, native time
grid, sampling settings, prescribed RNG/draw quotas, occupied slot alignment
and target eligibility retain the contracts in
[docs37](37_defog_repair_guidance.md).

A valid compatible repaired graph can become an explicitly different,
**detached guidance reference**. It never silently replaces the current
Boltz graph. Decoder categorical cross-entropy against its fixed labels can
supply a gradient to `z`; hard drawing, DeFoG diffusion/transitions, chemistry
inspection and target selection do not receive gradients. Keep the target
fixed throughout each gradient/finite-difference/line-search objective;
refresh only at the declared boundary. No DeFoG-likelihood gradient,
reparameterized hard sampling, exact expected-repair derivative or validity
guarantee is claimed.

## Gated recovery sequence

The implemented sequence is:

1. Run the corrected immutable release's CPU engineering suite and prepare
   eleven source-authenticated scientific configurations. Check
   launcher/provenance, environment selection, protocol matching, target
   alignment and failure handling. Engineering fixtures do not prove a
   working pretrained repair transition.
2. Run serial allocated GPU process diagnostics: first an exclusive-process
   control with `mps=no`, then the managed-MPS positive probe with `mps=yes`.
   Keep a real parent CUDA tensor alive while the exact configured isolated
   DeFoG interpreter allocates its own tensor. Record parent/child allocation,
   PID, interpreter/prefix, device, visibility and MPS environment. Compare
   GPU UUIDs when both runtimes expose them; otherwise disclose the weaker
   inherited single-device identity evidence. Retain unexpected control
   success rather than forcing the hypothesized failure. These probes load
   no pretrained model or molecule and cannot validate native repair.
3. Only after CPU DONE and an authenticated **positive managed-MPS probe
   DONE**, submit the actual native allocation. It repeats the concurrent
   CUDA probe on its allocated device before loading models; that process
   exits before the separate scientific controller starts. Then run the
   **actual six-update prior0 gate** before any long case. Require
   a native accepted compatible target, fixed-target repair-gradient checks
   and a nonzero guided accepted update under the existing gate. Record
   coverage/absence and all prescribed attempts. A native launch pass alone
   or a Boltz-only fallback is not a positive guided-learning result.
4. Only after that gate passes, run **ten prescribed random starts serially,
   at most 100 updates each**, retaining the original source063 saved
   unfiltered latent starts, initial occupied slots, weight ramp and quotas.
   Keep raw argmax/sampled chemistry, target edits, weights, gradients,
   oracle costs, stalls and eligible fresh-native endpoint scores separate.
   Invalid raw outputs are not silently repaired for the reported score.

The inherited scientific configuration uses a late-half guidance ramp from
0.001 to 0.1 across the declared accepted-update schedule, with fixed initial
occupancy and no trainable model parameters. Exact limits remain those in
the source069 protocol until authenticated new configuration receipts exist.
Search/repair eligibility can fail scientifically even after launch succeeds;
keep that distinction from engineering failures. Time-censored/unattempted
cases remain explicit, not claimed as ten completed trajectories.

## Implemented APIs, requests and launch order

[submit_ramped_random.py](../scripts/hpc/submit_ramped_random.py) now has
`--stage cpu`, `--stage diagnostic` and `--stage native`. Its default prints
the authenticated plan; `--submit` creates jobs with immutable intents and
receipts. It verifies the source archive/manifest, base/start asset hashes,
guidance YAML and prepared protocol before any dependent stage. CPU
preparation uses [boltz_ramped_random_preflight.sh](../scripts/hpc/boltz_ramped_random_preflight.sh).
The public [gpu_process_probe.py](../src/latentmol/gpu_process_probe.py) supplies
both standalone allocated diagnostics and the repeated native runtime check.

| Stage | Queue and host resources | GPU policy | Limit and dependency |
| --- | --- | --- | --- |
| CPU tests/preparation | `hpc`, two slots, 4 GB per slot, one host | None | 30 minutes |
| Exclusive control | `gpuv100`, four slots, 4 GB per slot, one host | `num=1:mode=exclusive_process:mps=no:j_exclusive=yes` | 10 minutes; CPU DONE |
| Managed-MPS probe | Same V100 resources | `num=1:mode=exclusive_process:mps=yes:j_exclusive=yes` | 10 minutes; CPU DONE and control ENDED |
| Native gate and serial cohort | `gpua100`, four slots, 16 GB per slot, one host; `select[gpu80gb]` | `num=1:mode=exclusive_process:mps=yes:j_exclusive=yes` | 23 hours; CPU DONE and positive probe DONE |

The negative control can complete with the specifically observed child-CUDA-
unavailable outcome, or retain an unexpected child success. Unrelated failures
remain failed. Native submission additionally authenticates the managed-MPS
receipt's source/preparation identity, successful status, exact worker
interpreter and matching parent/child LSF job IDs. A DONE scheduler record
without a matching positive probe is insufficient. The helper never changes
device visibility or starts a manual MPS daemon.

After the new archive and unique remote paths are authenticated, use this
plan-only template from its release directory. Variables must contain the
actual verified identities/paths from the records below:

```bash
PYTHONPATH="$task_release/src" /work3/s234818/latentmol/env/boltz311/bin/python \
  scripts/hpc/submit_ramped_random.py --stage cpu \
  --release "$task_release" --archive-sha256 "$task_archive_sha" \
  --base "$task_base" --base-sha256 "$task_base_sha" \
  --starts "$task_starts" --starts-sha256 "$task_starts_sha" \
  --prepared-root "$task_prepared" --receipts "$task_receipts"
```

Add `--submit` to submit CPU preparation. After its authenticated DONE
receipt, use the same arguments with `--stage diagnostic`, the actual
`--protocol-sha256` produced by preparation, and `--submit`; this submits
the two serial process probes. Only after the same-identity positive probe
passes and is DONE, use `--stage native` with that protocol digest and
`--submit`. Exclusive intent/receipt handling prevents a blind duplicate
submission after an uncertain scheduler reply. This is a command template;
actual submitted/completed stages are recorded in the status and ledger.

[boltz_ramped_random_run.sh](../scripts/hpc/boltz_ramped_random_run.sh)
performs the native-allocation probe before launching the unchanged scientific
controller. Its six-update prior0 gate and ten saved-seed/100-update study
keep the source069 scientific schedule: gate timeout 7,200 s, long optimization
6,000 s/case plus 900 s fresh-pose reserve, 79,200 s controller budget and
6,960 s full-case boundary reservation. No automatic continuation is promised.
The gate's four draws per refresh/interval five, long cases' two draws per
refresh/interval ten, raw argmax plus eight draws per state and conditional
fresh-pose eligibility remain declared in docs38. Passing the lightweight
CUDA process probe does not shorten or waive the real native guided gate.

## Identity, resource and result records

The original source069 archive SHA-256 is
`405df53db752530dc67ff7322c5a46a363dd027b5e8f12840e9f4daeffbed8ff`,
manifest `d3740595b959184573d7249e3afac0b9ba02e8884b33fe9f01b67e68dd7837d3`;
prepared protocol SHA-256 is
`43b578dd30378a9d659f78f28f1a788fca7bce3223481f1783e598da1ad0e1ac`.
These identify the failed historical intervention, not a validated new
runtime. The separate source076 attempt001 archive SHA-256 is
`8bf9c547862840a3a53451dd0bf2ec5562d0a2b13beec23f6c780aaeab812943`,
manifest `8e1050b4241d7b9fa242f0d035f6ed65d349d1ab3bb9aef606d7343dff242d45`,
prepared protocol `292e41db3b36041c18150c06850b8897396749cc993dfdcfbd99591ee5d68fe6`.
The eight selected source069 scientific files retain identical hashes.
Receipt/preparation root is
`/work3/s234818/latentmol/submissions/source076-defog-mps-attempt001`;
scientific output root is
`/work3/s234818/latentmol/runs/source-076-defog-mps-recovery-attempt001-random-study`.
Post-seal documentation updates do not change executable-source identity.

CPU ran 08:07:55–08:08:39 UTC, control 08:15:19–08:15:27, positive probe
08:15:36–08:15:44. Both GPU interpreters report Python3.11.9 and
Torch2.5.1+cu121/CUDA12.1. Native29634091 uses the A10080GB23-hour request
above, with CPU/MPS DONE dependencies. It must repeat process admission on
its own allocated GPU, then pass the actual six-update guidance gate before
the ten long cases. At **08:17:33 UTC**, it is queued with no native result.
The verified 48-member
[runtime evidence bundle](../outputs/hpc-results/source076-defog-mps-recovery/runtime-evidence001.tar.gz)
has SHA `7ccdf13928b191dcc830a99f68bf33b3978dc491825b3ab95f8568edbf8711a4`.
The [native receipt](../outputs/hpc-results/source076-defog-mps-recovery/extracted-evidence001/receipts/native-receipt.json)
and [scheduler snapshot](../outputs/hpc-results/source076-defog-mps-recovery/extracted-evidence001/scheduler_snapshot.json)
distinguish the queued study from completed launch diagnostics.

Append exact commands, source/config/checkpoint/data identities, allocated
resources, compute mode, environment, worker identities and oracle counters
to the [ledger](05_experiment_ledger.md) as evidence becomes available.
Distinguish submission, diagnostic pass, native guided gate pass, completed
optimization and chemical/binding utility. No earlier engineering or
decoder-training improvement can substitute for this study's native gates.
