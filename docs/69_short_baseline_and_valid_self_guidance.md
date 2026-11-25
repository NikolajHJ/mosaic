# Three-epoch original-loss baseline and separate valid self-guidance test

The [21 October authorization](70_epoch36_collapse_mechanism.md) permits a new
independent weak-test plan regardless of baseline failure. It supersedes the
earlier dependency for future tests only; source131's failed outcome and blocked
weak stage below remain unchanged. The new [source132 CPU submission](70_epoch36_collapse_mechanism.md#source132-actual-immutable-staging-and-cpu-submission)
has separate immutable identities and passed full CPU/profile plus the completed,
restored 100-update mechanism replay. At 11:19 DTU-local, both quarter-step
baseline and independent weak proof arms failed the 200/2,500-update validity
guards after native passes; no training was submitted. See the [current terminal outcomes](70_epoch36_collapse_mechanism.md#both-quarter-step-proof-arms-failed-at-update-200).

Status, 21 October 2026 09:22 DTU-local: **source131 longer proof FAILED at
100/2,500 updates after passing native readiness**. The weak test and baseline
training did not start. Epoch 36 remains the latest completed checkpoint.
All numerical fixtures, preprocessing, updates and evaluations
remain on allocated DTU HPC; local work is editing, static review, packaging
and saved-evidence inspection only.

The user requested three more epochs from completed epoch 36 using the loss
that previously worked, once checks pass, and a **separate test** of weak
guidance for already-valid molecules. These are distinct objectives and run
identities. Preserve source127's user stop, source128's failed benchmark and
source129's sealed diagnostic/evidence; no historical job is restarted or
relabelled by this authorization.

## Source131 original-loss proof failed early fixed-panel retention

The [07:21:48 UTC status collection](../outputs/hpc-results/source131-short-callback-result-attempt001/status-news20261021-001.json)
and [07:22:47 UTC fixed-panel outcome](../outputs/hpc-results/source131-short-callback-result-attempt001/fixed-panel-outcome-news20261021-001.json)
show that GPU **29764169** ran on 20 October and failed the longer proof at
relative update **100/2,500**, absolute update **237556**. The proof run spans
20 October **18:57:26–19:09:49 UTC**. Current LSF records have aged out;
persisted logs, failed-run records and gate evidence establish the outcome,
not a fresh live EXIT query. The earlier queued observation below is history.

All **85 CUDA fixtures passed**, zero skips/failures/errors, in **37.93s**.
All **18 native benchmark gates and 11 supplementary gates passed**, including
runtime, resume and execution parity. This native readiness did not establish
longer stability. At the 100-update fixed panel, only starting raw-validity
retention and strict-teacher-validity retention were false; the other **16 of
18 early-health checks passed**. The fixed 1,024-prior trajectory was:

| Disposable updates | Raw valid | Strict teacher valid |
| --- | --- | --- |
| 0, authenticated epoch 36 | 641/1,024 (62.5977%) | 641/1,024 (62.5977%) |
| 32 | 674/1,024 (65.8203%) | 674/1,024 (65.8203%) |
| 100 | 357/1,024 (34.8633%) | 354/1,024 (34.5703%) |

At update 100, classification records **667 chemical failures** (534
sanitization, 133 disconnected), **three representation failures** (two
roundtrip, one canonical identity), and **zero input failures**. Raw chemical
and representation acceptance are distinct. Exact and chemically valid
held-out reconstruction remains **256/256 at every panel**. Sulfur frequency
was 0.015069966 at baseline and 0.012059477 at 100; expected size was
22.33288 and 22.30303 respectively. These observations do not establish the
specific chemical/optimizer mechanism.

The original-loss baseline therefore **also fails stability under these
continued settings**; its earlier successful epoch does not waive this failed
proof. The controller's saved self-test decision stopped at predecessor
benchmark EXIT. The weak-test campaign is absent and no training segments
exist in the collected evidence: neither weak self-guidance nor baseline
epochs 37–39 ran. No disposable state is promoted; original epoch 36 remains
the latest completed parent.

The 100-update panel SHA256 is
`366d5daa045c8731a2f62c8e6c6af02eba12b900390f1598ef9583eaa58b1d1b`;
failed gate SHA256 is
`03cb363943252e28ee43b8f272e4cca86b147b123d2d3ccb8a098f70608820d8`.
This status turn collected saved evidence only: **no code change, new test,
submission or gate waiver**. Preserve every failure and the untested weak arm;
the cause remains unproven.

## Full CPU passed; baseline GPU proof queued

The [17:54:00 UTC snapshot](../outputs/hpc-results/source131-short-callback-result-attempt001/status-cpu005.json)
confirms CPU **29764158 DONE**, after-CPU **29764159 DONE**, saved `cpu_gate`
passed and original repair profile passed. JUnit retains **1,525 passed,
79 CUDA-only skips, zero failures/errors**. This supersedes the earlier
unit-only/profile-pending observation without rewriting it.

The saved after-CPU decision is `native_stability_submitted_not_passed` and
binds actual baseline GPU **29764169 PEND** plus callback **29764170 PEND**:
one 80GB A100, 16 CPU slots, **24h requested walltime/23h internal cap**, for
fresh original-loss native readiness and the **2,500-update disposable proof**.
It is not training or a native/stability pass. The separate **weak-test GPU
is not yet queued**, and **baseline training is neither queued nor started**.
Only the declared serial, authenticated path below can later submit original-
baseline epochs 37–39. Exact sealed identities/operator commands remain below.
Source129 was not queried again in this final status read; its last observed
state was PEND, with its existing diagnostic unchanged.

## Source131 unit suite passed; profile still running

The [17:52:04 UTC snapshot](../outputs/hpc-results/source131-short-callback-result-attempt001/status-cpu003.json)
records **1,604 collected: 1,525 passed, 79 CUDA-only skips, zero
failures/errors**, JUnit **200.278s** (SHA256
`9a14606529cd05ece67bdf0ea6469501d817b886a7ac89fc2c96b26f2ae4dfc6`).
The two previously failing controller fixtures now pass with tests unchanged.
CPU **29764158 remains RUN** for the original repair profile; after-CPU
**29764159 is PEND**. No `cpu_gate` result is saved yet: the full CPU gate
includes the profile and has **not** passed. No GPU validation or training
is submitted. This unit proof does not replace the required later gates.

## Source131 actual callback-interface retry

[New preparation](../outputs/hpc-submissions/source131-short-callback-result-attempt001-preparation.json)
derives from source130 with exactly one overlay,
`scripts/hpc/chemistry_campaign.py`, SHA256
`a173618cd308b8000919cf6bfe3bd148e34e77246a7d9b5c2640cfdce2ca1424`.
It returns the saved `audit` on the first short-branch callback invocation,
without changing the historical interface for other plans. Objective,
protocol, weak loss, tests and scientific settings are unchanged.

The [actual staging/queue receipt](../outputs/hpc-submissions/source131-short-callback-result-attempt001-stage-and-queue-001-output.json)
returned 0 and authenticated **603 members**, then submitted CPU **29764158**
and after-CPU **29764159** as `submitted_not_passed`. This is a unique source131
release/plan/campaign, not a rewrite of source130's failed allocation. There
is no completed CPU pass, native GPU submission or training start yet.

| Source131 identity | SHA256 |
| --- | --- |
| Archive | `7894bd4795e8c93df0d3cd1a85eb762e2e8de34ceeb84aaf3f59596cd9da537f` |
| Manifest | `bf2cdd3adc90b45f9cbbc73a28126584855aed380b6d094ecb1d4b812f779c64` |
| Baseline plan | `d95a264eccb737b5cc08e32ad7cb5e29e2a49c23b215488255f4cbc6f8372077` |
| Separate weak-test plan | `1d4ddaa08d21ac28fcc667c0e0fc44e91a3c63daa35af16505e93b245f129303` |

Bundled-CPython operator command tails (same local runtime recorded below):

```text
outputs/hpc-submissions/prepare-source131-short-callback-result-attempt001.py
outputs/hpc-submissions/operate-source131-short-callback-result-attempt001.py stage-and-queue
```

These are packaging/byte staging/scheduler work, not local numerical tests.
The serial baseline proof → separate weak proof → conditional baseline
37–39 path and every gate below remain required.

## Source130 CPU failure retained; source131 correction in preparation

The [17:46:43 UTC snapshot](../outputs/hpc-results/source130-short-baseline-valid-self-attempt001/status-cpu004.json)
records CPU **29764146 EXIT** and after-CPU **29764147 EXIT**, the latter
stopping on the failed CPU gate. JUnit reports **1,604 collected: 1,523 passed,
79 skipped, two failures, zero errors**, duration **228.374s**. No native GPU
or training was submitted.

Both failures are new short-controller fixture assertions expecting the
returned decision object. The inherited `Campaign.advance()` first invocation
returned `None` while writing the proper saved decision; idempotent replay
returned that dictionary. The failures concern this controller API behavior,
not a recorded numerical assertion or a chemical-efficacy finding. They are
nevertheless genuine CPU contract failures and block progression; no native
or longer-proof pass follows from other fixture successes.

The prepared fix returns `audit` after `finally` **only for the explicit short
plan**, preserving the historical `None` interface for other campaigns. The
next source131 is to derive from immutable source130 with one controller-module
overlay; weak loss and tests remain unchanged. Source130's source, receipts
and failed results are preserved. This records a code correction in preparation,
not a new seal, allocated test, pass or restart.

## Source130 actual release and submission

[Immutable preparation](../outputs/hpc-submissions/source130-short-baseline-valid-self-attempt001-preparation.json)
binds 26 overlays (24 Python files and two configurations). The
[stage-and-queue receipt](../outputs/hpc-submissions/source130-short-baseline-valid-self-attempt001-stage-and-queue-001-output.json)
records return code 0, **603 members authenticated**, and actual CPU
**29764146**/after-CPU **29764147** submitted as `submitted_not_passed`.
The [17:43:20 UTC snapshot](../outputs/hpc-results/source130-short-baseline-valid-self-attempt001/status-cpu001.json)
shows CPU 29764146 **RUN** on 16 CPU slots since approximately 19:42 DTU-local,
and after-CPU 29764147 **PEND**. Pytest progress is partial, not a completed
test count/pass. **No native GPU, stability result or training start exists.**

| Identity | SHA256 |
| --- | --- |
| Source130 archive | `86441c76f39857d5131fd53e4717677c370c9aebd015c2ad8dec726e776266aa` |
| Source manifest | `4c4ed3b477ea00304d239179e34d7d458833ee07f1fe322bf4ea76f025cf82b9` |
| Baseline plan | `342a33d9656065ed41e408cd67684a0a973d772cb6f8de935c450f2dc08ebf9e` |
| Separate weak-test plan | `0fda2e7a59ea7fcb8d458d5f18d5effce9c5266554f48785fb74bc864d5aa51a` |

The [baseline plan](../outputs/hpc-submissions/source130-short-baseline-valid-self-attempt001.json)
declares `baseline_continuation.version = "epoch36-original-repair-v1"` and
`short_continuation.version = "three-epoch-baseline-v1"`, purpose
`baseline_training`, start 36/end 39. The [weak-test plan](../outputs/hpc-submissions/source130-short-baseline-valid-self-attempt001-self-test.json)
has purpose `self_guidance_test` and explicitly adds
`valid_prior_self_guidance={version:"valid-prior-self-nll-v1",weight:0.01}`.
Both retain the original source121 loss family, ratio, reference/ramp and
parent state. One-time startup is an explicit execution-only policy; its
presence is not a passed warmup or runtime gate.

Root executed these operator command tails through bundled CPython
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`:

```text
outputs/hpc-submissions/prepare-source130-short-baseline-valid-self-attempt001.py
outputs/hpc-submissions/operate-source130-short-baseline-valid-self-attempt001.py stage-and-queue
```

Local execution was standard-library AST parsing of 24 Python files,
hashing/packaging, byte staging and scheduler submission, with no project-import
or numerical/model test. A preliminary PowerShell `-c` attempt to derive the
collector failed quoted-command parsing before execution; an explicit metadata
collector then succeeded. This did not change jobs or sealed scientific source.
Source129's [19:30 DTU-local snapshot](../outputs/hpc-results/source129-chemistry-validity-runtime-diagnostic-attempt001/status-shortbranch001.json)
still showed its original diagnostic PEND; that existing job was not changed.

## Declared serial sequence and promotion boundary

1. Complete the allocated CPU contracts/profile.
2. Pass fresh baseline native readiness and its **2,500 disposable updates**
   from original epoch 36, with the fixed early panels/gates below.
3. Run the separate weak-self native readiness and **2,500 disposable updates**
   from the same original epoch 36, matching input/teacher/RNG scope, not from
   the baseline proof endpoint. Weak-test states can never promote training.
4. Once the weak stage is authenticated terminal, conditionally submit only
   the original-loss baseline for epochs **37–39**. A failed weak test is
   preserved and does not block an independently passing baseline, but it
   never waives a baseline gate, a terminal-state check, quota or seriality.
   Discard every proof state and restore original epoch 36 before training.

Both proofs record unfiltered generation before/after: fixed **1,024 prior
latents with four categorical draws each**, plus argmax outcomes, saving raw
valid and invalid graphs. Evaluation RNG is separate/restored. The planned
baseline epoch-39 endpoint includes deeper monitoring; none is a result yet.

## Clean baseline: original source121 objective, stop at epoch 39

Restore the authenticated completed epoch-36 checkpoint, SHA256
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`,
including its native AdamW moments, native RNG and separate fresh-prior RNG.
Use the original successful source121 atom-preserving repair objective, without
the later failure-guidance fallback/extra-witness terms or the proposed
already-valid self-target term. Preserve the frozen epoch-35 reference and
original ramp, 256 real/512 prior (1:2), 0.25/0.75 repair branch means, other
weights/composition/occupancy/KL, and original bond-search semantics.

The baseline's endpoint is **absolute epoch 39**, meaning epochs 37, 38 and 39
after parent 36, not three additional epochs after any disposable test.
The global maximum remains 100; this branch must stop at 39 and must not queue
epoch 40 automatically. The endpoint is a short continuation, not an epoch-100
completion or an isolated causal demonstration. Retain all completed/partial
epoch records and actual controller/segment receipts.

Keep microbatch 128, 12 search workers plus four main Torch threads in 16 CPU
slots, serial training and at most 24h per allocation/23h internal runtime.
Any execution-only pooling/startup/clock optimization requires explicit new
configuration, identity and parity proof; no unsealed helper is silently
adopted. Preserve source/input/native-gradient/finite-value, reconstruction,
validity, composition/sulfur, memory, resume, runtime and fresh-quota checks.

## Readiness and the longer disposable proof

Seal a unique baseline source/configuration and prove objective/target/loss/
gradient parity with the original source121 path on matched inputs. Reusing
its historical successful epoch is motivation, not a new gate pass. Retain
allocated CPU/profile and applicable native CUDA/readiness checks.

Run the **2,500-update disposable proof** from original epoch 36, restoring
the production shuffle, epoch-37 learning rate and exact AdamW/native/fresh-RNG
state. It must span the earlier roughly 1,500–2,300-update deterioration window;
an eight-update benchmark is insufficient. Evaluate the fixed 1,024-prior and
256-reconstruction panels at baseline, after 32 updates and every multiple
of 100 thereafter. Keep classified chemical versus representation failures,
raw versus strict-teacher validity, bounded real failure evidence and
nonpromotable failed-state diagnostics.

All applicable old health/retention, native/parity, runtime, resume and quota
gates remain mandatory. Assertions requiring a fallback component that is
deliberately absent may be explicitly **not applicable**, with a versioned
applicability rule; do not omit general validity/gradient checks or convert a
missing measurement into a pass. Include fixed-panel overhead and conservative
runtime estimates under the declared guard. Stop on a failed check, exception,
nonfinite value or exhausted budget.

Only after the complete proof passes may the controller discard every test
model/optimizer/RNG change, restore and authenticate the original epoch-36
checkpoint, recheck no competing continuation and fresh quota, and submit a
new serial baseline continuation ending at 39. A submitted test or passed CPU
suite is not proof completion, training start or a completed epoch.

## Separate exploratory weak self-target test

The declared test coefficient is **alpha 0.01**. For each prior molecule satisfying
the declared already-valid teacher rule, detach its current hard atom,
occupied-charge and unordered-bond labels as a fixed self-target. Use full
categorical NLL divided by `N + K + K*(K-1)/2` (`N` maximum slots, `K` occupied
slots), then average across **all attempted prior rows**; nonselected rows
contribute zero. Do not normalize only by valid-row count. Add this as
**`+ 0.01 * mean_over_all_prior(normalized_full_self_NLL)`**, independently
of the existing 0.25/0.75 repair branch weights and repair ramp. Eligibility
uses strict-valid self-targets; it creates **no new repair witnesses or repair-
witness counts**. Target identity, graph mask and probabilities must remain
authenticated rather than being resampled during a loss call.

This is a separate test, with its own configuration and evidence, not a
coefficient selected for the baseline or an automatic training replacement.
Compare alpha-zero/original and declared weak-self settings on matched states,
inputs and RNG using the separate 2,500-update proofs. Preserve the
baseline checkpoint and discard exploratory updates. No learned scorer or
new prior is introduced.

Evaluate both **raw argmax and categorical-sample validity**, separate strict
schema/roundtrip acceptance, and diversity/unique-valid counts using declared
fixed seeds, sample counts and all-attempt denominators. State whether reported
samples are repaired; primary raw comparisons must not hide invalid draws.
Report reconstruction, size/composition/sulfur and applicable health/runtime
gates as well. Self-target fitting may reinforce existing modes or lower
diversity; a lower self-NLL is not proof of improved chemical validity,
sampling quality or useful molecules. Keep any result exploratory and scoped
to the tested panels and budget.

Source/plan identities and CPU submissions are recorded above. Allocated
test/proof outcomes and any conditional baseline training receipts will be
recorded when they exist. Authorization is not an
executed experiment or successful restart.
