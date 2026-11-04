# Adaptive validity and score-transfer research

User authorization: 3 November 2026, continue experiments autonomously, using
results to select follow-up tests and a quiet scheduled HPC monitor. Preserve
existing experiments. Develop on `codex/valid-molecule-research`; promote a
concise result document to main only after a useful finding survives an
informative larger comparison. Do not merge the research branch wholesale.

Question: which changes produce valid discrete molecules whose useful native
scores survive decoding, rather than only improving a relaxed proxy?

First test: native Protenix contact on selected saved soft graphs, their raw
argmax and their saved valid repair targets. Keep key17, frozen model/context,
precision and scoring definition identical. Include fixed-logit temperature
controls. No optimizer updates or teacher recomputation in this audit. Invalid
hard graphs remain explicitly invalid observations. Fixed-context contact is
not affinity or experimental binding.

Follow-up decisions:

1. If saved repair targets retain useful hard scores, test short optimization
   plus explicit valid candidate selection/projection using the existing builder.
2. If a few persistent defects dominate, inspect their directional repair versus
   binding gradients, then compare focused edited-bond guidance with the existing
   full-target loss. Keep atom choices trainable and targets detached and fixed
   within each step.
3. If valid candidates are lost through oscillation, compare a lower LR alone
   and noise off alone from one common checkpoint at fixed T and chemistry.
4. If soft gains fail to transfer, prioritize common-graph scoring or constrained
   candidate construction over another unchanged long run.

Use at most two new model jobs concurrently for this research; start with small
causal probes (normally <=2,500 updates per arm) before selecting one promising
method for a matched 20k-50k confirmation. This is a budget policy, not a stopping
rule for existing runs. Use measured runtime and compatible available GPUs.
Compare equal budgets and a small number of seeds, preserving negative results.
Judge best-valid hard score, valid identity discovery, argmax and unfiltered
sampled validity, and wall/model-call cost. Keep a best-valid diverse archive;
report early-best and final separately.

All substantive numerical/model work runs on allocated HPC. Use proportional
portable tests, the shared own-output storage helper, fresh sealed snapshots,
one writer per output, and actual accepted submission receipts. Reconcile any
unknown submission before retrying. Never modify live/sealed sources or other
projects' jobs. No broad qualification campaigns or cleanup for routine polls.

The scheduled task reads `research-state.json` and actual receipts. An unchanged
RUN/PEND state needs no tests, rebuild, cleanup, documentation or commit. An SSH
outage is not a failed run. A useful result needs a matched baseline and larger
confirmation before promotion. Keep routine notes here on the research branch;
main receives only a compact verified finding with limitations and evidence links.

First outcome, 3 November 2026: audit GPU **29884143 DONE/V100**, 16 checkpoints,
63 native forwards, 29.94 seconds of worker evaluation. Actual same-job LSF,
source/plan/result bytes, input identities and frozen model/context were verified
independently. See [verification](audit-verification.json) and copied allocated
[point measurements](audit-results.json). No optimizer updates occurred.

At sparse update30k, the saved valid repair scored0.390706 hard contact versus
0.367041 for the raw valid50k endpoint. At20k, relaxed contact0.878212 fell to
0.324068 on the invalid hard graph and0.235913 on its valid repair. Structural1
at30k scored0.854596 soft versus0.208217 invalid hard/0.216255 valid repair.
Thus score transfer is a substantial issue, and existing repair candidates can
already retain useful contact. These selected checkpoints do not establish a
generally superior method, true affinity or a full pose-rescoring result.

Next: paired2k-update pilot from the same saved sparse30k state. Compare original
logits with its saved repair initialized as98% one-hot plus2% uniform. Both use
T0.4, chemistry30, LR0.3, no noise and the same original model/context/key17 and
repair builder/loss. These are explicitly new experiments with local steps0-2k,
not edits or continuations of the saved50k parent. The parent checkpoint was
recorded atT0.52; the common fixedT0.4 is a declared pilot condition. Score hard
valid candidates at a shared retained cadence and retain unfiltered samples.
Existing sparse/VAE runs continue independently.

Paired pilot **29884187 RUN/V100**, submitted3November21:14CET,90min,
8CPUx4GiB/oneexclusive32GBV100. Its first arm had131 committed updates at21:16CET;
the second runs serially afterward. Seventeen portable controls passed, and both
actual metadata startup and own512MiB entry storage passed. Accepted held receipt,
source/plan hashes and owned release were independently verified. See
[dispatch](pilot-dispatch-verification.json). No terminal outcome is claimed.
The intervention combines valid repair with a finite-support confidence reset;
a pure projection attribution would require a confidence-matched control.

Commands from this worktree (fresh launch tags only):

```text
py -3 experiments/brd4-jq1/protenix-random-chem/scripts/launch_score_transfer_audit.py --inputs INPUT_PACKET --tag NEW_TAG --kind projection-pilot
py -3 experiments/brd4-jq1/protenix-random-chem/scripts/observe_score_transfer_audit.py --tag ACTUAL_TAG
```

The first launch tag20261103-score-transfer-audit-01 is completed: never replay it.
The active scheduled task is `mosaic1` / Adaptive molecular validity research,
every two hours, operating on this branch and research-state record. Update its
actual next action/receipts before leaving any submitted or unknown outcome.


3 November23:02CET terminal audit: pilot29884187 actual same-job LSF/bhistDONE.
Both2k arms and2,001-state contiguous prefixes complete; source5files,57remote
file bindings, input bytes and frozen identities independently verified. Valid
repair initialization retained1,793valid states versus0 from original logits,
but found only its initial valid identity. Every retained valid hard score was
0.390705585; relaxed contact rose from0.383700 to0.611472 at best, without a
corresponding hard gain. Endpoint unfiltered valid draws were5/128 for the
repair arm versus21/128 for original logits. Thus validity retention improved,
while distribution robustness and useful discrete optimization remain unresolved.
See pilot-terminal-verification.json. No larger confirmation is justified yet.

Next causal test:2k updates perarm from the SAME saved valid-repair mixture,
T0.4/chem30/LR0.3/no noise/key17. Compare relaxed binding forward with raw-argmax
binding forward plus an explicitly BIASED identity probability-gradient estimator
(straight-through), followed by the unchanged softmax/logit pullback. This is NOT
the derivative of discrete argmax and is not derivative certification. Chemistry
still uses relaxed probabilities and detached step-fixed original targets.
Only the binding input/estimator changes. Both arms use the same inherited
update budget and the same21 retained hard evaluations at0/100/.../2000;
no extra best-soft candidate forward can bias the model-call budget. Both start
valid; invalid later hard graphs remain labeled and never become binding claims.
The inherited report's historical soft-score label is inapplicable to the hard
arm; its owned score-scope.json explicitly corrects it. Do not publish through
the canonical report adapter until this input schema is supported. Record actual
terminal native hard scores and new valid identities before selecting a larger
confirmation. One serial pairedGPUjob,90min/512MiB; earlier pair took34.1min.
Twenty portable metadata/ownership controls passed;3actual analytic forward/
surrogate controls run only inside the new allocation before native execution.


Actual successor29885132 accepted/held receipt persisted before ownedrelease;
PEND/V100 at23:07:27CET. Source5files/manifest97ea75244d0ad5ca603a2553ee3dc2f027e37e1d72c91b887f48e2ea8e465157,
plane9c2c81878cfe4bbb2b63aa2fc1b1f2562ff815bce47a01af404831003d6b278.
Actual startup metadata/command/resources/heldrelease/owncapacity independently
verified in binding-forward-dispatch-verification.json. No native/gradient result
yet. Do not replay the launch. Current account sparse/VAE producers remainRUN;
their previous startup IDs are historical, actual VAE DBMol successor29883315
with callback29883316 is read-only context. No other-owner job was modified.


4 November 2026, 01:05 CET — Binding-forward comparison verified complete.
Job29885132 actual DONE, same-job LSF/bhist success, both 2,000 updates and
2,001 contiguous states. Five source files and58 remote file bindings verified;
input/frozen identities preserved. All three allocated surrogate controls passed.
Hard-forward retained1,108 valid states versus1,793 relaxed, and both retained
one valid identity with the same best native hard contact0.390705585. The identical
21-point hard evaluation grids do not show discrete improvement. Hard-forward
endpoint samples were16/128 valid versus5/128, a separate distributional metric.
This mixed/negative result does not justify a larger confirmation. Stop expanding
the biased estimator for now. See binding-forward-terminal-verification.json.

Next question: do the already-saved categorical proposals contain better valid
hard candidates than the single argmax identity? A read-only allocated native
audit will score existing endpoint samples from original-logit refinement,
repair+relaxed refinement and repair+hard-forward refinement. Each group retains
all128 unfiltered draws and its invalid denominator; no new sampling or repair,
no optimizer updates. Bind each raw NPZ to its actual completed producer/report,
then verify validity and score at the original model/context/key17. Cache only
identical slot-labelled graphs; canonical SMILES alone does not imply identical
native input. Bound unique sample scoring to64, deterministic input order, with
any cap-censored valid draws explicit. Include the saved30k soft/raw/repair point
as a common reference, at most67 native forwards total. This is candidate selection,
not an equal-update efficacy comparison; unequal valid proposal counts remain
visible. Contact is still geometry-off, fixed-context surrogate, not affinity.
Request20min/V100,4CPUx8GiB, own128MiB results/256MiB complete group. The preceding
63-forward audit took29.94s worker time. Twenty-two portable metadata controls
passed, including producer/report/raw-byte closure and capacity/denominator bounds.
Actual native result is pending; only a demonstrated useful candidate direction
will receive a matched larger confirmation. Existing live experiments unchanged.


Actual new audit29886311 accepted and owned-release acknowledged at01:09CET.
Immediate01:09:42 scheduler still reportsPSUSP, so allocation/start is not claimed;
retain the successful bresume receipt and do not blindly resume again. Next check
reconciles actual scheduler/history. Four source files, metadata startup, held
receipt-before-release, original report/raw sample input closure and capacity
passed independent dispatch verification. Source53c71b95070e60eb045effa92258a649a0e0f4682d86827ac4df671eb184cac4,
plan758cce4263370f9fc4a197c93a2baaf8d04d4ae51fdf8d4885abe70c7a695c1a.
See candidate-audit-dispatch-verification.json; no native findings yet. All live
sparse/VAE jobs remain read-only context and RUN at the initial account capture.


4 November 2026, 03:09 CET — Saved candidate audit29886311 verified DONE.
Actual same-job LSF/history, four source files,19 remote file bindings and frozen/
input preservation passed. All384 saved draws were accounted for (valid21/5/16
across the three128-draw groups). Thirteen distinct slot-labelled valid graphs
needed13 native sample evaluations plus3 reference forwards; elapsed19.26s, no
censoring, no optimizer updates. Best valid sample hard contact0.446152449 versus
saved-repair argmax0.390705585. This is one saved-parent candidate-selection signal,
not improved affinity or evidence that a new optimizer works. See
candidate-audit-terminal-verification.json. Original PSUSP was transient; actual
job started01:09:37 and finished01:10:00CET, with no extra bresume needed.

Next causal pilot:2,000 updates perarm, starting from the SAME saved30k valid
repair with2% uniform support, T0.4/chem30/LR0.3/no noise/key17. Both use four
500-step blocks through the original sealed optimizer, preserving every scalar
and snapshot. Block-local counters and explicit global offsets are recorded;
fixed policies have no cooling/noise schedule to reset. At each boundary, both
sample128 unfiltered categorical proposals using the same global-step seed rule
and score ALL128, including invalid/duplicate draws, plus the current hard graph.
Both keep the best valid archive. Only treatment feeds that best valid candidate
back into the next block if current is invalid or a different candidate has a
higher hard score. Same valid identity never triggers confidence-only resetting.
Projection uses the tested2% uniform mixture; atom choices remain trainable,
chemistry targets are detached and fixed within each optimizer trial. Projection
is an explicit outside-trial intervention, with before/selected records.

Equal budgets:2,000 gradient updates and517 hard diagnostic forwards perarm
(one initial plus4x129). No early quality stop or invalid-graph abort. Compare
best-valid hard score, distinct valid identities, valid update-source fraction
(2,000 sources perarm, excluding duplicated block endpoints),128-draw validity,
reset count and runtime. Both methods deliver their best archive; treatment does
not receive privileged candidate selection. This isolates feedback within the
block protocol, not all differences from an old unsegmented trajectory. Baseline
runs first, so compilation warming limits runtime comparisons. No final reset is
applied after update2,000. Request one90min V100 job/8CPUx4GiB,512MiB complete cap;
prior2x2k took33–34min and additional forward cost fits the measured allowance.
Twenty-four portable metadata/selection/ownership controls passed; native outcomes
remain pending. This extends the existing runner/launcher, not a new controller.
After a useful pilot, run the authorized matched20k–50k confirmation; if feedback
fails, preserve it and select a different direction rather than tuning a grid.


Actual feedback pilot29887463 RUN/V100 at03:16:03CET, accepted held receipt
persisted before owned release. Source5files ce03e7e051a40d6999d8d260301c5c57e52a610ca87ed1884fb9852f8770fcef,
planb3ea03ab731cdae46364ace215a013f39150be4cb71b4be8fa924223cbc942a2.
Metadata startup, source bytes, exact90min/8CPUx4GiB/exclusive32GBV100 vectors,
held/release and full own512MiB entry capacity independently verified. Native
terminal outcome and completed gradients/updates are not yet claimed. See
projection-feedback-dispatch-verification.json. Never replay launch or release.

### Projection feedback result and matched confirmation

Job 29887463 finished both 2,000-update arms on V100. The terminal audit
verified 517 hard diagnostic forwards per arm, unchanged frozen scorer and
inputs, and successful LSF completion. See
[the terminal verification](projection-feedback-terminal-verification.json).

| Outcome | Archive only | Projection feedback |
| --- | ---: | ---: |
| Best valid hard contact | 0.414022 | 0.434865 |
| Final raw hard contact, both valid | 0.390706 | 0.434865 |
| Valid update sources | 1,792 / 2,000 | 1,706 / 2,000 |
| Distinct valid SMILES | 11 | 5 |
| Valid endpoint samples | 5 / 128 | 0 / 128 |
| Feedback resets | 0 | 2 |

The score gain justifies a 20,000-update-per-arm confirmation. The lower
diversity and endpoint sample validity make both measures essential to
judging the result. Hard contact measures the frozen scorer's contact
output. The comparison uses one starting parent and key, so it tests
persistence over a longer run at that start.

Keep the same 500-update blocks, 128 unfiltered draws at each boundary, and
5,161 hard diagnostic forwards per arm. Both arms start from the same saved
30k repair with 2% uniform support, temperature 0.4, chemistry weight 30,
LR 0.3, zero noise and key 17. The only intervention is feeding an eligible
archived candidate back into the treatment. Run both arms serially on one
32 GB V100. The pilot took 2,019 seconds overall. Scaling by ten, adding a
25% margin and 600 seconds for setup gives a minimum internal allowance of
25,836 seconds. Request eight hours with a 28,200-second internal limit and
eight CPU slots at 4 GiB each. Admission checks projected writes against
the 1 GiB complete cap and 8,000 files. All 24 focused local controls passed.

Confirmation job 29888169 started on V100 at 05:19:38 CET. The accepted
receipt and held proof precede release. LSF history records actual dispatch
and RUN after release. Metadata startup authenticated the measured pilot
and projected 277,159,752 bytes across 6,678 files. The dispatch and entry
capacity checks passed. The initial observation recorded zero committed
updates. See [dispatch verification](feedback-confirmation-dispatch-verification.json)
for the source, plan and receipts.
