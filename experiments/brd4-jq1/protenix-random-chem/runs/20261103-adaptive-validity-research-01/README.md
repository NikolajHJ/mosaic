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
