# Decoder guidance from bounded valid-repair witnesses

Status, 7 October 2026: the user explicitly requested a test of gradients
guided by the rule-based repair search. **Source071 is sealed/uploaded and
CPU engineering job 29623854 DONE: 109 checks passed, one CUDA-only check skipped.**
Native V100 replacement **29625193 PEND** runs the identical sealed source071
body/config after `done(29624915) && ended(29624920)`. LSF rejected `bmod` as
administrator-only, so the held original29623874 was retired/EXIT only after
the replacement receipt and fresh held-state check. Model mathematics is unchanged.
At the **22:06 DTU-local** recovery snapshot, seed17 control is fully trained/
evaluated, alpha0.1 has five saved epochs and is running; seed23 and this
search pilot are still PEND. The search study's output root is not created yet.
Actual-checkpoint CPU gradient probe **29623910 EXIT** before Python/model
execution due to scheduler quoting of its inline command. Source072 saved
launchers recovered CPU **29623958 completed**: the actual frozen FP32 CPU
latent gradient passes neighboring FD scales 0.003/0.001 in both tested
directions. No search-guided CUDA training or trained-model improvement is established.
It is a separate focused study. Source070's defect-proxy
decoder study and source069's DeFoG random-start study retain their sealed
code, queued jobs and original protocols.

The user-directed decoder extension in `AGENTS.md` authorizes this narrow
continuation of the existing VAE. No learned chemistry critic, activity
surrogate or replacement external model is trained. The existing encoder
and its buffers remain frozen; the original checkpoint is preserved. Every
numerical fixture, witness-search experiment, training run and evaluation
executes on DTU HPC. Local AST/JSON inspection and packaging are metadata work.

## What this experiment tests

The [source070 study](39_decoder_chemical_distance_training.md) trains with
expected local valence defects and sampled graph connectivity. A cheap
differentiable proxy can improve while raw chemical validity remains low.
This additional study instead asks whether **verified, probability-supported
repair witnesses can supply useful decoder targets** and improve raw prior
generation under an independent evaluation.

The bounded search itself remains discrete and outside autograd. For a
frozen current decoder output, it proposes a valid graph close to the raw
argmax graph, preserving its occupied slots. That detached graph is a
training target. Differentiation increases the decoder's categorical support
for its target labels; it does not differentiate the RDKit validator,
search decisions, hard argmax or cache refresh.

This is an alternating fixed-target objective. It is **not the exact gradient
of nearest-valid distance**, the probability of generating a valid molecule,
or an expectation over all possible repairs. Epoch-wise target updates and
search censoring make its coverage and bias part of the measured result.

## Initializer and matched arms

Use the same untouched source047 **epoch-89 best-validity** VAE:
`/work3/s234818/latentmol/runs/20261001T001617Z_vae-train_c008c090/best_validity.pt`,
SHA-256 `1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.
Protected dataset and split identities, architecture and original
reconstruction settings are in [docs39](39_decoder_chemical_distance_training.md).
The original decoder is also an untouched evaluation reference.

The prespecified pilot compares decoder-only continuations from identical
weights, with a frozen encoder, original reconstruction and the same
occupancy/aromatic reference anchors:

| Method | Additional decoder guidance |
| --- | --- |
| Matched anchor control | No additional chemical gradient |
| Current defect proxy | The declared local-defect/connectivity objective from source070 |
| Search-target NLL, weight 0.1 | Detached verified repair-target negative log probability |
| Search-target NLL, weight 1 | Same target objective and quota, stronger declared weight |

The exact protocol is [search_decoder_pilot.json](../configs/search_decoder_pilot.json):
**one seed, 17; four cases; 16,384 reconstruction graphs; three epochs;
768 updates per case**. New decoder-only AdamW uses learning rate 3e-5,
weight decay 1e-5 and gradient cap 5; reconstruction batch 64. The entire
protected training split supplies the novelty reference, but this pilot
fits only its same deterministic subset in every arm. It is not full-data
training or independent training-seed replication.

Each update computes the parent defect objective on 16 posterior points and
16 independent fresh prior points, with equal halves; its weight is 0 in the
control/search arms and 1 in the defect arm. The search pool has 256 standard
normal points, seed 71000; eight are decoded per update in a fixed cyclic
schedule. All arms receive original-reference occupancy and aromatic KL
anchors, weight one each, on both the fresh prior and that pool. Thus the
new control differs from source070's control by the common pool anchors and
reduced training budget. Source070 is historical context, not the sole
matched comparator for this pilot.

Within each training seed, all methods must share reconstruction data/order,
posterior noise, a prescribed **256-point standard-normal training latent
panel**, batch schedule and witness-search quotas. That panel is independent
of final evaluation points. All methods perform the same search procedure
for diagnostic fairness, even when their search-target coefficient is zero.
Equal search budgets are not equal realized checks or wall time: already-valid
outputs, beam pruning and failed witnesses can differ as the
decoders diverge. Record actual work and target coverage.
Each epoch refresh searches all 256 prescribed points, including duplicates;
there is no cross-point search cache, resampling or successful-only cohort.

## Epoch-wise search and fixed-target gradients

The intended sequence is:

1. Decode each training-panel latent with the current decoder outside the
   gradient path. Preserve its raw argmax graph, probabilities, occupied slots
   and checkpoint/source identity.
2. Apply the declared bounded probability-aware search to that graph. Store
   the observed repair witness and every edit, its cost, checks, policy and
   validity evidence. Do not label it globally closest.
3. Use only a verified, slot-aligned witness as a detached target during the
   declared epoch. Freeze it for all gradient/finite-difference probes of
   the same objective. No target refresh occurs inside a derivative probe.
4. Clear unsuccessful or unsupported guidance at the next refresh; do not
   reuse a stale target, invent a valid label or replace the prescribed latent
   with an easier sample. Already-valid graphs need no repair guidance and
   are recorded separately.
5. Refresh targets at the next declared epoch boundary using that method's
   actual current decoder. Targets can differ across methods after updates;
   identical quotas do not imply identical targets.

The existing [probability-aware search](../src/latentmol/probability_repair.py)
uses `edit_count + sum(max(0, log(p_old / p_new)))` at probability weight one,
with a disclosed probability floor in **search ranking only**. Its fixed
occupancy preserves slot identity and permits element, formal-charge and
unordered bond-label edits. Its pruned, capped witnesses are upper bounds;
no witness means censored search, not proof that no valid molecule exists.
The pilot uses the **best observed probability-weighted witness**, with
maximum three edits, 256 candidate checks per depth, beam width four and
probability weight one. Record candidate checks separately from explicit
raw/witness inspections; those counters are not all internal RDKit operations.

Target NLL uses the **unconditioned** factorized decoder law: all N=50 atom
decisions including NO_ATOM, K occupied charges and K(K-1)/2 occupied
unordered bond decisions including NONE. Divide each target's full summed
NLL by N+K+K(K-1)/2, then average over all eight scheduled points with zeros
for unavailable targets. No separate per-head means or success-only
renormalization is hidden in the loss. NO_ATOM gets a direct gradient.
Stable log-softmax from live logits uses no probability floor. Search ranking
and target likelihood are different quantities and have separate receipts.

Occupancy is fixed **within** each raw-to-witness repair and held interval;
the current argmax occupancy may change at the next epoch refresh. Record
slot IDs, counts and changes. The original-reference anchors apply identically
in every arm and discourage count/aromatic shortcuts; they do not guarantee
that every generated identity or count is preserved.

Any cache must authenticate actual graph labels, exact probabilities, graph
schema, policy and refresh/checkpoint identity. Approximate or stale cache
hits must not silently turn one epoch's target into another epoch's search
result. Chemistry conversion and search never rewrite the raw evaluation
molecule or become a differentiable scorer adapter.

## Engineering gates and scientific outcomes

CPU regression fixtures come first, followed by actual-checkpoint native
gradient probes and a small matched training smoke inside the GPU allocation.
Engineering evidence must establish frozen encoder/reference states, finite
losses and decoder updates, matched RNG/budget handling, correct target/mask
alignment, valid target acceptance and retained failed/censored search records.
Fixed-target analytic gradients are checked with the target frozen; refreshing
the search during finite differences would measure a different function.
The CPU suite includes correct probability-gradient, deliberately wrong
backward, zero-gradient and no-witness fixtures. The wrong backward keeps one
latent coordinate correct and erases another: the gradient's own direction
can agree while the independent seeded direction disagrees.

Native probes use the actual frozen FP32 VAE, the first accepted training-pool
witness and both a gradient unit direction and seeded random unit direction
(seed 71002), at epsilons 0.01/0.003/0.001. Two neighboring informative
epsilon checks must agree in **both** directions for verification. The
scalar repeat/ULP roundoff proxy is not a rigorous network arithmetic error
bound. Coarse-radius mismatch can be curvature; absent targets, zero signal
or no agreement plateau is uninformative/inconclusive, not proof of bad
autograd or a passing search-gradient result. Coverage itself does not gate
the pilot's scientific outcome. A real CUDA 512-graph, one-epoch control/search
smoke checks finite training, frozen encoder and matched RNG budgets.

**Search success and absence are scientific data.** Zero guidance because all
graphs are already valid, or because no bounded witness is found, is not by
itself an engineering failure. Passing an invalid target as valid, changing
its occupied slots unnoticed or silently retaining stale guidance is an
engineering failure. Report weighted **eligible** guidance updates separately
from actual nonzero gradients; availability does not prove a nonzero gradient.
A finite zero-guidance smoke is not evidence that repair-guided learning works.

Independent final evaluation uses **4,096 unfiltered latent
points, each argmax plus four categorical draws**, full held-out deterministic
mean reconstruction on **7,347 test graphs**, and the first **64 prescribed
argmax graphs** for a separately budgeted probability-repair audit. Invalids
and duplicate candidates remain in the denominators. Raw validity, unique
valid identities per attempt, novelty against the entire training split,
size/element/bond concentration and reconstruction retention determine
utility; cheap repairs or improved target NLL cannot substitute for raw
generation improvement. Validation prior is 256 points/seed81000; final prior
uses seed91000 and sampling seed91001. The fitted 256-point pool is evaluated
separately with four draws/seed71003 to expose memorization versus generalization.
Exploratory gates require positive paired-cluster validity CIs in argmax and
sampled modes versus both original/control, no loss of unique-valid yield and
at most one percentage point deterministic-mean reconstruction validity/exact
loss. Report search versus matched defect directly too. This is one-seed
pilot evidence without multiple-comparison correction; see docs39.

Record training-panel target coverage per epoch, already-valid/censored/
unsupported counts, actual checks, edit count/probability regret,
NLL before/after updates, gradient magnitudes and decoder/encoder hashes.
Search can encourage repeatedly easy repairs or reduce diversity, so compare
these distributions with both the matched control and untouched original.
Do not claim Boltz binding, synthesis feasibility, global latent smoothness
or a guarantee of valid generated graphs from this study.

## What the actual-checkpoint gradient diagnostic measured

The CPU diagnostic used the unchanged pretrained FP32 VAE and the prescribed
training prior bank. Search found 90 verified repair targets among 256 raw
graphs, with 162 censored and four already-valid outputs skipped. It then
selected the **first accepted point, latent ID 3**, rather than a target with
the best loss or the most favorable derivative. The original model parameters
remained frozen. Its target graph and occupied slots were fixed for every probe.

At that point the normalized target NLL is **0.0991265997**. Autograd computes
its derivative through the live decoder logits with respect to the 256-dimensional
latent vector, with norm **0.1665399671**. For each unit direction `d`, the test
compares `dot(autograd_gradient, d)` with a central finite-difference slope:

```text
FD = [L(z + epsilon*d, fixed_target) - L(z - epsilon*d, fixed_target)] / (2*epsilon)
```

The first direction follows the gradient itself. An independent direction
uses a prescribed random vector/seed71002, normalized to unit length. The
second check matters because an incorrect backward can agree along its own
reported direction while missing another component; the engineering suite
includes precisely that deliberately incorrect-backward case.

| Direction | Epsilon | Autograd directional slope | Central FD slope | Declared check |
| --- | --- | --- | --- | --- |
| Gradient unit | 0.01 | 0.1665399373 | 0.1615274698 | Failed |
| Gradient unit | 0.003 | 0.1665399373 | 0.1644119620 | Passed |
| Gradient unit | 0.001 | 0.1665399373 | 0.1657418907 | Passed |
| Seeded random unit | 0.01 | -0.0015427149 | -0.0016350299 | Failed |
| Seeded random unit | 0.003 | -0.0015427149 | -0.0015497208 | Passed |
| Seeded random unit | 0.001 | -0.0015427149 | -0.0015608966 | Passed |

The implemented gate requires informative agreement at **two adjacent scales
in both directions**. Its per-row allowance is 3% of the analytic directional
magnitude plus twice the recorded scalar ULP/repeat-noise proxy and 1e-7.
Repeated baseline values had zero observed difference. The allowance is an
engineering rule; a scalar ULP estimate is not a rigorous bound on every
floating-point operation in the network. Coarse 0.01 probes failed while both
smaller neighboring probes passed. A finite interval averages a nonlinear
slope, so that coarse mismatch alone does not establish broken autograd.

The [saved probe report](../outputs/hpc-results/source072-search-checkpoint-probe/extracted-completed001/actual-probe/probe/probe.json)
establishes local agreement for **one CPU latent point and a fixed valid target**.
It does not differentiate search, establish an exact nearest-valid gradient,
verify all latent points or decoder-parameter gradients on this pretrained
model, perform an update, or demonstrate improved generated molecules. The
bounded 90-target coverage is search evidence, not raw decoder generation
validity. CUDA probes and the matched decoder-training comparison remain
separate required experiments.

## Source071 identity and execution stages

The pilot is one serial V100 allocation, four cores and 16GB/core host memory,
with an eight-hour scheduler cap, seven-hour controller budget and one-hour
case budgets. Reserve 1,800 seconds before starting each case; oracle/evaluation
durations are not hard-killed and the reserve cannot guarantee all cases fit.
Retain incomplete/censored/unattempted cases rather than ranking their partial
checkpoints. Native submission requires CPU DONE and **ended(29622699)**,
so this GPU study follows the currently running source070 study.

Core files are [search_repair_guidance.py](../src/latentmol/search_repair_guidance.py),
[search_decoder_study.py](../src/latentmol/search_decoder_study.py) and the
optional guidance hook in [decoder_finetune.py](../src/latentmol/decoder_finetune.py).
The CPU command is `bash scripts/hpc/search_decoder_tests.sh`; the native
entrypoint is `python -m latentmol.search_decoder_study --config ... --sha256 ...
--output-root ...`. Submission uses `scripts/hpc/submit_decoder_validity.py
--kind search`, authenticated sealed releases and duplicate-safe receipts.
Source071 archive SHA-256 is
`3ca7c93a13a1ca91c942159baeac3576ac1085edc2699cf5c357e578799408ec`;
scientific manifest SHA-256 is
`998c62b8bc0223874e3bae7a3006732a71c47d02285bdc38bedd97ed3855914b`;
config SHA-256 is
`c51a0c3657d070a66ff2c6702fe370daa474cc0f80b4d5ba99665452256ffdf0`.
The immutable uploaded release is
`/work3/s234818/latentmol/releases/source-071-search-decoder-attempt001`,
with receipts at
`/work3/s234818/latentmol/submissions/source071-search-decoder-attempt001`.
CPU **29623854 DONE** passed 88 foundation fixtures/one CUDA-only skip in
12.78 seconds and 21 search/pipeline/gradient fixtures in 9.43 seconds:
**109 passed, one skipped**. These include correct, deliberately incorrect,
zero-gradient and no-witness cases and establish engineering behavior only.
CPU run is
`/work3/s234818/latentmol/runs/source-071-search-decoder-attempt001-cpu/20261007T141150Z_decoder-validity-cpu_351565a5`.
Native **29623874 PEND** has exact dependency
`done(29623854) && ended(29622699)`. Source070 was observed RUN and source069 PEND in the
coordinator's current authenticated check; their protocols remain unchanged.
The downloaded [engineering/submission archive](../outputs/hpc-results/source071-search-decoder/engineering-evidence.tar.gz)
has SHA-256 `e09977b5aa5a71d462478b71191ccdfa8b0f047827235a8f31ef5fdf92ace185`;
all 14 data-file hashes in its 15 members were verified by local metadata-only
checks. The fifteenth member is the digest manifest; the outer SHA covers all.
See the [completed CPU run/environment/command](../outputs/hpc-results/source071-search-decoder/extracted/cpu/run.json),
[pytest stdout](../outputs/hpc-results/source071-search-decoder/extracted/cpu/command.log),
[native receipt](../outputs/hpc-results/source071-search-decoder/extracted/receipts/native-receipt.json)
and [scheduler snapshot](../outputs/hpc-results/source071-search-decoder/extracted/scheduler_snapshot.json).

The first attempted actual-checkpoint **CPU probe 29623910** requested
four cores, 4 GB/core and 45 minutes in queue `hpc` after CPU gate DONE.
It loads the authenticated FP32 original VAE on CPU, searches all 256 training
prior points and invokes the sealed fixed-teacher gradient probe without
training. Inline orchestration is retained in submission/ExperimentRun
receipts; the sealed source and config are unchanged. Output is
`/work3/s234818/latentmol/runs/source071-actual-checkpoint-cpu-probe-attempt001`,
with receipts `submissions/source071-actual-checkpoint-cpu-probe-attempt001`.
It **EXIT before Python or model execution**: LSF mangled the multiline
`python -c` command, producing an unmatched-double-quote EOF and a
`CPU: command not found` shell error. This failure does not test the gradient.
Its original submission/logs remain preserved. Recovery uses saved
[search_actual_checkpoint_probe.py](../scripts/search_actual_checkpoint_probe.py),
[search_checkpoint_probe.sh](../scripts/hpc/search_checkpoint_probe.sh) and
[submit_search_checkpoint_probe.py](../scripts/hpc/submit_search_checkpoint_probe.py)
in immutable source072. Recovery CPU **29623958 submitted** requests 45 minutes,
four cores and 4 GB/core (16 GB host total) in `hpc`. It reruns the affected
21 search/pipeline/gradient fixtures before the actual pretrained FP32
256-point search and fixed-target latent gradient probe. No decoder is trained;
an actual CPU model result is distinct from CUDA or held-out quality validation.

[Source072 archive](../outputs/source-releases/source-072-search-checkpoint-probe-attempt001.tar.gz)
SHA-256 is `7883456f7cb90f4ceb6da82ad15c4afa0f534d162c841a20d9279730a99dbb20`;
scientific manifest is
`1ad3c4b7cef9bb25a9ce410b92d2e2526f47f0282a24f574bc9555915f91512c`.
The config retains source071's SHA
`c51a0c3657d070a66ff2c6702fe370daa474cc0f80b4d5ba99665452256ffdf0`.
Release and receipts are respectively
`/work3/s234818/latentmol/releases/source-072-search-checkpoint-probe-attempt001`
and `/work3/s234818/latentmol/submissions/source072-search-checkpoint-probe-attempt001`.
Output is `runs/source-072-search-checkpoint-probe-attempt001-probe`, wrapper
is `runs/source-072-search-checkpoint-probe-attempt001-cpu-probe`, and logs are
`logs/source-072-search-checkpoint-probe-attempt001-cpu-probe.29623958.{out,err}`.
Exact `bsub` command is in the durable `actual-probe-intent.json`.
The downloaded [recovery snapshot bundle](../outputs/hpc-results/source072-search-checkpoint-probe/cpu-probe-evidence-snapshot001.tar.gz)
SHA-256 is `0bab3c82028556d5511c4517a4819b007ce1b456b6dfdd29cb9eb21fdd4febcc`.
All 15 data-file hashes in its 16 members were checked by local metadata-only
safe extraction; the remaining member is the digest manifest. At
**14:36:40 UTC / 16:36:40 DTU-local**, recovery **29623958 RUN** had passed
21 engineering fixtures in 10.63 seconds; actual-checkpoint phase had no
summary. See [scheduler snapshot](../outputs/hpc-results/source072-search-checkpoint-probe/extracted-snapshot001/scheduler_snapshot.json),
[exact intent/command](../outputs/hpc-results/source072-search-checkpoint-probe/extracted-snapshot001/receipts/actual-probe-intent.json),
[partial wrapper metadata](../outputs/hpc-results/source072-search-checkpoint-probe/extracted-snapshot001/wrapper/20261007T143110Z_search-actual-checkpoint-cpu_025a322c/run.json)
and [pytest stdout](../outputs/hpc-results/source072-search-checkpoint-probe/extracted-snapshot001/wrapper/20261007T143110Z_search-actual-checkpoint-cpu_025a322c/command.log).
The bundle also retains the failed launch's receipts/stderr. Final 16:37 local
check was still RUN/PEND; SSH closed normally. No actual-checkpoint gradient,
CUDA result, decoder training or chemical improvement was established at that
earlier snapshot.

**Completed recovery:** [actual summary](../outputs/hpc-results/source072-search-checkpoint-probe/extracted-completed001/actual-probe/summary.json)
and [FD rows](../outputs/hpc-results/source072-search-checkpoint-probe/extracted-completed001/actual-probe/probe/probe.json)
verify the frozen FP32 **CPU latent gradient at first accepted pool point ID 3**,
norm 0.1665399671. Both directions pass adjacent epsilons 0.003/0.001; 0.01
fails. Search found 90 accepted/162 censored/four already valid among 256
points, with 193,005 candidate checks. Wrapper completed at 14:44:21 UTC,
about 13 minutes 10 seconds after starting. Verified
[completed-evidence001.tar.gz](../outputs/hpc-results/source072-search-checkpoint-probe/completed-evidence001.tar.gz)
SHA is `d3a169040ff574de4d13c94029867bef45cf2f4f323c07139a9b6a124afc19df`.
This validates one fixed-teacher latent derivative, not decoder learning or
improved raw generation.

The [17:58:17 DTU-local timing snapshot](../outputs/hpc-results/source072-search-checkpoint-probe/extracted-completed001/status_and_timing_snapshot.json)
still shows source071 PEND behind source070 RUN. Source070's first case is
epoch6/10 at 8,348.42 seconds: roughly 23 minutes/epoch projects ten epochs
at 3 hours 52 minutes, beyond its sealed three-hour cap. Censoring is likely
if pace holds, but no timeout is observed yet. The controller retains censored
cases and continues to the next arm; partial checkpoints are not completed
comparisons. The search pilot's `ended` dependency releases after source070
DONE or EXIT. That read changed no protocol/job. Subsequent hold/retirement
and recovery actions are in [docs41](41_decoder_budget_recovery.md); current
pilot state is PSUSP pending dependency reconciliation. No CUDA training or
chemical improvement is established.
Every submitted or failed stage
must also be retained in the [experiment ledger](05_experiment_ledger.md).
