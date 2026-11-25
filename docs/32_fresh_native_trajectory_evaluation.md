# Fresh-native evaluation along saved latent trajectories

Status, 6 October 2026: **corrected native-evaluation jobs 29609963/29609964
submitted and queued**, using immutable source059 and the unchanged
source058 fresh-native protocol/budgets. Source059 passed 112 HPC fixtures,
all 13 actual historical-parent readers and its separate chemistry audit.
The known-broken original jobs 29604253/29604252 were cancelled while
pending at 09:52:20/21 DTU-local; their release, earlier snapshots and CPU
preflight evidence are preserved below. No new fresh-native outcome is
established by the replacement submissions.
The user requested a comprehensive checkpoint evaluation using ordinary
Boltz-2 native prediction, fixed argmax/categorical selections and multiple
fresh-pose seeds. This document describes the implemented boundaries;
the checkpoint grid, draw budget and scope below were selected before
native outcomes. Source/config identities and CPU/GPU receipts are now
recorded. The CPU audit reused existing native predictions; no new source058
native prediction or scientific result has been established.
Follow the [experiment ledger](05_experiment_ledger.md)
for verified execution records.

The existing [random-start study](30_sequential_random_start_study.md) and
[carbon/protein panel](31_carbon_protein_panel.md) remain unchanged. This
evaluation reads saved hard candidates and runs forward-only native
predictions outside the differentiable optimization branch. It trains no
new model and does not feed future native scores back into the saved runs.
The strict protocol is
[configs/boltz_fresh_trajectory.yaml](../configs/boltz_fresh_trajectory.yaml).
Config SHA-256 is
`ec6418cf2bd99e35c15e11946b35fccabf1714e35b1adc4aec315f8cc0b67ee1`.
The ledger records the attempt001 archive/source hashes and all receipts.

## Verified execution snapshot

**Corrected native submissions, 10:12 DTU-local:** primary **29609963**
requires `done(29609935) && ended(29604187)`; resume **29609964** requires
`done(29609963)`. Both were accepted at 10:12:19 and remain PEND in the
saved scheduler snapshot. Each requests one exclusive `gpua40` GPU,
four cores, 8 GB per core, one host and 23 hours. The maximum requested
sequential budget stays 46 hours; full request-list completion remains
separate from allocation completion.

The release is
`/work3/s234818/latentmol/releases/source-059-diagnostics-attempt001`,
archive SHA `7941c38554eebd2dd928c5d896bc5ae13bbe5cf30bd92c546e1f37ebbffa37f6`,
source-manifest SHA
`d29e5a4e1ba0a08382c9fa6b46bc57c21e8f354680e671c0978d88349bd6f694`.
The original protocol SHA above, checkpoints/draws/native settings and
`/work3/s234818/latentmol/runs/source058-fresh-trajectory` output root are
unchanged; the previously pending jobs created no native study there.
CPU 29609935 completed at 10:12:36 after its fixture/reader checks and
separate 124-state probability-repair audit. That audit makes no Boltz
calls and is not a native score-retention or gradient confirmation.
Exact command/receipt JSON and `scheduler-new-jobs.txt` are retained under
`outputs/hpc-results/source059-diagnostics-attempt001/`, with full identities
and outcomes in the ledger. The original numerical gradient failures are
still being diagnosed separately; queued six-case replay **29609958**
requests L40S rather than the original A40 hardware, and has no outcome yet.

**09:41 DTU-local update:** source056 processed all ten random seeds,
with nine finite-difference failures and one completed two-arm pilot
(seed 7) whose rescore failed before prediction. Its parent configuration
was hashed with integer `contexts` keys; JSON reload produces string keys
and changes sorted-key ordering. Read-only restoration of only those key
types reproduces the stored hash exactly. Sealed source058 checks the
same hash before pilot status for **all ten** random parents, and selection
does not catch this error. It would abort before new native predictions.
The CPU preflight below covered C13 artifacts rather than these random
records, so its pass does not cover this incompatibility. A corrected,
separately sealed release must preserve the integrity check under a
consistent representation; bypassing it is not an acceptable fix.
No code fix, queued-job modification or resubmission has occurred.
The [ledger](05_experiment_ledger.md) records the exact hashes and artifact
paths. Source057 is still running, with 35 completed, three failed, one
optimizing and 61 unstarted cases; its continuation remains queued.

**Recovery implementation staged, not applied to cancelled source058:** the
reusable [configuration identity module](../src/latentmol/config_identity.py)
normalizes JSON object keys before sorting/hashing new records and declares
`canonical-json-config-v1`. It rejects integer/string key collisions,
unsupported key/value types, nonfinite values and unknown hash versions.
Historical active-set readers may explicitly reconstruct only unversioned
`contexts` count keys 1–50, and only when that exact reconstruction matches
the stored digest. Changed values, unrelated key changes and versioned
mismatches still fail. The verifier returns the used scheme without
rewriting historical records or hashes. HPC fixtures and existing-artifact
validation are required before a separately sealed replacement can be
submitted; no pass or queued-job replacement is claimed here.

**Historical intermediate reader-validation snapshot:** source059 retry CPU job **29609935** passed
112 regression fixtures and all **13 actual historical-parent readers**,
including exact legacy-context identity recovery. Its probability-repair
audit was still running at that intermediate snapshot; this is separate from fresh-native scoring and a
gradient gate pass. The original failed CPU launch 29609924 used a malformed
65-character copied protocol digest and failed before fixtures; unchanged
code was retried with the file-derived digest. Validated source059 reader
code can support a separately receipted native replacement, but no such
job or new native outcome is established by this reader validation alone.

**09:52 DTU-local update:** both original GPU jobs were verified `PEND`
immediately before owner cancellation, resume 29604253 first and primary
29604252 second. They had zero execution; `TERM_OWNER` is not a native
prediction failure. Cancellation evidence is under
`outputs/hpc-results/source059-diagnostics-attempt001/`. The historical
immutable release and CPU evidence are retained, and source057 is unchanged.
Corrected-release fixture/parent validation must precede replacement
submission; no replacement job or new native prediction is established.

Immutable attempt001 contains 234 executable source/config/script/test
files. CPU job **29604247** completed on `n-62-30-23` in queue `hpc`,
00:25:28–00:26:07 DTU-local on 6 October, with a 39-second job wall time.
The five selected fixture modules passed **62 tests in 11.61 seconds**.
The processed-input preflight verified **1,818 historical hard candidates**
and all **four** existing full-C13 native predictions were retention-eligible.
Both outer workflow and inner audit records are `completed`, with null
errors. These are engineering checks of saved evidence, not new predictions.
Environment: HPC Python 3.11.9, GCC 12.3.0, Boltz 2.2.1, Torch 2.5.1+cu121
and RDKit 2024.9.6. The retained limits of the audit below still apply.

CPU completion was **manually checked before GPU submission**. Primary
job **29604252**, accepted at 00:27:13, depends on `ended(29604187)`;
there is no scheduler dependency on the already completed CPU job. Resume
job **29604253**, accepted at 00:27:37, was submitted with
`done(29604252)` so a fatal primary failure does not automatically trigger
the continuation. Each requests one exclusive `gpua40` GPU, four cores,
8 GB per core on one host and 23 hours. The two sequential limits total
46 hours; completion remains governed by the sealed request-list state,
and the controller exits early when all requests are terminal.

Downloaded evidence is in
`outputs/hpc-results/source058-fresh-trajectory/submission/` and
`outputs/hpc-results/source058-fresh-trajectory/preflight/`. The latter
contains LSF stdout/stderr, outer and inner run records, the command log
and all four identity audits. Queued GPU jobs have not passed their own
fixtures or completed any fresh-native request at this snapshot.

## Evaluation question

Does an improved relaxed fixed-pose objective remain improved after hard
decoding, ordinary native preprocessing and generation of new poses?
Compare predeclared checkpoints with each trajectory's own start, keeping
binary and continuous affinity outputs separate. A positive binary-score
change is not an improvement in both heads or evidence of measured binding.
The continuous head's intended activity comparisons remain restricted to
actives; its increasing log-IC50 direction does not indicate stronger
predicted activity.

## Predeclared checkpoints and sample scope

Use states **0, 10, 25, 50, 75 and 100** and native-pose seeds **17, 23
and 31**. Select hard candidates by recorded draw index before seeing
fresh-native predictions; no candidate is replaced by a later valid,
higher-proxy or higher-native-scoring draw.

| Scope | Draws per checkpoint | Maximum hard candidate memberships |
| --- | --- | ---: |
| All 100 source057 target/length cases | Argmax | 600 |
| C8 and C13 on all ten source057 proteins | First four categorical samples in addition to argmax | 480 |
| Completed source055 full C13 run, both arms | Argmax plus all eight categorical samples | 108 |
| Ten source056 random starts, both declared arms | Argmax plus first four categorical samples | 600 |

Thus the full planned selection has **1,788 memberships**, or at most
**5,364 logical membership/pose-seed outcomes**. Invalid or unavailable
memberships remain explicit; deduplication can reduce actual native
predictions. Repeated states, paired arms and reused predictions are not
independent replicates. The parent scopes retain 100 carbon-panel cases
and ten original random draws, even when some never optimize or have
no native-eligible starting graph.

## Native settings and resource plan

The declared fresh-native budget is **100 diffusion steps**, one sample,
zero structure recycling; affinity uses **100 diffusion steps**, one
sample and its native recycling budget **5**. Use `bf16-mixed`, an MSA
subsampling budget of 128 and **no molecular-weight correction**. Preserve
the declared target sequence/MSA/model identity when creating each native
input. These settings differ from the zero-recycle fixed-pose proxy, so
native-minus-proxy changes combine hard decoding, preprocessing, native
recycling and pose changes. They do not isolate which stage loses signal.
The exported `proxy_binary` is the original relaxed latent-state score;
it is shared by every saved hard draw at that state. Source058 does not
compute a separate fixed-pose proxy for each hard candidate. A sample's
native score therefore must not be presented as retention of that sample's
own hard fixed-pose score.

The native MSA parsing cap is **8,192 sequences**, distinct from the later
128-row subsampling budget. Native affinity recycling and mixed precision
are pinned defaults; unsupported modes are rejected rather than described
as effective CLI overrides. Both confidence and affinity checkpoints are
pinned.

After CPU preflight passes, the plan is **two serial 23-hour GPU allocations**:
primary after source057 continuation job 29604187 ends, followed by a
continuation of the same request list. This initially requests at most
46 GPU wall hours. Each driver has a **22-hour budget** and reserves
**2,220 seconds** before starting another request: 1,800 for native
prediction, 300 for identity audit and 120 for cleanup/persistence. The
maximum logical-outcome count is not a guarantee of full completion in
those allocations. Additional continuation, if needed, requires its own
verified receipt and ledger entry. No parallel per-candidate jobs or
compute-node submissions occur.

## Selection and serial native pipeline

The [selection module](../src/latentmol/boltz/trajectory_selection.py)
constructs **122 declared roles**: 100 panel cases, twenty arm roles from
ten random starts and two full-C13 arm roles. Missing parents/checkpoints
remain unavailable memberships; malformed or changed existing evidence is
an integrity failure. Candidate chemistry is recomputed from saved hard
labels, verifying graph/schema/draw identity, parent source/config and
target/MSA, and passed-context gates. Completed parents also receive
full-budget/frozen-state checks. Available partial/failed-parent states
may be evaluated, but retain their pilot/parent status and
`full_budget_verified: false`; scoring does not complete the optimization.

Selection seals parent file sets and freezes `selection.json` before new
native outcomes. Verification detects additions/deletions as well as
changed bytes. Requests are keyed by target sequence/MSA, canonical SMILES,
pose seed, native settings and model checkpoints. Repeated identities can
reuse one prediction without becoming independent replicates. Deterministic
request order covers initial argmax states first, then increasing checkpoint,
draw and role order; no proxy or native score controls that order. Invalid
graphs remain members without entering an incompatible native interface.

The [controller](../src/latentmol/boltz/fresh_trajectory.py) verifies the
frozen source/protocol/model assets, records study and selection hashes,
then processes one content-addressed request at a time:

1. Write a fresh protein/SMILES YAML and `expected.json`, preserving the
   exact declared target/MSA. A new output directory prevents upstream
   "already predicted" reuse of another request.
2. Run ordinary Boltz prediction outside autograd, logging the command,
   subprocess outcome, elapsed time and new generated artifacts.
3. Retain both raw affinity heads. A zero CLI exit without an affinity
   output is a native failure; missing/nonfinite or inconsistent scalars
   stop the allocation as corrupt evidence.
4. Run the processed chemical/target audit below. Readable identity changes
   remain `identity_ineligible` observations with raw scores, excluded from
   retained-gain comparisons. Corrupt audit evidence or audit-subprocess
   failure stops execution for investigation.
5. Save `result.json` and an independently hashed terminal attempt seal.
   Reports refresh at startup, every ten terminal outcomes, and at the
   allocation boundary. A completed allocation can still have unfinished
   requests; `all_requests_terminal` is the authoritative processing flag.

Process-group timeouts also terminate preprocessing workers. Pinned assets
and MSA identities are checked around requests; parent seals are rechecked
at the boundary. Resume verifies terminal request/file-set identities,
skips completed and ordinary failed outcomes, and preserves any interrupted
attempt before starting a new uniquely named one. It does not redraw
candidates, choose another parent or silently repair molecules.

## Processed chemical and target identity audit

The [audit module](../src/latentmol/boltz/processed_identity.py) strengthens
the previous input-standardization-only check. It links input YAML,
manifest/record, processed structure/molecule/MSA, pre-affinity pose and
affinity output by request identity and hashes. It loads only trusted
request-generated native artifacts.

The full pre-crop protein sequences in processed structure and pose must
match the declaration. Both the MSA query and full pinned native parse of
its input are checked, using the 8,192-sequence parsing cap. The first
contract allows one unmodified protein and one ligand; extra targets,
templates and pocket/contact constraints are rejected.

Ligand identity includes stereo-free heavy atoms, **per-atom formal charge
and typed/aromatic bonds**. Harmless atom reordering is permitted; equal
net charge alone is insufficient. Full input/pose, tokenizer and affinity
crop must preserve all ligand heavy atoms through injective atom-name maps.
Reconstructed native element, local-charge, typed-bond and connectivity
features must agree with the generated RDKit molecule; affinity masks and
finite pose coordinates are checked.

These are retained-artifact and reconstructed-feature checks with pinned
native code, **not an in-forward tensor hook**. Score association is
record/path/manifest/pose linkage, not an embedded score-input hash.
Stereochemistry, isotope/radical identity, explicit-hydrogen geometry,
protonation ensembles and physical pose quality remain outside the supported
scope. Stronger identity evidence does not establish affinity correctness.

## HPC preflight and execution interfaces

The [CPU preflight](../src/latentmol/boltz/trajectory_preflight.py) runs
selected fixtures then audits **all four** saved native predictions from
the completed full-C13 follow-up. An ineligible audit fails the gate. It
uses existing artifacts and makes no new model prediction or optimizer
update; it does not select only successful audits.

The [CPU wrapper](../scripts/hpc/fresh_trajectory_preflight.sh) and
[GPU wrapper](../scripts/hpc/boltz_fresh_trajectory.sh) run
`test_fresh_trajectory`, `test_processed_identity`,
`test_boltz_active_set_rescore`, `test_random_start_study` and
`test_protein_panel` before their stages. All numerical/native tests run
on HPC. Local parsing alone is not a fixture pass.
The [CPU submission helper](../scripts/hpc/submit_fresh_trajectory_preflight.sh)
records its distinct preflight receipt before any GPU gate claim.

The [submission helper](../scripts/hpc/submit_fresh_trajectory.sh) uses
one-shot receipt directories and requests one exclusive `gpua40` GPU, four
cores, 8 GB per core, one host and a 23-hour limit. These symbolic examples
explain interfaces; actual commands/receipts belong in the ledger:

```bash
bash scripts/hpc/submit_fresh_trajectory.sh \
  RELEASE_DIR ARCHIVE_SHA256 PROTOCOL PROTOCOL_SHA256 RECEIPT_DIR [AFTER_JOB] [resume]

python -m latentmol.boltz.fresh_trajectory \
  --config PROTOCOL --sha256 PROTOCOL_SHA256
```

Continuation adds `--resume` under the same source/protocol/selection
identities. The wrapper's resume mode initializes the original protocol
only when no study directory exists; it never replaces an existing frozen
selection. Actual CPU/GPU IDs, dependencies and gating outcomes require
verified receipts.

## Reading the report

The [report module](../src/latentmol/boltz/trajectory_report.py) writes
`summary.json`, `memberships.csv` and `report.md`. Every planned
checkpoint/draw/pose outcome remains visible, including invalid,
unavailable, unattempted, failed, identity-ineligible and partial-parent
observations. Raw native heads remain for ineligible predictions; their
retained-gain deltas are unavailable rather than zero.

The **primary baseline** is each role's **step-0 argmax at the same native
pose seed**. Secondary `draw_start_delta_*` fields use the same saved
categorical draw index at step 0; they do not imply coupled molecular
noise or create new draws. If a starting argmax has no eligible prediction,
later eligible native scores remain unpaired. An eligible sample-start may
support its separately labeled secondary comparison.

Groups separate role/checkpoint and argmax from categorical samples. They
report planned/valid/invalid/unavailable, native-eligible and paired counts,
unique valid SMILES, raw heads and deltas. Mean, population SD, minimum and
maximum are descriptive **membership-weighted** statistics: duplicate
samples retain sampling weight even though computation is deduplicated.
They are not confidence intervals over independent molecules. Favorable
head counts retain their eligible denominator; these carbon/prior starts
are not established actives.

The original endpoint remains its predeclared step-100 graph. Checkpoint
scores do not reselect it or create a native/proxy ratio with unjustified
meaning. Read request IDs, logs, identity audits, terminal seals and
allocation records alongside aggregate reports.

## Requirements before sealing the protocol

- Declare the input study/run identities and all intended parent cases,
  including failed, partial and unstarted cases where applicable.
- Freeze checkpoint indices, argmax/sample draw indices, pose seeds and
  native prediction settings before seeing fresh-native outcomes. Specify
  whether sample comparisons use a starting argmax or matching sample.
- Verify original graph labels, hashes and target/MSA/native-source/model
  identities. Ordinary chemistry conversion stays outside autograd.
- Preserve invalid and missing candidate records. Successful native
  standardization must retain the evaluated chemical identity; changed
  identity is not retained-score evidence for the original candidate.
- Declare deduplication by target, chemical identity, pose seed and native
  budget. Reused predictions remain reused predictions, rather than
  independent replicates.
- Keep original candidate/checkpoint/case denominators and native-eligible,
  attempted, failed and paired subsets separately. Unavailable predictions
  do not receive a fabricated zero score.
- Save both raw native heads, signed objective comparisons, per-pose values
  and failures before any descriptive aggregation. A best observed native
  checkpoint must not silently replace the predeclared endpoint.
- Use a unique immutable source/config release and unique HPC receipts.
  Boundary continuation must preserve completed/failed attempts and verify
  their identities before skipping them.

## Limits of the conclusion

Fresh prediction removes reliance on a saved reference pose for this
evaluation, but the same pretrained Boltz model supplies optimization and
confirmation. It is not independent experimental validation. Input target,
single-query-MSA, missing-cofactor, graph-schema and stereochemistry limits
remain those of the parent studies unless explicitly changed and recorded.
No synthesis, selectivity or clinical binding conclusion follows from this
model-only evaluation. Final execution and scientific claims wait for the
retained HPC preflight and native artifacts.
