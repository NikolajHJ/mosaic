# Dense decoder comparison through 100 completed epochs

For the later cross-experiment explanation and its causal limits, see
[sulfur bias in chemical guidance](84_sulfur_bias_in_validity_guidance.md).
The results and original source identities below remain unchanged.

Current verified endpoint, **10 October 2026 21:24 DTU-local**: the single
seed-17 coefficient-1 decoder **completed all 100 epochs and final evaluation**.
Its final checkpoint SHA is
`c1c9d692024ee1d47d0fb1ba4308ec4a5312b75d550d3f9e57434429da68bcf4`.
The persisted scheduler log records job 29645586 successfully completed at
**06:51:32 DTU-local**, after 32,041 seconds for the final segment; its
training run completed at 06:47:26. Finished jobs have aged out of lookup,
so the saved records establish completion.

The same unfiltered 4,096 latent points, four categorical draws per point
and 7,347 held-out reconstructions give:

| Decoder | Valid argmax / 4,096 | Valid categorical / 16,384 | Valid reconstruction / 7,347 | Exact reconstruction / 7,347 |
|---|---:|---:|---:|---:|
| Untouched original epoch 89 | 64 (1.56%) | 124 (0.76%) | 7,220 (98.27%) | 7,160 (97.45%) |
| Epoch 25 | 661 (16.14%) | 1,794 (10.95%) | 7,237 | 7,163 |
| Epoch 50 | 786 (19.19%) | 2,387 (14.57%) | 7,243 | 7,167 |
| Epoch 75 | 817 (19.95%) | 2,644 (16.14%) | 7,247 | 7,169 |
| Epoch 100 | **840 (20.51%)** | **2,743 (16.74%)** | **7,263 (98.86%)** | **7,182 (97.75%)** |

At epoch 100 there are 835 unique valid argmax outputs and 2,715 unique
valid categorical outputs. Reconstruction loss increased from
0.002641222666 to 0.003052592256 despite the slightly better hard validity
and exact counts. Most random-prior outputs remain invalid. Saved paired
receipts have positive validity intervals and retained unique yield, but
still mark `advantage_established=false` / `pending_reconstruction_retention`;
the standalone controller retention assessment is not in this export.
Report these observed metrics rather than a protocol-certified advantage.

**Composition changed substantially.** Occupancy-weighted expected carbon
count fell from 16.4789 in the original to 4.1972 at epoch 75 and 3.6838
at epoch 100; expected sulfur rose from 0.39095 to 16.8928 and 17.5771.
Expected heavy-atom size stayed near 22.09 (epoch 100: 22.0683). The canonical
element order is atomic numbers 6, 7, 8, 9, 15, 16, 17, 35, 53. These are
relaxed probability-weighted counts across all prior states, not counts
restricted to valid molecules. The raw validity improvement is accompanied
by a major sulfur preference and does not establish general chemical utility,
training-distribution fidelity or binding performance. Static review of the
maximum-allowed-valence proxy identifies a plausible incentive: neutral
sulfur permits valence 6, while carbon permits 4, nitrogen 3 and oxygen 2.
Changing element labels could reduce that penalty. This is an untested
causal hypothesis about an incomplete chemistry objective, not evidence
that allowed sulfur valence or RDKit validation is incorrect. Aromatic and
ring consistency are not established by this local proxy. Extra fine-tuning,
chemical loss and the frozen-encoder regime remain confounded in this
single-candidate comparison against the zero-additional-prior original.
The new decoder remains experimental pending composition auditing. The
completed DeFoG integration used the untouched original epoch-89 decoder;
no substitution or new DeFoG pilot was implemented/submitted in this turn.

The [verified capsule](../outputs/hpc-results/status-20261010-check002.tar.gz),
SHA `f30dce44b5252157feaa4a7ddb2824f4568c4694cbc5f43b8db7ead4a1116784`,
exported `2026-10-10T19:23:54.437421+00:00`, contains 96 safe members / 95
verified files. The [final summary](../outputs/hpc-results/status-20261010-check002/decoder/e100/summary.json),
[checkpoint metadata](../outputs/hpc-results/status-20261010-check002/decoder/e100/training/20261009T195751Z_decoder-s17-a1_b0f81d1b/latest_checkpoint.json),
[completed training run](../outputs/hpc-results/status-20261010-check002/decoder/e100/training/20261009T195751Z_decoder-s17-a1_b0f81d1b/run.json),
[scheduler success log](../outputs/hpc-results/status-20261010-check002/scheduler-logs/source078-single-seed17-seed17-a1-e100.29645586.out)
and [original evaluation](../outputs/hpc-results/status-20261010-check002/original/original.json)
preserve identity and outcomes. Local [verification](../outputs/hpc-results/status-20261010-check002.tar.gz.verification.json)
used Python 3.12.14 / Windows 11 for bytes/hashes/metadata only. This status
turn changed no source, data, model, scientific setting or scheduler job.

## Static RAM and sulfur-bias investigation, 10 October 2026

An A100's **80 GB is GPU memory (VRAM)**, separate from host RAM used by
preprocessing, dataset arrays and the loader. The [dense continuation submitter](../scripts/hpc/submit_decoder_density_long.py)
and [native matched-pilot submitter](../scripts/hpc/submit_defog_matched_pilot.py)
request four cores at 16 GB per core, or **64 GB of host RAM** for their GPU
stages. DTU specifies job memory per core in its [resource guidance](https://www.hpc.dtu.dk/?page_id=4976)
and [batch-job documentation](https://www.hpc.dtu.dk/?page_id=1416).
GPU model/batch memory does not directly grow with total dataset size, but
the current all-record preparation, array stacking/archive buffer and full
cache decompression can grow host memory with the corpus. Peak host RAM
for approximately 1.94 million molecules has not been measured. Chunked,
sharded or memory-mapped preparation/loading is a future option, not an
implemented change or a demonstrated fit in the current allocation.

The [implemented local penalty](../src/latentmol/chemistry_regularization.py)
computes element/charge-probability-weighted `E[max(V - maximum_allowed, 0)]`.
For fixed bond, occupancy and neutral-charge distributions, substituting
neutral sulfur's maximum 6 for carbon's maximum 4 never increases this
local excess term. At bond-order sum 5, the carbon excess is 1 and sulfur
excess is 0. This proves a structural escape route in that component;
it does **not** establish how much of the measured carbon/sulfur shift
(16.48/0.39 to 3.68/17.58 expected atoms) it caused. Expected size stayed
near 22.09 to 22.07 across the same 4,096 prior points. Sulfur's legitimate
expanded valences should not be falsified to prevent this shortcut; see
[RDKit's allowed-valence rules](https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences).
The maximum-valence proxy is not full chemical validation.

The current prior anchors constrain occupancy and the aromatic/nonaromatic
partition, **not element identity or formal-charge distribution**. Original
reconstruction still supervises real graphs; that does not pin chemistry on
random-prior points. Zero undercoordination cost accommodates unrepresented
implicit hydrogen. Counting aromatic bonds as order one remains optimistic
and cannot establish ring consistency or Kekulization. The unchanged loss
and original checkpoint are retained; the causal contribution of each term
has not been tested.

Possible future comparisons, all **unimplemented and unselected**, are:

- A bond-focused chemistry surrogate with explicitly stopped atom/charge
  gradients in that term, while reporting its limited gradient scope.
- Bounded valid-repair fixed targets that first preserve candidate atom
  identity, then permit explicitly penalized identity edits. Witnesses remain
  bounded observations, not globally nearest-valid molecules.
- A weak aggregate empirical/reference element-and-charge trust region,
  declared as a distribution prior rather than a ban on sulfur.

Stopping atom/charge gradients removes only their immediate path in that
term; updates to shared decoder parameters can still shift those outputs.
Fixed valid-repair targets supply a surrogate gradient, not an exact validity
probability or nearest-valid distance. Even bond-first selection needs
acceptance, edit and composition audits because element classes that are
easier to repair can be overrepresented among accepted targets. A population
trust region deliberately introduces a reference/dataset distribution
preference: validity alone cannot promise typical druglike chemistry or
element-neutral output.

These alternatives need counterfactual/component-gradient checks and
legitimate sulfur examples before a large-corpus 100-epoch continuation.
They are not guarantees of validity or bias removal. This investigation
used source/docs/saved-metadata review and primary documentation only:
no implementation, numerical experiment, local model test, HPC login,
submission, deletion or dataset download occurred in this turn.

## Conceptual loss, batch and budget discussion, 10 October 2026

The one-sided excess `max(V - maximum_allowed, 0)` deliberately leaves
undercoordination unpenalized because implicit hydrogen is absent from this
graph representation. Using `abs(V - maximum_allowed)` or its signed-gap
square would wrongly drive every atom toward its maximum heavy-atom bond
order, including legitimate hydrogen-bearing atoms. Squaring the **positive
excess** is a defensible separate loss choice, but changes tail weighting
and still leaves the carbon-to-sulfur maximum-cap shortcut. It is neither
implemented nor a demonstrated correction here.

For a prospective comparison, the preferred first chemical term is bounded
fixed-target repair NLL alongside reconstruction and the existing common
anchors, rather than making an extra element-distribution guard mandatory.
Monitor raw validity, repair coverage/edits, composition, target availability
and reconstruction. If repair coverage is adequate, omitting the local
valence auxiliary is a candidate ablation. Missing or already-valid targets
must remain explicit; repair-only guidance is not complete validity
supervision or a bias-removal guarantee. This is a conceptual preference,
not a changed configuration or authorization to start another experiment.

An optional aggregate element/charge guard can reuse the original-prior
decoder output already computed for common anchors. It adds categorical
reductions, not another decoder forward in that existing path; exact cost
and benefit are unmeasured. A repair-only implementation without reference
anchors would not automatically have this free forward available. Current
local valence DP and four-draw connectivity have nontrivial tensor loops;
bounded repair refresh has potentially substantial CPU/oracle cost, while
held-target NLL itself makes no per-update chemistry calls. The present
matched search trainer still computes common diagnostic/draw quotas at
coefficient zero; omitting their execution requires a newly declared policy.

Existing dense timing evidence is for **64 real + 64 fresh-prior points**
on an L40S, not an A100 or repair-only benchmark. The [gate receipt](../outputs/hpc-results/source077-dense-chemistry/extracted-evidence001/runs/source-077-dense-chemistry-attempt001-density-study/gate/passed.json)
records six warmed full updates: 0.40530 seconds (control) / 0.38747 seconds
(chemistry), with peak allocated memory approximately 5.13 GB. It excludes
validation/evaluation and most run overhead. The [final segment metrics](../outputs/hpc-results/status-20261010-check002/decoder/e100/training/20261009T195751Z_decoder-s17-a1_b0f81d1b/metrics.jsonl)
retain 215,000 examples / 3,360 updates per epoch and cumulative elapsed
30,503.4107 / 31,774.5726 seconds at epochs 99/100; these are existing
receipts, not a new throughput measurement.

Retain four cores × 16 GB = 64 GB host RAM as a profiling baseline. Four
cores × 32 GB = 128 GB is a possible request only if eligible hardware and
measured host-memory needs justify it; more cores are not intrinsically
required just to obtain RAM. Extra cores may help explicitly parallel repair
search. Peak RAM for the larger corpus remains unmeasured.
Proposed A100 batch checks are equal real/prior pairs 64/64, 128/128,
256/256 and 512/512, choosing measured throughput **and quality**, not the
largest fitting batch. No such benchmark has run. Using an illustrative
1.9 million examples, 100 epochs imply 190 million real-example
presentations and 8.84 times the current per-epoch corpus. Arithmetic
update counts are 29,688 / 7,422 / 3,711 per epoch at real batches
64/256/512; they are not
runtime predictions. A future 100-epoch limit should be a maximum staged
budget with validation, not an assertion that 100 is optimal. Larger batches
alter update count and optimization behavior and need matched learning-rate
and evaluation decisions. Optional BF16 network execution with FP32
probabilities/losses and `torch.compile` are future benchmark ideas requiring
numerical validation, not active changes.

The original checkpoint and frozen encoder remain the prospective first
comparison's reference. Architecture inspiration is discussed in
[docs24](24_atom_pair_decoder_comparison.md). This turn changed no code,
loss, batch, precision, epoch budget, data or submitted jobs; it ran no
numerical tests, models or HPC commands.

## Historical progress snapshots

Historical verified snapshot, **10 October 2026 00:10 DTU-local**: the single
seed-17 chemical decoder completed **epoch 75** and has saved **epoch 81**;
epoch-100 **29645586 RUN** continues. Epoch-75 prior validity is
**817/4,096 argmax (19.95%)** and **2,644/16,384 categorical (16.14%)**,
versus original **64/4,096 (1.56%)** and **124/16,384 (0.76%)**. Held-out
reconstruction is **7,247/7,347 valid (98.64%)** and **7,169/7,347 exact
(97.58%)**. These remain one-candidate raw gains, with most prior outputs
invalid and no isolated fresh-prior causal claim; epoch 100 is not evaluated
in this snapshot. The [verified capsule](../outputs/hpc-results/status-20261010-check001.tar.gz),
[epoch-75 summary](../outputs/hpc-results/status-20261010-check001/decoder/e075/summary.json)
and [epoch-81 checkpoint metadata](../outputs/hpc-results/status-20261010-check001/decoder/e100/training/20261009T195751Z_decoder-s17-a1_b0f81d1b/latest_checkpoint.json)
preserve exact evidence. Source/config and checkpoints are unchanged; no
numerical work or scheduler changes occurred during status retrieval.

Historical status, **9 October 2026 10:19 DTU-local**: the single seed-17 chemical
decoder completed its **epoch-50 milestone** and saved **epoch 52** while
epoch-75 **29645585 RUN** continues; epoch-100 **29645586 PEND**. Prior validity
is **786/4,096 argmax (19.19%)** and **2,387/16,384 categorical (14.57%)**, versus
the original **64/4,096 (1.56%)** and **124/16,384 (0.76%)**. Held-out
reconstruction is **7,243/7,347 valid (98.58%)** and **7,167/7,347 exact (97.55%)**.
These are one-candidate raw before/after gains; most prior outputs remain
invalid, and no isolated fresh-prior causal advantage or epoch-100 result is
claimed. The [verified capsule](../outputs/hpc-results/status-20261009T081813Z.tar.gz)
and [epoch-50 summary](../outputs/hpc-results/status-20261009T081813Z/decoder/e050/summary.json)
preserve the result; exact checkpoint/source identities and byte-verification
commands are in the latest ledger entry. No numerical work or submissions
occurred during this status-only retrieval.

Historical status, **8 October 2026 23:30 DTU-local**: the seed-17 coefficient-1
decoder completed its authenticated **epoch-25 milestone**. The epoch-50
allocation **29645584 RUN** has a saved **epoch-28 checkpoint**;
epoch-75/100 **29645585/29645586 PEND**. Epoch-25 job and original ten-epoch
prefix have aged out of scheduler lookup, but their persisted completed
records verify the reused prefix and completed segment. On the same unfiltered
prior panel, validity improved from **64 to 661 of 4,096 argmax molecules
(1.56% → 16.14%)** and **124 to 1,794 of 16,384 categorical draws
(0.76% → 10.95%)**. Held-out reconstruction remains 98.50% valid/97.50% exact.
These are one-candidate raw gains; most prior outputs remain invalid and no
isolated fresh-prior causal advantage is claimed. No epoch-100 result exists.

[Verified status capsule](../outputs/hpc-results/status-20261008T212940Z.tar.gz),
[epoch-25 summary](../outputs/hpc-results/status-20261008T212940Z/decoder/e025/summary.json),
[epoch-28 checkpoint metadata](../outputs/hpc-results/status-20261008T212940Z/decoder/e050/training/20261008T201028Z_decoder-s17-a1_3a73ef29/latest_checkpoint.json)
and [scheduler snapshot](../outputs/hpc-results/status-20261008T212940Z/scheduler_snapshot.json)
preserve the evidence. Capsule SHA is
`083543834a9886eeb8435468e6d941a7457a8dc61a4a35b7708771b82440693e`;
47 safe members/46 file hashes verified. This was read-only metadata retrieval,
not a new numerical test or submission. Exact identities and commands are in
the latest ledger entry.

Historical scope/submission snapshot, after restored access at 14:07 DTU-local: the user
reduced the 100-epoch scope to **one seed-17 coefficient-1 dense chemical
decoder**, benchmarked against the untouched original VAE. All sixteen former
source078 reservations and unused source077 seed-23 prefixes were canceled
and observed EXIT. The existing seed-17 chemical prefix **29638859 remains
RUN**. One replacement trajectory is now **submitted as four serial
allocations: 29645583/29645584/29645585/29645586**, ending at epochs
25/50/75/100. The first waits for actual prefix 29638859 and the passed CPU/
native gates; each later allocation also waits for its predecessor.
The byte-verified **14:10 DTU-local** snapshot confirms all four replacements
PEND, preserved prefix RUN and all eighteen cancellations EXIT. No 100-epoch
result or new quality result is established.

The sealed/uploaded **source078 attempt002** and scientific settings remain
unchanged. CPU **29642004 DONE** passed **183 engineering checks/three
CUDA-only skips**; native L40S **29642352 DONE/passed** verified exact parameter,
AdamW, four-stream and frozen-encoder parity. Prefix authentication, explicit
original restart when necessary, milestone/latest-two checkpoint retention
and lossless row-log compression remain in force. Attempt001's quota failure,
the source077 control's first-checkpoint failure, and all canceled receipts
remain history. Content-preserving deduplication left official usage at
88.45 GiB of the 100 GiB hard quota in the last measured storage snapshot.
All numerical work remains on DTU HPC.

## Current single-candidate scope and baseline

The untouched epoch-89 original VAE is the benchmark and frozen anchor
reference. It was trained with real-molecule reconstruction, posterior
sampling and VAE KL, **without additional unconditional fresh-prior training
points**. Historical coefficient-zero fine-tuning controls included
fresh-prior occupancy/aromatic anchors; they are not this baseline.

Only seed-17 coefficient-1 is continued, with all 215,000 real training graphs
per epoch, the frozen encoder, and 64 real posterior / 64 fresh-prior points
per full update (actual-size 1:1 tail). Loss and half-posterior/half-prior
scaling are unchanged. The original comparison at the same unfiltered 4,096
latent points (argmax plus four categorical draws) and all 7,347 held-out
reconstructions remains prescribed at milestones 25/50/75/100. Epoch 100 is
the final endpoint, not a test-selected checkpoint.

This is an exploratory before/after comparison. Extra fine-tuning epochs,
chemical guidance and the frozen-encoder regime change together; a gain
cannot be attributed solely to training on random points. There is one
training seed, so paired latent-cluster intervals quantify evaluation-panel
variation rather than training-seed replication. No new matched control is
available; any missing same-epoch control comparison must remain unavailable.

The sealed four-case protocol is preserved as history. A separate execution
scope selects **segment indices 4–7 only** (seed17/chemistry), using the same
scientific archive/configuration, gates and campaign. Remote scope journal:
`/work3/s234818/latentmol/submissions/source078-single-seed17-attempt001/scope.json`,
SHA-256 `160da96041ac742ccaebc38619f702713e134f72c9fa7bda723deaa0daeecb38`.
Replacement receipts are under the same submission root's `receipts/`.

| Single candidate stop | Replacement job | Additional dependency |
| --- | --- | --- |
| 25 | 29645583 | `ended(29638859)` |
| 50 | 29645584 | `ended(29645583)` |
| 75 | 29645585 | `ended(29645584)` |
| 100 | 29645586 | `ended(29645585)` |

Every job also requires `done(29642004) && done(29642352)` and retains the
23-hour L40S allocation, four slots/16 GB per slot, one exclusive GPU/host,
19-hour training cap and 1.5-hour evaluation reservation. The first segment
must authenticate its own completed epoch-10 chemical prefix, or explicitly
restart from the original if that prefix failed/is missing. It must not
credit unsaved work. The earlier cancellation script stopped before `bsub`
on stale PEND scheduler observations; the first recovery upload failed
authentication before any SFTP put. After the user reconnected VPN, the
recovery script was uploaded/hash-authenticated, confirmed no prior replacement
receipts, and submitted the four jobs above. No numerical tests or scientific
source changes accompanied this scheduler recovery. Evidence003 is now
downloaded and verified below; evidence002 remains the historical pre-reduction
queue snapshot.

### Verified single-candidate submission and original baseline

[Single-evidence003](../outputs/hpc-results/source078-dense100/single-evidence003.tar.gz)
SHA-256 `6e8a0ad7f17ab6dd27f5f6e9f4068f483849826f1dbb51640f7b46c7badaf28d`
contains **73 safe members / 72 individually verified files**. Export time is
**2026-10-08 12:10:11.989349 UTC / 14:10:11 DTU-local**. The
[passed metadata audit](../outputs/hpc-submissions/source078-single-audit003.json)
authenticates the single scope, eighteen terminal cancellations, four
replacement identities/dependencies, and complete existing original 4,096-point
prior-panel bytes/hashes. Local verification used Python 3.12.14 / Windows 11
and byte/hash/metadata operations only; exact commands are in the ledger.
This supplies no new numerical evaluation or training outcome.

Direct records are [scope.json](../outputs/hpc-results/source078-dense100/extracted-single003/scope/scope.json),
[first replacement receipt](../outputs/hpc-results/source078-dense100/extracted-single003/scope/receipts/seed17-a1-e025-receipt.json),
[last replacement receipt](../outputs/hpc-results/source078-dense100/extracted-single003/scope/receipts/seed17-a1-e100-receipt.json)
and [scheduler snapshot](../outputs/hpc-results/source078-dense100/extracted-single003/scheduler_snapshot.json).
The [original training metadata](../outputs/hpc-results/source078-dense100/extracted-single003/original/training-run.json),
[original prior metrics](../outputs/hpc-results/source078-dense100/extracted-single003/original/original-prior/metrics.json)
and [panel identity](../outputs/hpc-results/source078-dense100/extracted-single003/original/original-prior/panel.json)
preserve the benchmark provenance. Scope SHA is unchanged. CPU/native proofs
were reused, scientific bytes did not change, and SSH closed normally after
export. Current states are prefix RUN, four replacements PEND, all eighteen
former reservations EXIT.

## Original four-case design and protected lineage

The original sealed design retained seeds **17 and 23**, matched coefficients
**0 and 1**, and all
**215,000 real training graphs per epoch** from
[the dense comparison](44_dense_chemistry_decoder_comparison.md). Continue
each method's own authenticated dense training state where eligible; do not initialize the
chemistry arm from its control or substitute a historical 16/16 checkpoint.
The untouched epoch-89 VAE remains the original evaluation/anchor reference,
and the encoder's parameters and buffers remain frozen in every trained arm.
Only decoder parameters continue learning.

The requested endpoint is **100 cumulative dense fine-tuning epochs**.
An eligible source077 ten-epoch prefix is preserved and needs 90 additional
epochs. For an explicitly failed/missing prefix, attempt002 instead plans a
fresh start from the protected original VAE and all 100 epochs. These
are fine-tuning epochs counted from the original protected initializer, not
new training of the encoder or a replacement VAE prior. The prefix may be
used only after its own epoch/update/checkpoint identity is authenticated;
source077's control lacks a completed epoch-10 checkpoint. Failed, partial
or unavailable parents and the resulting fresh-start decision must be
reported explicitly. No prefix from a different arm/seed may be substituted,
and a corrupt or unverified identity is not silently accepted as a resume.

Preserve the 64-real/64-fresh-prior rule, including an actual-size 1:1 tail,
original reconstruction weights, occupancy/aromatic reference anchors at
weight one, and the local chemical objective's half posterior / half prior
mean scaling. Coefficient 0 remains reconstruction plus the same anchors and
computed diagnostic/draw budget. Coefficient 1 uses the unchanged local
valence/connectivity proxy. This continuation introduces neither search-target
NLL nor DeFoG guidance, a new learned scorer, or a silently larger chemical
loss weight. Keep learning rate, optimizer and other scientific settings
unchanged unless a later explicit protocol declares a separate comparison.

The new scientific fine-tuning configuration declares `epochs = 100` for
every segment. Its execution-only `stop_after_epoch` selects a boundary;
it does not rewrite the scientific endpoint or loss. Restoration from the
source077 prefix must authenticate the explicit **10 → 100 epoch extension**
and reject changes to every other scientific field. A failed/missing-prefix
restart uses new decoder AdamW and the prescribed seed/streams from the
original initializer; it is not labeled a restore. Later resumes must retain
that same epoch-100 configuration.

At batch size 64, the declared full-data budget is **3,360 updates per epoch**,
**336,000 cumulative updates per method at epoch 100**. These are requested
counts, not evidence of executed work. Reports must separate restored prefix
updates, newly executed updates, validated cumulative epochs and unfinished
minibatches.

## Epoch-boundary continuation and HPC segments

Use unique immutable releases and run directories, retaining every source077
checkpoint/metric/evaluation. Authenticate the original initializer, data,
split, dense loss configuration, seed, coefficient, parent source, parent
checkpoint and completed-epoch metadata before restoration. Restore decoder
weights, frozen encoder identity, **AdamW state and the four dedicated
loader/posterior/prior/connectivity RNG streams**. Loading weights alone is
not an optimizer/RNG continuation.

Segments must stop/resume at a fully saved, completed epoch. Unsaved partial
epochs are discarded and recorded; a latest partial file cannot stand in for
a completed boundary. The prior [budget recovery](41_decoder_budget_recovery.md)
restored the dedicated optimizer/stream path, without guaranteeing global
RNG restoration or cross-device bitwise equivalence. The new dense/native
resume gate must verify its own supported path; an old engineering pass is
not a new native validation claim.

Each GPU allocation stays within the **23-hour scheduler cap**, with explicit
reservations for checkpoint verification, validation, milestone evaluation
and shutdown. The selected schedule for each method is:

| Authenticated starting epoch | Segment stop | New full epochs | Role |
| --- | --- | --- | --- |
| 10 | 25 | 15 | First continuation/milestone |
| 0, if failed/missing prefix | 25 | 25 | Declared fresh restart/milestone |
| 25 | 50 | 25 | Continuation/milestone |
| 50 | 75 | 25 | Continuation/milestone |
| 75 | 100 | 25 | Final fixed endpoint |

The original, now-canceled queue had **16 scientific segments** across four methods, all run
serially. None may begin before all four source077 jobs have ended and the
new native resume gate is DONE/passed. The last original job is **29638861**.
An `ended` scheduler dependency alone is insufficient: each method must
independently authenticate an eligible completed epoch-10 parent or record
the failed/missing-prefix original restart. Failed or partial parents cannot
be relabeled as restored work. The short native
engineering gate may execute while the original cohort continues; the long
scientific segment chain must wait. Order is seed 17/control through all four
stops, seed 17/chemistry through all four stops, seed 23/control, then seed
23/chemistry. Each segment has a **19-hour training cap** and **1.5-hour
evaluation reservation** inside its 23-hour allocation. The reservation is
headroom, not a hard evaluation timeout or guaranteed completion.

Every resume must carry an **immutable cumulative metrics receipt** for
epochs 1 through its current boundary, stitching the authenticated parent
history with new segment metrics. Record its digest, checkpoint digest and
cumulative updates; later segments must verify both the state and the full
history instead of assuming an arbitrary local metrics file is complete.
This preserves earlier work and enables repeated continuation without
rewriting any parent record.

A controller must retain completed work and explicitly mark failed,
time-censored or unattempted segments. No endpoint claim follows merely
because an allocation ends or a dependency releases. **The fixed milestone
protocol does not automatically resume from an arbitrary interrupted epoch.**
Checkpoints/completed history are retained, but failure before the declared
stop requires explicit authenticated recovery to that milestone; subsequent
fixed segments fail closed on its missing boundary. The short source077
timing forecast is not a guarantee that training plus full milestone
evaluation fits. Resources, launcher commands and the initial CPU receipt
follow; the downloaded new native/segment receipt capsule is verified below.

## Implemented interfaces and passed engineering gates

The [configuration](../configs/decoder_density_100_study.json) and
[metadata-only long protocol](../src/latentmol/decoder_density_long_protocol.py)
pin source077 archive, executable manifest, configuration, campaign and
receipt roots. They authenticate its protected initializer/data/split and
successful native dense profile. The
[immutable history helper](../src/latentmol/decoder_continuation.py) seals
contiguous cumulative epoch records and resume envelopes. The
[trainer](../src/latentmol/decoder_finetune.py) separates constant scientific
`epochs` from `stop_after_epoch`; the
[resume validator](../src/latentmol/decoder_resume.py) requires explicit
`epoch_extension` fields `from_epochs`/`to_epochs` for the first extension.
The [long study](../src/latentmol/decoder_density_long_study.py) records
intermediate versus final endpoints, authentic parent histories, milestone
evaluation and failure phases. Source077's executable/config/checkpoint/job
bytes remain unchanged.

New allocated CPU fixtures must check epoch-extension restrictions, stop
boundaries, cumulative history authentication, repeated resume, dedicated
stream parity, source/data/seed/method identity and failure retention. An
actual **50-slot full-width GPU checkpoint gate**, seed 17/coefficient 1,
compares **three uninterrupted
epochs** against **one completed prefix epoch → resume through epoch 2 →
resume through epoch 3**. Each tiny epoch uses 513 real graphs: eight full
64-point updates and a one-point tail, with the 1:1 posterior/prior chemistry
coverage. The prefix uses one scientific epoch, then explicitly authorizes
1 → 3 while stopping at 2; the second resume retains the epoch-3 configuration
and stops at 3. It must compare final parameters, AdamW state and all four dedicated
RNG states exactly in its tested environment. **This actual native gate has
now passed**, as recorded below. Its engineering proof is not a quality
result or certification of global RNG/cross-device equivalence.

The [submission helper](../scripts/hpc/submit_decoder_density_long.py) has
`cpu`, `gate` and `segments` stages. Default invocation prints a verified plan;
`--submit` creates duplicate-safe intents/receipts. Each allocated command
reauthenticates the submitted source archive and executable manifest.

| Stage | Resources/limit | Dependency and proof |
| --- | --- | --- |
| CPU | `hpc`, two slots, 4 GB/slot, one host, 45 minutes | New fixture suite; no CUDA equivalence claim from a CPU pass |
| Actual repeated-resume gate | `gpul40s`, four slots, 16 GB/slot, one host/GPU, two hours | Same-identity CPU DONE; positive `gate/passed.json` required |
| Four selected seed-17 chemical segments | Same L40S host/GPU request, 23 hours each | CPU DONE, native gate DONE/passed, actual chemical prefix ended, then previous replacement ended |

GPU requests use `num=1:mode=exclusive_process:j_exclusive=yes`. The current
first segment dependency includes
`done(29642004) && done(29642352) && ended(29638859)`;
later replacements also include `ended(previous_new_segment)`. Such dependencies
serialize reservations and leave failed/ended cases explicit; they do not
waive source/checkpoint/epoch eligibility. Authentication must still reject
an incomplete prior segment.

## Implemented launch/artifact contract and original cohort submission

From the extracted new immutable HPC release in the Boltz environment, set
`RELEASE`, `ARCHIVE_SHA` and `RECEIPTS` using the **attempt002** receipt below.
All three original attempt002 stages were submitted in authenticated gate order;
the actual CPU/native gates passed before that cohort's segment submission.
The full-cohort `segments` command below is historical and must not be rerun
for the current one-candidate scope; use its authenticated subset recovery
and unique receipts instead:

```bash
python scripts/hpc/submit_decoder_density_long.py --stage cpu \
  --release "$RELEASE" --archive-sha256 "$ARCHIVE_SHA" --receipts "$RECEIPTS" --submit
# After the same-identity CPU gate is DONE:
python scripts/hpc/submit_decoder_density_long.py --stage gate \
  --release "$RELEASE" --archive-sha256 "$ARCHIVE_SHA" --receipts "$RECEIPTS" --submit
# After the actual native gate is DONE and its passed.json is verified:
python scripts/hpc/submit_decoder_density_long.py --stage segments \
  --release "$RELEASE" --archive-sha256 "$ARCHIVE_SHA" --receipts "$RECEIPTS" --submit
```

The CPU wrapper [decoder_density_long_tests.sh](../scripts/hpc/decoder_density_long_tests.sh)
runs the existing dense suite, then new long-resume/protocol fixtures. Expected
fixture counts are not passed results. Native
[decoder_density_long_run.sh](../scripts/hpc/decoder_density_long_run.sh)
calls `python -m latentmol.decoder_density_long_study` with exact protocol
path/hash, output root and `--stage gate` or `--stage segment --segment 0..15`.
Numerical entry requires LSF and immutable source identity.

Output is `/work3/s234818/latentmol/runs/<new-release-name>-density100-study`.
It retains the native `gate/passed.json`, then per-case `seed17-a0`,
`seed17-a1`, `seed23-a0`, `seed23-a1` directories with `e025`, `e050`, `e075`
and `e100` subdirectories. Each has a protocol, unique training run,
checkpoint/profile, immutable `completed-epochs.jsonl`, milestone prior
evaluation and `summary.json`. Successful intermediate summaries use
`completed_epoch_segment`; epoch 100 uses `completed_final_endpoint`.
Failure receipts preserve the phase and whether training already completed,
so an evaluation failure does not erase its completed training boundary.
The source077 original and epoch-10 evaluations remain separate references.

Local preparation checked eight Python ASTs, JSON and shell LF/BOM bytes
only; the exact command/receipt are linked below. No model imports or
numerical tests ran locally. Actual CPU/native validation passed on HPC as
recorded below; local syntax checks do not supply that proof.

## Historical source078 sealing/CPU submission snapshot, 8 October 2026

The authenticated archive was uploaded/extracted with **415 safe regular
members**. Source078 archive SHA-256 is
`59df917c6fae075104bcd00634132250eddee96c78060ed906e0e35cbe386a31`;
executable manifest is
`0fd7796b5f31521abf2fd7105e7bd3611a45ebfe764041eb46249ef5f3c8ada8`;
configuration is
`849a6504ae0de10b320f25f66f301379ef99ccf65f8c4c7c23079802be3e1011`.
Paths are:

- Release: `/work3/s234818/latentmol/releases/source-078-dense-100epoch-attempt001`.
- Receipts: `/work3/s234818/latentmol/submissions/source078-dense-100epoch-attempt001/receipts`.
- Output: `/work3/s234818/latentmol/runs/source-078-dense-100epoch-attempt001-density100-study`.

The exact HPC submission used `PYTHONPATH=$PWD/src` and
`/work3/s234818/latentmol/env/boltz311/bin/python` to invoke the helper with
`--stage cpu`, the archive digest above, the receipt path above and `--submit`.
CPU **29640092 RUN**, n-62-31-13, two slots/4 GB per slot/45 minutes; native
gate/segments are unsubmitted and no epoch-10 parent has yet finished in the
latest source077 snapshot. Original four jobs remain unchanged.

[Static review](../outputs/hpc-submissions/source078-static-review.json)
records stdlib-only AST/JSON/shell line-ending scope, Python 3.12.14 /
Windows 11. Exact command is the bundled Python executable followed by
`outputs/hpc-submissions/source078-static-check.py`, recorded in the ledger.
Remote shell syntax passed for both wrappers, not numerical model validation.
The scientific release remains sealed; future gate/segment results require
actual verified receipts.

## Historical attempt001 CPU completion/quota-blocked submission

CPU **29640092 DONE** with **140 passed/three CUDA-only skipped**: foundation
88/one skip in 12.58 s, dense/resume 25/one skip in 14.87 s, long fixtures
27/one skip in 25.00 s. The attempted native-gate submission failed before
`bsub` while exclusively creating `gate-intent.json`, at
`submit_stability_study.py:89`, `intent_path.open('x')`:
`OSError: [Errno 122] Disk quota exceeded`. There is no native-job receipt
or queued 100-epoch segment. Source078 bytes remain unchanged; its failed
metadata write is distinct from successful CPU fixture work.

The original control **29638858 EXIT** has a retained stale `running` child
record. Subsequent scheduler/child logs confirm **disk quota failure during
the first epoch's `torch.save` checkpoint write**, with no completed epoch-10
checkpoint. This is now direct evidence of its cause, superseding the earlier
unconfirmed snapshot. Original chemistry **29638859 RUN**, the other two cases
PEND, remain the last observed statuses. Recovery must preserve these failure
histories and authenticate actual saved work.

Global work3 free capacity does not measure the user's remaining quota.
The official query identifies user s234818's **work3 pool 6, 100 GiB hard
limit, 96.27 GiB used** after removing an abandoned approximately 2.1G cache.
Audited reproducible `cache/uv` downloads have also been removed, retaining
installed environment hard links and all experiment results. Refreshed
quota accounting after that maintenance is pending. Receipts are
`quota-recovery.json` and `uv-cache-maintenance.json` under
`/work3/s234818/latentmol/submissions/source078-dense-100epoch-attempt001`;
their local export link remains pending.

## Implemented attempt002 storage and prefix-recovery policy

Attempt002 is a new sealed/uploaded immutable recovery; the failed sealed
attempt001 remains history. Its endpoint/loss/data/seed/method settings stay
the same. It explicitly chooses an authenticated completed epoch-10 training
prefix where eligible, or records a failed/missing-prefix fresh restart from
the original checkpoint. This changes the first segment's new-work count to
15 or 25 epochs, without falsely crediting failed/unsaved work. Later segments
restore the completed preceding milestone, optimizer, four streams and
contiguous history as before.

For **new runs only**, `checkpoint_keep_recent = 2` and declared milestone
epochs keep milestone checkpoints and
the latest two recoverable checkpoints, with all epoch metrics and cumulative
history retained. It must identify checkpoint/history relationships and avoid
deleting an active resume parent. Detailed row logs will be **losslessly
gzip-compressed**, preserving every raw row/denominator and integrity digest.
Existing paired-comparison readers receive verified materialized rows; neither
compression nor materialization constitutes new numerical evaluation.
Legacy defaults and all existing experiment results stay unchanged.

These changes are implemented/sealed, with **183 CPU engineering passes and
the actual native repeated-resume gate passed**. Engineering checks cover checkpoint retention, exact row
round-trip/checksums, paired-reader compatibility and explicit prefix/restart
decisions before scientific use. Maintenance/cleanup alone does not pass a
native resume gate. No retention/compression can recover a checkpoint that
never finished writing. New source/config/resource/gate/segment receipts will
be recorded when actually obtained.

The [storage helper](../src/latentmol/decoder_checkpoint_storage.py) writes
checkpoint-deletion identities before pruning only the newly owned run
directory; it never traverses older parents. `archive_panel_rows` checks raw
`rows.jsonl` against evaluation metadata, verifies a lossless gzip round trip,
then records original/compressed digests and byte counts in `row-storage.json`.
`materialized_panel_rows` authenticates both identities, temporarily restores
exact rows for the unchanged paired evaluator, and removes only its own
restored copy afterwards. Denominators/draws are preserved rather than
filtered. These are implemented engineering boundaries, not new numerical
passes.

## Historical attempt002 release/initial CPU-retry snapshot

Archive SHA-256 is
`c6772b9c625274e9d6b577cf4aae1e8587103cf1c2017c349168151558b3d373`;
executable manifest is
`b2a5fb2f97cb6287931c60929769cc6fd42ef198bde3b5f49951cd3a73db6295`;
long configuration is
`24943ba12ffa0d8b60597830391f5d5182fd59e800d1714ccdc7dee6bb40f0b9`.
The uploaded authenticated archive contains **417 safe regular members**.
Release/receipts/output paths are:

- `/work3/s234818/latentmol/releases/source-078-dense-100epoch-attempt002`.
- `/work3/s234818/latentmol/submissions/source078-dense-100epoch-attempt002/receipts`.
- `/work3/s234818/latentmol/runs/source-078-dense-100epoch-attempt002-density100-study`.

CPU **29642004 submitted**, `hpc`, two slots, 4 GB/slot, one host, 45 minutes;
actual result is pending. The helper uses the same CPU command shape as
attempt001 with this new archive digest/receipt directory; the exact command
is in the ledger. New native/segment stages remain unsubmitted and must use
their own same-identity gates, not attempt001's CPU pass.

[Static review002](../outputs/hpc-submissions/source078-static-review002.json)
records ten ASTs/JSON/shell LF checks, Python 3.12.14 / Windows 11. Exact
command is bundled Python plus `outputs/hpc-submissions/source078-static-check.py`.
Remote `bash -n` accepted both wrappers. All are syntax/metadata-only checks;
no model imports or numerical work ran locally. New HPC fixture/native proofs
are pending, and the scientific release bytes remain sealed.

### Historical storage-maintenance progress snapshot

The latest official query after UV-cache removal is **96.35 GiB used of
100 GiB**; installed hard links meant little byte freeing, so no additional
free-space claim is made. Filesystem-only deduplication in completed runs has
a persisted **1,707-duplicate / 10,153,946,786-byte potential** plan. It
preserves every checkpoint filename and verified content digest through hard
links, with plan-before-change and completion-entry journals under the
attempt001 `checkpoint-deduplication` receipt directory. Actual total reclaim
and final accounting remain pending; no experiment data was deleted. This
maintenance does not establish a resume or generation-quality result.

## Historical attempt002 CPU completion/storage-recovery snapshot

**29642004 DONE**, n-62-31-4, **13:20:39–13:21:36 DTU-local**. Allocated
fixtures passed **183 checks**, with **three CUDA-only skipped**: foundation
88/one skip in 12.79 s, dense/resume 25/one skip in 13.20 s, long/storage
70/one skip in 24.43 s. The new attempt002 source/config is unchanged.
Actual full-width GPU resume still needs its independent gate. Its helper
invocation is dispatched, accepted job identity pending; no long segment has
been submitted at this snapshot.

The checkpoint hard-link maintenance **completed 1,707 identical-copy
replacements**, preserving filenames/content SHA and completion journals.
Official accounting now reports **88.45 GiB used of 100 GiB, 11.55 GiB free**,
in `quota-after-dedup.txt` under attempt001 maintenance receipts. The measured
96.35 → 88.45 GiB change is separate from the 10,153,946,786-byte potential
plan; do not present the latter as observed charged-quota savings. This is
file/hash/link maintenance without numerical model work, not a learning or
validity result. Evidence-export links will be added once verified locally.

## Historical native-gate RUN snapshot, 8 October 2026 13:22 DTU-local

**29642352 RUN** on L40S, `gpul40s`, four slots/16 GB per slot, one host and
one exclusively reserved GPU, two-hour limit, `done(29642004)` dependency.
It executes actual stable-decoder uninterrupted versus two-resume training
with lossless validation-row compression enabled. CPU **29642004 DONE**;
native equality/compression proof has **not yet passed**, and all sixteen
long scientific segments remain unsubmitted. Archive/manifest/config are the
same sealed attempt002 values above; no scientific source edits occurred.
Exported verified receipts/results will be linked when available.

The measured quota record is
`/work3/s234818/latentmol/submissions/source078-dense-100epoch-attempt001/quota-after-dedup.txt`;
the completed 1,707 filename/content-preserving replacements remain
maintenance evidence, not scientific results.

## Historical native-gate pass and original scientific segment queue

**29642352 DONE/passed**, NVIDIA L40S, Torch 2.5.1+cu121 / CUDA 12.1,
**48.331108863 s**. Actual full-width three-epoch uninterrupted versus
1 → 2 → 3 repeated-resume training on 513 graphs/tiny epoch passed exact
final parameters and AdamW, four dedicated RNG streams and frozen encoder
equality, with lossless validation-row compression enabled. This is the
declared native engineering proof in that environment, not quality or
cross-device certification.

All sixteen scientific reservations are **submitted/PEND**:

| Case | `e025` | `e050` | `e075` | `e100` |
| --- | --- | --- | --- | --- |
| Seed 17/control | 29642812 | 29642813 | 29642814 | 29642815 |
| Seed 17/chemistry | 29642816 | 29642817 | 29642818 | 29642819 |
| Seed 23/control | 29642820 | 29642821 | 29642822 | 29642823 |
| Seed 23/chemistry | 29642824 | 29642825 | 29642826 | 29642827 |

They use the same sealed attempt002 source/config and receipt/output roots.
Every job requests 23 hours, `gpul40s`, four slots/16 GB per slot, one
exclusive GPU/host. Every dependency includes
`done(29642004) && done(29642352) && ended(29638861)`; each later reservation
also waits for its previous new job to end. The 19-hour training cap and
1.5-hour evaluation reservation remain unchanged. Actual prefix/restart
lineages are not yet executed. No checkpoint/result is inferred from queued
reservations; verified evidence002 links follow.

## Historical verified evidence002 and pre-reduction queued snapshot

[Evidence002](../outputs/hpc-results/source078-dense100/evidence002.tar.gz)
has SHA-256
`8c2402261b3811948d052660e23d83ba410a776fc39e36537ef28c47a7fe44a8`:
**165 safe members, 164 verified data-file digests**, exported at
**11:26:00 UTC / 13:26:00 DTU-local, 8 October**. The
[verification receipt](../outputs/hpc-results/source078-dense100/evidence002.tar.gz.verification.json)
records Python 3.12.14 / Windows 11, byte/hash/metadata scope only, no local
numerical work. The
[receipt audit](../outputs/hpc-submissions/source078-receipt-audit002.json)
passed all sixteen actual IDs, source/config/initializer/data identities,
serial dependencies, native proof and 1,707 maintenance observations. Exact
verification/audit commands are recorded in the ledger.

[Native passed proof](../outputs/hpc-results/source078-dense100/extracted-evidence002/runs/source-078-dense-100epoch-attempt002-density100-study/gate/passed.json)
supports the 48.331108863-second exact parameter/AdamW/four-stream/encoder
resume gate. The [common identity](../outputs/hpc-results/source078-dense100/extracted-evidence002/attempt002/submissions/receipts/identity.json),
[first scientific receipt](../outputs/hpc-results/source078-dense100/extracted-evidence002/attempt002/submissions/receipts/seed17-a0-e025-receipt.json)
and [last scientific receipt](../outputs/hpc-results/source078-dense100/extracted-evidence002/attempt002/submissions/receipts/seed23-a1-e100-receipt.json)
retain actual source/resources/serial dependencies. The
[scheduler snapshot](../outputs/hpc-results/source078-dense100/extracted-evidence002/scheduler_snapshot.json)
confirms all sixteen scientific jobs **PEND**, CPU/native **DONE** and the
original last job still pending. No long-training endpoint is inferred.

The [official quota text](../outputs/hpc-results/source078-dense100/extracted-evidence002/attempt001/submissions/quota-after-dedup.txt),
[deduplication plan](../outputs/hpc-results/source078-dense100/extracted-evidence002/attempt001/submissions/checkpoint-deduplication/plan.json)
and [completed observations](../outputs/hpc-results/source078-dense100/extracted-evidence002/attempt001/submissions/checkpoint-deduplication/completed.jsonl)
preserve the maintenance provenance and measured 88.45 GiB usage. Maintenance
does not supply quality evidence. Old attempts and uncertainty snapshots
remain historical; actual prefix/restart decisions and all full milestone
results are future work. Both SSH sessions closed normally, with no new source
changes or jobs after this snapshot.

## Planning time estimate

The source077 six-update native profiles imply about **36–38 hours of
training updates per method through all 100 epochs**, including its original
ten-epoch prefix where eligible. The former four-method cohort implied a rough
six-to-eight-day planning scale. The current scope retains only one candidate:
90 new epochs if its ten-epoch prefix is authenticated, otherwise a declared
100-epoch original restart. No verified finish ETA is available. Queue waits
are additional, and the estimate omits validation, repeated full evaluations, data/metrics overhead,
segment transfers and failure/retry time. The remaining work is 90 epochs per
authenticated ten-epoch parent or 100 for a declared original restart; no already completed prefix is charged again
as newly executed work. Actual full-data epoch timing should replace this
short-profile extrapolation when available.

## Diagnostics and fixed evaluation milestones

Continue epoch-level training metrics, reconstruction validation on **256
fixed graphs**, and the fixed **256-point validation prior** every epoch.
Keep their data/latent identities independent of optimization
draws and identical across matched methods. Log each objective component,
examples/update counts, frozen-encoder and anchor identity, gradient norms,
clipped-update fraction, elapsed training/validation time and actual memory.
Chemical coverage/count metrics must still verify the 1:1 full and tail
minibatches. Record checkpoint digest and cumulative RNG/update metadata at
every completed epoch.

Preserve source077's completed epoch-10 evaluations **where available** and perform the
predeclared full milestones at **25, 50, 75 and 100**:

- Deterministic posterior-mean reconstruction on all **7,347 held-out test
  graphs**, retaining validity and exactness denominators.
- The same independent **4,096 unfiltered prior points**, each argmax plus
  four categorical draws: raw validity, unique-valid yield per attempt,
  novelty and concentration of atom counts/elements/bonds.
- Probability-aware bounded repair on the fixed first 64 argmax points,
  with unchanged search limits, reporting witness/censored counts and cost
  conditional on found witnesses rather than a population nearest distance.
- Paired latent-cluster intervals versus the original, measured training/
  evaluation cost, and the ten-epoch chemical result if available. The
  canceled same-epoch controls provide no current matched-control comparison.

Epochs 25/50/75 are diagnostic milestones; **100 is the sole final endpoint**.
These checkpoints are **not opportunities to
select the best test checkpoint or change the loss using test results**.
Retain the epoch-100 endpoint even if an earlier milestone looks better;
report deterioration and overfitting rather than replacing the prespecified
endpoint. The implemented protocol must authenticate this milestone set and
the unchanged panel/search quotas before submission.

To determine whether more training is helping, report recent validation
trends alongside independent prior validity/diversity and held-out
reconstruction. A falling training objective alone does not show improved
generation, and a flat proxy does not prove convergence. Useful summaries
include the last ten completed epochs' validation/chemistry trends, widening
train/validation gaps, chemical gains accompanied by reconstruction loss,
gradient clipping frequency and declining diversity. These are diagnostics,
not an unvalidated automatic early-stopping criterion; the user requested
100 epochs.

## Interpretation and evidence required

The current comparison has one training seed and uses the same 4,096 latent
points for candidate/original pairing. It provides no training-seed
replication or isolated causal estimate of the fresh-prior contribution.
Repeated milestone comparisons do not establish family-wise significance.
Preserve exploratory uncertainty and the existing reconstruction/diversity
criteria. More epochs may improve raw generation, plateau, collapse or
overfit; no improvement is assumed. The chemical proxy and bounded repair
witnesses do not establish universal validity, synthesis, affinity or a
globally nearest-valid distance.

Required records are the new protocol/release hashes, authentic parent
checkpoints and completed epochs, CPU/native resume gates, every segment
intent/receipt/resources/dependency, runtime environment, restored/new
updates, fixed milestone evaluations and failures. Add actual events to the
[ledger](05_experiment_ledger.md) as they occur. The existing source077 jobs
and their ten-epoch results remain immutable history, and any chosen final
decoder must be frozen before later molecular-latent optimization.
