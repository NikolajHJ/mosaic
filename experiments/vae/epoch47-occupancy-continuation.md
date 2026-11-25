# Recovering Gaussian generation during epoch 47

Can the large VAE keep its improved PubChem reconstruction while recovering
variable molecule sizes and chemical validity on fresh Gaussian samples?
This experiment continues from the affected checkpoint. The earlier 24-update
decoder test moved 125/128 full graphs to zero full graphs, but only 9/128
draws were valid and distinct afterward. Count and chemical validity are
therefore separate outcomes. [Earlier measurements](occupancy-collapse.md)

## Training boundary

The last saved epoch-47 segment completed 47,682,960 of 60,000,000 real
molecules. The file is `sparse-02/checkpoint-000524177.pt`, with SHA-256
`408b6904281fbe5d9f92e8b40c62e301d099b0fdad87109ef0ca5d90ce172922`.
The checkpoint's completed-epoch field is 46. Its nonzero row locates the
next real batch within epoch 47. The remaining budget is **51,321 updates**
of 240 real molecules and 16 fresh Gaussian draws. The new branch reads the
saved model, AdamW moments, global RNG, fresh-prior generator and data cursor.
It regenerates the original epoch permutation from the original shuffle seed.

The runner authenticates the designated checkpoint and its source config.
It permits the prior-guidance change at that boundary and writes a
`recipe_handoff.json` receipt. Checkpoints from subsequent fixed-loss
allocations resume under the new config digest. A fresh output directory
keeps the old checkpoint available for comparison.

## Objective

Real training molecules still receive reconstruction and KL. For each of
the 16 fresh training vectors, both the student and the frozen epoch-46
reference decode that same vector. The prior objective has three terms.

| Term | Weight | Target and normalization |
| --- | ---: | --- |
| Structural bond repair | 0.375 | On successful repairs, mean categorical cross-entropy on unordered, occupied pairs. Average over the whole prior batch, with zero contribution from graphs without a repair target. |
| Occupancy | 1 | Binary cross-entropy between each live slot's occupied logit and the reference's soft occupancy, averaged over all draws and slots. |
| Reference chemistry | 1 | On reference argmax graphs passing chemical and schema checks, match conditional element and formal charge on occupied slots and bond type on unordered occupied pairs. Atom, charge and bond coefficients are 1, 1 and 2. The no-bond class weight is 0.1. Scale by the fraction of reference draws accepted. |

The live occupied logit is `logsumexp(real_element_logits) - no_atom_logit`.
It retains a corrective gradient even when the live decoder is very confident
that a slot is occupied. The reference assigns different occupancies to
different vectors. It is only used during training. The repair search changes
bonds, so its targets supply bond labels; element and charge targets come
from the independently frozen model. The older reference can still pass on
its own limitations.

## Reading the run

### 5% dropout in the next A100 allocation

DTU accepted A100 worker **30114866** and callback **30114867** on 19 November 2026. The superseded pair **30112466/30112467** exited with zero running seconds. The new worker was eligible and waiting for a free job slot at verification. [Scheduler handoff](../../outputs/hpc-submissions/vae-pubchem60m-e47-dropout-20261119a/scheduler-handoff.json) · [CPU optimizer check using the sealed release](../../outputs/runs/vae-large-dropout-20261119a/sealed/summary.json)

The [dropout recipe](epoch47-pubchem60m-parent-guidance-dropout.json) resumes the 3.94M-parameter student from update **524,177** for the remaining **51,321** updates of epoch 47. In each of its four decoder refinement rounds, independent dropout with probability 0.05 is applied to the output of the atom-update MLP immediately before adding it to the residual path. The student uses dropout during both real reconstruction and random-prior training. Evaluation, the frozen epoch-46 teacher, and ordinary inference use the unmasked network. The large run keeps its 240 real molecules and 16 prior vectors per update, its guidance objective and its checkpoint and monitoring schedules. [Refinement code](../../src/latentmol/models/refined_decoder.py) · [Trainer and handoff](../../src/latentmol/training/combined_continuation.py)

The handoff authenticates both the historical checkpoint recipe and the approved occupancy-recovery recipe. A local CPU optimizer step on the actual checkpoint restored all 249 AdamW parameter states and used 240 real and 16 prior examples (microbatch 8). Before the update, the original and dropout-enabled student gave identical evaluation logits on four fixed vectors for elements, charges and bonds. Two training-mode decodes differed, the joint optimizer step was finite, and the frozen teacher retained its checksum. The fixed panel will measure whether validity and diversity improve as training proceeds. This continuation has a single dropout arm. [Optimizer check](../../outputs/runs/vae-large-dropout-20261119a/summary.json)

### Fixed-panel measurements

The fixed panel uses 128 Gaussian vectors and 64 held-out validation molecules.
It compares the starting checkpoint with updates 10, 100, every 1,000 and the
end of the allocation. Read the fraction of prior graphs at 50 atoms alongside
the count histogram, raw valid fraction, number of distinct valid structures,
and exact held-out reconstruction. The panels use their own random generators.
They provide early evidence of recovery; a larger independent panel is needed
for a precise final comparison.

With the production panel's seed (2026111909), the CPU starting baselines are:

| Checkpoint | Valid draws | Distinct valid structures | Full 50-slot graphs | Median atoms |
| --- | ---: | ---: | ---: | ---: |
| Frozen epoch 46 | 91/128 | 91 | 0/128 | 22 |
| Partial epoch 47, before the loss change | 7/128 | 7 | 125/128 | 50 |

The 121 invalid draws at the handoff comprised 99 sanitization failures and
22 disconnected graphs. We also restored the real handoff checkpoint on CPU
and compared the training weights, all 249 AdamW parameter states, global
CPU RNG and fresh-prior RNG with their saved values. All matched.
[Baseline counts and checkpoint hashes](../../outputs/runs/vae-epoch47-guidance-baseline-20261119a/summary.json)

One CPU update from those saved weights and AdamW moments exercised the full
240-real/16-prior loss using 240 sampled training rows and a CPU microbatch of
8. All 16 live priors received a bond repair. Eleven reference graphs passed
the strict chemistry/schema check. BCE was 7.764, conditional chemistry loss
was 5.150 and the unweighted repair loss was 0.0102. The pre-clip gradient
norm was 70.48, the student updated and the epoch-46 reference matched its
initial checksum. This checks that the objective is active. It does not
measure sustained recovery or A100 throughput.
[One-update result](../../outputs/runs/vae-occupancy-production-smoke-20261119a/summary.json)

The production entry point is
`python -m latentmol.training.combined_continuation`, with the [recovery
config](epoch47-pubchem60m-parent-guidance.json), `--resume`,
`--resume-sha256`, `--batch-size 240` and `--microbatch-size 120`.
The training run uses the existing A100 launcher.

LSF accepted GPU job **30112466** and checkpoint callback **30112467**
on 19 November 2026. One A100 worker is requested for up to 24 hours per
allocation. The callback can schedule the next segment from its saved
checkpoint if the allocation ends before epoch 47 finishes. Outputs live
under `/work3/s234818/latentmol/runs/vae-pubchem60m-e47-parent-guidance-20261119a/sparse-01`.
LSF recorded zero running seconds for the superseded worker (30105550), and
its callback (30105551) exited. The new worker was eligible and waiting for
an A100 slot when checked.
[Sealed run plan](../../outputs/hpc-submissions/vae-pubchem60m-e47-parent-guidance-20261119a/plan.json)
· [Scheduler receipt](../../outputs/hpc-submissions/vae-pubchem60m-e47-parent-guidance-20261119a/cutover.json)

## Epoch 48 design proposal, 20 November 2026

Fork two production arms from the same completed epoch-47 checkpoint. The control uses the four trained refinement rounds (3,941,860 parameters). The treatment appends three independently weighted triangle and bond-message rounds, each with initially zero atom and pair residual gates (4,301,290 parameters). Train the encoder and decoder on every update in both arms using AdamW. Retain the optimizer state for the inherited parameters. Set aside the intermittent encoder-freezing schedule.

Each arm would see the same 60M real training graphs and the same 60M fresh Gaussian draws, with 240 of each per update for 250,000 updates. Real graphs retain reconstruction and KL. The candidate prior objective uses valence projection on all 240 random decodes and the epoch-46 frozen-reference occupancy and chemistry supervision on a matched subset of 16. Average each term over its own population. Use the same chosen recipe and evaluation panel for both arms. Compare distinct valid raw decodes, exact reconstruction, atom counts and 50-slot saturation, connectivity, and GPU-hours. The projection and full 1:1 step still need implementation and end-to-end timing before submission.

**Proposed faster chemistry target.** Reuse each live random decode. Take its detached hard graph and hold occupied slots, elements and formal charges fixed. On each graph, try up to eight single-edge edits to reduce excess valence. Prefer downgrading a bond order; also allow removing a single bond outside a protected spanning forest of the original occupied graph. Construct the forest once, preferring high-confidence existing bonds. Preserving it keeps connected components from splitting. The first version protects aromatic bonds and omits aromatic centers and unknown element/charge environments from excess calculations. Implicit hydrogens make unused capacity legitimate.

Rank eligible edits by the loss of log probability under the detached conditional bond distribution per unit of excess removed. Accept only strict improvements. Use the final labels on edited unordered pairs as detached targets for cross-entropy on the original live symmetric bond logits. Average per edited pair within each graph, then over all prior graphs, including zeros for unedited graphs. A partly repaired graph can teach a local correction. No other pair or atom receives a direct projection target. Keep the existing occupancy and frozen-reference chemistry losses separate. This surrogate teaches selected upper-valence corrections; it does not certify the graph, resolve all aromatic errors or join disconnected components.

Before choosing a loss for epoch 48, replay identical prior decodes with both teachers and measure the fraction of known-valid inputs changed, partial-repair coverage, raw and projected strict validity, connectivity, sulfur and bond counts, and gradient scale alongside reconstruction. Compare end-to-end optimizer-step throughput and CPU use on the actual recipe. The old 15:1 V100 timing found about 10 microseconds of exposed repair wait per update because the CPU search overlapped real-molecule training. The current frozen-reference branch adds its own compute. This proposal has no measured speed or quality result. [Existing structural proposals](../../src/latentmol/guidance/repair_proposals.py) · [Large-run timing](large-corpus.md#benchmarks-completed-epoch-47-queued)

### Epoch 48 paired first leg

Both arms fork from the same authenticated, completed epoch 47 and train the whole VAE on 32 GB V100s. Each first allocation targets 25,000 updates with 240 real graphs and 240 fresh Gaussian draws per update. Both use microbatches of 48. The production worker saves a checkpoint on a wall-time stop. Compare quality at common update counts and elapsed training time on the same GPU class. The launcher submits only the first leg.

The [shared recipe](epoch48-paired-leg1.json) explicitly selects graph-matched reconstruction. The inherited large checkpoint used canonical targets with slotwise cross-entropy. The new recipe aligns atoms, charges and both bond axes together before computing real-graph reconstruction. Exact relabelings receive an exact correspondence. On incorrect graphs the alignment search is approximate. See [training update](../../src/latentmol/training/projected_continuation.py), [graph matcher](../../src/latentmol/training/matching.py) and [bond projection](../../src/latentmol/guidance/valence_projection.py).

The frozen-reference loss supervises 16 of the 240 random draws per update. Bond projection supervises all 240 with coefficient 0.375. Both arms retain the same AdamW step policy and inherited optimizer moments. The first-leg evaluation uses 512 fixed prior draws and 256 held-out real molecules at initialization, after 10 and 100 updates, periodically and at the endpoint. It reports raw validity, unique valid structures, disconnected graphs, 50-slot saturation and exact reconstruction.

Local CPU replay used the cached partial epoch-47 checkpoint at update 524177 with the authentic epoch-46 reference. The initial three heads were bit-identical across arms, both arms took an optimizer step on 32 real graphs and 32 random draws, and all six new gates moved. Projection edited 21 of 32 priors and zero of 128 known-valid real graphs. Nineteen priors still exceeded a modeled valence cap after projection. These checks measure behavior on a small earlier-state panel. [Replay](check_projected_fork.py) · [Results](../../outputs/runs/vae-e48-pair-20261120c/graph-matched-smoke.json)

The [original submission script](submit_epoch48_pair.py) sealed and submitted the pair. The CPU gate follows the epoch-47 callbacks across allocations and releases the two held GPU workers when it verifies the finished checkpoint. Both workers verify that checkpoint again before training.

On 20 November LSF accepted V100 control **30131204**, A100 seven-round arm **30131205**, and CPU gate **30131206**. A subsequent `bjobs -l` confirmed both GPU workers in PSUSP and the gate waiting on `done(30114867)`. Epoch-47 job **30114866** was pending. Nineteen focused CPU regression cases passed, including exact matches with shuffled slots and equal losses and gradients across logical-batch chunking. [Submission receipt](../../outputs/hpc-submissions/vae-e48-paired-leg1-20261120c/scheduler.json)

Later on 20 November, the seven-round allocation moved to a 32 GB V100. The four-round job is **30131204**, the seven-round job is **30131363**, and their checkpoint gate is **30131364**. LSF confirmed the original A100 worker and gate exited with zero runtime. The replacement archive differs only in the seven-round device requirement. Both workers read the same parent receipt in the original control directory. Seven-round V100 memory use and throughput await its first allocation. [Verified scheduler state](../../outputs/hpc-submissions/vae-e48-paired-v100-20261120d/verified.json) · [Queue transfer](../../outputs/hpc-submissions/vae-e48-paired-v100-20261120d/cutover.py)

### Functional-group loss in the seven-round arm

The [seven-round recipe](epoch48-seven-round-motifs.json) adds the weak batch functional-group loss to the first 25,000-update allocation. The goals are **8% valid aromatic-ring yield** and **12% valid amide yield**, each as a fraction of all sampled graphs. A graph can count toward both. Every fourth optimizer update the loss examines 48 of the 240 already-decoded priors. Its coefficient is 0.30 on those updates. A full first leg therefore includes 6,250 motif measurements on 300,000 sampled graphs. The training log records valid attempts, aromatic and amide hits, and whether each group contributed a gradient. [Estimator, prior calibration and limitations](weak-functional-groups.md#higher-floors-at-the-same-weight)

The four-round production control uses the original epoch-48 objective. A difference between it and the seven-round production arm can reflect both decoder depth and functional-group guidance. The separate 256-update four-round pilot tests the effect of the lower 5%/8% floors under a matched architecture.

A matched local CUDA replay exercised the production batch shape on the cached epoch-47 checkpoint at update 541677. On the first scheduled motif step, the extra gradient had 0.59% of the base gradient norm. The batch contained 19 valid sampled molecules, one with an aromatic ring and none with an amide. All six new residual gates moved. The two arms used identical training random streams and retained the frozen reference and source weights. Ten focused unit and integration tests also passed. [Local seven-round replay](../../outputs/runs/vae-e48-r7-motif-production-check-20261121a/result.json)

On 21 November DTU accepted seven-round job **30151761** on a 32 GB V100. Callback **30151762** releases it and the original four-round control **30131204** after verifying the completed epoch-47 checkpoint. The scheduler rejected an in-place command edit as administrator-only, so we verified the replacement command and dependencies before retiring old worker **30131363** and gate **30131364**. Both ended with zero runtime. The new worker writes to `/work3/s234818/latentmol/runs/vae-e48-r7-motifs-20261121a/r7`. [Scheduler verification](../../outputs/hpc-submissions/vae-e48-r7-motifs-20261121a/reservation-verify.json)

### Epoch 48 continuation and functional-group comparison

The seven-round continuation resumes the first allocation's exact terminal checkpoint, including AdamW state, training random streams and the shuffled data cursor. It targets the remaining rows of the 60M pass and writes a checkpoint before each 24-hour V100 allocation ends. One callback queues the next allocation after verifying the worker and checkpoint. The four-round arm retains its original first-leg budget. [Continuation controller](epoch48_segment_chain.py) · [Submission script](submit_epoch48_continuation.py)

At the first handoff, a matched comparison examines the original motif loss and the new local-neighborhood loss. Both train for 512 updates from the same checkpoint with the usual 240 real molecules and 240 Gaussian draws per update. The candidate reinforces the atoms and bonds of an aromatic ring or amide already present in a valid prior decode at a nearby latent vector. The comparison uses 4,096 fresh, paired categorical draws and the existing 256 held-out reconstructions. Adoption requires at least four extra valid aromatic molecules and a difference greater than one paired standard error. Allowed declines are at most one percentage point in overall validity, half a percentage point in amide yield, and two exact reconstructions. These are exploratory adoption thresholds. The selected checkpoint continues immediately, including its optimizer state. The other 512 updates count as experiment cost. Later allocations read the recorded choice. [Recipe](epoch48-seven-round-motif-followon.json) · [Pilot results and limitations](training_data_eda/epoch48-motif-completion.md) · [Decision implementation](../../src/latentmol/training/motif_activation.py)

On 22 November at 08:58 CET, DTU reported the original seven-round job **30151761** running. Worker **30211418** was pending on `done(30151761)` with a 1,440-minute V100 limit. Callback **30211420** was pending on `ended(30211418)`. The production comparison runs inside the worker. [Scheduler verification](../../outputs/hpc-submissions/vae-e48-r7-motif-followon-20261122b/live-verification.json) · [Release and checkpoint lineage](../../outputs/hpc-submissions/vae-e48-r7-motif-followon-20261122b/plan.json)

### Ring sizes in the second seven-round allocation

Both arms start at update **597604**, after 5,305,440 of the 60M training rows. Every fourth update they evaluate 240 prior draws with the aromatic and amide targets measured in a uniform audit of 50,000 training molecules (87.056% and 47.988%). The treatment also penalizes the excess proportions of valid molecules containing a three-membered ring or a ring of eight or more atoms. It matches each training reference rate to the heavy-atom counts in the generated batch and uses the same chemical evaluations for both ring indicators. The internal ring coefficient is 0.1 and the shared motif coefficient is 0.30. [Recipe](epoch48-seven-round-ring-sizes.json) · [Reference counts, gradient measurements and local training smoke](training_data_eda/ring-topology-probe.md)

The V100 worker trains both arms sequentially for 512 updates each, then compares 4,096 paired fresh categorical draws and 256 validation reconstructions. To carry the ring loss forward, both size-adjusted excesses must hold steady or fall. The candidate also needs at least four net ring improvements exceeding one paired standard error among draws that remain valid and cyclic at the same atom count. Safeguards monitor raw validity, ring presence, aromatic and amide yield, unique nonempty scaffolds and exact reconstruction. The selected arm supplies the checkpoint, optimizer state and training cursor for the rest of epoch 48. The control uses the same corpus targets, neighbor guidance and sampling budget. [Decision rule](../../src/latentmol/training/ring_activation.py)

DTU accepted the 24-hour, eight-core V100 worker **30249335** and CPU callback **30249336**. At 16:53 CET on 22 November the worker was pending with no predecessor dependency. Its callback waits for that exact worker and checks the saved checkpoint before advancing. The superseded worker **30211418** and callback **30211420** both exited without starting. [Verified scheduler and parent checkpoint](../../outputs/hpc-submissions/vae-e48-r7-rings-20261122a-s02/live-verification.json)

### Timing an encoder update every fifth step


On 20 November 2026 a local RTX 4060 Laptop GPU timed the four-round large architecture at eight real molecules and eight prior vectors. There were six warmup pairs and 28 measured pairs with alternating execution order. Both modes used canonical PubChem inputs, FP32, randomly initialized weights, real reconstruction and KL, and a proxy prior bond cross-entropy. They differed in whether the real branch retained the encoder backward graph. Median network times were 109.24 ms with a trainable encoder and 106.46 ms frozen. Weighting those medians for four frozen steps per joint step gives 107.02 ms, or 2.0% less time. The frozen branch was faster in 21 of 28 pairs. The benchmark excludes optimizer steps, reference guidance, projection, and I/O. It does not measure A100 epoch time, model quality or the seven-round decoder. [Timing script](../../outputs/runs/encoder-one-in-five-timing-20261120a/time_network.py) and [raw observations](../../outputs/runs/encoder-one-in-five-timing-20261120a/timing.json).

### Epoch 48 segment 3 · generated graph reconstruction

On 23 November the ring comparison in segment 2 chose its control. The ring-size loss failed both topology criteria (net −30 on 2,527 matched valid cyclic graphs of equal size). Both arms reconstructed 247/256 validation molecules. The control therefore carries forward the corpus-frequency aromatic and amide loss and the previously trained motif-neighbor guidance. [Ring decision and live scheduler evidence](../../outputs/hpc-submissions/vae-e48-r7-roundtrip-20261123a-s03/live-verification.json)

Segment 3 adds one encoder objective. Every fourth update, take the argmax graphs of the first 16 already-decoded Gaussian priors in training mode and keep up to eight in draw order that pass strict chemistry and a lossless canonical graph mapping. Each provides a fixed graph target. Encode it to the mean, decode that mean with frozen decoder parameters and dropout disabled on this branch, and average the normal graph-matched atom, charge, bond and size reconstruction over the selected parents. The coefficient is 0.10. Empty selections contribute zero and their rejection reasons are recorded. The original reconstruction, KL, chemistry and motif branches still update their usual parameters. The production batch is 240 real and 240 random, with 48 per microbatch. The neighbor term still uses one correlated Gaussian perturbation per selected center (correlation 0.995), preserving only the selected aromatic ring or amide and its local connectivity. The inherited neighbor coefficient is 3 inside the existing 0.30 motif weight. [Production recipe](epoch48-seven-round-generated-roundtrip.json) · [Round-trip implementation](../../src/latentmol/training/generated_roundtrip.py) · [Trainer](../../src/latentmol/training/projected_continuation.py)

Quality panels at the start, +10, +100, absolute multiples of 1,000 updates and the segment end include 128 fixed evaluation prior vectors. The monitor reports how many generated graphs are eligible, how many re-encoded means are valid and exact, and how many extra and missing bonds occur in canonical slots. One panel regenerates the parents with the current checkpoint; another holds the first panel's eligible molecules fixed within the allocation. The original 512 prior decodes, held-out real reconstruction on 256 PubChem molecules, and motif/size diagnostics remain in the report. [Monitor](../../src/latentmol/training/occupancy_monitor.py)

A local RTX 4060 smoke ran two matched steps per arm from the selected control checkpoint at update 598116, including its optimizer moments. It used smaller batches (48 real + 48 prior, microbatch 16), with real training rows sampled locally rather than the production shuffle. On the scheduled update eight parents qualified after scanning nine candidates. The round-trip term added 0.01775 to the loss. The pre-clip total gradient norms were 1.525 and 1.571. Both arms ended with 58/64 exact real reconstructions, 117/128 valid prior decodes, and 20/111 exact on the fixed generated parents. The smoke measures execution and gradient flow; longer-term molecular quality is pending. [Replay](../../outputs/runs/e48-generated-roundtrip-20261123a/smoke.py) · [Step logs](../../outputs/runs/e48-generated-roundtrip-20261123a/smoke.json) · [Quality panels](../../outputs/runs/e48-generated-roundtrip-20261123a/roundtrip/quality-panel.jsonl)

DTU queued 24-hour, eight-core V100 worker **30263213** behind successful completion of **30249335** and callback **30263214** behind the new worker. The old pending callback **30249336** exited with zero runtime. At 09:10 CET the original worker was running at update 599280. The successor authenticates that worker's terminal checkpoint and resumes its optimizer, RNG streams, selected motif policy and training cursor. On a successful partial allocation its callback schedules the subsequent segment until epoch 48 is complete. [Scheduler receipt](../../outputs/hpc-submissions/vae-e48-r7-roundtrip-20261123a-s03/scheduler.json) · [Sealed plan](../../outputs/hpc-submissions/vae-e48-r7-roundtrip-20261123a-s03/plan.json)

### Switching at the next checkpoint on 23 November

The operator requested the generated-graph reconstruction loss in the middle of the allocation. Segment 2 published update **600116**, after **5,908,320** of the 60 million real training rows in epoch 48. A local CPU reload verified the checkpoint SHA-256, model weights, all 354 AdamW parameter states, six generator states, the saved CUDA RNG and the selected ring-control policy. DTU disables suspension of running workers. The final segment-2 log reached 600137 while the checkpoint was verified and the job terminated. The successor will process those 21 updates from the saved state. [Checkpoint verification](../../outputs/runs/e48-s02-early-stop-600116/payload-validation.json)

The replacement reuses the archived segment-3 training code and its 24-hour V100 request. Its worker accepts the authenticated early stop only for this handoff. Later segments still advance on successful completion. Use `submit_epoch48_continuation.py` with `--early-roundtrip` for the operator actions `stage`, `retire`, `stop` and `release`. The superseded worker and callback exited before starting. At 10:03 CET worker **30264092** was eligible for scheduling, with **30264093** waiting on it. [Release receipt](../../outputs/hpc-submissions/vae-e48-r7-roundtrip-early-20261123a-s03/scheduler-release.json) · [Live verification](../../outputs/hpc-submissions/vae-e48-r7-roundtrip-early-20261123a-s03/live-verification.json)
