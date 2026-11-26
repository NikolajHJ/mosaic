# Three-block encoder comparison in epoch 48

Does a more graph-aware encoder recover generated molecules and leave them easier to edit, and how fast do these measures improve on the real training stream?

The candidate adds explicit per-atom counts of single, double, triple and aromatic bonds, three independently weighted residual graph-attention blocks and a weak excess-confidence loss. Each block has four heads, width 96, a 192-wide feed-forward hidden layer, and learned attention biases for direct bond type and shortest-path distance capped at five. Canonical ordering and the 256-dimensional latent are shared. The encoder grows from 2,173,216 to 2,398,084 parameters. The seven-round decoder has 2,128,074 parameters in both arms. Zero-initialized output projections and bond-count projection preserve initial outputs. The checkpoint graft preserves inherited AdamW moments, parameter order on reload, and training RNG streams. [Encoder](../../src/latentmol/models/encoder.py) · [Graft](../../src/latentmol/training/encoder_expansion.py)

Every fourth update the candidate reuses 16 Gaussian prior vectors for a separate evaluation-mode decode and selects up to eight strictly valid, losslessly canonicalizable parents. For exact re-encodings it compares winner–runner-up logit gaps for occupied atoms and existing undirected bonds. Only excess above the source gap plus one logit incurs a penalty. Correspondences respect graph symmetry. The branch updates the encoder through a decoder with frozen weights. Its effective coefficient is 0.01 on scheduled updates. The ordinary generated-parent reconstruction and the joint encoder–decoder training retain their existing targets and weights. [Confidence loss](../../src/latentmol/training/confidence_margins.py) · [Integration](../../src/latentmol/training/generated_roundtrip.py)

## Local matched pilot, 23 November 2026

Both arms took 16 joint updates from checkpoint 600116 with 16 real PubChem molecules and 16 fresh prior vectors per update. Training rows, starting weights of inherited layers, optimizer moments and RNG streams matched. This short run used a smaller training batch than production.

| Fixed panel | Before | Current encoder | Three changes |
|---|---:|---:|---:|
| Exact held-out reconstructions / 128 | 118 | 117 | 120 |
| Valid random argmax graphs / 128 | 114 | 119 | 119 |
| Exact recovery of independently generated parents / 54 | 10 | 10 | 9 |
| Exact recovery on the initially recoverable panel / 32 | 32 | 31 | 28 |
| Parents with a distinct valid edit in four radius-1 directions / 32 | 6 | 7 | 4 |

The last row counts an unrecovered parent as zero successful edits. The three blocks and the bond-count projection all learned. Five of 32 selected parents contributed to the confidence loss over the four scheduled updates. The small panels provide mixed early evidence. Local timings are confounded by control-first warmup. [Pilot command and raw data](../../outputs/runs/e48-encoder-upgrade-local-20261123a/) · [Totals](../../outputs/runs/e48-encoder-upgrade-local-20261123a/summary.json)

## Next production leg

LSF accepted candidate **30302772** on the 32 GB V100 queue. At 15:28 CET on 23 November, segment-3 worker **30264092** was running. The candidate waits for callback **30264093** to successfully schedule the ordinary segment-4 control. Both arms resume segment 3's authenticated final checkpoint. The candidate writes its own checkpoints under the home quota. Its single allocation has a 24-hour scheduler limit, an 81,000-second internal budget and a cap of 25,000 updates. Both arms train encoder and decoder on the next real 60M-dataset rows with 240 real and 240 prior points per update. [Submission receipt](../../outputs/hpc-submissions/vae-e48-encoder3-nextleg-20261123a/e48-encoder3-s04.receipt.json) · [Pilot recipe](epoch48-encoder-upgrade.json)

Both arms measure fixed panels at the fork, +10, +100, every absolute 1,000th update and the endpoint. A CPU comparison is scheduled by the candidate worker once the production control job ID exists. It compares changes per matched update and per measured training hour, reports confidence-loss exposure, and measures distinct valid edits from shared parents under four common radius-1 perturbations. It labels each endpoint's actual training budget. Protein binding and fresh protein rescoring are separate experiments. These two arms measure the combined change. [Training entry point](encoder_upgrade_pilot.py) · [Comparison](compare_encoder_upgrade.py) · [Launch and dependencies](launch_encoder_upgrade.py)

## Early fork with an eight-hour control

On 23 November the operator requested an early switch and a shorter control. The revised comparison gives candidate **30330582** one 24-hour V100 allocation and control **30330583** one eight-hour V100 allocation. Each has eight CPU slots and one 32 GiB GPU. The trainer budgets are 81,000 and 28,200 seconds respectively, including the final save reserve. Both resume the same copied checkpoint, optimizer moments, RNG streams and data cursor. Both retain the initial panel, +10 and +100 checks, regular quality panels and rolling checkpoints. These are single allocations. The original encoder runs once for comparison.

The common fork is update **611116** at row **8,548,320** of the epoch-48 pass. Its SHA-256 is `96804708c82851c068d3462b7032bdf59f01892f1cd55f89b2df7abbca161497`. After the checksum-verified copy, LSF stopped worker **30264092**. Its log reached update 611122, so the new arms repeat six logged updates. The original callback **30264093** and candidate reservation **30302772** exited before dispatch. At 22:59 CET both replacements were released and pending on gpuv100. [Verified stop and checkpoint](../../outputs/hpc-submissions/vae-e48-encoder3-early-8h-20261123a/stop.json)

CPU job **30330584** reads the fixed-panel curves after the control exits and records which training updates both arms have reached. Its report can be read while the candidate is still running. CPU job **30330585** performs the full paired evaluation after both workers exit. Its learning curves compare matched updates. Its geometric editability results label the training budget for each endpoint. The two branches share the fixed generated cohort established at their fork. Segment 3's 111 fixed generated parents were selected at an earlier checkpoint and form a separate cohort.

[Scheduling receipts](../../outputs/hpc-submissions/vae-e48-encoder3-early-8h-20261123a/scheduler.json) · [Bounded training entry point](early_encoder_arms.py) · [Eight-hour report](encoder_interim.py)

## Recovering generated parents with a frozen decoder

Can optimization recover a known generated graph from its encoded mean? The 24 November pilot uses a local copy of the three-block candidate at update 623116. It screens the first 128 of the production monitor's 512 seeded Gaussian draws. It retains every valid graph that can be converted to the training representation without changing its labels. The same checkpoint generates the target, encodes it and decodes the refined latent. The original generating vector is retained solely to check that the frozen model can reproduce the target.

Starting at the encoded mean, each molecule receives 50 Adam updates to its own latent. The objective uses the checkpoint's reconstruction weights and recomputes the training pair assignment at each step. The learning rate is 0.02 and the maximum Euclidean displacement from the mean is 4. Dropout is off. No repair is applied to the output. Readouts at 0, 1, 5, 10, 20 and 50 updates retain every parent, including failures and previously recovered parents. Exact recovery compares canonical stereo-free SMILES. Extra and missing bonds are counted once per pair in canonical training slots, with separate counts after graph matching. Matching on an inexact graph is approximate.

```powershell
$runDir = 'outputs/runs/e48-inverse-recovery-623116-20261124a'
python experiments/vae/frozen_decoder_recovery.py --checkpoint "$runDir/checkpoint-000623116.pt" --sha256 504fccee8954f684bace24c84e3b04af239260927531248b3ce5f974725d5df8 --output "$runDir/recovery.json"
```

Of 128 draws, 114 distinct parents qualified. Seven were disconnected, two failed sanitization and five could not be canonicalized with labels preserved. All 114 original generating latents reproduced their targets. The following are paired endpoint counts starting from the encoded mean.

| Latent updates | Exact / 114 | Valid / 114 | Extra bonds | Missing bonds |
|---:|---:|---:|---:|---:|
| 0 | 27 | 58 | 317 | 50 |
| 1 | 32 | 59 | 302 | 32 |
| 5 | 51 | 72 | 199 | 7 |
| 10 | 64 | 84 | 147 | 3 |
| 20 | 77 | 92 | 89 | 0 |
| 50 | 94 | 103 | 42 | 0 |

At 50 updates, 67 of 87 initially missed identities were recovered and all 27 initial matches were retained. Of the original 317 spurious edges, 279 disappeared and four new ones appeared. Canonical-slot and graph-aligned aggregate edge counts agreed at each recorded step. All 20 remaining errors retained an extra bond and reached the displacement limit within 0.001. The limit's effect on recovery was not tested. The original VAE state and buffers matched the checkpoint after the local GPU run. This measures target-guided recovery for generated molecules, with a known target and a frozen model. [Runner](frozen_decoder_recovery.py) · [Per-parent results](../../outputs/runs/e48-inverse-recovery-623116-20261124a/recovery.json)

## One-pass encoder distillation at 623116

Can an encoder learn from refined latent vectors and recover new generated parents in one pass? The pilot screens 384 independent Gaussian draws. For each eligible parent, it saves the first latent that reproduces its chemical identity within 20 teacher updates, using Adam at 0.02 and a radius of 4. A parent already recovered exactly receives its original mean as the target. Failed refinements are counted. Both training arms see the same qualified examples.

Both arms start from the three-block encoder at 623116 and use its frozen decoder in evaluation mode. Each receives 128 Adam updates at 0.00002. Each minibatch contains 16 generated parents supervised by graph-matched reconstruction and 16 distinct real training examples whose posterior means are anchored to the original encoder (mean squared error, weight 2). The distilled arm also regresses the generated means toward their teacher latents (mean squared error, weight 2). The encoder variance head stays frozen; its shared inputs can still move. These are isolated posterior-mean experiments. Evaluation uses direct argmax decoding and a single encoder pass, with no repair or latent optimization.

The prespecified endpoint is update 128, with an intermediate readout at 32. Every eligible unique parent from 512 independently seeded draws enters evaluation, regardless of baseline difficulty. The same 512 held-out PubChem molecules are scored in both arms. All evaluation identities are excluded from student training. Extra and missing bonds use the canonical-slot convention above. [Pilot](encoder_distillation.py) · [Teacher selection](refined_latent_bank.py) · [Evaluation](encoder_distill_panel.py)

```powershell
$out = 'outputs/runs/e48-encoder-distill-623116-20261124a'
python -m experiments.vae.encoder_distillation prepare --output $out
python -m experiments.vae.encoder_distillation fit --output $out
```

The local RTX 4060 pilot found 348 eligible teacher parents among 384 draws. The original encoder recovered 88 immediately and latent refinement rescued another 168. The remaining 92 had no exact teacher within budget. All 256 usable parents were disjoint from evaluation, and each arm also used the same 2,048 distinct real training examples. The fresh evaluation seed gave 454 eligible unique molecules out of 512 draws. Both final decoder states and variance-head weights matched their original values exactly.

| One-pass result at the 128-update endpoint | Starting encoder | Graph loss + refined-latent targets | Matched graph-loss control |
|---|---:|---:|---:|
| Exact generated identities / 454 | 110 | 142 | 152 |
| Chemically valid re-encodings / 454 | 249 | 257 | 253 |
| Extra / missing bonds on generated parents | 1,119 / 180 | 1,025 / 154 | 1,064 / 151 |
| Exact held-out reconstructions / 512 | 480 | 478 | 476 |
| Valid held-out reconstructions / 512 | 491 | 492 | 490 |

Relative to baseline, the distilled arm gained 39 fresh generated identities and lost seven. It lost two held-out identities and gained none. The control gained 56 fresh identities and lost 14; on held-out molecules it gained two and lost six. Directly comparing the two endpoints, ten fresh molecules were exact only with distillation and 20 only with the control. The refined targets reduced spurious edges and preserved more of the starting encoder's successes. This pilot provides no observed incremental benefit in exact recovery from adding latent regression to the shared graph objective.

On the 168 rescued **training** parents, the distilled encoder recovered 53 in one pass and the control recovered 56. The mean Euclidean distance to the teacher for those parents was 1.37 initially, 1.52 after distillation and 2.19 in the control. Exact graph recovery can improve even as distance to a particular first-hit vector increases. These numbers come from one teacher bank, training seed and paired evaluation panel. [Summary](../../outputs/runs/e48-encoder-distill-623116-20261124a/comparison/summary.json) · [Per-molecule endpoints and encoder weights](../../outputs/runs/e48-encoder-distill-623116-20261124a/comparison/)

## Single next leg with focused feedback on invented bonds

On 24 November, LSF accepted V100 job **30359417** with dependency `done(30330582)`. It restores the three-block candidate's terminal checkpoint, including both networks, AdamW moments, RNG and data cursor. It has one 24-hour allocation with eight CPUs, an internal budget of 81,000 seconds and a 25,000-update cap. Its first quality panel establishes the leg baseline; subsequent panels track the same parents and a freshly decoded cohort. Each reports exact identities and extra/missing bonds separately for targets with 0–2, 3–4 and 5+ independent cycles. This single trajectory measures changes during the leg. The two simultaneous objective changes cannot be separated causally by this run.

Every fourth update the trainer decodes 16 prior vectors with dropout disabled and keeps up to eight valid, losslessly canonicalizable parent graphs. It reuses this cohort and one posterior decode for ordinary graph reconstruction, the existing confidence margin and focused bond supervision. For each parent, it uses the ordinary reconstruction's atom assignment and considers each undirected pair whose two endpoints exist in both graphs. Where the parent has no bond and a predicted bond wins, it selects up to four pairs with the largest winning-logit margin over NO_BOND. The focused term is the mean softplus of these margins per parent, averaged over all selected parents including error-free graphs. Its inner weight is 0.1, giving a coefficient of 0.01 after the existing outer round-trip weight. This branch guides the encoder through a frozen decoder; the ordinary real-molecule and prior objectives continue joint training. [Policy](epoch48-encoder-targets.json) · [Loss](../../src/latentmol/training/spurious_bonds.py) · [Resume entry](encoder_next_leg.py)

A single local RTX 4060 update on a copy of checkpoint 623116, with 16 real examples and 16 prior draws, exercised the scheduled branch. Seven of eight eligible parents had extra bonds. The loss selected 26 of 37 false-positive pairs and contributed 0.01492 to a joint loss of 0.35832. The generated-parent branch had a finite encoder gradient and zero decoder parameter gradients. The same step updated the decoder through the ordinary objectives. [Spot check and command](../../outputs/runs/e48-fp-joint-spot-623116-20261124a/) · [Verified submission](../../outputs/hpc-submissions/vae-e48-encoder-invented-bonds-20261124a/e48-encoder-invented-bonds.receipt.json)

### Operator-requested early start

On 24 November the operator requested an immediate switch. Candidate 30330582 exited after logging update 623651. Its newest published checkpoint was 623116 (11,428,320 real rows into the epoch), copied and SHA-256 verified both before and after the stop. The successor restores its model, optimizer and RNG and reprocesses the 535 later logged updates under the new objective. The original run retains its observed quality panels. Its sealed end-of-leg comparison expects a normal terminal summary, which an operator stop does not produce; the interim report can still compare recorded common updates.

DTU restricted `bmod` to administrators, so the original dependent reservation 30359417 was retired and its exit verified. The single replacement **30359446** was submitted without a dependency and moved to first among the user's pending V100 jobs. The scientific release differs only in the resume entry point, which checks the explicit operator-stop receipt and pinned checkpoint. Scheduler resources and training objectives are inherited. [Cutoff and checksum](../../outputs/hpc-submissions/vae-e48-encoder-bonds-early-20261124a/operator-stop.json) · [Replacement receipt](../../outputs/hpc-submissions/vae-e48-encoder-bonds-early-20261124a/e48-bonds-early.receipt.json)

## Wider atom attention and direct bond attention

On 25 November the operator requested both architecture changes on the production leg. The three encoder atom-attention layers and four decoder atom-attention layers now use ten heads of 24 features. The 96-feature atom states and 48-feature pair states keep their original widths. After the seven refinement rounds, two independent residual blocks let each possible bond attend to the other possible bonds incident to either endpoint. The endpoint operations share weights, use soft atom occupancy and average their outputs symmetrically. Each block has ten 24-feature heads and a 48→96→48 feed-forward network. Both residual output projections start at zero. All prediction heads still use the existing graph vocabulary. [Atom attention](../../src/latentmol/models/refined_decoder.py) · [Bond attention](../../src/latentmol/models/bond_attention.py)

The expansion adds 532,494 parameters for a total of 5,058,652. Each trained atom head initially has two or three copies. The graft divides each head’s outgoing contribution among its copies with a small asymmetry so the sum preserves the original output while the copies can learn different patterns. Unchanged AdamW parameters retain their history. The graft maps first and second moments for expanded tensors by head and retains step counters. Their mapped values are a warm start. The new bond blocks acquire moments on the first update. [Migration](../../src/latentmol/training/attention_expansion.py)

Job **30359446** last logged update 631195 and exited following the stop request. The saved checkpoint at **631116** is SHA-256 verified, with the cursor at 13,348,320 rows in epoch 48. The new leg replays the 79 unsaved updates. DTU accepted and released job **30363630** on `gpuv100`, with one 32 GB V100, eight CPUs and a 24-hour allocation. The effective batch remains 240 real and 240 prior, microbatch 48. The training entry restores the inherited motif winner, generated-parent and false-positive-bond losses, shuffle cursor, RNG, quality panels and original 25,000-update/81,000-second segment limits. This single fork tests both attention changes together. Its fixed generated-parent cohort is established again at the start of this leg. [Resume entry](attention_next_leg.py) · [Verified cutoff](../../outputs/hpc-submissions/vae-e48-attention-20261125a/cutover.json) · [Scheduler submission](../../outputs/hpc-submissions/vae-e48-attention-20261125a/submission.json)

On a CPU scratch copy of the actual cutover checkpoint, 8/8 reconstructed hard graphs and 8/8 hard prior decodes matched immediately after expansion. The maximum absolute bond-logit difference was 0.000178. Two reduced-batch updates exercised the new bond layers and the inherited generated-parent loss. These checks cover the migration and local training path. The production panel measures reconstruction, random validity, aromatic/amide yield and generated-parent recovery as training progresses. [Checkpoint replay](../../outputs/runs/vae-attention-cutover-20261125a/cpu-631116-graft.json)

Two additional scratch optimizer steps on an RTX 4060 used the saved AdamW state, graph-matched reconstruction and KL. Both new residual output projections received gradients on the first update. On the second, both internal Q/K/V projections received nonzero gradients. The latent and all seven atom-attention layers received gradients on both updates. This small check used a different GPU and batch size from the queued production allocation. [CUDA smoke results](../../outputs/runs/vae-attention-cutover-20261125a/local-full-model-cuda.json)

### Bond-attention backward failure on the V100

Job 30363630 started at 23:34 CET on November 25 and exited after update
631117. Its initial quality panel and first update were written. The latest
durable training checkpoint is still 631116, before the attention graft.

The generated-parent round-trip branch uses a functional decoder call with
detached parameters. CUDA activation checkpointing runs the bond-attention
endpoint again during backward. The old endpoint read the live module after
the functional call restored its trainable parameters. Linear backward then
saved an extra input tensor, shifting the saved-tensor sequence and triggering
PyTorch's metadata check. The two-molecule reconstruction and KL smoke did
not invoke the round-trip loss or reach the 256-row recomputation threshold.

The fix passes the exact LayerNorm and linear parameter tensors as arguments
to the recomputed function. The focused CPU tests passed (13 passed, one CUDA
case skipped in that environment). A separate local RTX 4060 replay used the
verified 631116 checkpoint and restored optimizer, two real molecules, and
eight selected generated parents on each of two scratch steps. The
round-trip loss produced finite encoder gradients and no gradient on the
checked decoder attention projection. A full V100 leg still requires a
replacement allocation and a new release containing this fix.

On November 26 DTU accepted replacement V100 job **30387778** from the same
verified checkpoint at update 631116. The sealed source archive changes only
the bond-attention recomputation implementation. The recipe, dataset,
architecture, GPU resources and checks use the original submission settings.
The replacement has its own output directory and was released to scheduling
at 08:37 CET.
[Submission receipt](../../outputs/hpc-submissions/vae-e48-attention-20261126a/submission.json)
