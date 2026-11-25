# Teaching ring size with sparse edge edits

**Question.** Could an auxiliary objective change the sizes of rings in Gaussian decodes while preserving atom identities, connectivity and existing functional groups? This CPU probe reuses the exact 256 saved Gaussian graphs from seven-round checkpoint 580498. RDKit 2024.09.6 reproduced every saved strict readout. It accepted 223 graphs, of which 209 contained a triangle or a ring of at least eight atoms. The 33 invalid graphs were retained in the attempt denominator and received no topology target. The VAE weights were frozen.

**Target construction.** We compared three searches on each eligible raw graph. The first can cut one single bond on a disfavored ring. The second can add a single-bond chord across a long ring. The third can rewire one endpoint of an existing single bond. It removes one edge and creates one, preserving total edge count. We tested up to 32 candidates per arm, with a deterministic preference for six-membered closures. A target must reduce the count of triangles or rings of size eight or greater, preserve the atom sets of every perceived five- or six-membered ring, contain at least one such ring, and avoid increasing the count in any of the 3, 4, 7, ≥8 or ≥12 bins. It must pass the shared strict graph readout, including connectedness and absence of radicals. Existing aromatic ring atom sets and amide matches must survive. Ring sizes use RDKit's symmetric smallest-ring set. A change of ring basis can alter this readout even when some longer cycles remain in the graph.

| Graphs with each feature, out of the same 223 valid inputs | Raw | One cut | One closure | One rewire | Expected at matched atom counts in training |
|---|---:|---:|---:|---:|---:|
| Three-membered ring | 160 | 133 | 160 | 130 | 5.30% |
| Six-membered ring | 137 | 137 | 155 | 220 | 93.38% |
| Ring of size ≥8 | 165 | 78 | 108 | 51 | 0.51% |
| Ring of size ≥12 | 67 | 28 | 67 | 10 | 0.079% |
| Aromatic ring | 7 | 7 | 7 | 7 | |
| Amide | 26 | 26 | 26 | 26 | |

Cutting or closing one bond supplied an accepted target in 184/209 eligible graphs. The rewire search found a target in all 209, including the 25 where neither one-edit arm succeeded. A rewire added at least one perceived five- or six-membered ring in all 209, and removed every triangle and ≥8 ring in 63. These are smallest **observed** edit counts within the tested proposal families. Finding the lowest-cost edit across every possible molecular graph was outside this comparison. Searching rewires took about 2 ms per eligible graph on the local CPU. This timing excludes model forward/backward passes, data transfers and GPU synchronization.

**Interpretation.** One rewire often corrected the most conspicuous large ring while preserving atom count, bond count and existing motifs. After rewiring, triangle prevalence was still 58.3%, and six-membered rings occurred in 98.7%. Authentic cyclopropanes and macrocycles occur in the training set. A candidate loss would therefore activate only on a batch distributional discrepancy among Gaussian priors, supervise just the changed pair logits, and retain the reconstruction loss on real molecules. Strict validity, reconstruction, size-conditioned ring rates, scaffold diversity and fresh aromatic/amide yields would all need a matched training comparison. These edits establish target feasibility on one panel. No production training adopted them.

**Gradient replay.** We reproduced the first 32 eligible raw decodes exactly, then applied cross-entropy to the two edited symmetric bond logits (NONE for the cut and SINGLE for the closure). Both the first acceptable rewire and the lowest cross-entropy acceptable rewire among the same first 32 proposals gave finite, nonzero gradients to the bond head and latent in 32/32 cases. Probability ranking reduced median CE from 12.52 to 5.50. The median probability assigned to the desired cut increased from 0.000002 to 0.082; the closure increased from 0.000004 to 0.00062. The latter is still far from the winning bond category. A weak auxiliary weight and a held-out generation check would be important before adopting this in training.

### Reproduce

Set `PYTHONPATH=src` and use a new output directory. Omitting the checkpoint runs the topology searches without model evaluations. Supplying it replays the first 32 eligible inputs and measures gradients of sparse two-edge cross-entropy through the frozen decoder on CPU.

```sh
python experiments/vae/training_data_eda/ring_topology_probe.py \
  --graphs outputs/runs/vae-aromatic-mechanism-20261121a/decoder/rows.jsonl.gz \
  --size-reference outputs/runs/vae-aromatic-mechanism-20261121a/overview/size-matched-ring-sizes.json \
  --checkpoint outputs/runs/vae-aromatic-mechanism-20261121a/checkpoint.pt \
  --out outputs/runs/vae-ring-topology-probe-20261122c
```

[Results](../../../outputs/runs/vae-ring-topology-probe-20261122c/summary.json) · [Raw-input-aligned edit targets](../../../outputs/runs/vae-ring-topology-probe-20261122c/edits.jsonl.gz) · [Original aromatic diagnosis](epoch48-aromatic-mechanism.md)

### Training reference and paired ring-frequency gradient

On 22 November 2026 we tested a batch-level objective on the actual seven-round first-leg checkpoint, update 597604. The two indicators count valid molecules containing a triangle or a ring of eight or more atoms. Each reference rate averages the empirical probabilities from real training molecules at the exact heavy-atom counts of the batch's supported valid draws, including acyclic molecules. The uniform 50,000-row training panel had 2,919 triangle hits and 591 large-ring hits, with observations at sizes 4 through 50. Unobserved sizes have zero reference denominator. All size-specific counts agreed with the separate archived molecular profiles. The packaged reference reproduces the earlier 223-molecule comparison above, yielding 5.302234% triangles and 0.509665% large rings. [Packaged counts and provenance](../../../src/latentmol/guidance/ring_size_reference.py) · [Reference builder](make_ring_size_reference.py)

On the local RTX 4060 we sampled seven independent 240-draw pools. The 240-draw corpus-frequency control and the ring branch sampled identical graphs and dropout masks. Independent chemistry readouts matched the training implementation in every pool. Of 1,680 draws, 1,063 were strictly valid and 1,059 contained any ring. Their triangle rate was 790/1,063 (74.3%), versus a size-matched training estimate of 5.25%. The large-ring rate was 760/1,063 (71.5%), versus 0.51%. We used ring weight 0.10 inside the existing outer frequency weight of 0.30. Across the pools, 1,047 draws received a nonzero additional score coefficient.

In one full decoder backward replay, the control frequency gradient norm was 0.1975 and the combined frequency gradient norm was 0.2238. Their difference, the weighted ring gradient, had norm 0.0733 and cosine +0.198 with the control. All weights remained frozen. These gradients isolate the pooled frequency loss from reconstruction and the other training terms. Six alternating-order paired prepass timings after warmup gave a median additional 0.042 seconds per scheduled call; the mean was 0.078 seconds, or 0.020 seconds per optimizer update when called every fourth update. This used 16-row training-mode decoder chunks and does not establish V100 allocation throughput or a learned improvement in the generated molecules. [Paired gradient and timing evidence](../../../outputs/runs/vae-e48-ring-size-loss-20261122a/gradient-replay.json) · [Replay code](ring_size_score_replay.py)

```sh
python experiments/vae/training_data_eda/make_ring_size_reference.py \
  --panel outputs/runs/vae-training-data-eda-20261120a/inputs/panel.tsv.gz \
  --output src/latentmol/guidance/ring_size_reference.py
python -m experiments.vae.training_data_eda.ring_size_score_replay \
  --checkpoint "$LEG1_CHECKPOINT" --receipt "$LEG1_RECEIPT" \
  --base-config experiments/vae/epoch48-seven-round-corpus-motifs.json \
  --panels 7 --output "$NEW_RESULT_JSON"
```

### Checkpoint training smoke at update 597604

We also applied the ring-size score to the previously saved, unedited categorical endpoint panel of the local corpus-frequency control. Its first 960 of 1,024 draws formed four full 240-draw batches. Among 637 supported valid attempts, 487 had a triangle, 466 had an eight-membered or larger ring, and 632 contained any ring. Within each batch, the two scores had informative coefficients on 153–161 valid cyclic rows. Each exact heavy-atom count contributed a zero-sum contrast; acyclic and invalid rows received zero direct ring coefficients. The sums of absolute weighted ring coefficients were 0.0189–0.0211 per batch before the outer motif factor of 0.30. These are score coefficients, not model gradient norms. [Per-batch measures](../../../outputs/runs/vae-ring-size-score-20261122a/summary.json)

The CUDA smoke compares two optimizer updates per arm at production shape. Its input is the authentic seven-round checkpoint and optimizer state at update 597604. The earlier corpus-frequency arm supplies the matched control, including saved unedited outputs and training RNG digests. The ring arm uses the same 240 real rows plus 240 Gaussian draws per update, 48-row decoder microbatches, 240 motif draws on the scheduled update, and 48 neighbor candidates. [Pilot runner](corpus_motif_update_pilot.py)

The unscheduled update 597605 reproduced every logged control metric except wall time. At the scheduled update 597606, 147 of 240 molecules were valid. Triangles occurred in 105/147 (71.43%) against a size-matched reference of 5.24%, and rings of eight or more in 95/147 (64.63%) against 0.51%. All 147 valid molecules were cyclic. The score coefficients supplied a ring gradient on 134 rows. The motif shortfall was 0.23213444 and the ring term added 0.1 × 0.21231581 once, yielding 0.25336602 before neighbor guidance and the outer 0.30 weight. The full pre-clipping gradient norm went from 1.31221 to 1.32282. Ending RNG digests matched and the reference teacher and starting model retained their parameter hashes.

After this single exposure the 1,024 paired fresh categorical draws showed identical indicator counts in both arms: 677 valid, 672 containing any ring, 517 with a triangle, 494 with a ring of eight or more, 15 aromatic and 36 with an amide. Both reconstructed 118 of 128 held-out molecules exactly. The smoke verifies training mechanics. A longer run is needed to measure whether the frequency of these rings changes. [Paired endpoint evidence](../../../outputs/runs/vae-ring-size-score-20261122a/update-pilot/paired-counts.json) · [Training telemetry](../../../outputs/runs/vae-ring-size-score-20261122a/update-pilot/ring-updates.jsonl)

```sh
python -m experiments.vae.training_data_eda.corpus_motif_update_pilot \
  --checkpoint "$LEG1_CHECKPOINT" --receipt "$LEG1_RECEIPT" --teacher "$TEACHER" \
  --config experiments/vae/epoch48-seven-round-ring-sizes.json \
  --control-config experiments/vae/epoch48-seven-round-corpus-motifs.json \
  --real-panel outputs/runs/vae-training-data-eda-20261120a/inputs/panel.tsv.gz \
  --heldout outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json \
  --saved-corpus-control outputs/runs/vae-e48-corpus-floors-20261122a/update-pilot/summary.json \
  --output "$NEW_DIRECTORY" --updates 2
```
