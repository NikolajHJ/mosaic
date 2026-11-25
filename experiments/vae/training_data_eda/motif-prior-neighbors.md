# How much of a real decoded motif survives near its latent seed?

We started with 1,024 independent standard-normal vectors decoded at update 580498. Strict argmax evaluation gave 885 valid graphs, including 28 aromatic and 70 amide molecules. We randomly selected 24 positive seeds per motif. Four appeared in both sets. All 44 distinct seeds reproduced their original atom slots, charges and full edge labels when replayed on CPU.

For each seed we drew four independent Gaussian perturbation directions and kept each direction fixed across the three correlations. The neighbor vector was `rho * z + sqrt(1 - rho²) * epsilon`. Each group and correlation thus has 96 seed–direction pairs. The motif-positive seed selection deliberately enriches these neighborhoods. Their yields describe local retention, and the 96 outcomes are clustered within 24 seeds.

| Seed motif | Correlation | Strictly valid | Same original motif site | Exact slot graph | Within 2 label edits | Median label edits |
|---|---:|---:|---:|---:|---:|---:|
| Aromatic | .995 | 81/96 | 63/96 | 5/96 | 40/96 | 3 |
| Aromatic | .98 | 79/96 | 46/96 | 2/96 | 11/96 | 7 |
| Aromatic | .95 | 79/96 | 25/96 | 0/96 | 3/96 | 12 |
| Amide | .995 | 83/96 | 64/96 | 7/96 | 41/96 | 3 |
| Amide | .98 | 82/96 | 45/96 | 2/96 | 14/96 | 6 |
| Amide | .95 | 86/96 | 35/96 | 0/96 | 4/96 | 10 |

Same-site retention requires a strictly valid graph with all atoms of an original site in the same slots with the same element and formal charge. An aromatic ring must have the same edges and RDKit-perceived aromaticity, including Kekulé encodings. An amide must reproduce the same typed carbonyl carbon, oxygen and nitrogen in their respective roles. Slot edit distance adds atom-label changes (including occupancy), charge changes on shared occupied slots, and changed undirected bond labels. It is a distance to the actual latent seed graph in the model's slot ordering.

At .995, 58 aromatic neighbors and 57 amide neighbors changed their hard graph while retaining the original site. The aromatic seed group produced 70 distinct valid non-parent canonical molecules across its neighbors, and the amide group produced 74. Of the 33 neighbors that lost the aromatic site or became invalid, only three lay within two label edits of their seed. For amides this was seven of 32. A verified whole-molecule seed is a nearby target for a subset, but training all neighbors to reproduce that entire graph would remove many valid changes that already retain the motif. A site-specific objective, or a whole-graph target restricted by a measured edit budget, would make a clearer pilot.

The run used CPU only and evaluated 528 unique neighbors in 138 seconds. All reported valid SMILES passed a readback check. The saved per-draw output retains failed attempts.

### Local training signal

The optional neighbor objective takes at most two naturally aromatic and two naturally amide-positive centers from the current prior batch. When the same center supplies both groups it gets one additional decode. At correlation .995 the neighbor predicts the center's mapped motif atom and charge labels and original conditional bond types, including Kekulé forms. A weaker term supervises the bonds directly attached to those atoms. An occupancy-gated Bernoulli KL term compares all incident soft edge probabilities to their detached center values, balancing present and absent pairs. Each site's mean core bond CE, shell CE, atom CE, charge CE and topology KL receives weight 1, .25, .25, .10 and .5 respectively, and the sum is divided by the full prior batch size. The neighbor pass uses evaluation mode with gradients and a private Gaussian generator.

An isolated CPU backward pass at update 588498 selected two rings and two amides from 48 Gaussian priors. Its auxiliary loss was 0.0342 and the final bond-head weight gradient norm was 0.131. Training mode and the global RNG state were preserved. Two targeted tests cover subgraph locality, normalization, attachment and empty batches. The smoke test only measures the gradient path. [Loss implementation](../../../src/latentmol/guidance/motif_neighbors.py) · [Smoke results](../../../outputs/runs/vae-e48-natural-neighbor-smoke-20261122a/summary.json)

### Check at the production handoff

The optional `functional_groups.activation_gate` runs the original policy and the proposed neighbor policy sequentially in one allocation for 512 updates each from the exact same authenticated checkpoint. Each receives the same real rows, optimizer state, dropout RNG and fresh-prior RNG. The two endpoints are compared on 4,096 common Gaussian vectors and categorical sampling draws, retaining every invalid draw in the denominator. Reconstruction uses the existing 256 held-out rows. Each arm keeps individual readouts, including invalidity reason, atom count and presence of a ring of size eight or greater. The decision file records gains, losses and denominators.

Adopting the candidate requires at least four additional valid aromatic outputs, with the net gain also strictly exceeding one paired standard error. Validity may drop by at most one percentage point, amide yield by half a point and exact reconstruction by two of 256. A tie, inconclusive result, or insufficient allocation time for the comparison selects the control. Nonfinite updates and incompatible artifacts remain execution failures.

The selected checkpoint supplies all weights, optimizer moments, RNG streams and the next data row. Only its trial history joins the main training history. The discarded updates are reported separately. Each production checkpoint carries the decision digest and selected policy so later allocations continue that arm without repeating the pilot. The extra training cost of a completed gate is 512 updates overall. [Gated recipe](../epoch48-seven-round-motif-followon.json) · [Decision and continuation logic](../../../src/latentmol/training/motif_activation.py)

```sh
python experiments/vae/training_data_eda/motif_prior_neighbors.py \
  --checkpoint outputs/runs/vae-aromatic-mechanism-20261121a/checkpoint.pt \
  --start-rows outputs/runs/vae-aromatic-mechanism-20261121a/teacher-train-pilot-a/start.jsonl.gz \
  --out outputs/runs/vae-e48-prior-motif-neighbors-20261122a
```

[Counts and seed identities](../../../outputs/runs/vae-e48-prior-motif-neighbors-20261122a/summary.json) · [Per-draw readouts](../../../outputs/runs/vae-e48-prior-motif-neighbors-20261122a/neighbors.jsonl.gz)
