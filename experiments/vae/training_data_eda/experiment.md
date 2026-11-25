# What chemistry does the VAE train on?

The [generation and reconstruction audit](generation-motifs.md) compares the training frequencies with the large VAE's decoded outputs.

20 November 2026. The [illustrated report](../../../outputs/runs/vae-training-data-eda-20261120a/analysis/report.html) contains the full size histogram, functional-group frequencies, scaffold structures, chemical-property distributions, PCA and ICA.

## Population and sampling

The [large epoch-47 recipe](../epoch47-pubchem60m-parent-guidance-dropout.json) and [compact training recipe](../compact435k-count-v100.json) use the same real-molecule training view. It holds 60,000,000 distinct, connected, stereo-free graphs selected by heavy-atom count from the 81,409,264-graph PubChem, ChEMBL and MOSES parent. Each selected identity has equal weight here. The [manifest record](../../../outputs/hpc-results/pubchem-large-corpus-20261105a/cache-and-view-20261106T094305211844Z.json) contains both size distributions.

One 1-core DTU CPU job (30131132, 48 seconds) scanned the packed records, verified the train file hashes and counted independent cycles as bonds minus atoms plus one. Positive cycle rank means at least one ring in a connected graph. The histogram matched the manifest at each heavy-atom count. From the same split we sampled 50,000 rows uniformly without replacement, seed 20261120. All rows agreed with RDKit 2024.09.6 on connectivity, atom count, cycle rank and ring presence. [Census](../../../outputs/runs/vae-training-data-eda-20261120a/inputs/census.json) · [Panel verification](../../../outputs/runs/vae-training-data-eda-20261120a/inputs/verification.json)

## Results

- **57,497,175 / 60,000,000 (95.83%)** contain a ring and 80.89% have at least two independent cycles. Median size is 25 heavy atoms (middle half 18–32). Ring prevalence reaches 98.64% at 25 atoms and falls to 93.26% at 50. Among sampled 41–50-atom molecules, the 114 acyclic graphs have a median of 34 rotatable bonds, compared with 9 among the 2,945 cyclic graphs.
- **46,868 / 50,000 (93.74%)** have both a ring and at least one of 20 SMARTS-defined functional-group families (95% Wilson interval 93.52–93.95%). This is 97.76% of the 47,944 sampled cyclic molecules. Requiring a group center outside every ring gives 90.73% of all sampled molecules. Amide-type carbonyls occur with a ring in 46.94% of the panel, amines in 42.99%, carbon-bound halogens in 39.37% and ethers in 38.13%. [Group definitions](../../../outputs/runs/vae-training-data-eda-20261120a/analysis/group_definitions.csv) · [Counts and denominators](../../../outputs/runs/vae-training-data-eda-20261120a/analysis/ring_by_group.csv)
- A six-membered ring occurs in 88.44% of the panel, an aromatic ring in 87.06% and a heterocycle in 74.30%. The cyclic sample has 32,423 distinct Murcko scaffolds, of which 29,675 occur only once in this sample. The benzene scaffold is the most frequent (3,298 molecules, 6.60% of the entire panel). Median molecular weight is 360.46 Da and median calculated logP is 3.24.
- PCA needs **9 of 18 standardized chemical descriptor components** for 90% of the variance and 10 for 95%. PC1 tracks molecular size and PC2 contrasts aromatic atom fraction with polar surface area and oxygen count. For 1,024 mean-centered radius-2 Morgan fingerprint bits on a uniform subsample of 15,000, the same thresholds take **662 and 811 components**. These measure two different representations of the same chemical population. [Full spectra](../../../outputs/runs/vae-training-data-eda-20261120a/analysis/pca_variance.csv)
- For the large VAE's **256 posterior-mean coordinates**, 55 PCs capture 90% of variance and 61 capture 95% in a random 4,096-molecule training panel. The same thresholds held when fitting on 2,048 rows and evaluating on the other half. DTU CPU job 30131210 encoded the cached graphs with the frozen partial epoch-47 checkpoint at update 524,177. PC1 correlates with heavy-atom count (r = 0.93), and PC2 correlates with aromatic atom fraction (r = 0.76). Together the first two explain 7.64%. The two clouds visible along PC1 separate sharply near 26–27 heavy atoms in this panel. The result measures variance in this checkpoint's code distribution. Choosing a bottleneck width for a new VAE would require measuring reconstruction and generation at that width. [Latent projection](../../../outputs/runs/vae-training-data-eda-20261120a/latent/latent_pca.png) · [Encoder and split-half results](../../../outputs/runs/vae-training-data-eda-20261120a/latent/latent_summary.json)
- ICA used the first six descriptor PCs (77.24% of descriptor variance). Both simultaneous FastICA fits hit the 500-iteration limit. Deflation extraction on the same two seeded 12,000-row draws converged both times. The charge, flexibility, hydrogen-bond donor and halogen directions reproduced at absolute r above 0.998. The remaining two matched at about 0.74. [Deflation projections](../../../outputs/runs/vae-training-data-eda-20261120a/analysis/ica_deflation.png) · [Stability across fits](../../../outputs/runs/vae-training-data-eda-20261120a/analysis/ica_sensitivity.json)

## Reproduction

From the repository root, with Python, NumPy, pandas, Matplotlib, SciPy, scikit-learn, threadpoolctl and RDKit available. Use new output directories. No model is used to compute chemistry descriptors or fingerprints.

```sh
# On an allocated CPU with the published view and its parent cache accessible
python experiments/vae/training_data_eda/sample_view.py \
  --view /work3/s234818/latentmol/data/pubchem-hac60m-view-20261106a \
  --dataset-sha256 64247d04ec1f94d00320bb1e5a0d9c5540419eb37830050d3100775bf4385e18 \
  --sample-size 50000 --seed 20261120 --output NEW_RUN/inputs

# On a CPU with the sampled files
python experiments/vae/training_data_eda/profile_panel.py \
  NEW_RUN/inputs/panel.tsv.gz NEW_RUN/analysis --workers 6
python experiments/vae/training_data_eda/ica_sensitivity.py NEW_RUN/analysis

# On an allocated CPU with Torch, the project's model code and the training cache
python experiments/vae/training_data_eda/latent_panel.py \
  --panel NEW_RUN/inputs/panel.tsv.gz \
  --view /work3/s234818/latentmol/data/pubchem-hac60m-view-20261106a \
  --checkpoint /work3/s234818/latentmol/runs/vae-pubchem60m-prior15-v100-20261111b/sparse-02/checkpoint-000524177.pt \
  --output NEW_RUN/latent --count 4096 --seed 2026112003 --threads 2

# With the three result directories under NEW_RUN
python experiments/vae/training_data_eda/report.py NEW_RUN \
  outputs/hpc-results/pubchem-large-corpus-20261105a/cache-and-view-20261106T094305211844Z.json
```
