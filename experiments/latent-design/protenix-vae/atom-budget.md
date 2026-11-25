# Protenix count budget comparison

Does a soft count penalty prevent size drift without degrading the score of the
selected hard molecule? Both arms start at the frozen epoch-45 encoder mean of
the same JQ1-derived 25-heavy-atom molecule. The raw VAE reconstructs the start
exactly. The BRD4 pocket comes from the saved 3MXF/JQ1 fixture. The modified
ligands are hypothetical analogues.

The four optional choices are the original chlorine and three proposed
fluorines. The 16 supported molecules span 24–28 heavy atoms. Protenix-Tiny
scores each native FILE input once using the negative mean log contact
probability over the same 24 retained ligand atoms and the same pocket residues.
The optimization objective is the exact expectation of the 16 native scalar
scores under four independent Bernoulli occupancies. The starting chlorine
occupancy is calibrated to 0.8 and the three padded slots to 0.2 by constant
logit offsets shared by both arms. Their occupancies stay attached to the
latent. The proposed fluorine identities and common core are fixed by the
panel. Their probabilities are distinct from the unmodified decoder output.

Both arms take up to 64 normalized latent updates within radius 4. Only the
budget-on arm adds `0.01 * max(0, abs(expected_atoms - 25) - 2)^2`. The slot
surcharge is zero. The count term applies to the expected count of the supported
topology family, while the raw VAE expected count, hard count and chemistry are
also saved. A 28-atom hard selection is possible even when the soft count stays
under 27, so the result includes the hard count, probability of 28 atoms, and
whether the penalty region was reached.

Fresh geometry scoring uses a separate fixed ETKDG seed for the shared start
and each final selected topology. The raw VAE endpoint receives a fresh score
when its chemical identity matches a supported molecule. The panel contact
scores are structure-model proxies.

```bash
python -m latentmol.protenix_occupancy.budget \
  --config experiments/latent-design/protenix-vae/atom-budget-dtu.json \
  --output <new-output-directory>
```

The runner checks the pinned one-atom follow-up receipt before model work.
Results include `panel.json`, both trajectories, and `result.json`.

The protein pocket and reference scaffold come from [RCSB 3MXF](https://www.rcsb.org/structure/3MXF).

## First submission

DTU job **29893904** was submitted at 22:50 CET on 4 November 2026 to
`gpua10` with one A10, four cores and a one-hour limit. It was pending
scheduling at the first check. The source SHA-256 is
`e3cd4881ff03c4e29fad5e2c785df5f891f3776015601fc7c0bd3af376ef8ca2`.
The output directory is
`/work3/s234818/latentmol/protenix_occupancy/runs/protenix-atom-budget-20261104-a01`.

59 focused CPU tests passed. The verified epoch-45 checkpoint exactly
reconstructed all labels of the shared start, and all 32 declared conformer
preparations succeeded locally. A synthetic reward favoring larger molecules
gave final controlled hard counts of 28 without the budget and 27 with it,
demonstrating latent reachability under this update budget.

## Native result

Job **29893904** finished successfully at 23:03 CET on 4 November 2026.
Both saved trajectories are identical. The controlled expected count stayed
between 24.455 and 25.400, within the 23–27 free band, so the count penalty
contributed zero throughout. The raw hard count was 25 at all 65 observations
in both arms. Each arm had 28 valid raw decodes, and the final raw decode
failed sanitization, leaving no fresh raw score.

Both arms selected the same 24-atom panel molecule. Its fresh Protenix
contact loss was 2.32107, versus 2.36163 for the shared 25-atom start
(lower is better). This run did not exercise the size penalty under native
growth pressure. [Saved result](../../../outputs/hpc-results/protenix-atom-budget-20261104-a01/result.json).
