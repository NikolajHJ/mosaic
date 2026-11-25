# Architecture study: interim HPC results, 29 September 2026

This document is the **29 September historical snapshot**. Source042 wide
AE/VAE, source043 wide fixed-`z`, and source047 stabilized AE/VAE results
completed later; see the [1 October ledger update](05_experiment_ledger.md)
for their audited outcomes and provenance.

As of about 16:30 DTU local time, four full-data AE/VAE workflows and their
matched fixed-`z` diagnostics had completed. The study uses the same
215,000/7,347/7,347 train/validation/test graph cache. Checkpoint selection
used validation before held-out test evaluation. The complete job, source,
config, checkpoint and artifact identities are in the
[`experiment ledger`](05_experiment_ledger.md).
The reported end epochs are validation-governed stopping decisions under
this schedule, not proof of convergence.

At the **best-validation-reconstruction checkpoint**, each test result below
is a decode of a supplied held-out molecule, with 7,347 attempts. `Valid`
means the raw decoded graph passed the recorded chemistry checks; `exact`
means it reconstructed the target molecule.

| Model | Training completed | Test valid | Test exact | Reconstruction loss |
|---|---:|---:|---:|---:|
| Residual AE | 80 epochs | 7,009 / 7,347 | 6,860 / 7,347 | 0.010285 |
| Pair-refine AE | 80 epochs | 7,136 / 7,347 | 7,064 / 7,347 | 0.004419 |
| Residual VAE | 100 epochs | 6,788 / 7,347 | 6,587 / 7,347 | 0.008156 |
| Deep-trunk AE | 80 epochs | 6,998 / 7,347 | 6,879 / 7,347 | 0.005847 |

A separate **best-validation-validity** checkpoint gives different test
counts: residual AE 7,070 valid / 6,912 exact; pair-refine AE 7,219 / 7,143;
residual VAE 6,833 / 6,656; deep-trunk AE 7,150 / 7,072. These are not
the checkpoints used for the fixed-`z` comparison below.

The fixed-`z` diagnostic encoded 128 validation molecules to their latent
means and made 16 temperature-1 categorical hard draws per molecule. Every
attempt is counted, without repair or resampling for validity; 128 argmax
decodes form a separate comparator. These are **conditional reconstructions
around known molecules**, not samples from an unrestricted prior.

| Model | Categorical valid | Categorical exact | Unique valid | Argmax valid / exact |
|---|---:|---:|---:|---:|
| Residual AE | 1,960 / 2,048 | 1,892 / 2,048 | 139 | 123 / 120 of 128 |
| Pair-refine AE | 1,988 / 2,048 | 1,973 / 2,048 | 133 | 124 / 124 of 128 |
| Residual VAE | 1,788 / 2,048 | 1,682 / 2,048 | 161 | 113 / 107 of 128 |
| Deep-trunk AE | 1,896 / 2,048 | 1,827 / 2,048 | 146 | 119 / 116 of 128 |

Among these four completed arms, pair-refine AE has the highest held-out
and fixed-`z` valid/exact reconstruction counts. The VAE's encoded-mean reconstructions
are substantially better than its separate standard-normal prior sample:
only **18 of 1,024** prior draws were valid at its best-reconstruction
checkpoint. Drawing `z` from the standard-normal prior therefore does not
currently work well. The fixed-`z` draws do not measure whether paths
between encoded molecules stay chemically or semantically coherent; that
remains untested. The runs also use one data split and seed, have different
parameter counts and ended at different validation-governed epoch counts.
Several planned arms are still incomplete, so these numbers do not settle
the architecture comparison.

At the snapshot, source037 pair-refine VAE was RUN; source040 global-bond
AE, deep-trunk VAE and global-bond VAE were PEND. Source042 wide GPU smokes,
full jobs and dependent source043 diagnostics were PEND. The corresponding
fixed-`z` diagnostics cannot be interpreted before their parent full jobs
finish. No result here measures Boltz binding, synthetic accessibility,
off-target selectivity or experimental activity.
