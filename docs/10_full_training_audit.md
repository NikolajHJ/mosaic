# Full-training audit and checkpoints for the prior study

Recorded 28 September 2026. This is a documentation-only Stage 0 audit of the
completed source015 campaign. The coordinator fixed the next-stage selections
to each model's **best-validation-validity** checkpoint: AE epoch **108** and
VAE epoch **94**. Test results are reported below, not used to select a new
checkpoint or change the training protocol.

## Completion and provenance

Both scheduler outputs report successful completion. AE job **29503898**
finished at **2026-09-28 01:41:13 UTC** after 120 epochs, with decision
`budget_reached_without_recent_gain`. VAE job **29503899** finished at
**00:30:56 UTC** after 100 epochs, with decision
`stop_after_validation_plateau`. These are different final compute budgets.

The common data split is **215,000 training / 7,347 validation / 7,347 test**,
seed 17, with the earlier pilot split memberships preserved. The source015
archive SHA256 is
`375c7f6a5a67f229d5c95d307413047b26e244de82a1c71ff09a0471bc192f12`.
Source, dataset and split identities and the training protocol are recorded in
[the ledger](05_experiment_ledger.md) and
[the full-training protocol](07_full_training_protocol.md).

The coordinator verified all checkpoint hashes on HPC and confirmed that common
epoch-60 and epoch-80 checkpoints exist. Each workflow sealed its
`selection_before_test.json` before opening test evaluation. Four checkpoint
roles were retained per model—common epoch 60, best validation reconstruction,
best validation validity and last—and all **eight final validation/test
evaluations per model completed**. Common checkpoints support equal-budget
comparisons; the selected later checkpoints below do not have equal budgets.

## Fixed checkpoints for Stage 1

Best-validity selection uses the fixed 1,024-molecule validation cohort:
maximize validity, then exact reconstruction, then prefer lower full-validation
reconstruction loss; complete ties retain the earlier epoch. It does not mean
that every possible chemical or latent-quality metric is maximal.

| Model | Selected epoch | HPC checkpoint |
|---|---:|---|
| AE | 108 | `/work3/s234818/latentmol/runs/20260928T002738Z_ae-train_2f61b913/best_validity.pt` |
| VAE | 94 | `/work3/s234818/latentmol/runs/20260927T231648Z_vae-train_a0e7eba0/best_validity.pt` |

AE checkpoint SHA256:
`1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95`.

VAE checkpoint SHA256:
`80b22b9c5a5e9010457b49246f830f0425c5578458eab28ddf2a10f5f632cee9`.

Freeze both selected encoders and decoders for the prior study. Preserve these
identities in reference-bank, calibration and optimization artifacts.

## Selected-checkpoint results

Reconstruction uses deterministic latent means and raw hard decoding without
repair. Each row includes all **7,347** molecules in that split. Validity and
exactness are topology-only, within the admitted stereo-free graph schema.

| Model / split | Reconstruction loss | Valid / 7,347 | Exact / 7,347 |
|---|---:|---:|---:|
| AE epoch 108 / validation | 0.02128245 | 7,185 | 7,045 |
| AE epoch 108 / test | 0.02110486 | 7,161 | 7,032 |
| VAE epoch 94 / validation | 0.01078273 | 6,759 | 6,514 |
| VAE epoch 94 / test | 0.01103442 | 6,755 | 6,511 |

The selected VAE's standard-normal prior produced **13 valid molecules from
1,024 draws (1.2695%)**, seed 17; all 13 were unique and novel relative to the
prepared training split. Validation and test evaluation records repeat the same
seeded prior diagnostic, not two independent sets of draws. The AE has no
trained standard-normal prior, so this diagnostic is inapplicable to it.

AE validation loss worsened from **0.00924933** at its best-loss epoch **61**
to **0.02128245** at selected epoch **108**, while full-validation validity
increased from **6,895** to **7,185** and exact reconstruction from **6,695**
to **7,045**. Loss and hard-chemistry rankings therefore differ. Both models'
last trend records flag `overfitting_warning: true`; neither the plateau rule
nor the budget cap proves global convergence.

These results support testing local search around encoded molecules. They do
not establish which representation best preserves useful changes during
optimization, and the VAE prior remains weak. The next study must compare
valid-and-changed outputs, diversity, size and cost under matched perturbations
and frozen checkpoint choices, as specified in
[the next-experiment plan](09_next_experiment_plan.md). The original test set
must not be adaptively reused to tune those methods.

## Evidence

The [downloaded completion record](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/full-training-completion-20260928.json>) contains
workflow status, checkpoint identities, all final summaries and scheduler tails.
The selected evaluation run IDs are:

- AE validation: `20260928T014016Z_evaluation_3f3e1b54`; test:
  `20260928T014031Z_evaluation_f63b72e0`.
- VAE validation: `20260928T002955Z_evaluation_a0f906f2`; test:
  `20260928T003011Z_evaluation_84e78e95`.

This audit adds no training, tests, code changes or job submissions. New Stage 1
execution will be recorded separately with its actual source and job identities.
