# Removing conflicting teacher bond types on verified six-membered C/N rings

The optional helper preserves the full atom and charge losses, every teacher edge-existence term, and conditional bond-type supervision away from a selected mask. It qualifies both teacher and student graphs first. It protects every already-perceived aromatic six-membered C/N ring on the student, including Kekulé representations. The existing completion detector can also contribute one near miss per graph when changing its ring labels produces a strict aromatic molecule. Each undirected pair is counted once. The ablation applies only where the teacher actually supplies two occupied slots and the SINGLE edge label.

The implementation evaluates the native teacher loss and subtracts the selected bonds’ conditional SINGLE cross-entropy. It retains the native factor of two on bonds, original eligible-row scaling, and original sum of all eligible pair weights (1 for a present bond and 0.1 for NONE). Bond existence therefore retains exactly the same supervised target and coefficient even on a masked pair.

We replayed the original update-580498 panel of 128 Gaussian vectors and archived epoch-46 teacher outputs, in eight groups of 16. Of the 95 teacher-valid and 85 jointly valid rows, 29 had an eligible ring. The mask covered 155 edges on 28 possible aromatic completions and five edges on one existing aromatic ring. Nine teacher NONE and five teacher AROMATIC labels within the verified ring sets remained supervised.

| Measurement | Result |
|---|---:|
| Mean derivative for a uniform increase in the student's AROMATIC logit, native | +0.004190 |
| Same derivative after selective ablation | +0.001811 |
| Reduction in this direction | 56.8% |
| Largest loss error vs independent existence/type decomposition | 2.78 × 10⁻¹⁷ |
| Largest gradient error in float64 replay | 6.51 × 10⁻¹⁹ |
| CPU mask selection time per 16 priors, median / maximum | 46 / 65 ms |

Atom, charge, all off-mask bond and all NONE-logit gradients agreed exactly. A focused test checks both explicit and alternating encodings of an already-aromatic ring with mixed teacher labels. The measured positive derivative means the average native pressure against raising aromatic logits weakened but remained. This frozen-output direction does not establish a training or generation improvement. The shared entry point is `conditional_ring_teacher(decoded, hard, valid, spec)`.

```sh
python -m experiments.vae.training_data_eda.ring_teacher_ablation \
  --checkpoint outputs/runs/vae-aromatic-mechanism-20261121a/checkpoint.pt \
  --teacher-graphs outputs/runs/vae-generation-motifs-20261120a/checkpoint-000325498.jsonl.gz \
  --teacher-summary outputs/runs/vae-generation-motifs-20261120a/summary.json \
  --output outputs/runs/vae-e48-ring-teacher-20261122a/summary.json
```

[Shared helper](../../../src/latentmol/guidance/ring_teacher.py) and [independent replay](ring_teacher_ablation.py) · [Per-group measurements](../../../outputs/runs/vae-e48-ring-teacher-20261122a/summary.json) · [Original teacher decomposition](epoch48-aromatic-mechanism.md)
