# Stage 2 with a one-query DeFoG prior

Does a detached DeFoG target improve hard chemistry under the same Boltz-2
objective and Stage 2 projection?

`configs/stage2-pair.yaml` uses the imported 1,200-update run's connected
projection, valence caps, fluidity, seed and step sizes for 500 updates per
arm. Both arms use the same prepared 25-slot DTU BRD4 pose. Its context digest
differs from the imported study's. The treatment adds weight 0.5 times the sum
of mean per-row squared distances for atom and bond probabilities. Each state
gets one frozen DeFoG clean prediction at time 0.799995, including a terminal
evaluation. Only the current probabilities receive gradients from the prior.
The Boltz model, its context and the DeFoG checkpoint are frozen.

Run `bash hpc/stage2-pair.sh SOURCE_ROOT OUTPUT_ROOT` from an allocated DTU job
with the A100 profile and a fresh output directory. The coordinator uses CPU
JAX and the Boltz worker owns the GPU. The run checks finite graphs,
derivatives, checkpoint identities and the teacher's forward count. The
finite-difference gate is disabled for this exploratory pair.

`pair/comparison.json` compares sanitized connected readouts, sulfur counts,
hard graph edits, native soft and valid hard affinity, and the native score
change on discretization. Both arms record all readouts.

## Imported baseline

In the [500-update matched study](../../../stage2/evidence/overnight-20261103/matched/comparison.json),
the unprojected control had 45 of 501 valid connected readouts. Valence-only
had 3, connected projection had 54, and connected projection with sparsity had
60. All four final readouts failed sanitization. The unprojected control had
the lowest final relaxed loss of these arms, so the new comparison reports
the best affinity among valid connected hard readouts separately.

The [1,200-update connected arm](../../../stage2/evidence/round3-20261104/long-1200/run_summary.json)
reported best loss −1.562 and mean hard-minus-soft loss +0.232 on its pose.

## Submission

DTU job 29891369, gpua100, A100 40 GB or 80 GB, two-hour limit, submitted
4 November 2026 at 15:25 CET. Source revision 880d5d9. Output root
`/work3/s234818/mosaic/runs/stage2-defog-pair-20261104a/output`.
