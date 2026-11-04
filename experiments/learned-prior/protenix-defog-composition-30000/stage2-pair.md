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
