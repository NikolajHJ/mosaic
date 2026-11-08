# How many children can share one A100 80 GB?

The [pilot config](restart-batch-throughput.json) uses the frozen epoch-45
decoder, BRD4 Protenix setup, LR 60, a 0.25 step cap, and chemistry and
valence excess weights of 1. It restores two archived parents and applies
four screened latent kicks to each. Every trial replays the same eight
starts, active masks, and update noise seeds for 40 updates.

One resident JAX worker scores batches of 1, 2, 4, or 8 children. Each child
receives its own score and gradient. The CPU comparison varies one, two,
or four branch workers and tests overlapping CPU repair search with native
scoring. The scalar baseline uses four Torch threads. The other modes use
one Torch thread so the CPU workers can share the eight allocated slots.
The report records the CPU model and affinity for interpreting the result.

Two repetitions of nine modes give 5,760 timed updates. The warmups add
30 updates. The first and second native calls for each batch width are
reported separately from timed throughput. Each trial records wall time,
phase timings, sampled GPU memory and utilization, score and latent-gradient
agreement at the same initial states, endpoint differences, and validity.
Timed trials include CPU chemistry and hard graph checks. A separate
probe fresh-scores up to 12 distinct valid molecules after the JAX worker
closes. The throughput trials use compact summary writes.

The allocated entry point is `python -m latentmol.design.protenix_restart_benchmark`.
The [submission directory](../../../outputs/protenix-restart-batch-throughput-20261108a/)
holds the source archive, GPU plan, checks, and submission receipt. The
runner writes `timings.jsonl` as trials finish and `result.json` after
the fresh probe.

DTU accepted job **29931253** on `gpua100` at 23:04 CET on 8 November 2026.
It requests one A100 80 GB, eight CPU slots, a four-hour wall limit, and a
45-minute startup allowance. It was pending at the first check. Twenty
focused tests passed. A CPU smoke with the verified decoder and mock
scorer matched serial and overlapping two-child trajectories. Native
batch compilation, numerical agreement, and throughput are pending.
