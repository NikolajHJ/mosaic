# Sparse cooling/noise: two independent seed replicas

Question: was the valid connected endpoint of the sparse seed17 run a one-off?
The user authorized two new seed replicas on 3 November 2026. Each runs all
50,000 updates from fresh sparse random initialization.

| Initialization seed | Noise seed | Frozen model key | GPU | Actual job | Latest state |
| --- | --- | --- | --- | --- | --- |
| 23 | 230023 | 17 | A100 | 29880094 | PEND |
| 41 | 230041 | 17 | L40S | 29880095 | PEND |

Scheduler and receipts verified at 17:04:21 UTC; zero committed updates then.
Existing VAE jobs and the completed seed17 trajectory are unchanged.

The [completed seed17 protocol](../20261102-sparse-annealed-50k-01/README.md)
is the control: LR0.3, N25 occupied neutral slots, original frozen Protenix/JQ1
model/context/key17, sparse connected tree plus2% extra edges, structural repair
ahead of unchanged search fallbacks and categorical repair loss. Composition and
confidence margin remain off. No LR decay, clipping, projection, backtracking or
gradient normalization is introduced. Repair targets stay detached and fixed
within each optimizer step.

- Chemistry coefficient1 through20k, geometric increase to30 by30k, then hold30.
- Temperature linearly0.7 to0.4 across50k.
- Independent row-centered binding-gradient-scaled logit noise0.4 to0 by30k,
  then20k without added noise; each run has its own checkpointed PCG64 stream.
- Retain every loss/state record; matrices every20 and endpoints/latest/best.
  Save128 unfiltered categorical samples every1000 and at the endpoint.

Primary comparison: raw argmax sanitization **and connectivity at50k**, fraction
of valid trajectory states, sampled validity/diversity and saved contact curves.
There is no conditional rescue extension for these replicas: an invalid50k
endpoint remains an informative outcome. Changing temperature changes relaxed
scorer inputs, and changing chemistry weight changes the trade-off. GPU families
differ, so runtime and floating-point differences are recorded. Three seeds give
an early robustness signal, not a precise population success-rate estimate.

The two capsules reuse all156 non-plan files of the completed seed17 source
byte-for-byte. Only the two stochastic seeds, output ownership and resource
metadata change. Two real metadata-only CLI startups, all input hashes, immutable
source hashes, exact accepted submission vectors and held-before-release proofs
passed. Native finite/frozen/startup checks run inside normal production; no
separate qualification campaign was added. All50k updates are requested with
15h15 allocations, based on the original measured11h39 runtime plus25% and
setup/cleanup. New seed/device runtime is not yet measured.

Own remaining-output capacity passed:4,831,838,208required versus13,894,004,736
available bytes. Group cap4GiB+256MiB/15,000files, including2GiB per arm. Entry and
periodic accounting remain active. No cleanup was performed.

[Configuration](configuration.json), [current jobs/source](campaign.json),
[independent receipt/byte audit](verification.json). Generated preparation,
submission and raw evidence are under
`results/hpc/20261103-sparse-annealed-seeds-01`.

Both runs are registered in the canonical report selection
`experiments/reporting/jq1-live-progress/configs/current-completions.json`.
After completion, use `scripts/reporting/report.py` and the existing cache;
queued runs do not trigger a rebuild or replace the current report. Include the
new IDs `annealed_sparse_50k_s23` and `annealed_sparse_50k_s41` in the next focused
collection once published. Presentation and proxy scores do not certify binding.

Read-only follow-up: send the existing generated `observe.py` through the DTU
helper and save a fresh local label. Inspect actual `run/jobs.json`, each
`production/ARM/submission.json` and `release.json`, logs and committed prefixes.
Do not rerun `prepare.py`, `stage-and-submit.py` or the accepted submission calls.
