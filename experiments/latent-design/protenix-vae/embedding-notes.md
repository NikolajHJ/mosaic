# Epoch-45 VAE hidden states to Protenix embeddings

Question: can a small mapping from the frozen VAE's final atom/pair hidden
states replace the relaxed-graph input to the frozen Protenix contact scorer,
while the ordinary decoded molecule remains observable?

**Completed3November2026 at13:22CET:** corrected job29877165 is DONE and all
10,000 main updates completed. Main elapsed6767.9s (~1h53m), mean0.676s/update;
whole fit/pilot/main pipeline7491.7s (~2h05m). All finite-path and frozen-model
checks passed. Of10,000 source states,5576 (55.76%) were raw-valid. The final
graph is invalid and has an accepted structural repair target; that repaired
graph was not substituted into the scorer. Final mapped contact0.8853 versus
same-state decoded relaxed contact0.2582 still shows a large representation gap.
These contact surrogates do not establish affinity or binding improvement.
[Terminal results](../../../outputs/hpc-results/protenix-embedding45-20261103b/news-1793710563822115500.json).

Decoded graphs/probabilities were saved every10 updates. Automatic HTML rendering
failed because the source snapshot omitted the static report template; the
scientific results, checkpoints and graph observations are intact. No models
were rerun during collection.

The completed matched10k control has2.02% raw-valid source states versus55.76%
for mapping. Mapping's best valid *observed relaxed-graph* contact is0.390198
at step180; the direct run's best valid saved score is0.345866 at27. Those
scores are not fresh discrete hard-graph predictions; see the
[final comparison](embedding-comparison.md#completed10k-trajectory-comparison)
for sampling-cadence limits and endpoint results.

This is a separate exploratory experiment using completed epoch45 checkpoint
`/work3/s234818/latentmol/runs/source156-vae-epoch45-structural-attempt002/epoch0045/epoch_0045.pt`,
SHA256 `5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be`.
The VAE, native model and prepared context stay frozen. The mapping alone is
trained first, then frozen during latent optimization. No existing continuation
or saved trajectory is changed.

The native assets are the existing [BRD4/JQ1 scene](native-dtu.json). The starting
latent is the same epoch45 standard-normal seed17 used by the completed relaxed
Protenix study, with its 25 highest initial occupancy slots fixed. This differs
from the older epoch36/Boltz single-molecule start; copying the old latent across
VAE checkpoints would not preserve its molecule. This arm also differs from the
completed epoch45/Protenix relaxed-graph 50k trajectory because the binding path
uses learned native embeddings. The ordinary decoder probabilities remain the
source of chemistry guidance and saved molecule observations.

Fit data comprise 2,048 standard-normal training latents and 256 disjoint
development latents at seed17045. The mapping uses width128, AdamW LR0.001,
batch8 and 2,000 updates. Development error is recorded without a quality-based
abort threshold. A fit is not proof that the mapped embedding always represents
the decoded molecule; decoded/mapped disagreement and finite gradients are
reported at the tested points.

The [pilot config](embedding-pilot.json) requests 100 latent updates after fit
in an initial allocation of at most four hours on one 32GB V100. After the
pilot's finite/attached-path, shape and frozen-state checks pass, the
[full config](embedding-full.json) requests 10,000 updates from the same initial
latent; pilot updates are discarded. This is exploratory admission, without a
native-parity or detailed derivative-qualification campaign.

Both configs retain the previous Protenix contact objective, structural repair,
SGD LR0.3, step bound0.5, seed17/noise_seed230017, temperature0.7→0.4 and latent
noise schedule. They keep the original 50,000-update schedule horizon instead of
compressing it into the new budget. Consequently the chemistry weight remains1
through the 10k endpoint, temperature reaches0.64, and noise ratio reaches~0.267;
the late chemistry ramp to10 is not exercised. Temperature applies to the
decoded graph and chemistry guidance; the learned binding path has no softmax
or categorical temperature. Snapshots of the decoded graph
are saved every10 updates, resumable checkpoints every100, plus endpoints.

The scorer's contact surrogate is negative mean ligand best-pocket contact
probability below8Å, geometry-off Protenix-Tiny with model key17. It is not the
older Boltz DBMol affinity objective. Raw decoded validity, decoded identity,
mapping error and the sampled contact scores are observations; no binding-gain
or derivative-correctness claim follows from a successful engineering pilot.

Submitted 3 November 2026 at11:01CET as **29877051**, queue `gpuv100`, one
exclusive32GB V100, eightCPUslots/32GiB hostmemory, four-hour wall budget.
The single allocation runs the fit and100-update pilot, then starts the10k arm
only after the pilot's technical checks pass. At11:02CET it was PEND, with no
native smoke, mapping-fit or latent-update result. Twenty-three focused local
tests passed for the VJP transport, hidden mapping and shared latent loop.
The initial hard-graph/native-projection smoke is pending GPU execution.
Outputs: `/work3/s234818/latentmol/runs/protenix-embedding45-20261103a`.
Snapshot SHA256 `7d84e85f8ce76819d49a7f2ae87074d8fedd35964293d70647a2c867bc11d84a`.
The [pipeline](embedding_pipeline.py) and configs capture the command and budget;
local submission receipts are under `outputs/hpc-submissions/protenix-embedding-20261103a`.

At11:16CET, job29877051 was confirmed EXIT after starting11:03:28 and
stopping11:03:44. It failed before native loading or mapping training because
the configured checkpoint string reached a hash helper requiring `Path`.
The new `load_frozen_decoder` boundary normalizes the path before hashing and
loading; its local string-path regression passes (four focused mapping tests).
The failed snapshot and outputs remain intact. New snapshot
`d0165a81c79ef633dffc3025e77a92257fec0bf3941455b18d27e9a4b4033fec`
was submitted as **29877165**, with the same checkpoint, configuration and4h
V100 budget. At11:17CET it was PEND and had no mapping updates yet. Outputs use
`/work3/s234818/latentmol/runs/protenix-embedding45-20261103b`.

At11:51CET on3November, corrected job29877165 is RUN. All2000 mapping updates
finished in~8.5minutes; development MSE decreased~96.8% on average. The one hard
native/projection smoke differed1.49e-7. The100-update pilot completed with
finite paths and frozen-model checks, and the main automatically advanced to
last logged update1930/10000. Current raw graph is invalid;583of1931saved
source states were valid. At1930 the mapped contact score0.8695 differs greatly
from relaxed-graph contact0.2142, so mapped-score gains are not evidence of a
better decoded molecule. The bounded run continues as planned. A native
integer-scatter futurewarning is retained; no nonfinite failure is reported.

At12:28CET the main job29877165 remains RUN at last logged5214/10000, with
2555raw-valid source states out of5215 (~49.0%). Latest source is invalid.
At5210 mapped contact0.8769 versus decoded relaxed contact0.2081 still shows
a large score gap. The [matched-gradient investigation](embedding-comparison.md)
completed its seven-point native probe; the learned gradient is not consistently
better aligned with the decoded objective and the current implementation is
slower per warm gradient evaluation. The bounded run continues unchanged.
