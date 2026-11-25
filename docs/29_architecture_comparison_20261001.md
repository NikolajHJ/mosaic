# Graph-decoder architecture comparison, 1 October 2026

The completed wide VAE continuation adds a 180-epoch endpoint to the graph
AE/VAE comparison. It improved its own validation-selected held-out
reconstruction over its 120-epoch endpoint, but it does not establish that a
wide VAE is the best latent representation. The strongest current case for a
VAE is the stabilized VAE's reconstruction plus the earlier, separate
128-dimensional interpolation result. Neither establishes smooth semantic
interpolation for these 256-dimensional models or useful binding optimization.

All rows below use the same `zinc215k-seed17-v1` graph cache, one seed, and
215,000 training / 7,347 validation / 7,347 test molecules. The source
manifest reports 249,455 source rows, 18,195 representation-roundtrip
rejections and 1,566 duplicates, leaving 229,694 retained rows. The graph
schema is stereo-free and the split is not scaffold-separated. The test
reconstruction count is the number of raw decoded graphs passing the recorded
chemistry checks or exactly reconstructing the supplied held-out molecule,
out of 7,347 attempts. Selection of each *best validation reconstruction*
checkpoint was sealed using validation data before the test evaluation. The
[ledger](05_experiment_ledger.md) retains immutable source/config/data hashes,
job IDs, checkpoint paths, failed runs and evaluation run IDs. This report
compares outcomes; it does not select a new checkpoint using the test set.

| Decoder family | AE: selected epoch; test valid / exact | VAE: selected epoch; test valid / exact | VAE standard-normal prior valid / 1,024 |
|---|---:|---:|---:|
| Residual control | 63; 7,009 / 6,860 | 87; 6,788 / 6,587 | 18 |
| Pair refinement | 61; 7,136 / 7,064 | 114; 7,190 / 7,103 | 18 |
| Deep pair refinement | 61; 6,998 / 6,879 | **Failed in epoch 7**; no test result | — |
| Global bond context | 62; 7,170 / 7,107 | 99; 7,162 / 7,090 | 12 |
| Wide global bond context | 81; 7,220 / 7,169 | **178 after continuation**; 7,054 / 6,948 | 8 |
| Stabilized deep pair refinement | 61; 7,114 / 6,996 | 86; 7,225 / 7,159 | 8 |

The pair/global test summaries downloaded under
`outputs/hpc-results/architecture-20261001-status/`, the historical
[29 September report](25_architecture_results_20260929.md), and the
source042/047 entries in the ledger supply the rows. Every count in this
table uses a different checkpoint selected within its own arm; later epochs
are **not equal-update comparisons**. The failed original deep-pair VAE run
remains a failure. Source047's stabilized variant is a separate model, not a
retroactive completion of it. The wide VAE's source042 epoch-114 endpoint
before continuation was 6,990 valid / 6,890 exact; its 180-epoch continuation
is reported separately below. Among these selected test evaluations,
stabilized VAE has the largest valid count (7,225), while wide AE has the
largest exact count (7,169). These observations do not establish a general
architecture winner from one split, one seed, unequal budgets and several
model comparisons on the same test set.

The common **epoch-60** checkpoints provide a narrower equal-update view
for the four source042/047 arms with downloaded held-out summaries:

| Arm | Test valid / exact at epoch 60, of 7,347 | Reconstruction loss |
|---|---:|---:|
| Wide AE | 5,671 / 5,095 | 0.022703 |
| Wide VAE | 5,851 / 5,313 | 0.015042 |
| Stabilized AE | 6,777 / 6,554 | 0.009434 |
| Stabilized VAE | 7,052 / 6,920 | 0.004439 |

The epoch-60 rows share the update count and split, but the decoder families
have different parameter counts. They also do not answer whether subsequent
latent optimization finds valid, distinct molecules with retained native
binding scores.

## Wide VAE continuation: the comparable checkpoint roles

The source048 job **29539692 completed** 180 epochs by resuming the exact
source042 wide VAE epoch-120 `last.pt`, including optimizer/RNG state. The
source042 model/training files were byte-identical in the source048 release;
the continuation used a new config and run directory. The coordinator is
`20261001T091512Z_vae-epoch-continuation_00e6b73f`, training run
`20261001T091512Z_vae-train_ce9f56be`, immutable source048 archive SHA-256
`c1c556ecc775920699d58664646507baf0a44f5f607e7d2690f7c047a05c04de`.
The downloaded coordinator summary and six completed evaluation
`run.json`/`summary.json` pairs are under
`outputs/hpc-status/20261001-wide-vae-complete/`. Checkpoint hashes and
their validation-sealed roles are in the ledger.

| Checkpoint role | Epoch | Validation valid / exact | Test valid / exact | Test reconstruction loss | Test prior valid / 1,024 | Test evaluation run ID |
|---|---:|---:|---:|---:|---:|---|
| Best validation reconstruction | 178 | 7,039 / 6,945 | 7,054 / 6,948 | 0.003472 | 8 | `20261001T151506Z_evaluation_f008ff4f` |
| Best validation validity | 176 | 7,043 / 6,957 | 7,065 / 6,975 | 0.003529 | 10 | `20261001T151544Z_evaluation_281f7bdd` |
| Last | 180 | 7,031 / 6,944 | 7,061 / 6,966 | 0.003293 | 10 | `20261001T151621Z_evaluation_c562904f` |

The validation runs for these three rows are respectively
`20261001T151447Z_evaluation_fd8d5426`,
`20261001T151525Z_evaluation_004a53f3` and
`20261001T151602Z_evaluation_7473b0b8`. The epoch-178 test result improves
on the separately selected source042 epoch-114 wide VAE checkpoint by
**64 valid and 58 exact** reconstructions, after 60 additional epochs. The
epoch-180 last checkpoint happens to have a lower test reconstruction loss
than epoch 178; choosing it on that basis would use held-out data for model
selection, so the best-validation-reconstruction row remains the primary
continuation comparison. The final-window validation trend recorded
`extend=false`, `reason=no_threshold_crossed` and an overfitting warning.
That schedule-specific stop signal is not proof of convergence.

## Latent-space evidence and what remains open

At the selected source048 epoch-178 test evaluation, **78 of 256 latent
dimensions** exceed the recorded encoded-mean variance threshold of 0.01.
Only **8/1,024** standard-normal prior draws were valid. At a relative latent
perturbation scale of 0.2, **240/256** neighborhood decodes were valid, but
only **2/256** were valid graphs changed from their unperturbed baseline.
High local validity here largely reflects decoding the same graph again;
this is not evidence for useful movement through latent space. The
corresponding source042 epoch-114 prior result was 13/1,024, so extending
training improved conditional reconstruction while leaving unrestricted
prior sampling poor. The completed 180-epoch checkpoint has **no new
matched 128 × 16 fixed-`z` categorical diagnostic or interpolation study**.
The neighborhood perturbations use a relative scale: their reference latent
RMS standard deviations differ sharply, for example about 264.51 for
stabilized AE, 0.436 for stabilized VAE, 0.303 for wide AE and 0.452 for
the continued wide VAE. A raw `z` step or trust radius is therefore not a
matched movement budget across these models.

Existing fixed-`z` studies of earlier best-reconstruction checkpoints drew
2,048 temperature-1 hard graphs from 128 encoded validation means per arm,
with no repair or validity-conditioned resampling. They found 1,995 valid /
1,982 exact for wide AE and 1,886 valid / 1,841 exact for the **older
120-epoch** wide VAE; these numbers must not be assigned to the 180-epoch
continuation. A separate source051 equal-cohort fixed-`z` quality-v1 panel
used **best-validation-validity** stabilized checkpoints, seed 20261001,
and found 2,007/2,048 valid AE draws and 1,963/2,048 valid VAE draws.
That comparison is matched within the stabilized family; its checkpoint
roles and seed differ from the earlier best-reconstruction wide panel, so
the counts should not be pooled as a cross-family ranking. The source051
score is a chemistry-readiness diagnostic, not a binding or diversity
objective. The requested equal-third graded-quality
v2 real-checkpoint panel has **not run on HPC**; its tests were only small
local engineering smokes under the user's temporary exception.

An earlier **128-dimensional**, different-source paired interpolation audit
found a +11.510 percentage-point VAE intermediate-validity difference on
223 paths with valid endpoints in both models (669 intermediate attempts per
model; anchor-clustered 95% interval 7.692–15.408 points). That result
supports sampled intermediate validity in that historical setup. It does
not demonstrate semantic consistency or transfer the finding to these
256-dimensional decoders; the [stage-1 audit](11_stage1_latent_study.md)
and ledger retain its scope.

For the next frozen-model tests, the current
[provisional candidate panel](26_next_test_protocol.md) uses the stabilized
VAE epoch-89 checkpoint as primary and same-family stabilized AE epoch 63
as control, with wide AE as an optional strong reconstruction comparator.
That panel has historical test exposure and must be treated as provisional.
The decisive evidence is still missing: valid **changed** molecules under
matched latent/search budgets, interpolation chemistry and similarity,
graded repair cost, and retained fresh-native Boltz scores after hard
projection. No architecture result here measures binding, synthesis,
off-target selectivity or experimentally useful molecules.
