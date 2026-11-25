# Latent boundaries and temperature

Do narrow decoder transitions prevent binding-guided steps from reaching
distinct valid molecules? Both tests freeze the epoch-45 VAE and use the
prepared 25-slot BRD4 Protenix-Tiny contact context. Contact scores are model
proxies. These experiments use one protein context.

## Boundary sweep

The [boundary config](boundaries.json) uses two 25-heavy-atom validation
molecules each from MOSES and ChEMBL, two fresh prior draws, and two saved
trajectory latents. It evaluates temperatures 0.7 and 0.4 at six L2 radii
from 0.03 to 4 along normalized contact descent and two opposite random
directions. Active slots, model and pocket are fixed. The 304-call maximum
includes center gradients. The summary distinguishes raw and fixed-slot
graphs and records the first observed hard edit and valid changed graph at
the sampled radii. Per-center files retain categorical margin, probability
motion and graph changes. The local smoke used the pinned VAE and an
analytic scorer.

```bash
python -m latentmol.design.latent_boundaries --checkpoint EPOCH45 \
  --config experiments/latent-design/protenix-vae/boundaries.json \
  --mosaic-config experiments/latent-design/protenix-vae/native-dtu.json \
  --mosaic-python MOSAIC_PYTHON --output NEW_OUTPUT
```

DTU job **29896185** was submitted on 5 November 2026 to `gpuv100` with
four cores, one exclusive GPU, 16 GiB requested VRAM and a 30-minute limit.
LSF started it at 09:02 CET. All 304 evaluations were written. The launcher
then exited on a 15-second process wait after the scorer had acknowledged
the frozen-state check. The eight downloaded center files match their DTU
hashes and contain every requested radius for every ray.

At each temperature the descent ray reached a valid fixed-slot edit in all
four held-out encoded cases, versus zero of eight random rays. Only one of
the four descent cases reached that edit by radius 1. The first valid edit
improved the relaxed contact score in three of the four cases at either
temperature. The two prior and two saved starts have different behavior and
remain separate in the [recovered summary](../../../outputs/hpc-results/protenix-latent-editability-20261105a/boundary-29896185/recovered-summary.json).
This is an eight-center exploratory panel and the scores use relaxed input.

The [paired hard-graph and relaxed-loss figure](../../../outputs/runs/vae-boundary-readout-20261122a/epoch45-boundaries.png) shows all four encoded holdouts at temperature 0.7 along their original contact-gradient directions. Columns are the six evaluated distances from each posterior mean. A cell labeled “Valid edit” means that the fixed-slot molecule differs from the starting graph and passes the chemistry check. The lines connect measured relaxed scores and do not locate the exact boundary between samples.

## Matched temperature trajectories

The [trajectory config](temperature-study.json) compares a compressed
0.7-to-0.4 schedule with constant 0.7 and 1.0. The short run compares
temperature behavior over its own budget. It does not replay the 50,000-step
cooling timescale. Each arm uses 32 updates from one encoded JQ1 analogue
and one fresh prior seed, for 192 updates in total. It records fixed-slot
hard graphs and scores, and unconditioned hard-decode validity. Each arm
has the same structural-repair loss and seeded noise schedule.
The [fresh scoring config](temperature-fresh-dtu.json) applies the existing
native FILE contact scorer to selected valid starts and novel candidates
after the trajectory worker exits. It rebuilds ligand conformers and native
features in a separate process and reports its own matched contact deltas.
This score has a different contact reduction from the optimization loss.
It uses the native trunk distogram without diffusion.

```bash
python -m latentmol.design.temperature_study --checkpoint EPOCH45 \
  --config experiments/latent-design/protenix-vae/temperature-study.json \
  --mosaic-config experiments/latent-design/protenix-vae/native-dtu.json \
  --mosaic-python MOSAIC_PYTHON --output NEW_OUTPUT
python -m latentmol.design.temperature_fresh --run NEW_OUTPUT \
  --config experiments/latent-design/protenix-vae/temperature-fresh-dtu.json
```

DTU job **29896199** was dispatched on `gpuv100` and exited before model
loading because the scheduler expanded an inline shell argument. The retry
uses the [two-stage shell launcher](run_temperature.sh), which passed a
local check for both launch commands and failure propagation.

Replacement **29896250** was submitted to `gpuv100` with four cores, one
exclusive GPU and a 45-minute limit. It dispatched at 09:12 CET. The worker
now closes its request stream after the frozen-state acknowledgement and
can reclaim the process after a teardown timeout, with an explicit warning.
The frozen-state check and unexpected worker errors remain fatal.
At 09:14 CET the worker had passed its first native backward and saved all
three encoded-start trajectory results. The fresh native FILE stage was
still pending within this allocation. Twenty focused trajectory tests and
nine native subprocess tests passed locally.

### Temperature result

Job **29896250** completed successfully at 09:19 CET in 428 seconds. All
six trajectories reached 32 updates. The encoded JQ1 analogue was valid at
all 33 observations and never changed identity in any arm. For the prior
start, the fixed-slot and unconditioned hard-decode counts agreed.

| Temperature | Valid prior observations including start | Distinct valid edits | Final prior graph |
| --- | ---: | ---: | --- |
| 0.7 → 0.4 | 10/33 | 2 | Valid |
| Constant 0.7 | 1/33 | 0 | Sanitization failure |
| Constant 1.0 | 22/33 | 4 | Valid, same identity as cooling |

Cooling and constant 1.0 shared both their first valid edit and final hard
molecule. The shared first edit at update 2 improved the fresh native FILE
contact loss from 2.606645 to 2.599261. Their shared final molecule had a
worse fresh score of 2.664897, despite improved hard fixed-context loss
from −0.197830 to −0.388924. Four unique molecules were freshly scored
across both starts. The extra intermediate edits at temperature 1.0 were
outside the predeclared fresh candidate selection. One prior seed and one
encoded start give an exploratory comparison of editability.

[Verified trajectories](../../../outputs/hpc-results/protenix-latent-editability-20261105a/temperature-29896250/trajectories.json)
and [verified fresh scoring](../../../outputs/hpc-results/protenix-latent-editability-20261105a/temperature-29896250/fresh.json).

## Warm start versus constant 1.0

The [follow-up config](temperature-warm-study.json) compares 1.5 cooling
linearly to 1.0 by update 16, then held, with constant 1.0. Both arms share
three standard-normal starts and three encoded starts, 32 updates per start,
the epoch-45 checkpoint, the same fixed-25 context and the previous pilot's
optimizer, structural-repair objective and seeded noise. That gives 396
relaxed scorer calls, including the initial and final evaluations, across
12 trajectories. The encoded panel includes the JQ1 analogue and one
validation molecule each from MOSES and ChEMBL. The random panel uses the
original seed 191411 plus 191521 and 191522. The test varies only temperature
within a matched pair.

The [fresh rescore](temperature-warm-fresh-dtu.json) includes each valid
starting graph and up to four distinct valid edits per trajectory, including
intermediate observations. It selects by occurrence along the trajectory.
It reuses a single fresh score for duplicate SMILES and reports which valid
edits fit within the cap. Start-relative changes require a valid scored start.
Judge each schedule by distinct valid edits and fresh contact improvements.
The fresh score uses re-featurized discrete molecules and ligand conformers
regenerated with the same seed under the pinned protein pocket.

The existing [two-stage launcher](run_temperature.sh) runs both parts
sequentially on one GPU. Pass `temperature-warm-study.json` and
`temperature-warm-fresh-dtu.json` as its study and fresh config arguments.

DTU job **29897058** was submitted on 5 November 2026 to `gpuv100` with
one exclusive GPU, four cores and a 45-minute limit. The previous matched
pilot dispatched promptly in this queue. The live snapshot before this
submission showed the accessible GPU queues occupied. The initial check
reported pending scheduling.

Twenty-one focused trajectory tests and three candidate-selection and
rescoring tests passed locally. The authenticated epoch-45 model produced
identical initial hard graphs within each pair. Prior seed 191522 starts
with a sanitization failure in both arms and remains in the panel as an
invalid-start rescue case.

### Warm-start result

Job **29897058** completed at 11:09 CET in 996 seconds. The 1.5 start
increased the initial Protenix gradient norm 6.6–39× on the three encoded
molecules. Each still decoded to its original hard graph throughout both
arms, and warm-arm final latent displacement stayed below 0.006 in L2.

| Prior start | Distinct valid edits, fixed 25, warm / constant | First valid edit, warm / constant | Fresh improvements among scored edits, warm / constant |
| --- | ---: | ---: | ---: |
| 191411 | 3 / 4 | 2 / 2 | 0/2 / 3/4 |
| 191521 | 4 / 4 | 5 / 14 | 3/3 / 4/4 |
| 191522 | 12 / 7 | 1 / 3 | Initial graph invalid in both arms |

The prior arms yielded 88/99 versus 80/99 valid fixed-slot observations.
Summed within trajectories, fixed-slot distinct valid edits were 19 versus
15. The corresponding unconditioned hard decodes yielded 18 versus 19
distinct valid edits. Fresh scoring succeeded for 25 of 27 unique selected
graphs. Two warm candidates failed conformer embedding. With a valid
starting score, 3/5 warm and 7/8 constant scored edits improved. Warm's
best improvement for seed 191521 was −0.1328, versus −0.1222 at constant
1.0. This small panel did not show a consistent fresh-scoring advantage
for either schedule across starts.

[Verified trajectories](../../../outputs/hpc-results/protenix-latent-editability-20261105a/temperature-warm-29897058/trajectories.json)
and [verified fresh scoring](../../../outputs/hpc-results/protenix-latent-editability-20261105a/temperature-warm-29897058/fresh.json).

## Bounded search along the latent gradient

The [step-search pilot](step-search-study.json) asks whether useful
directions around the encoded molecules need larger steps. It uses frozen
epoch 45, temperature 1.0, the same BRD4 context and structural repair
objective as the temperature pilot. The budget is 31 optimization scorer
calls per start in each arm, counting the endpoint. The current update
spends these on 30 updates. The search uses six center evaluations and
four radius probes per center. Its radii are 0.1, 0.3, 1.0 and 2.0 in L2.
The repair target and active slots are held fixed within each comparison.

The search uses a deterministic gradient direction. The current rule
includes its existing binding-gradient-scaled noise. This comparison tests
the two update policies at matched scorer cost. Report valid distinct edits
on the committed trajectory separately from edits seen only at trial points.
The two-stage launcher freshly scores the valid starts and up to four
distinct valid edits per arm across all paid evaluations, with trial-only
provenance retained. The four starts are the encoded JQ1 analogue, the
held-out MOSES and ChEMBL encodings, and prior seed 191522.

DTU job **29898952** was submitted on 5 November 2026 to `gpuv100` with
one GPU, four cores and a 45-minute limit. Initial status was pending.
Twenty-three focused CPU tests passed. The tests cover the fixed repair
target, matched call counts and acceptance of valid improving trials.

### Bounded search result

Job **29898952** completed in 447 seconds. All eight trajectories used
exactly 31 Protenix calls. On the three encoded starts, the current rule
committed zero distinct valid edits and normalized search committed two,
one each for the MOSES and ChEMBL starts. The MOSES committed edit improved
fresh contact loss by 0.02959. A trial-only MOSES edit improved it by
0.03455. The ChEMBL candidate could not be embedded for fresh scoring.
The invalid prior start produced seven distinct valid fixed-slot edits
under the current rule and zero committed under normalized search. Two
valid search trials failed the acceptance rule. This is a small matched
panel of two update policies.

[Verified trajectories](../../../outputs/hpc-results/protenix-latent-editability-20261105a/step-search-29898952/trajectories.json)
and [verified fresh scoring](../../../outputs/hpc-results/protenix-latent-editability-20261105a/step-search-29898952/fresh.json).

## Same molecule from three latent starts at update 597604

This comparison holds the starting molecular identity, BRD4 pocket, native atom order and scorer budget fixed. Each distinct 25-heavy-atom parent begins as a standard Gaussian draw from the seven-round VAE. The other two starts are the canonical re-encoding of that molecule (posterior mean) and a single sample at the encoder's learned posterior variance. Eligibility requires all three raw and fixed-slot hard graphs to represent the same parent. The original Gaussian and posterior noise seeds, rejected proposals and slot maps are saved. The cohort therefore describes molecules recoverable from all three starts. It does not estimate success on arbitrary prior vectors or arbitrary ligands.

Each start gets six normalized search rounds at temperature 1.0, with four latent L2 radii (0.1, 0.3, 1, 2), totaling 31 Protenix search evaluations including the endpoint. The structural-repair and contact objectives are the same in all arms. Native input order is aligned against the same hard parent while each arm keeps its own relaxed probabilities. Fresh scoring uses the common parent and at most four distinct valid edits per arm selected by occurrence on the evaluated trajectory, independent of fresh scores. Each candidate gets a new reference conformer and native FILE features in the same pocket. A lower fresh contact loss than the parent counts as a confirmed improvement. The report separates edits accepted by the trajectory from those exposed only in a search trial. Unscorable pairs remain visible. These are predicted contacts under one protein context.

The two GPU allocations cover disjoint slices of the same ordered twelve-parent panel. The [analysis command](analyze_same_molecule.py) accepts one or more downloaded run directories, checks the budgets and parent identities, and labels incomplete coverage. [Search config](same-molecule-597604.json) · [Fresh-scoring config](same-molecule-597604-fresh-dtu.json) · [Panel selection](../../../src/latentmol/design/same_molecule_starts.py)

**Panel qualification.** In deterministic seed order, 763 of 11,862 Gaussian proposals decoded with 25 occupied atoms. Of these, 670 passed the original chemistry, raw/fixed and canonical-label checks. Re-encoding reproduced the original graph at the posterior mean in 49/670 cases. For the others, 503 failed sanitization, 114 decoded to another chemical identity and four changed atom count. One independently seeded draw at the learned posterior variance also reproduced the graph for 12 of the 49. These twelve distinct matches are the full paired search panel. This conditions the optimization comparison on a successful three-way reconstruction. The screened population and reasons remain available for interpreting generalizability. All 36 baseline states passed a CPU replay covering the raw and fixed-slot graph, canonical position mapping and latent provenance. [Frozen panel, actual vectors and screening outcomes](../../../outputs/runs/same-molecule-597604-20261122a/gpu-prefilter-extended/same-molecule-597604-panel.json)

### What would copying a valid random molecule to its neighbors teach?

At update 597604, 45 of 48 independent standard Gaussian vectors decoded to a valid hard molecule. Of the other three, two failed sanitization and one was disconnected. All 45 valid centers entered this probe, with no atom-count or re-encoding filter. We drew three isotropic unit directions per center and reused the directions at each Euclidean distance. Each row is 135 decodes around the same 45 centers. Identity uses the repository's canonical non-stereochemical graph representation.

| Latent distance | Same molecule | Different valid molecule | Invalid | Parents with at least one valid edit |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 112 | 22 | 1 | 13/45 |
| 1.0 | 29 | 98 | 8 | 42/45 |
| 2.0 | 5 | 122 | 8 | 44/45 |

At 0.1, whole-graph reconstruction of the center would push 22 valid neighbors back toward their parent. Only one neighbor failed the validity check. Matching the center's existing aromatic ring or amide leaves the rest of the graph free to vary. Whether those edits improve a protein score and how training would change them are separate questions.

Reproduce with `python -m experiments.latent-design.protenix-vae.inspect_same_molecule_neighbors --checkpoint CHECKPOINT --prior-draws 48 --device cpu --out OUTPUT.json` with `src` on `PYTHONPATH`. The independent prior seed was 2026112300 and the direction seed was 2026112299. [Per-neighbor outputs](../../../outputs/runs/same-molecule-597604-20261122a/unselected-neighbors.json) · [Earlier motif-neighbor training results](../../vae/training_data_eda/epoch48-motif-completion.md#teaching-neighborhoods-of-motifs-the-decoder-already-generates)

The same probe on the 12 matched parents reused each parent's directions across its three latent vectors. At distance 1.0, the original random vectors yielded **26/36** different valid molecules, the re-encoded means **2/36** and posterior samples **3/36**. Unchanged/invalid counts were respectively 7/3, 34/0 and 32/1. The same starting graph can occupy neighborhoods with different editability. These are directional observations on 12 selected parents, with no protein score involved. To reproduce, replace `--prior-draws 48 --device cpu` above with `--panel PATH_TO_EXTENDED_PANEL --device cuda`. [Matched neighbor decodes](../../../outputs/runs/same-molecule-597604-20261122a/neighbors.json)

**Submission status at 19:59 CET, 22 November.** Both disjoint V100 releases were sealed and verified locally. The DTU helper was denied SSH authentication during staging before any remote submission command ran. There are no job IDs for this experiment. The scheduler advertised a service window from 20:00 CET until 06:00 CET on 23 November. When login returns, use `submit_same_molecule.py upload first` followed by `submit_same_molecule.py submit first`, then the same commands for `remaining`. Each submission uses a unique scheduler intent and receipt. The native Protenix ranking is pending.

### Why a valid generated graph can fail after re-encoding

On 23 November 2026 we froze checkpoint 597604 and re-encoded all 670 eligible 25-atom generated parents from the above screening record, plus the separate saved 512-molecule PubChem validation panel. Both used canonical input order, posterior mean and conditional bond argmax. On the local RTX 4060, all 670 generated exact-identity decisions agreed with CPU screening. The earlier 503 sanitization failures counted only graphs with 25 predicted atoms. This replay also sanitized the four changed-count predictions, bringing the total to 507 valence exceptions. Exact identity uses canonical stereo-free SMILES throughout.

| Posterior-mean readout | Valid generated / 670 | Exact generated / 670 | Valid PubChem / 512 | Exact PubChem / 512 |
| --- | ---: | ---: | ---: | ---: |
| Raw | 163 | 49 | 488 | 474 |
| Existing valence projection, up to eight edits | 633 | 69 | 502 | 479 |
| Multiply the relative odds of NO_BOND by two | 273 | 73 | 493 | 477 |
| Multiply the relative odds of NO_BOND by four | 494 | 100 | 500 | 484 |
| Give the decoder the reference atoms and charges | 165 | 51 | 489 | 479 |
| Remove precisely the predicted bonds absent from the reference | 651 | 292 | 508 | 500 |
| Give the decoder the reference bond matrix | 651 | 600 | 510 | 505 |

The last three rows use the known parent in its canonical training slots to isolate errors. The two odds shifts add a fixed logit offset to NO_BOND at readout. The projection and odds shifts use only the predicted labels and scores. Every method gave 30/30 valid and 29/30 exact on the 25-atom subset of the real validation set. The small control matches by size only. These generated graphs were originally valid and are selected by the study's size and representation filters.

The raw generated round trips had 3,510 extra bonds, 620 missing bonds, and ten disagreements about the order of a bond present in both matrices. These counts compare upper-triangle entries in canonical training slots. Of the 507 invalid reconstructions, removing the extra bonds restored validity to 488, while supplying reference atoms and charges restored it to two. Validity can therefore be recovered while quite a few parent bonds still differ.

Independent cycle count (bonds minus atoms plus one per connected molecule) also tracked the performance gap. Exact recovery was 15/34 at zero to two cycles, 18/89 at three, 12/135 at four, 4/158 at five and 0/254 at six or more. None of the 30 size-matched PubChem examples had five or more cycles. This association does not isolate a training cause.

We also regenerated each original Gaussian vector from its saved seed. Raw decoding and the existing valence projection each preserved all 670 valid starting identities. Doubling NO_BOND odds on these same original vectors preserved 194 identities and left 632 valid molecules. Quadrupling those odds preserved 85 identities and left 600 valid. A constant shift that helps posterior-mean reconstruction can therefore damage already-valid prior decodes.

The neighborhood test measures a different failure. At random latent distance 0.1, 19 of 22 changed valid decodes changed RDKit's perceived ring count. Sixteen did so at constant heavy-atom count. Median radius-2 Morgan fingerprint similarity (2,048 bits, Tanimoto) was 0.458 among the 22 changed valid directions. This supports measuring bond or scaffold preservation alongside validity when judging the size of a latent edit. [Saved neighbor graphs](../../../outputs/runs/same-molecule-597604-20261122a/unselected-neighbors.json)

Reproduce with `python -m experiments.latent-design.protenix-vae.diagnose_roundtrip --checkpoint CHECKPOINT --panel SCREENED_PANEL --heldout PUBCHEM_VALIDATION_PANEL --device cuda --out OUTPUT.json` from the repo root with `src` on `PYTHONPATH`. [Replay code](diagnose_roundtrip.py) · [Per-molecule outcomes](../../../outputs/runs/same-molecule-597604-20261122a/roundtrip-with-prior-control.json)

## Does warming change the useful Protenix gradient direction at update 600116?

The frozen seven-round VAE uses constant temperature 1.0 or 1.5. The study normalizes the combined Protenix contact and structural-repair gradient before each search, so inflation of its norm alone cannot buy the warmer arm a larger latent step. Both arms have six centers with trials at latent L2 radii 0.1, 0.3, 1.0 and 2.0, plus one endpoint. This costs 31 native search calls per arm per start. The same parent latent and 25 active slots enter both arms. Temperature changes the relaxed graph supplied to both objectives and can also change which trial they accept. [Study config](temperature-600116.json)

The two held-out MOSES and ChEMBL molecules from the earlier step search each reconstructed exactly in local CPU replay. The two Gaussian starts were the first distinct valid 25-atom graphs that agreed between raw and fixed-slot decoding and passed canonical alignment, scanning consecutive seeds from 2026112300 through 2026112332. This small panel conditions on a valid baseline. The fresh score separately rebuilds the ligand conformer and native features in the pinned BRD4 pocket, and compares the discrete predicted contact loss with the same scored parent. Each arm can supply four distinct valid edits per start. Save both accepted and trial-only provenance and count failed fresh representations separately. [Fresh scoring config](temperature-600116-fresh-dtu.json)

For this experiment the existing submission utility takes `seal temperature`, `upload temperature`, and `submit temperature` in that order. Twenty-eight local tests passed, including a matched two-temperature search with equal scorer budgets. DTU accepted job **30276244** on 23 November 2026, with four cores, one V100 and a 35-minute limit. It was pending at 11:49 CET. [Submission utility](submit_same_molecule.py) · [Scheduler receipt](../../../outputs/hpc-submissions/protenix-temperature-600116-20261123a/submit.json)

## Can a warmer decoder reach a specified hard edit at update 600116?

On 23 November we froze the seven-round checkpoint and used 24 held-out PubChem posterior means and 24 unfiltered Gaussian draws (seed 2026112289). Twenty-three initial molecules in each group passed raw validity, and 22 of the 24 encoded starts reproduced the source molecule. The target was exactly one additional nitrogen or oxygen atom with the original heavy-atom count. The smooth objective summed squared errors on both expected counts. RDKit independently checked both integer counts on the canonical, connected, sanitized raw argmax molecule. This composition target permits other structural changes.

Each direction received three trial decodes. The short ladder used latent L2 radii 0.025, 0.05 and 0.1 and the wide ladder used 0.1, 1 and 2. Each search started from the same parent, using one normalized gradient direction or one isotropic unit direction. Every arm selected its best trial by the same T=1 relaxed score and accepted it only if the decrease exceeded the FP32 resolution threshold. The table counts selected molecules that were valid, differed from their parent, and met both integer targets. Each cell uses the 23 valid parents in that group.

| Direction | Encoded, short | Encoded, wide | Gaussian, short | Gaussian, wide |
|---|---:|---:|---:|---:|
| Random | 0 | 0 | 0 | 3 |
| T=1 | 0 | 4 | 12 | 12 |
| T=1.5 | 0 | 3 | 9 | 10 |
| T=2 | 0 | 3 | 9 | 10 |
| T=1 forward, T=1.5 derivative | 0 | 3 | 11 | 12 |
| T=1 forward, T=2 derivative | 0 | 3 | 12 | 13 |

The short T=1 search lowered the smooth score on **all 23 encoded parents**, yet produced zero distinct valid edits. The wide T=1 search exposed a distinct valid edit on five encoded parents and selected a hard target hit on four. Those four successes required radii 1–2. Eleven of the twelve selected Gaussian successes used radius 0.1. Ordinary warming gave no additional encoded hits. In the last two rows the size and N/O counts at the starting point retain their T=1 forward values, while a warmer softmax supplies a proposal derivative. This surrogate added one selected Gaussian success at T=2 over ordinary T=1, with one fewer encoded success. This small composition study measures editability and continuous-to-discrete transfer. The independent Protenix study above tests protein contact scores.

Run `python -m experiments.latent-design.protenix-vae.local_temperature_edits --device cuda --output OUTPUT` from the repository root with `src` on `PYTHONPATH`. Add `--warm-backward-only` and a fresh output path for the surrogate. [Probe code](local_temperature_edits.py) · [Temperature and random controls](../../../outputs/runs/vae-600116-temperature-edits-20261123a/results.json) · [Warm backward surrogate](../../../outputs/runs/vae-600116-temperature-edits-warm-backward-20261123a/results.json)
