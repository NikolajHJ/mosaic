# Boltz-2 diffusion on a saved soft molecule

Question, 3 November 2026: can the structural Boltz-2 diffusion stage generate
coordinates directly from the same probabilistic graph shown in the fixed-pose
inspection, without hardening its chemistry first?

Authorized as a small separate test. Native job **29876935** submitted to
`gpuv100`, one32GB V100, four CPU slots, 15-minute wall limit. This run does not
resume or modify any optimization/training campaign. Source, input graph and
launcher are isolated in `/work3/s234818/latentmol/runs/soft-diffusion-20261103-attempt001`.
[Plan](../../outputs/hpc-submissions/soft-diffusion-20261103-attempt001/plan.json),
[submission](../../outputs/hpc-submissions/soft-diffusion-20261103-attempt001/submission.json).

## Inputs and budget

- Historical source120a004 Boltz-only VAE graph at225, 22 occupied slots,
  full saved element/charge/bond distributions, unchanged temperature1.
- Exact prepared count22 context SHA `9a6cd38c...74d570`; structural checkpoint
  `boltz2_conf.ckpt` SHA `090e82ac...428e1`.
- Pinned native Boltz source `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`.
- One sample, seed17, zero recycling steps, 40 diffusion steps, float32,
  standard Torch kernels. No chemistry/physical/contact/FK steering.
- Existing continuous feature adapter and expected typed-bond trunk; native
  diffusion conditioning and sampler. No argmax, SMILES, RDKit conversion,
  latent updates, model training, or differentiation through diffusion.

Reference conformer, atom names, occupancy, crop, MSA, templates and other
prepared fields remain conditioning. Diffusion starts its own noisy coordinates
and generates a new protein/ligand pose; the old bound coordinates are retained
only for later comparison. Probabilistic chemistry is an exploratory extension
of a model trained on discrete inputs, not a mixture of independently sampled
valid chemical species. Chemical validity and binding are not established.

## Command and checks

Implementation: [`latentmol.boltz.soft_diffusion`](../../src/latentmol/boltz/soft_diffusion.py).
The uploaded source archive SHA is
`efd325dbf12627d4786b31e61c09477b1d7c89f5327700ea7a4c07b1f9d4b6af`.
The exact allocated command and scheduler requests are in the remote
`submission-intent.json`; its essential command is:

```sh
python -m latentmol.boltz.soft_diffusion --context PREPARED.pt --context-sha256 SHA --probabilities probabilities.json --probabilities-sha256 SHA --checkpoint boltz2_conf.ckpt --checkpoint-sha256 SHA --steps 40 --seed 17 --output NEW_OUTPUT
```

Local CPU check loaded the actual context/soft graph and confirmed continuous
element mapping exactly:1504 native atom rows,214 tokens, slot9 nitrogen53.9%
and carbon45.9%. CLI loading passed with `PYTHONPATH=src`. Native inference
checks finite trunk/coordinates, shape/schema/probability compatibility,
consumption of unchanged soft elements by diffusion conditioning, unchanged
adapted feature tensors, frozen parameter/buffer versions, consumed input-file
hashes, output collisions and the exact40-step budget. Intermediate saved frames
are the native denoised-coordinate estimates, not the noisy sampler state.

Launch preparation initially could not find `bsub` outside a login shell;
no submission process had been created. The recorded submission used the
configured login shell and produced one job. No duplicate retry was made.

## Native result

**Completed:** scheduler29876935 DONE, native `status=completed`, all40 denoising
calls recorded. Total worker67.23s (scheduler runtime77s), prediction5.48s after
loading, peak Torch GPU allocation2.49GiB on TeslaV100-SXM2-32GB. Finite trunk,
intermediate/final coordinates, soft-conditioning consumption, frozen tensor
versions, adapted feature identity and input hashes passed. Downloaded result,
probability graph and coordinate NPZ match remote SHA256 values.

The new final pose differs from the fixed reference: ligand RMSD5.31Å after
a proper rigid fit on receptor atoms; receptor RMSD2.94Å. The soft7–21 link
(88.4% presence) spans1.546Å in the prediction versus13.771Å in the old scaffold.
This is a coordinate response, not evidence that the model repaired chemistry.
The closest ligand pair is1–6 at0.983Å, and one greater-than50% link remains
3.987Å long. The original soft graph's hard endpoint is invalid. No chemical
validity, force-field minimum, confidence calibration, native derivative parity,
or affinity improvement was established.

All protein/ligand atoms were generated; the inline view displays the22 ligand
slots. The geometry selector includes the fixed reference, native denoised
estimates10/20/40 and the final sampler output. Probability rings and bond
probabilities are unchanged across those geometries. Each frame is aligned using
receptor atoms only; a common rigid camera basis is used, with no ligand fit,
energy minimization or coordinate repair.

- [Exact native coordinates and40 intermediate estimates](../../outputs/hpc-results/soft-diffusion-20261103-attempt001/prediction/coordinates.npz)
- [Native result and soft-conditioning evidence](../../outputs/hpc-results/soft-diffusion-20261103-attempt001/prediction/result.json)
- [Download identities and scheduler DONE](../../outputs/hpc-results/soft-diffusion-20261103-attempt001/download-verification.json)
- [Rotatable preview](../../outputs/reports/soft-diffusion-20261103-attempt001/preview.html),
  [screenshot](../../outputs/reports/soft-diffusion-20261103-attempt001/preview-desktop.png),
  [visual data and distance diagnostics](../../outputs/reports/soft-diffusion-20261103-attempt001/visual-data.json),
  [browser QA](../../outputs/reports/soft-diffusion-20261103-attempt001/browser-qa.json).

Presentation assembly uses `python -m latentmol.reporting.soft_diffusion
--prediction SAVED_PREDICTION --output NEW_JSON`. A local rigid-transform fixture
confirmed that receptor alignment preserves ligand displacement. Browser checks
passed for all five geometries, unchanged chemistry, default final prediction,
rotation, atom selection, and320/360px overflow; no runtime errors. Desktop/mobile
screenshots were visually inspected. Native Lightning emitted a nonfatal warning
that the checkpoint producer2.5.0.post0 is newer than the installed2.5.0 loader;
strict checkpoint loading and inference nevertheless completed.

## Follow-up: stretched and compressed displayed links

The user's distance concern prompted a read-only audit of the saved final
coordinates. There are41 displayed links atP(bond)≥10%,30 above50%, and25
above90%. The longest displayed link18–19 is5.163Å with27.8% presence, and its
modal category is NONE; a faint line is an uncertain marginal, not an asserted
covalent bond. However,10–18 has50.7% aromatic probability at3.987Å, and1–6
has97.9% presence (mostly SINGLE) at0.983Å. The problems are therefore not
solely the display threshold or a viewing projection. These are actual3D
distances before any rigid display alignment.

The input has competing/invalid hard connectivity, retained carbon reference
metadata and a short diffusion schedule. Their individual effects have not been
isolated; this single run does not establish which caused each distortion.
A matched hard-input control and schedule comparison would be separate tests.
[Complete distance audit](../../outputs/reports/soft-diffusion-20261103-attempt001/bond-distance-audit.json).

## Standalone 3D report, 6 November 2026

The report embeds the saved graph, all 40 denoising estimates, the final sample,
and the fixed reference. Comparison uses synchronized rotation and scale.
The pair inspector reports 3D distances and the saved bond and element
probabilities. The plot and viewer share the selected pair.
[Open the interactive report](../../outputs/reports/relaxed-molecules-20261106/relaxed-molecules-3d.html).

```sh
python -m latentmol.reporting.soft_diffusion --prediction outputs/hpc-results/soft-diffusion-20261103-attempt001/prediction --html outputs/reports/NEW_REPORT/relaxed-molecules-3d.html
```

Rendering uses the saved artifacts and a browser. Offline browser checks
covered the 42 frames, rotation, comparison, playback, pair selection, bond
threshold, and layouts from 320px to desktop. The earlier JSON payload matched
the saved payload exactly.

## Five-molecule gallery, 6 November 2026

Question: how do the saved element, charge and bond distributions relate to
the predicted geometry across several archived experiment molecules?

| Atom slots | Archived input | Probability source |
| --- | --- | --- |
| 19 | Epoch-45 matched evaluation, posterior anchor 100 | Redecoded with epoch 45, T=1 |
| 21 | VAE / Protenix raw occupancy, accepted update 483 | Redecoded the guided latent with epoch 45, T=1 |
| 22 | Boltz-only optimization, step 225 | Saved graph, T=1 |
| 22 | Boltz + DeFoG optimization, step 279 | Saved graph, T=1 |
| 25 | VAE / Protenix 50k endpoint | Saved graph, T=0.4 |

The two redecoded inputs use their argmax occupied slots with the conditional
element, charge and bond distributions. The guided decode reproduced its saved
SMILES. The compact case is the smallest eligible chemically valid archived
anchor with a prepared count context and a material soft atom mixture, after
excluding the counts already selected. The 21-slot input required rebinding
template slots to the archived IDs. All five inputs use count-matched Boltz
carbon scaffolds and their prepared receptor context.

Job **29905474** completed four 40-step samples at seed 17 on an allocated
A100, using the original structural checkpoint and sealed inference source.
Prediction took 5.67–6.29 s per sample after loading, with 2.49–2.54 GiB peak
Torch allocation. All four finite-coordinate, soft-input, frozen-state and
input-identity checks passed. Downloads matched remote hashes. The earlier
V100 submissions 29905214 and 29905312 reached EXIT after retirement while
pending for host slots. The interactive allocation used a 900 s worker deadline.
[Case selection and input bindings](../../outputs/hpc-submissions/soft-diffusion-gallery-20261106-a01/panel/cases.json).
[Native completion and download evidence](../../outputs/hpc-results/soft-diffusion-gallery-20261106-a01/download-verification.json).

The gallery shows five final predictions, their references and all 40 denoising
estimates per case. The element and formal-charge tables show each categorical
distribution, and the bond table selects any bond or an individual channel.
The matrix and viewer share the measured pair. Offline browser checks covered
all five cases, slot labels, matrix values, playback, comparison and mobile
layout. [Open the five-molecule report](../../outputs/reports/relaxed-molecules-gallery-20261106/relaxed-molecules-3d.html).

```sh
python -m latentmol.reporting.soft_diffusion --gallery outputs/reports/relaxed-molecules-gallery-20261106/gallery-inputs.json --html outputs/reports/NEW_REPORT/relaxed-molecules-3d.html
```
