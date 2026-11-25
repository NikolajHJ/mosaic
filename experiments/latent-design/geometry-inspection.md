# Inspecting soft molecules in 3D

Question, 3 November 2026: what does a relaxed (probabilistic) molecule look
like inside the frozen scorer, before hardening atom and bond categories?

The AE/VAE `RelaxedGraph` contains element, charge and bond probabilities, not
Cartesian coordinates. The inspected Boltz affinity/composite paths reuse a
fixed, detached reference pose and disable structure diffusion. The VAE/Protenix
contact scorer is likewise geometry-off. A trunk distogram is a distribution
over pairwise distance bins, not itself a set of atom coordinates. This view
overlays the saved soft graph on its actual fixed scaffold; it is not a newly
inferred pose of soft chemistry.

## Saved example and command

Used the historical Boltz-only VAE trajectory `20261017T231228Z_long-500-boltz_only_3c998ded`,
arm `fixed_initial_mask`, saved states 0, 100 and 225. Its 22 occupied slots
map to a native carbon-chain reference pose. The local prepared snapshot SHA
`9a6cd38cfbaf114c09aa31da756b9db472b47ed0afe2de1edf9ef7792c74d570`
exactly matches this run's recorded count-22 context. Slot identity is retained;
there is no atom matching, conformer generation, chemistry repair or
force-field minimization. The camera uses a rigid principal-axis basis only.

```powershell
& ./.venv/Scripts/python.exe experiments/latent-design/extract_relaxed_geometry.py --run outputs/hpc-results/source120a004-staged-003/runs/source120a004-boltz-defog-staged-24h-attempt004/boltz_only/optimization/20261017T231228Z_long-500-boltz_only_3c998ded --context outputs/hpc-submissions/source138-context-restore-20261023/payloads/runs/source056-random-start-study/20261005T225808Z_vae-random-starts-056_37735924/contexts/count-22/prepared.pt --output outputs/reports/relaxed-3d-20261103-001/snapshot.json
```

Budget: local CPU artifact extraction and browser presentation checks only;
zero decoder/scorer evaluations and no HPC operations. The extractor checks
context identity, slot mapping, schema, finite normalized probabilities and
bond symmetry, and rejects output collisions. Consumed-file hashes were
rechecked unchanged. [Extracted data](../../outputs/reports/relaxed-3d-20261103-001/snapshot.json)
includes all selected atom/charge/bond distributions and original Å coordinates,
with display probabilities rounded to seven decimal places.

## Result and limits

The rotatable view shows element probabilities as ring sectors and bond-presence
probabilities as line width/opacity, displaying links with probability at least
10%. The atom selector exposes numerical probabilities and expected charge.
Coordinates remain identical across the three saved states. At step225, slot9
is 53.9% nitrogen and 45.9% carbon. Slots7–21 have an 88.4% bond-presence
probability but are 13.77Å apart on the fixed scaffold. Soft connectivity and
the retained reference geometry need not agree.

The original trajectory stalled at225 accepted updates of500 requested, and
its endpoint `argmax_smiles` is null; fresh native pose confirmation was
`required_not_run`. The picture diagnoses the scorer input, not a valid molecule,
a relaxed physical conformation, or improved binding. No new native/model/
derivative validation is claimed. This historical example does not describe
the outcome of the separate, recently completed VAE/Protenix50k experiment.

Presentation: [standalone preview](../../outputs/reports/relaxed-3d-20261103-001/preview.html),
[screenshot](../../outputs/reports/relaxed-3d-20261103-001/preview-desktop.png),
[browser QA](../../outputs/reports/relaxed-3d-20261103-001/browser-qa.json).
Headless Edge checks passed for state changes, rotation, atom selection, and no
horizontal overflow at320/360px; no browser runtime errors. Desktop/mobile
screenshots were visually inspected. Rendering uses saved data only.

Follow-up: the separate [soft-input diffusion test](soft-diffusion.md) now
generates actual native Boltz-2 coordinates from this saved probabilistic graph.
The original view above remains a fixed-input diagnostic. Embedding predicted
distance distributions into3D would instead be a derived geometric approximation.
