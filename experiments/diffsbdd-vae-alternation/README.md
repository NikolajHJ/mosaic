# Frozen DBMol VAE / DiffSBDD alternating study

Implementation belongs to the **DBMol** repository on
`codex/diffsbdd-vae-alternation-hpc`. The original DBMol and Mosaic checkouts,
VAE continuation, checkpoints, existing jobs and environments are preserved.

**Execution is blocked before staging.** Initial HPC storage admission failed;
no experiment job, runtime provisioning, mapping fit or alternating trajectory
has run. The numerical code and fixtures are unvalidated. See
[current status and evidence](../../docs/80_diffsbdd_vae_alternation.md).

The [protocol](../../configs/diffsbdd_vae_alternation_v1.json) binds completed
epoch 39, the public conditional full-atom CrossDocked checkpoint, crystal
3MXF/JQ1 pocket, grouped mapping data, all six pilot arms and hard gates.
[The original proposed plan](proposed-plan.md) is retained as historical source
material copied from Mosaic; its old ownership/status paragraphs and relative
links do not describe this implementation. This README and the DBMol protocol
are the implementation entry points.

## Boundaries

`src/latentmol/diffsbdd/` contains separate owners for chemical heads, paired
data, fitting, native core/state/noise, canonical codec, crystal pocket, pose
transport, fresh native scorer features, scorer qualification, guidance,
held-out denoiser audit and cohort execution. Existing VAE/relaxed-graph/Boltz
implementations are reused without modifying their historical gates.

The small head predicts six element probabilities from the frozen decoder's
96-dimensional atom features. The unchanged native normalization/noise/atom
encoder then produces DiffSBDD's early representation. The diffusion state
remains noisy coordinates and atom features. There is no hidden-state inverse
or native diffused bond channel.

At a qualified handoff, the native builder, ordinary VAE reconstruction and
adapter must describe the same complete molecule with a verified atom map.
Only the latent changes. Return transport carries a pose for element-only T1
and uses bounded, deterministic restrained UFF for T2 bond edits. A fresh hard
native objective must not worsen. No gradient crosses the detached conversion,
builder, force field or diffusion sampler.

The H0 arm performs actual eligibility/identity/frame/no-op checks with zero
scorer-gradient calls. Updating arms retain fresh native scalar, component and
independent AD checks. Same-seed H0/native tensor and Gaussian replay is exact.

## Execution

Local standard-library actions:

```powershell
py -3.13 scripts/diffsbdd_campaign.py admission
py -3.13 scripts/diffsbdd_campaign.py prepare --output outputs/diffsbdd-releases/attempt001
py -3.13 scripts/diffsbdd_campaign.py stage --seal outputs/diffsbdd-releases/attempt001 --output outputs/hpc-submissions/diffsbdd-vae-alternation/staging-001.json
```

`stage` repeats admission and returns nonzero **before any remote write** if it
fails. `submit --stage NAME --seal DIR --output FILE` repeats admission, verifies
the sealed source and every predecessor, records intent before `bsub`, and
refuses an attempted/ambiguous stage. Numerical stages run only in allocations.
`status`, `collect` and `report` preserve saved results and never infer passes
from queued/missing artifacts. No automatic successor is scheduled.

The [runtime admission notes](hpc/README.md) describe the two new required
environments. They have not been provisioned. Their measured footprint must
also fit before provisioning; an environment or sampler replay pass must never
be inferred from a static implementation or an older experiment's certificate.

## Interpretation

All attempted inputs, invalid graphs, skipped opportunities and failures remain
in their denominators. Learned, exact-six and full-native conversions are
reported separately, including excluded probability mass and unsupported modes.
Feature fitting alone cannot qualify useful optimization. The initial eight
seeds on one pocket can establish feasibility; model predictions do not
establish measured affinity or binding.

The exact [3MXF input](inputs/3MXF.pdb) is copied from the existing Mosaic
crystal fixture with SHA256
`534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f`.
Its bytes are protected from Git newline conversion. Public source/weight
provenance is in [public-assets.json](public-assets.json); checkpoint bytes and
transport bundles stay outside Git.
