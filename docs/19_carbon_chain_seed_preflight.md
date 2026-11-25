# Carbon-chain seed: representation and fresh-context preflight

Updated 28 September 2026. Immutable **source027** has been uploaded and extracted.
CPU jobs **tests 29507353, AE 29507354 and VAE 29507355 are DONE** on `hpc`,
each with two CPU slots, 4 GB per slot and a 30-minute limit. Both molecule
preflights depended on `done(29507353)`. **42 tests passed in 8.07 seconds**,
and **both deterministic encodings decoded to the exact valid C13 chain**.
No Boltz prediction/pose or latent update occurred in source027. The separate
source029 native preparation below has now completed; no chain latent gradient
or optimization has run.

HPC source archive:
`/work3/s234818/latentmol/releases/source-027-carbon-seed.tar.gz`.
Source SHA256:
`f1e9b09e1b86bf5a6d394e93c3ea182e6af7e1f96bfd6125468cc9b02586f2a4`.
Scheduler logs use
`/work3/s234818/latentmol/logs/{carbon-seed-tests-027,ae-carbon-seed-027,vae-carbon-seed-027}.%J.{out,err}`.

## Source029 fresh native C13 preparation: completed

Source archive `/work3/s234818/latentmol/releases/source-029-chain-native-prep.tar.gz`
was uploaded, extracted and SHA256 verified:
`6509aae8666092376178d696185aa8be051690796ff61c88c9b9853a440800e7`.
CPU tests **29507404 are DONE**: **12 passed, one skipped in 7.63 seconds**,
run `20260928T125320Z_tests-chain-native-029_b541af1f`. Resources were `hpc`,
two CPU slots, 4 GB per slot and 30 minutes.

Dependent native preparation **29507405 is DONE**, completed run
`20260928T125333Z_chain-native-prepare-029_279cccee`, with **zero failed native
structure/affinity examples**.
Resources are `gpua10`, one exclusive GPU, four CPU slots, 8 GB per slot and
two hours. The attempt directory is
`/work3/s234818/latentmol/boltz-preparation/source029-chain-attempt001`;
logs use `/work3/s234818/latentmol/logs/{chain-native-tests-029,chain-native-prepare-029}.%J.{out,err}`.

This release factors reusable official protein/MSA input preparation and supplies
the explicit C13 ligand to a fresh native Boltz prediction. It uses a reduced
20-step engineering pose budget, followed by the established atom/feature parity
snapshot. Static review caught and corrected the explicit `--subsample_msa`
flag before release. The saved `prepared.pt` SHA256 is
`fe91865528cfb4dbe3bfeb552d4e28b14a8ea5ea089cefe4ae4dd6f96f5d6f49`.
`prepared.json` records seed `CCCCCCCCCCCCC`, fixed occupancy **13** and
**hard-feature parity true**. Native pose SHA256 is
`87cba880b1153c0b020f5b3cb98e41a38957d2fef8978e9779f531825ab72d8c`;
input YAML SHA256 is
`420b688e06c6e9b46b52124e5925a343f13d79ac8e0d6abaa25a0bed5da47883`.
Boltz commit `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` and checkpoint repository
revision `6fdef46d763fee7fbb83ca5501ccceff43b85607` are pinned.

**No chain gradient or latent optimization has run.** Feature parity is not
derivative verification; the reduced-budget pose is engineering evidence, not
a binding observation. This context is separate from source027 encoding and
the earlier tyrosine-derived context. The complete HPC export
`/work3/s234818/latentmol/exports/chain-native-source029-results-20260928.tar.gz`
and local `outputs/hpc-results/stage2/chain-native-source029-results-20260928.tar.gz`
have matching SHA256
`b59227f85f28da06d29938dae44363a6a955b54675d129d262923d4721df9684`.
The local archive hash was rechecked.

## Observed representation preflight

Test run: `20260928T124511Z_tests-carbon-seed-027_a23f7535`.
AE run: `20260928T124528Z_ae-carbon-chain-seed-preflight_4bcb2f5b`.
VAE run: `20260928T124528Z_vae-carbon-chain-seed-preflight_fb92a698`.

| Measurement | AE | VAE |
|---|---:|---:|
| Valid and exact C13 hard decode | Yes | Yes |
| Hard / expected heavy-atom count | 13 / 13.0 | 13 / 13.0000153 |
| Latent dimensions | 128 | 128 |
| Standardized latent norm | 5.8665986 | 4.8342209 |
| PCA residual / training-reference limit | 3.0033274 / 9.6629944 | 3.4391828 / 8.4074020 |
| Ordered-graph decoder log probability | 0.0 | -0.033050686 |
| Reconstruction loss | 0.0 | 0.00035546 |
| Mean posterior standard deviation | Not applicable | 0.7488142 |

Each result concerns one explicit seed molecule. VAE decoding used the mean,
not a posterior sample. The residual limits are empirical reference diagnostics,
not a proof of chemical-domain coverage. The AE's computed zero log probability
and loss reflect the recorded finite-precision decoder output; they are not a
calibrated molecular probability or evidence of a useful local gradient.

Both models pass the initial identity gate, so latent inversion is not required
merely to obtain this chain at the tested mean. A new native chain pose/features
and chain-specific parity/gradient checks are still needed before scoring or
optimization. The tyrosine context remains distinct.

The coordinator verified the results and transferred the archive from
`/work3/s234818/latentmol/exports/carbon-seed-source027-results-20260928.tar.gz`
to `outputs/hpc-results/stage2/carbon-seed-source027-results-20260928.tar.gz`.
Matching HPC/local SHA256:
`f4219acf47dc1358bfaf07a830ab40d1596183fda2c9e9156c9044f1895f3969`.
The local archive hash was rechecked for this report.

## Declared seed and background

The user clarified that the intended molecule-level optimization starts with
a **linear 13-carbon chain**, represented here by neutral, single-bonded
`CCCCCCCCCCCCC`. Optimize a separate molecular latent variable while keeping
the chosen encoder/decoder and downstream model weights frozen.

DBMol Figure 1 visualizes its initial graph as a simple carbon chain, while
Algorithm 1 specifies a simple discrete initial molecule without fixing that
figure's exact structure as a universal initialization policy.
[DBMol, Figure 1 and Algorithm 1](https://arxiv.org/pdf/2607.19237).
Our source017 experiments instead used the official **tyrosine-derived
13-heavy-atom context**, not a carbon-chain seed. Matching atom count does not
make their molecules, poses or model inputs equivalent. This preflight specifies
C13 explicitly in response to the user's clarified initialization.

## First check: frozen AE/VAE encode and decode on HPC CPU

Use the same validation-selected checkpoints as prior stages: AE epoch 108,
SHA256 `1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95`,
and VAE epoch 94,
SHA256 `80b22b9c5a5e9010457b49246f830f0425c5578458eab28ddf2a10f5f632cee9`.
Verify their identities, graph schema and matching training-reference artifacts.
The VAE reference SHA256 is
`c811b600d1031c2a890fe6106a29ac12d5bb30f114e18926dce64f40f5f1c435`.

Convert the stated chain through the existing topology-only canonical graph
codec: 13 occupied carbon slots, zero formal charges, 12 single bonds and
remaining slots absent within the 50-slot capacity. Require exact input
SMILES/graph round-trip before encoding. Use each model's deterministic encoded
mean, including the VAE; no posterior sampling or latent update is part of this
first check. Decode once and inspect the un-repaired hard graph.

Record each model's input/output canonical SMILES, valid/exact flags and failure
reason, hard and expected atom counts, full latent mean and checkpoint/config
hashes. Report its standardized latent position using that checkpoint's existing
training-only center and scale: `(z - center) / scale`, together with its norm
and the reference artifact hash. Reuse the saved scale floor; do not refit on
this chain or compare raw AE/VAE latent magnitudes as calibrated probabilities.
This position is descriptive, not a validity or molecular-likelihood score.

The denominator is **one declared chain per model**. Exact reconstruction is the
gate for calling the initial decoded molecule a carbon chain. If a decoder
returns another valid molecule or an invalid graph, retain that outcome. A
separately specified latent inversion would then be needed: optimize `z` against
the fixed chain graph under a frozen decoder, with its own budget and hard
reconstruction gate, before claiming chain-initialized Boltz optimization.
No such inversion is submitted by this document.

## Before any chain-based Boltz gradient

Prepare a **new native Boltz prediction, ligand features and pose snapshot** for
the declared chain in the chosen protein context. Retain standardization output,
checkpoint/MSA hashes, pose seed, feature mappings and any failure. Do not reuse
the tyrosine pose or its gradients as evidence for a chain, even though both
have 13 heavy atoms. The new prepared context requires its own native-forward
parity, finite/nonzero gradient and frozen-parameter/context checks before a
latent optimization claim.

Keep chain encoding, any required inversion, native preparation and later
optimization as distinct recorded stages. This initial plan tests fixed-size
initialization only; it does not implement atom birth/death or demonstrate
binding, synthesis feasibility or DeFoG projection. The completed source027 jobs
are recorded in the ledger. The separate source029 native preparation is recorded
above; no latent inversion was needed. Source031 chain-optimization submissions
are recorded separately below and do not change the source027/source029 results.

The source031 next step is specified separately in
[the submitted carbon-anchor latent pilot](21_carbon_anchor_latent_pilot.md):
exact chain identity, an anchor-specific derivative gate, one 10-update
target-only trajectory per model and a matched-displacement random control.
CPU tests 29507438 passed 17 tests. AE 29507439 failed its anchor binary
derivative gate before any trajectory; VAE 29507440 completed 10 updates with a
valid branched C13 endpoint and a higher fixed-context hard binary score.
Native smoke 29507451 matched both hard-forward heads exactly, then failed the
AE chemical-gradient threshold before raw-affinity finite differences. These
source031 outcomes and the source032 fresh-pose endpoint follow-up are detailed
in docs21. Source032 CPU preflight passed eight tests; native GPU rescore
29507658 completed four successful predictions, but the branched endpoint's
binary score improved at only one of two pose seeds, failing native confirmation.
