# One optional atom in Protenix-guided latent design

Can a binding proxy guide whether an atom is present, and does that change
survive hard decoding from the frozen epoch-45 VAE?

The [runner](../../../src/latentmol/protenix_occupancy/one_atom.py) evaluates two
native ligand topologies with Protenix-Tiny. For optional-atom occupancy
\(q\), its loss is \((1-q)L_{\mathrm{deleted}}+qL_{\mathrm{full}}\). Each loss
uses the same retained protein–ligand pair targets. Both native predictions
are constant during the latent updates. This gives exact endpoints for one
Bernoulli choice and keeps every feature of the absent atom out of the deleted
forward. It tests a two-forward reference, separate from modifying Protenix's
internal attention and feature pooling to support soft occupancy in one pass.

The [DTU config](one-atom-dtu.json) pins the checkpoint and three saved BRD4
endpoint bundles. Two native calls per endpoint cover the full, compact deletion
and separately prepared full and deleted geometries. The compact repeats check
the score and distogram probabilities at \(q=0\) and \(q=1\). Fresh geometry
tests whether the compact ranking survives rebuilding the input. All scores
are retained-pair distogram losses.

The VAE starts from the encoder mean of the native full ligand. An explicit
constant offset sets the uniquely mapped halogen slot to 10% occupancy.
The run checks the initial latent directional derivative and takes up to
16 updates of length 0.125 inside a latent radius of 1.0. Each saved update
records the raw expected and hard atom counts, controlled occupancy, and raw
and occupancy-controlled SMILES or chemistry failure. The two native scores
only apply to the two declared topologies. Other raw decoder outputs are
recorded separately and receive no score attribution from those endpoints.

A soft-budget follow-up requires the repeated endpoint checks to pass and
at least one useful latent gradient agreeing with the fresh ranking. Near 25
heavy atoms, the shared optimizer now exposes an opt-in
[atom budget](../../../docs/workflows.md#frozen-model-latent-design) with a
count tolerance and a separate penalty for occupancy outside the initial
slots. The first pilot uses smaller ligands. A native budget follow-up needs
a scorer setup that supports growth and shrinkage near the target.

Entry point inside a GPU allocation

```bash
python -m latentmol.protenix_occupancy.one_atom \
  --config experiments/latent-design/protenix-vae/one-atom-dtu.json \
  --output <new-output-directory>
```

## First submission

DTU job **29890674** was submitted to `gpua10` on 4 November 2026 at
13:40 CET with one A10, four cores and a one-hour wall limit. The scheduler
reported PEND and waiting for scheduling. The pilot uses 24 native forwards
and up to 16 latent updates per case. The five local CPU fixtures passed.
GPU endpoint results and the soft-budget decision are pending. The runner
writes one JSON per case and a `result.json` with `supports_budget_followup`.

The source archive SHA-256 is
`74b6730fe1874f2dcf7e833c0660a8f2c7108e7e35a7eea323ed8bf8aea6ef0f`.
The remote output directory is
`/work3/s234818/latentmol/protenix_occupancy/runs/protenix-one-atom-20261104-a01`.

## Completed result

Job **29890674** completed with all three endpoint-equivalence and gradient
checks passing. The calibrated occupancies rose from 0.10 to 0.817, 0.812
and 0.574. Fresh geometry favored the same occupancy direction in all three.
Only HH8's controlled final decode matched the preferred endpoint. Its raw
decode already matched at the start. Neither of the other two raw decodes
matched the native endpoints. The receipt sets `supports_budget_followup`
to true. See the [count budget comparison](atom-budget.md).

## Unshifted decoder preflight

The next gradient test can use paired decoder outputs with one slot absent
or present. For a selected slot, harden the other labels normally and use the
decoder's highest-scoring real element, charge and conditional incident bonds
in the present branch. Check both structures before native scoring. Use the
unshifted occupancy to weight the two endpoint losses. Success means that a
latent step changes the unmodified hard decode toward the better freshly
scored valid endpoint.

A local CPU check on 5 November used the frozen epoch-45 checkpoint and the
existing `hard_observation` function. The previous 25-atom anchor plus 12
seed-1729 directions at each radius 1, 2, 3 and 4 yielded 46 valid decodes
among 49 points. None of the eight nearest-to-flip slots per valid decode had
occupancy in [0.02, 0.98]. In a separate seed-1730 panel of 64 standard-normal
latents, 51 hard decodes were valid. Checking up to four nearest-to-flip slots
in [0.02, 0.98] per valid graph produced 115 valid pairs in 137 attempts.
The raw decode started on the absent side for 59 pairs and on the present
side for 56. This was a chemistry and decoder feasibility check.
