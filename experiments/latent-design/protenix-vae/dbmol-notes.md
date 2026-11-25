# Frozen VAE with DBMol loss, cooling and noise

Question, 3 November 2026: does replacing the Protenix contact branch with the
existing full DBMol-style Boltz affinity/contact/confidence objective improve
random-latent design with the same structural chemistry guidance and schedules?

Fresh campaign `vae-dbmol-20261103-a01`; existing runs and checkpoints unchanged.

**Latest correction:** a01 startup29877824/callback29877825 EXIT1 before native
model evaluation because a test fixture was omitted from the source archive.
The corrected exact capsule passed3 tests; replacement **29879133 PEND/A40** and
callback **29879134 PEND** now own `vae-dbmol-20261103-a02`. No applied updates
were lost. Scientific settings are unchanged. See [the repair and Protenix arm](repeat-20261103.md).
The initial submission description below is retained as history.
Initial GPU **29877824**, A100,20-minute allocation for20 real updates; own-ended
CPU callback **29877825** measures the warm update rate and submits the largest
remaining budget that fits the queue's72-hour limit. Those20 updates count toward
the50,000 total, and continuations retain latent, fixed occupied slots and noise
RNG state. The full remainder is submitted in one allocation whenever it fits,
with30% measured runtime margin and20 minutes for setup/cleanup. Native backward
success and optimization progress are pending; scheduler submission is not a pass.

## Declared settings

- Frozen epoch45 VAE, checkpoint SHA `5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be`.
- Fresh standard-normal256D latent, seed17;25 fixed occupied slots selected once.
- LR0.3 and maximum latent displacement0.5 per update, as in the preceding VAE run.
- Temperature linearly0.7→0.4 across50k; independent noise seed230017,
  norm0.4→0 times the binding-only latent update norm by30k.
- Existing structural repair builder/fallback and detached categorical loss;
  coefficient1 through20k, geometric1→10 by30k, then10. No composition term.
- DBMol `dbmol_fixed_pose_v1`: affinity prediction minus native binder logit,
  plus2.5 contact and2.5 confidence; original confidence subweights/signs.
  Native FP32 arithmetic, zero recycles, no molecular-weight correction or diffusion.
- Original fixed Boltz count25 context SHA `84a0d35dbb565a535a18a7ef04bda5fbd7ab589c4e1ea2ac30d3e3437130e9cb`.
  All native features and reference coordinates are frozen. Its slot map/seed is
  reindexed by category identity to the decoder's schema; existing native seed
  mapping parity checks the derived view. No decoder element/bond categories are
  dropped. Native charge input is the expected formal charge, a declared loss of
  distributional information. Pocket residues are selected once from this
  authenticated initial pose and then fixed.
- Every scalar loss/gradient norm and hard graph/latent retained; full shared
  probabilities every100 updates plus endpoints. Complete output cap2GiB/10kfiles
  uses the current common exploratory storage helper, without stacked account
  forecasts. Initial admission passed13,871,370,240 available versus2,362,232,013
  required bytes, including the helper's current proportional reserve.

The new scorer uses the existing `FrozenBoltzDBMol` and continuous feature adapter.
Only latent variables change. The two native checkpoints retain separate frozen
trunks. Same shared relaxed graph and temperature feed binding and chemistry;
repair targets are never substituted into binding. Runtime metadata labels this
as the DBMol composite rather than the Protenix contact objective. Exact inputs,
archive/plan hashes and current IDs are in [the run record](dbmol-run-20261103.json).

## Commands and evidence

```text
python -m latentmol.design.latent_boltz --config full.json --checkpoint EPOCH45.pt --checkpoint-sha256 SHA --native-config native.json --output NEW_OUTPUT --max-updates 50000 --wall-seconds BUDGET
```

The cluster uses a new immutable snapshot based on the completed source157 code,
with only the new scorer, campaign/wrapper/tests and shared-loop metadata additions.
It does not adopt unrelated working-tree changes. The home controller reuses the
existing submission and node-local launch helpers. Own durable intents/receipts
block unknown-outcome retries and duplicate writers; callbacks only continue
successful finite/frozen segments. Failed evidence remains in its output.

Local checks:20 small toy adapter/latent controls and17 submission lifecycle
controls passed. The first local run had18passes and2 Windows temporary-directory
setup errors; a fresh owned pytest directory resolved them. No native model ran
locally. GPU startup includes3 focused adapter/budget controls and the actual
decoder→DBMol→latent backward path before proceeding. No FD/parity qualification
campaign is imposed. A metadata staging caller initially had a syntax error
before remote execution/submission; its receipt is preserved, then corrected.

[Submission and current scheduler evidence](../../../outputs/hpc-submissions/vae-dbmol-20261103-a01/status01.json),
[source/control seal](../../../outputs/hpc-submissions/vae-dbmol-20261103-a01/seal.json),
[local controls](../../../outputs/local-checks/vae-dbmol-20261103-a01/pytest.log).

Fixed-pose proxy improvement does not establish binding of discrete molecules.
This changes the native scorer/context relative to the preceding Protenix VAE
run; it is not a controlled comparison of numeric loss magnitudes. Preserve
validity, repair-search coverage, component losses, identity changes and runtime
as outcomes, including negative or nonmonotone results. On completion use the
existing saved-report renderer/adapter for the new output, not another renderer.
