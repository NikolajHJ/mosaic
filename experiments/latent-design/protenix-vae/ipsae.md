# Boltz-2 directional ipSAE pilot

Question: does a differentiable protein-frame / ligand-score ipSAE term improve
the latent design trajectory, and how does its latent gradient compare with
the existing Boltz confidence loss?

The matched baseline and ipSAE arms start from the same frozen epoch-45 VAE,
seed 17 and fixed 25-slot Boltz-2 pose. Both use the existing full DBMol
affinity/contact/confidence and structural-repair objective, including the
same noise stream. The ipSAE arm additionally subtracts 0.3 times ipSAE inside
the confidence component; the component's existing 2.5 multiplier makes the
effective total-objective coefficient -0.75. Neither term is binding evidence.

Both arms request 500 updates and learning rate 3. The existing maximum latent
step norm of 0.5 remains active, so applied steps can be capped; report cap
frequency and actual displacements. Temperature, repair and noise schedules
retain the existing 50,000-step horizon and settings rather than being rescaled
to 500 steps. All other settings and the random seeds match between arms.

Configs: [optimization](ipsae-lr3-500.json), [baseline](ipsae-baseline-native.json),
[ipSAE](ipsae-ipsae-native.json). The separate gradient probe compares the
unweighted gradients of the original confidence loss and negative ipSAE at
common latent states; its norm ratios/cosines must not be read as the weighted
contribution to the full objective. Fixed detached coordinates mean gradients
are through the confidence head and soft graph, not through pose prediction.

Allocated command: `bash experiments/latent-design/protenix-vae/ipsae-hpc.sh
ARCHIVE ARCHIVE_SHA256 NEW_OUTPUT`. It verifies the source archive and CUDA
ownership helper, then runs each arm serially on its one allocated GPU. The
launcher refuses an existing output root. The initial separate submission is
GPU job **29996335**, queue `gpuv100`, six-hour limit, source snapshot SHA256
`968618c3c72e3ae24c86841035756c75e05cfd4fa727e72aa1adb81b6cea6420`;
output root `/work3/s234818/latentmol/runs/ipsae-20261111-a02`. The LSF job
output reports successful completion; both arms and the gradient probe wrote
completed results, as summarized below.

An independent 30-minute initial-state gradient-only probe is GPU job
**29996344**, queue `gpuv100`, source SHA256
`f8edd9e1a4b9920fccc682ac4172ec29a8531146d54b2e3be62c5e8b6d06bac1`,
with output root `/work3/s234818/latentmol/runs/ipsae-initial-20261111-a01`.
It performs no optimization; its separate result is not required for the
completed trajectory comparison below.

## Completed comparison, 12 November 2026

Both arms completed 500 updates in the single allocated job. The model and
decoder frozen-state checks, finite gradient checks and same-shared-graph check
passed for both. The checkpoint loader warned that the checkpoint's Lightning
producer was slightly newer than the installed loader. The fixed pose, seed,
schedule and chemistry guidance were matched; the ipSAE arm ran first.

| Result | Baseline | With ipSAE |
| --- | ---: | ---: |
| Valid update-source states, out of 500 | 132 | 19 |
| Distinct valid update-source SMILES | 12 | 14 |
| Final state | invalid | invalid |
| Updates hitting the 0.5 step cap | 338 | 489 |
| Median actual step norm | 0.500 | 0.500 |
| Best valid state's common soft-graph DBMol loss | 5.477675, step 269 | 3.534630, step 219 |
| Final common soft-graph DBMol loss | 5.745100 | 4.003677 |

The common loss is the original fixed-pose DBMol objective, excluding repair
guidance. In the ipSAE arm we add `2.5 * 0.3 * ipSAE` back to the saved binding
loss to remove the new negative term. The initial common losses agree to about
4e-7 (10.170163). These scores are calculated from **soft probabilities**, even
when their argmax decode is valid. They are not fresh scores of hard molecules.
The stronger best-valid soft score and greater number of distinct valid
identities in this one ipSAE trajectory came with far fewer valid states; both
endpoints were invalid. They do not show a binding improvement.

On the same latent and temperature, the gradients of negative ipSAE and the
original confidence loss had cosines 0.8821 at the shared start and
0.9767–1.0000 at the four later sampled states. The **unweighted** ipSAE gradient
norm was 2.28%–10.71% of the confidence gradient norm; with the 0.3 inner
weight it was 0.685%–3.214%. The shared 2.5 confidence multiplier cancels in
that ratio. No zero gradients were observed at these five distinct states. These
local measurements show mostly aligned, smaller gradients at the sampled
points; they do not isolate their causal effect over an entire trajectory.

The matched start scored ipSAE 0.05206. The fixed-pose ipSAE was 0.05336 versus
0.10411 at step 250 (baseline versus treatment) and 0.09020 versus 0.08648 at
step 500. A single matched seed, a highly active step cap and the soft/fixed-pose
surrogate limit the conclusions. No fresh pose, affinity or experimental
binding evaluation is included.

[Compact result and six component-gradient measurements](../../../outputs/hpc-results/ipsae-20261111-a02/summary.json)
and [full gradient vectors](../../../outputs/hpc-results/ipsae-20261111-a02/gradient-comparison.json).
Eleven downloaded receipts and metric files matched their remote SHA256 hashes.
To regenerate the summary from those files, run
`python experiments/latent-design/protenix-vae/ipsae_summarize.py --root outputs/hpc-results/ipsae-20261111-a02 --output NEW_SUMMARY.json`.
