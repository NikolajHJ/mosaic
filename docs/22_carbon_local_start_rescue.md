# Nearby exact-C13 latent starts: matched AE/VAE feasibility protocol

Dated 28 September 2026. Source033 is frozen and its CPU preflight passed.
**The AE failed its bounded local-start gate; the VAE completed its trajectory.**
The VAE produced the same hard molecule as source031, with no new native evidence. This is
a separate, bounded experiment after the source031 exact-anchor AE derivative
failure. It preserves that failure and the source032 VAE endpoint's failed
two-seed native confirmation; it does not retrospectively replace either result.
See [docs21](21_carbon_anchor_latent_pilot.md).

## Release, passed preflight and GPU status

Local archive: `outputs/source-releases/latentmol-source-033-carbon-local-start-rescue.tar.gz`.
HPC archive: `/work3/s234818/latentmol/releases/source-033-carbon-local-start-rescue.tar.gz`.
Matching SHA256:
`3d8d0441aee468cba4532d32f07ccc46d5a00dbef33980dbf8e58c3af5ba5bd5`.
Static modularity review passed.

CPU job **29507713 is DONE** on `hpc`, with two CPU slots, declared 4 GB
memory request and a 30-minute limit. It runs the selected carbon-start-rescue,
latent-pilot and adapter tests. Logs:
`/work3/s234818/latentmol/logs/tests-carbon-local-start-033.%J.{out,err}`.
**22 tests passed in 15.31 seconds**; run
`20260928T133633Z_tests-carbon-local-start-033_5578427a` completed with the expected
source033 archive SHA256 recorded in its manifest.

AE **29507718 exited with code 1** and VAE **29507719 is DONE** on `gpua10`, each with one
exclusive GPU, four CPU slots, 8 GB per slot and a two-hour limit. Both use
the same source033, prepared C13 context, affinity checkpoint and data manifest,
with model-specific selected checkpoint/reference. Logs use
`/work3/s234818/latentmol/logs/{ae,vae}-carbon-local-start-033.%J.{out,err}`.
The distinct AE/VAE outcomes are recorded below. CPU test success alone does
not establish a local-start gradient or trajectory result.
Numerical work runs only on DTU HPC.

## Observed AE bounded failure

Run `20260928T134642Z_ae-boltz-carbon-local-start-rescue_d04081b2` records
**`failed_feasibility_gate`**. All **32 nonanchor candidates** retained exact
native C13 arrays in both ordinary and fixed-occupancy interpretations. The
four per-radius finalists had finite, nonzero binary-objective gradient norms:

| Radius | Gradient norm | FD attempted? | Observed result |
|---|---:|---|---|
| 0.25 | 2.4256e-11 | Yes | All five numerical differences zero; no scale agrees |
| 0.5 | 2.5804e-10 | Yes | All five numerical differences zero; no scale agrees |
| 1.0 | 1.2506e-9 | No | Screened gradient only |
| 2.0 | 5.3058e-8 | No | Screened gradient only |

The fixed protocol tests the nearest two viable finalists, so radii 1 and 2
were **not** FD-tested despite their larger analytic gradients. The selected
start is null and **no target trajectory or random endpoint ran**. This outcome
does not prove that the AE has no viable nearby point; it says the predeclared
bounded selection/gate did not find one. Do not silently reinterpret larger
screened gradients as validated or replace the failed candidates with extra
attempts. All failed gate records remain part of the paired result.

The immutable `candidate_audit.json` also exposes the decoder response directly:

| Radius | Maximum secant response: active probability L2 / latent distance | Finalist combined active probability change |
|---|---:|---:|
| 0.25 | 2.23785e-9 | approximately 5.59e-10 |
| 0.5 | 4.33877e-9 | approximately 2.17e-9 |
| 1.0 | 1.86848e-8 | approximately 1.868e-8 |
| 2.0 | 5.25242e-7 | approximately 1.050e-6 |

Almost all measured change is in element probabilities; charge changes are
around `1e-22` and bond changes around `1e-11`. Ordinary expected atom count
remains near 13. These finite-displacement, decoder-only measurements directly
support a **locally near-flat AE decoder response around the C13 anchor**,
beyond the earlier downstream logit-gradient evidence. They are not Jacobian
estimates at every point and do not establish global incapability, absence of
other useful latent starts or failure for other molecules.

## Observed VAE trajectory: same hard endpoint as source031

Run `20260928T134848Z_vae-boltz-carbon-local-start-rescue_5700cf7b` completed.
**31/32 nonanchor candidates** passed both exact-C13 interpretations and all
four radii supplied finalists. The nearest finalist, **r0.25-d7**, had binary
gradient norm **0.00049346866**. It passed as the **first FD candidate**: relative
errors at epsilon 0.001/0.003/0.01 were **0.00403/0.01391/0.000487**, approximately
**0.403%/1.391%/0.0487%**, within the declared 2% threshold. One additional FD
candidate was therefore not used.

The selected start still hardened to the exact C13 chain. After **10 updates**,
the valid 13-carbon target was **`CCCCCCCCCCC(C)C`**; the equal-displacement random
endpoint remained **`CCCCCCCCCCCCC`**. The run used **31 scorer forwards and
14 backwards**, with frozen parameters and context confirmed unchanged.

| Measurement | Observed VAE result |
|---|---:|
| Encoded-chain fixed-context hard binary probability | 0.079036884 |
| Target fixed-context hard binary probability | 0.0926051587 |
| Target relaxed binary probability | 0.09267884 |
| Random relaxed binary probability | 0.07910313 |
| Target standardized distance from selected start | 0.87873 |
| Target standardized distance from encoded anchor | 0.94625 |

The local-start trajectory found **the same hard target identity as source031**.
The baseline, target and native prediction settings match source032's completed
fresh-pose audit, which favored baseline at seed 17 and target at seed 23 and
**failed the two-seed confirmation**. That evidence already applies to these
same inputs/settings; do not rerun identical predictions or count this result
as independent native improvement. The new evidence is the VAE local-start
selection and trajectory behavior, not a new molecule or native replicate.

The AE failure remains in the comparison denominator. A VAE pass on this single
chain/context does not establish general AE/VAE superiority or binding utility.
The complete source033 evidence bundle preserves the CPU run/log/XML, AE
failure, VAE completion, both GPU logs and source archive. HPC export
`/work3/s234818/latentmol/exports/carbon-local-start-source033-results-20260928.tar.gz`
and local `outputs/hpc-results/stage2/carbon-local-start-source033-results-20260928.tar.gz`
have matching SHA256
`f6d9ef892f8a4915b56f7a29bfcd2dbef72662730c709ae5f3f9bc733b5b22d5`.
The local archive hash was rechecked.

## Question and shared inputs

Does either model have a nearby latent coordinate that still decodes to the
**exact original linear C13 graph**, but provides a usable local binary Boltz
gradient? Moving within the same hard-decoding region changes the relaxed
representation; it is not yet a molecule change. Any later trajectory begins
from the unchanged hard chain, while its soft probabilities may differ from
those at the original encoded mean.

Use the validation-selected AE epoch 108 and VAE epoch 94 checkpoints and their
separate frozen training-only latent references. Verify checkpoint, mode,
dataset/split and graph-schema metadata and the 215,000 training-reference
count. Use the source029 native C13 context with prepared SHA256
`fe91865528cfb4dbe3bfeb552d4e28b14a8ea5ea089cefe4ae4dd6f96f5d6f49`.
The native seed must be `CCCCCCCCCCCCC`. The original reference anchor is the
deterministic encoded mean, including for the VAE, in the existing standardized
latent coordinates. Do not refit normalization on this molecule.

Encoder, decoder and Boltz parameters remain frozen. Boltz uses the same native
pose, reference/MSA features, explicit ligand maps, **13 fixed occupied slots**
and zero trunk recycles. Atom birth/death and new pose generation are outside
this experiment. Shared numerical direction seeds are a matched policy, not
evidence that AE and VAE coordinates have identical geometry.

## Decoder-only candidate search: 33 points per model

Inspect the anchor plus eight unit directions at each standardized radius
**0.25, 0.5, 1.0 and 2.0**. Direction seeds are **33000–33007**, reused at every
radius and for both models. This gives **33 decoder-only candidate evaluations
within the search**, not 33 total decoder calls including scoring and optimization.
Retain every candidate and its exact latent tensor, including rejected points.

An eligible candidate must reproduce all native **slot-ordered atom, charge and
bond labels, including padding**, under both ordinary decoder hardening and
the fixed-occupancy interpretation used by the Boltz adapter. Matching SMILES
alone is insufficient; agreement only after overriding occupancy is also
insufficient. Log both per-field equality audits and ordinary expected atom count.

Choose at most **one eligible finalist per radius** using a fixed decoder-only
criterion: the L2 change in active element probabilities, active charge
probabilities and active unordered-pair bond probabilities, combined and divided
by standardized distance from the anchor. Count each undirected bond once;
exclude NO_ATOM from the fixed active element vector. Ties use increasing
direction index. This is a finite-displacement **secant response**, not a local
Jacobian, chemical-validity score or Boltz preference. Selection at this stage
does not consult Boltz values or gradients. A radius with no eligible candidate
remains an empty screening slot; do not draw replacements.

## Four gradient screens, at most two finite-difference candidates

Screen the **up to four finalists** with one frozen Boltz binary-objective
forward/backward each. Require finite gradient entries, a finite norm and norm
greater than **1e-12**. Record scores and norms, but do not rank candidates by
predicted affinity or gradient magnitude.

Process viable finalists in increasing radius. Attempt finite differences on
at most the **first two viable candidates**, stopping at the first pass. Use
the negative binary probability objective and its normalized analytic-gradient
direction. Central-difference epsilon values are **0.0001, 0.0003, 0.001, 0.003
and 0.01**; require **at least two distinct scales within 2% relative error**.
Persist every scale and each unsuccessful candidate. Selecting the first
passing nearby candidate is a derivative-feasibility screen, not a high-score
candidate search.

No exact finalist, no finite nonzero finalist gradient, or no passing FD candidate
produces a retained failed-feasibility summary and stops before any trajectory.
Do not silently expand directions, radii, FD candidates or thresholds for one
model. A passing point does not show that the exact encoded anchor passed.

## One target trajectory and a movement control

If a start passes, run **one target-only trajectory of 10 updates** from that
point. Use normalized gradient steps of **0.25**, with total movement bounded
by **2.5 from the selected start**. Record distance from both selected start and
original anchor. Consequently the anchor distance need not remain below 2.5;
the selected radius and optimization displacement must both be reported.

Use the existing target trajectory and hard-score machinery. Save all 11 states
and inspect the un-repaired hard graphs. Compare endpoint scores with both the
selected start and original encoded-chain baseline. Both hard controls are
exact C13, but their relaxed scores may differ. A hard improvement requires a
valid changed molecule and the existing fixed-context binary improvement
greater than `1e-4` versus both controls; report the individual components.

The random control uses seed **39000**, starts at the same selected latent and
matches that model's target displacement norm. It has no gradient updates and
is a movement control, not an equal-oracle-cost control. Keep invalid, unchanged
and duplicate outcomes. No decoder repair or best-of-many endpoint selection
is added after the declared trajectory.

## Denominators, cost and durable records

Per model, report all **33 candidate points**, **32 nonanchor attempts**, four
radius slots, actual finalists/screens, at most two FD candidates, and either
one attempted trajectory plus one random endpoint or the failed gate that
prevented them. Never compare only the subset of models or starts that passed.
Both models receive the same caps; actual use may differ after rejection or
early success. The declared fixed-context scorer upper bound is **41 forwards
and 14 backwards per model**, including baseline, screens, both FD candidates,
the trajectory and random control. Record actual per-purpose calls and wall
time; invalid hard graphs may reduce calls. This search adds compute beyond
source031, so any improvement cannot be presented as a same-budget rerun of
that earlier experiment.

Save source/config/checkpoint/reference/context/data hashes, baseline reports,
`candidate_audit.json`, `candidate_latents.pt`, `decoder_finalists.json`,
`screen_slots.json`, `gradient_screens.json`, `fd_candidates.json`,
`selection.json`, trajectory/random tensors, `final_attempts.json`,
`oracle_cost.json`, summaries and errors. Completed trajectories must retain
the parameter/context equality checks; a failure before those checks is not
evidence they passed. Candidate and call journals preserve partial work.

Any changed hard endpoint still requires a separately declared fresh native
rescore against controls, with standardization identity checks and paired pose
seeds. Source032 showed why a fixed-context gain alone is insufficient. This
protocol tests local start feasibility only; it establishes no binding,
synthesis, selectivity, DeFoG compatibility or general AE/VAE advantage.
