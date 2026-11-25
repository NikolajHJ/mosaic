# Fixed-count Boltz-2 carbon-chain panel results

Evidence downloaded 7 October 2026. Of 100 prescribed trajectories, 94
completed all 100 updates and six failed the original derivative gate.
Final argmax graphs were valid in 93 of the 94 completed cases. Three
completed trajectories changed a carbon to a heteroatom; elemental changes
were rare despite many different final graph identities. This establishes that frozen Boltz gradients
can move the trained VAE representation and change hard graph identity.
It does not establish useful binders or fresh-pose activity improvement.

## What was tested

The [source057 protocol](31_carbon_protein_panel.md) uses ten canonical full
human proteins and ten linear carbon-chain starts, C6–C15, for 100 cases.
Each starts from an authenticated encoded chain, optimizes only molecular
`z`, and keeps the trained VAE and Boltz-2 weights frozen. There is one
fixed-initial-mask arm per case. Count stays fixed; this panel does not test
learning atom count or compare AE against VAE.

The objective maximizes `affinity_probability_binary` with zero recycling
steps in the frozen-pose differentiable interface. Learning rate is 400,
gradient-norm cap 0.000125 and global radius 3 around the original latent.
This historical run uses the legacy clipped update and original coarse
finite-difference gate, not the later `local_v1`/Armijo policies.

At every retained state, hard graph evaluation includes argmax and eight
categorical draws under the fixed occupied slots. The headline validity and
identity counts below concern **final argmax**, not all samples or the whole
trajectory. They describe the graph inspector's conversion/sanitization;
validity alone establishes neither synthesis feasibility nor binding.

Each target is a full monomer with an explicit single-query MSA, without
templates, domain cropping, metals/cofactors or pocket constraints. The
prepared pose/features stay frozen during optimization. Atom probabilities
are conditioned on occupied real elements; formal-charge information is
lossy in this Boltz input and `allow_lossy_charge: true` is declared.
These limitations remain part of every result.

## Coverage and outcomes

| Result | Count / declared denominator |
| --- | --- |
| Requested trajectories | 100 |
| Completed 100-update trajectories | 94 / 100 |
| Original derivative-gate failures | 6 / 100 |
| Unstarted trajectories | 0 |
| Valid final argmax graphs | 93 / 94 completed; 93 / 100 requested |
| Valid final SMILES different from the start | 72 / 94 completed |
| Increased endpoint frozen-pose proxy | 84 / 94 completed |
| Completed cases with a hard noncarbon atom | 3 / 94 completed |

The only invalid completed endpoint is `mapk1-c15`. The raw report's
`changed_final_smiles: 73` also counts that missing/invalid SMILES as a
difference; `changed_final_valid_smiles: 72` is the chemically interpretable
identity-change count. Failed cases are not removed from the original
100-case denominator.

Endpoint-minus-start proxy gain across the 94 completed cases has mean
**+0.0117513**, median **+0.00835757**, and range **−0.0295000 to +0.103765**.
These are changes in the relaxed frozen-pose binary proxy, not hard-candidate
fresh-native probabilities, measured affinity, or an improvement guarantee.
Ten completed trajectories did not increase that endpoint proxy; legacy
clipping did not enforce monotonic descent.

| Target | Completed / 10 | Failed | Completed hard heteroatom cases |
| --- | --- | --- | --- |
| AKR1B1 | 9 | 1 | 0 |
| CA2 | 10 | 0 | 0 |
| DHFR | 10 | 0 | 0 |
| FKBP1A | 8 | 2 | 0 |
| GSTP1 | 10 | 0 | 1 |
| MAPK1 | 10 | 0 | 1 |
| NQO1 | 9 | 1 | 0 |
| PIM1 | 9 | 1 | 0 |
| PPIA | 10 | 0 | 1 |
| PTPN1 | 9 | 1 | 0 |

## The three elemental changes

| Case | First hard noncarbon step | Final argmax SMILES | Proxy start → endpoint |
| --- | --- | --- | --- |
| GSTP1 C6 | 48 | `CCCCCS` | 0.206463 → 0.238558 |
| PPIA C6 | 80 | `CCCCC=S` | 0.0891586 → 0.0905061 |
| MAPK1 C14 | 60 | `CCCCCCCCCCCN1CC1` | 0.0898543 → 0.140532 |

GSTP1/PPIA each replace one carbon with sulfur; MAPK1 replaces one carbon
with nitrogen. Counts remain six/six/fourteen and all three final hard
SMILES are valid. Each saved sequence has 101 observed states and no native
context switch. At the changed endpoint slot, conditional sulfur probability
is approximately 0.61964 for GSTP1 and 0.51730 for PPIA; MAPK1's nitrogen
probability is approximately 0.99895. This is decoder support for the label,
not an empirical chemical-validity or binding probability.

These changes are evidence that atom-type logits receive usable latent
gradients, even from a carbon-chain start. Their rarity and the broader
carbon-only outcomes remain important initialization-bias observations.
The panel does not contain matched random-start controls at all ten targets,
so it cannot quantify the causal effect of that initialization bias.

## Failures and what the later diagnostics establish

All six failed parents record `Conditional Boltz finite-difference gate
failed`: `fkbp1a-c06`, `pim1-c06`, `ptpn1-c06`, `nqo1-c10`, `akr1b1-c11`
and `fkbp1a-c15`. They remain failed original experiments. Subsequent local
probe diagnostics explain scale/conditioning problems in selected saved
states; they do not retroactively pass or rerun every panel failure.

[Source061/source062 diagnostics](33_step_control_diagnostics.md) passed
twelve saved-state local gradient audits and accepted twelve matched
one-step proposals. Nine retained the full proposal; three required
shrinking, rescuing worsening legacy proposals. That supports the proposed
temporary safeguard at those states, not completed long trajectories or an
identified sharp-curvature operation.

The earlier ten-prior-random-start study still has zero successful full
workflows: nine original gate failures and seed 7's completed optimization
followed by the historical rescore bookkeeping failure. Its invalid hard
endpoints and missing fresh-native confirmation remain retained. New
[48-trajectory stability tests](34_stability_campaign.md) compare selected
matched update controls and both occupancy policies; their result is not
implied by this carbon-only panel.

## Provenance and remaining activity check

Downloaded evidence is
`outputs/hpc-results/status-20261007T073227Z/results.tar.gz`, 617 files,
locally verified SHA-256
`3f92ed546076700dc2f65dc32a7e6d5b8b0e568e6aa7ed01f78d2940d6ad54ae`.
Within its extracted sibling root, `runs/source057-protein-panel/` contains
`identity.json`, `cases.json`, final `summary.json`, parent metadata and
arm/start/gate summaries. The concise computed report is
`submissions/source063-stability-attempt001/panel-results-20261007.json`.
`results.identity.json` records hashes for the collected artifacts.

Source057 archive SHA-256 is
`9ab44499a73d37203df48c6280f1cc9d6b4cd55849e26d4e8dc2b105193ccb37`;
scientific source-manifest SHA-256 is
`81ebf49ec2e8ac918d1e516d202c1a4cf909058fef3211f75dbe87419e39b674`;
protocol SHA-256 is
`41694234e8a5af4c3a5fab8dd57ddaeab0e4b88ef5baea5e3283666802568c54`.
Frozen VAE decoder checkpoint SHA-256 is
`1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.
Original controller allocation metadata records completion at 20:44:48
DTU-local on 6 October, with authenticated per-case source/config identity.
All 94 completed parent summaries report frozen state unchanged, and all
94 arm receipts confirm 100 updates with zero context switches.
The queued panel continuation and fresh-native follow-ups are recorded in
the [ledger](05_experiment_ledger.md); completion of that controller does
not imply those separate inference stages ran.

Fresh-native start/trajectory/endpoint evaluation remains the next activity
check, using audited hard identity and both affinity heads under declared
pose seeds. This report has no new fresh-native panel predictions. No
experimental binding, synthesis or selectivity conclusion is supported yet.

The new 48-trajectory stability workflow has started its engineering/native
checks, not its full campaign at the **09:40:55 DTU-local 7 October** snapshot.
The initial source063 CPU fixture failure (220 passed/two failed) is retained;
its blocked GPU stages were cancelled without execution. Source064 fixes
only the malformed test helper. Retry **29619051 DONE** passed 222 fixtures,
native smoke **29619052 RUN**, primary **29619053 PEND** and continuation
**29619054 PEND** under smoke-success guards. Those stages use the same
scientific 48-case protocol and frozen models; no new training or non-Boltz
scoring branch was added. The [ledger](05_experiment_ledger.md) contains
downloaded receipts and exact immutable identities. A running smoke or passed
CPU suite is not evidence of completed optimization or improved binding.
