# Chemistry guidance on valid relaxed molecules

> **Scope and access note (22 October 2026):** this document contains an
> experiment protocol and dated operator evidence. Concrete account paths,
> environments, jobs and source hashes describe that operator/attempt; they
> are not shared-account defaults or a fresh status/acceptance claim. Use
> [your own HPC setup](HPC_GUIDE.md). Keep the scientific gates and historical
> outcomes below intact when preparing a separately validated candidate.
> Canonical working home: [brd4/chemistry-compatibility](../experiments/brd4/chemistry-compatibility/README.md).


20 October 2026. The user asked to investigate useful chemistry gradients
on already-valid molecules and to retain **LR 0.03**. A lower 0.01 rate is a
possible future comparison; the earlier 0.1 recommendation is withdrawn.
This is research design for an allocated CPU probe of saved states, not a
new default loss, changed scientific configuration or molecular run. HPC
access failed during the single status/quota attempt; no upload or job
submission occurred. Local prototype/static-test preparation and immutable
packaging are complete below, but all numerical comparisons remain unexecuted.

## Why a valid hard graph can still need guidance

The existing failure-aware protocol sets chemistry loss and gradient to
zero on route 0, `already_valid: skip_without_self_target`. Its complete
repair-witness route uses full atom-plus-bond NLL, while local valence-risk
guidance is enabled only on fallback 3/4. A valid hard argmax can therefore
sit near a probability boundary without a chemistry gradient resisting the
next affinity step. The saved mix50 run switched validity 280 times in its
last 500 updates, and state 9999 valid became state 10000 invalid. These facts
motivate the investigation but do not prove a broken derivative or identify
which new surrogate works best.

Keep the same shared six-slot neutral graph, cached native context and
saved snapshots. Candidate losses consume relaxed atom/bond probabilities.
Discrete reference preparation stays outside differentiation, and any caps
or targets held detached must be explicit and fixed during trial/probe
evaluation. No candidate repairs the displayed molecule or replaces the
saved native score with an independently decoded identity.

## Three candidate signals

| Candidate | Valid-state signal | Gradient scope | Main limitations |
| --- | --- | --- | --- |
| Always-on hard-cap DP | Penalizes probability mass above a local valence cap even when hard argmax is valid | Bonds only; caps come from detached argmax elements | No atom gradient; cap changes at element switches; broad sulfur cap, aromatic-as-one approximation and no connectivity guarantee |
| Atom-probability-weighted local compatibility DP | Weights local compatibility by current element probabilities | Atoms and bonds; smooth for fixed vocabulary/rules | Can favor high-cap sulfur; same local/aromatic/connectivity limitations; not full chemical validity |
| Valid-target bounded margin | Keeps a detached valid target separated from competing categorical choices near its decision boundary | Targeted atom and bond categories | Can lock identity or inhibit valid changes; a confidence preference, not a chemistry law |

**Always-on hard-cap DP.** The existing local routine uses a dynamic program
over incident categorical bond orders to obtain probability mass within a
detached per-atom maximum allowed cap. Applying it on every state removes
the valid-route switch that currently suppresses it. A soft distribution
can have unsafe local mass even when its highest-probability graph is valid.
This does not imply a nonzero gradient everywhere: deep safe states may
already have little risk. Atom probabilities do not receive a gradient
through detached argmax caps, and the global function is not smooth across
those cap switches.

**Atom-probability-weighted compatibility DP.** Instead of choosing one
element cap, combine the per-element local safe masses using each atom's
current categorical probabilities. This allows both element and bond
probabilities to affect the signal without an element-argmax switch.
The exact probability mixture, logarithm and normalization must be recorded
with the probe implementation; it is a new surrogate, not source-equivalent
activation of the old fallback term.

Both local DP candidates use the existing broad neutral maximum caps
(C4/N3/O2/S6/F1/Cl1), count aromatic bonds as order 1 and ignore global
connectivity. Broad sulfur capacity may lower the loss by shifting atom
probability toward sulfur rather than making the intended bonds better.
Local compatibility does not certify aromaticity, charge consistency,
connectedness or complete RDKit sanitization. Retaining neutral charges
does not remove these limitations.

**Valid-target bounded margin.** Extract the current valid hard graph as a
detached atom/bond target and keep a declared finite margin against
competing choices. Bound the target pressure rather than always pushing it
toward certainty. This can remain informative at a valid boundary while
becoming inactive past the margin. However, a previous valid target may
penalize another legitimate valid identity or edit; refreshing it can create
new discontinuities. Record target lifetime and freeze it within each
probe/trial. This is a stability preference, not a chemical feasibility rule
or a globally nearest-valid loss.

## Saved-only allocated probe and decision gates

The planned probe compares all three on the same authenticated saved states,
including the valid/invalid endpoint neighborhood. It reads original native
gradients rather than rerunning Boltz. All new numerical loss/gradient
calculations and tests stay on allocated DTU HPC CPU; local work is editing,
static review, packaging and byte/schema verification only. No new learning
rate, coefficient schedule, optimizer step or molecular trajectory is
selected by this design.

Record candidate values, atom and bond gradient norms, directional agreement
with the saved native gradient, target/cap identity and finite-difference
agreement under frozen references. Distinguish a zero gradient expected
from hard-cap atom detachment from an implementation failure. Test safe
interior behavior as well as near-boundary valid states; do not demand
nonzero pressure everywhere or mistake a nonzero gradient for useful
chemical guidance. Preserve every failed probe and exact normalization.

Before recommending a production candidate, inspect whether it acts on the
problematic bond, whether atom gradients exploit broad caps, whether a
margin suppresses valid edits, and whether local guidance leaves disconnected
graphs unaddressed. Saved-state probes cannot establish trajectory stability,
chemical validity or binding. Any later production change needs a distinct
versioned protocol, allocated engineering/native gradient/parity/finite/
identity checks, and a matched comparison that retains LR 0.03. The original
loss, completed run and report remain unchanged; no production candidate is
selected or called passed before actual results.

## Current execution status and working hypothesis

The single SSH status/quota attempt used BatchMode=yes and ConnectTimeout=20.
It emitted the login banner, then failed with a connection timeout to
192.38.95.132 port 22. No quota result, upload or allocated job exists from
this attempt, and no retry is being made. This is an access failure, not a
failed scientific test or a numerical comparison. Local work is limited to
static prototype/test authoring and immutable packaging. The local release
is now sealed; numerical validation remains pending.

The tentative design preference is the atom-probability-weighted soft local
compatibility candidate because it can guide both atoms and bonds without
an argmax-cap switch. That is a rationale for prioritizing validation, not
measured superiority. It remains subject to allocated checks, broad-cap
sulfur bias and the local/aromatic/connectivity limitations above. LR 0.03,
the production chemistry loss and all completed experiments remain unchanged.
No new molecular optimization is submitted.

## Concrete research prototype and local release

The standalone `scripts/chemistry_compatibility_probe.py` implements
`L_local = mean_i[-log(sum_a p_i(a) * C_i(a))]`, where
`C_i(a) = P(sum of incident bond orders <= neutral cap(a))` under the
factorized categorical bond probabilities. Both sums are computed in log
space. This uses the full incident-order distribution, rather than only
its expected value. Mass above six cannot satisfy any supported cap and is
discarded. Allowing lower heavy-atom valences accommodates implicit H.
It is a local compatibility surrogate, not a whole-molecule validity probability.

Two comparators use the same saved inputs: the DP conditioned on detached
hard atom labels, and a certified-valid target margin. The latter averages
atom and unordered-pair squared hinges equally, with hinge argument
`0.2 + logsumexp(rival logits) - target logit`. It stops sharpening past that
log-odds margin. Its reference does not certify validity by itself; the
driver authenticates existing preparation identities and routes 0/1/2.

The planned driver reads saved states 0/1000/1467/8294/9639/9998/9999/10000,
checking case, snapshot and reference identities. It measures candidate
values, separate atom/bond gradients and engineering finite differences.
Saved native gradients are used only for direction comparisons and
**unaccepted** proposed perturbations at rates 0.01/0.03 and additional
coefficients 0.1/1/3. No actual Boltz scalar or new native coupled pass is
inferred from these proposals. A separately weighted, always-on compatibility
term is the working candidate; its coefficient requires measured calibration
and should not automatically inherit the repair ramp.

Eleven planned tests cover independent three-atom enumeration, atom/bond
finite differences, permutation/logit-shift invariance, hard-cap atom-gradient
absence, local counterexamples, high-cap sulfur bias and margin behavior.
Ruff, Python AST parsing and Bash syntax checks passed; independent static
review found no blocker. **No numerical tests or probes have run.**

Local source release:
`results/releases/20261020-brd4-chemistry-compatibility-probe-01/source.tar.gz`,
four source files, 8,592 bytes, SHA
`edd16a3aafa5092e2decea16f13c9ba3d9575de37edfd6d54db7c4e61f0ff489`.
Manifest SHA
`53fb385712f37412f204018e49dfb0f4c6a2a9adde14ff3e5b9e8760d70fea29`.
Receipt:
`scripts/hpc/receipts/20261020-brd4-chemistry-compatibility-probe-01/campaign.json`.
The planned CPU allocation is hpc, two slots, 4 GiB per slot, ten minutes,
zero GPUs, subject to a fresh quota check. There is no uploaded release,
submission or job ID. Production source02, default repair behavior and
LR 0.03 remain unchanged.

The rule table and aromaticity limitations should be interpreted with the
[RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences).
