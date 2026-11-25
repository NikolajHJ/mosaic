# Alternating DeFoG repair references and decoder guidance

**Latest scoped validation, 10 October 11:19 DTU-local:** source085's A10
repair-only replay completed and passed: three of four projected charged
outputs were eligible, both charged priors had a selected target, and
their frozen-target decoder-NLL gates passed. Neutral strict/projected
outputs had exact native-label parity. All accepted targets were neutral;
one other output was correctly rejected for selenium outside the schema.
This validates the opt-in lossy charge view and explicit rule-assigned
target charges on saved requests, without Boltz calls or latent updates.
It does not establish full-run or binding improvement. See the verified
[ledger result](05_experiment_ledger.md) and [diagnosis](48_defog_comprehensive_diagnosis.md).

**Historical investigation, earlier 10 October:** source079 subsequently corrected the
native frame/diagonal boundary mistakes and passed its six-update native
repair/gradient gate. All ten long cases then failed through runtime caps
or later guards; source080 reproduced three Boltz FD mismatches without
establishing their numerical cause. See the
[comprehensive diagnosis](48_defog_comprehensive_diagnosis.md) and
[ledger](05_experiment_ledger.md); short-gate success is not long-run efficacy.

**Historical native contract failures after successful startup, 8 October 10:30
DTU-local:** source076 native **29634091 EXIT**. Its repeated A100 CUDA
coexistence probe passed, but the scientific gate failed
`Unsupported native confidence frame shape or coordinates`; all four DeFoG
attempts separately failed native continuation with
`Native transition changed count/shape/symmetry or emitted nonfinite data`.
**The ten long starts did not launch.** Causes of these native-contract
failures remain unconfirmed; no new fix or rerun is established. Source076's
161 engineering checks and allocated no-MPS/MPS startup evidence remain
valid, without implying native repair, a passed guided-gradient gate or
chemistry success. Failure metadata and passed A100 startup are now preserved
in the verified [source077 evidence capsule](../outputs/hpc-results/source077-dense-chemistry/evidence001.tar.gz).
See [docs43](43_defog_gpu_process_recovery.md) and the ledger for provenance.
The failed source069 run and source068 engineering snapshot remain history.

Historical status, 7 October 2026, 12:03:39 DTU-local: source068 CPU **29620811 DONE**,
with **121 engineering tests passed in 18.37 seconds** and two six-update
C13 baseline/guidance configs prepared at weights 0/0.01. Both retain the
isolated `env/defog311/bin/python` launcher; a standard-library-only HPC
metadata probe confirms its runtime environment. Source066's fixture failure
and source067's passed engineering but superseded native configs are retained
in the ledger. **No native DeFoG guidance job was submitted or run**; no new
native repaired-target, chemistry or binding improvement is established.
The implementation adds a
frozen external DeFoG repair operation between differentiable latent updates,
then uses a fixed repaired graph as a categorical cross-entropy reference for
the existing decoder. Existing source064/source065 experiments remain
immutable controls.

Current source068 archive SHA-256 is
`c2b4855b63ad692a4850981269cea0cb6dc79dc718612a4fb49121f78375469a`,
manifest `55d6870823067cf553e4d09c1df2f6116120187c8418ba964f437e9610e8e225`.
Prepared/receipt roots are
`/work3/s234818/latentmol/submissions/source068-defog-repair-attempt003/{prepared,receipts}`.
The [completed engineering evidence](../outputs/hpc-results/source068-defog-repair-guidance/engineering-evidence.tar.gz)
SHA-256 is `1bd8560c0db7af642ce29b9e4267bb8dfad2efbfc6687a056355f6def83f5b9d`.
[Prepared comparison](../outputs/hpc-results/source068-defog-repair-guidance/extracted/submissions/source068-defog-repair-attempt003/prepared/comparison.json)
SHA-256 is `73546234354eff165339a0d06d49f3d7c94459dbd96e767ccde6f1339847abc1`;
weight-zero/weight-0.01 config SHA-256 values are respectively
`b1bdcf5ecc46f357b7ca54e53b28776e3edbd383705ee563f1b1ededcae7e99b`
and `f3ce948e58df635438c283f766f84a0fc2f1b98b97b9fcc824fe25bd30f6f3b5`.
The ledger records exact HPC commands, environments, resource use and claim
limits. These authenticated configs are prepared for the later native gate;
the guard correction and fixtures do not pass that gate themselves.

Historical source066/source067 evidence follows; its identities must not be
mixed with the current source068 configuration:

Archive `outputs/source-releases/source-066-defog-repair-guidance-attempt001.tar.gz`
has SHA-256 `e126c24060a8f66eae45be9f809d74d2e495be98a11615cd38c684300567f57c`;
scientific manifest is
`8ed0e416b5acaf49cb8c0013d734c43d102fcdd326abba79f82a4f5434d9896a`.
Base source065 `carbon-c13-dbmol-fixed_initial_mask.yaml` SHA is
`142a3d3e08538271462b1e01c6da94b18f73b7d0064f90edda958331d112dd0c`.
The prepared/receipt roots are
`/work3/s234818/latentmol/submissions/source066-defog-repair-attempt001/{prepared,receipts}`.
Source066 did not produce a prepared comparison. Its failed fixture logs
and source067's completed engineering/preparation evidence are downloaded
in [this 22-file bundle](../outputs/hpc-results/source067-defog-repair-guidance/engineering-evidence.tar.gz),
SHA-256 `e51f4024010ac084a9a51acba19d99199a88911701efaeef914ba54b3c23a274`.
Source067 archive SHA-256 is
`d8682b90016cd0d6e449e07e472510dd6fc9ef56b0cfd5fca4baf2f0d9e1f7e6`,
manifest `ba6a3e5ea5f91cce69922185e480f79203154366d2f7227e983c97cd25a7808a`;
its prepared root is
`/work3/s234818/latentmol/submissions/source067-defog-repair-attempt002/prepared`.
[Comparison metadata](../outputs/hpc-results/source067-defog-repair-guidance/extracted/submissions/source067-defog-repair-attempt002/prepared/comparison.json)
SHA-256 is `9f0724867b0f5b68c1ae63356d2d78416ef7548e0a1cdac85efc87a2eceee4df`.
Its authenticated identities remain historical and its native configs are
superseded by source068;
choosing the composite base does not pass its pending native model gates.
The [ledger](05_experiment_ledger.md) preserves source/submission history and
the pre-submission login bootstrap incident.

This is a different derivative path from the failed historical
[soft-DeFoG interface](18_stochastic_defog_projection_protocol.md). It does
not require differentiating DeFoG's categorical sampler or its hard graph
auxiliary features. It does require validating that the new hard-input
candidate continuation and exact output mapping work; the earlier loader
audit did not establish a working projector.

## One current molecule and one constant repair reference

```text
current z -> frozen decoder -> one shared relaxed graph
  -> frozen Boltz objective -> gradient to z

refresh boundary, without autograd:
  detached graph -> prescribed hard draw -> exact DeFoG codec
    -> frozen native late-time continuation -> hard output + identity audit
    -> compatible, valid, slot-aligned repaired target held constant

inner differentiable updates:
  current decoder distributions -> CE against that constant target -> gradient to z
```

Boltz always receives the **current shared graph**. The repaired graph is an
explicit guidance reference, not a secretly substituted molecular input to
Boltz or an independently decoded molecule claimed to represent the same
current state. Record both identities and the edits between them. A later
native score of a repaired graph is a separately labeled candidate evaluation,
not the score of its unrepaired parent.

Decoder and external model parameters remain frozen; only molecular `z`
changes. Draws, DeFoG calls, chemistry conversion, repair eligibility and
target selection run outside autograd. The reference is detached and fixed
through its declared inner-update block. The guidance derivative comes from
the decoder probabilities/logits against those labels. There is **no gradient
through target refresh** or categorical sampling, and no reparameterized,
straight-through or exact expected-repair gradient is claimed.

## Official checkpoint support and exact identity mapping

The draft hard provider is
[candidate_repair.py](../src/latentmol/defog/candidate_repair.py), using the
existing exact codec and newly added `from_guacamol_labels` inverse. Its
versioned JSON request/receipt includes schema fingerprint, exact candidate
labels, occupied slots, RNG seed, native source/checkpoint identity and
sampling policy. Request hashing excludes only its own digest field. Returned
targets stay in full-capacity canonical labels; RDKit inspection does not
round-trip through a reordered canonical SMILES. Rejected outputs retain
native labels and failure evidence while only an accepted target populates
the actual target field.

Use the cached official GuacaMol DeFoG checkpoint only with authenticated
external source, checkpoint and native inference settings. Historical pinned
source commit is `365bda9affadd5c2307014a0532ddaa244399441`; checkpoint SHA-256 is
`bc97d52ccf33a20d991f98349a1d37662273ab1c3e3f9777b938f9f144bc61f5`.
Its audited restoration allowed only four explicit sampling-metric buffers
to be omitted and checked all 696 native state tensors. A new backend still
needs its own actual load/frozen-state receipt rather than inheriting success
from that historical audit.

[The existing exact hard codec](../src/latentmol/defog/codec.py) supports a
**neutral, fixed-occupancy subset**, with native heavy-element order
`C, N, O, F, B, Br, Cl, I, P, S, Se, Si` and bonds
`NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC`. The canonical graph currently has
`C, N, O, F, P, S, Cl, Br, I`, categorical charges and `NO_ATOM`; these are
not identical vocabularies. The public model has no formal-charge channel or
absent-atom category; this does **not** establish that its training data were
all neutral. The exact hard-input codec starts with zero diagonal/masked edge
feature vectors. After native categorical sampling, an occupied diagonal is
one-hot `NONE`, while absent padding stays zero; collapse represents diagonal
`NONE` with label zero. Input initialization and returned transition tensors
have distinct conventions, and both must be validated at the correct boundary.

Under the **strict v1/default**, remove only explicit absent padding and carry the ordered original occupied
slots into DeFoG. Reject nonzero charges, unsupported labels, malformed
symmetry/diagonals, empty graphs and inconsistent padding. **Do not neutralize
charged inputs, reinterpret absorbing noise as NO_ATOM, kekulize or silently
alter an element to fit.** A schema-compatible but chemically invalid hard
graph can be a repair input; chemical invalidity is distinct from unsupported
vocabulary or charge semantics.

Output mapping must be equally exact. Retain native node order and return
labels to the same occupied canonical slots, with unchanged node count and
neutral absent slots. Reject output elements missing from the selected
decoder vocabulary, nonrepresentable bond types, malformed labels or a
conversion that needs chemistry/charge coercion. A checkpoint's twelve
element classes do not authorize adding untrained output categories to the
decoder. Neutral output labels are the strict adapter's charge-preserving
interpretation of a model that does not output formal charges, not a proof
that the public model was trained only on neutral chemistry. Native molecule
postprocessing can infer formal charges or remove fragments; this adapter
does neither silently. It does not extend support to arbitrary charged input.

The initial guidance experiment fixes occupied slot identities/count. Do not
align graphs by a favorable permutation, remove fragments or score a different
mask. Refresh-time and inner-step evidence must preserve the same slot map.
Variable occupancy needs a separate policy for target invalidation/alignment
and is not established by this first fixed-count path.
The native worker also checks the actual fixed count against positive support
in the checkpoint's node-count distribution; tensor shape compatibility is
not permission to extrapolate to an unsupported size. Such exclusions remain
attempt records rather than triggering a new count or replacement molecule.

## Optional lossy formal-charge model view: conversion gate passed; inference pending

Source085 implements the separately versioned opt-in policy
`project_input_native_output_v1` in request version 2. Static reviews passed;
attempt001 CPU validation failed in two new request fixtures, and the
corrected immutable attempt002 has **CPU 29670553 DONE**, 124 fixture passes
and actual native-builder parity 11/11, with zero inference. Native
**29670554 PEND**; candidate repair and target gradients remain untested. It
preserves the charged canonical hard candidate and the relaxed Boltz graph,
but projects the candidate to the public model's atom-symbol/bond features,
omitting formal charge. This is declared feature loss, not silent canonical
neutralization. Full candidate and projected native-feature digests and
lost-charge metadata must accompany every request. The strict v1/default
above remains unchanged, including its historical charged-input rejections.

Pinned public GuacaMol training input represents atoms and bonds without
formal charge. A charge-aware checkpoint is therefore not indispensable for
this deliberately lossy view. It remains necessary to reproduce the distinct
DBMol charge-aware model faithfully. The [native GuacaMol dataset](https://raw.githubusercontent.com/manuelmlmadeira/DeFoG/365bda9affadd5c2307014a0532ddaa244399441/src/datasets/guacamol_dataset.py)
and [partial-charge builder](https://raw.githubusercontent.com/manuelmlmadeira/DeFoG/365bda9affadd5c2307014a0532ddaa244399441/src/analysis/rdkit_functions.py)
define the relevant public input and deterministic output conventions.

After a native continuation, deterministic native-style conversion can
assign +1 to N/O/S where the rule requires it. These target charges are
rule-derived, **not predicted by the neural model**, and need explicit
conversion provenance. Exact native atom/bond labels, occupied-slot order
and atom count must survive conversion. Fragment selection, reindexing,
bond repair and element substitution are forbidden. Exact native atom,
bond and assigned charge labels must also survive RDKit sanitization,
otherwise reject the output: sanitization is not permission for a silent
identity rewrite. Inspect the full
charged target for representability, connectivity and sanitization. A
rejected target is retained, without resampling or a quiet neutral fallback.

All projection, DeFoG calls, rule-based charge assignment and target
eligibility remain outside autograd. A returned target may differ from its
candidate and becomes only the detached guidance reference; Boltz continues
to score the original shared graph. Target NLL includes the assigned
categorical charge labels, with the same fixed-count graph policy and
decoder-to-z gradient boundary. No derivative through the omitted feature
or charge-inference rule is claimed.

Attempt001 is sealed but failed its CPU gate, not numerically validated.
Source085 validation is CPU
engineering checks followed by native repair-only replay of saved prior-8/9
requests and a neutral control, with input loss, exact identity, inferred
charges, eligibility and rejected-output counts recorded. Attempt001 reached
no completed CPU gate or successful native result. CPU **29670540 EXIT**
recorded 122 passes/two request-fixture failures; native **29670541** was
canceled while PEND. Builder parity did not run; the fixture-only retry is
submitted with the same policy/plan. **CPU 29670553 DONE** now has 124
fixture passes and actual native-builder parity 11/11; **native 29670554
PEND** with its dependency fulfilled. These prove conversion engineering,
not sampler repair or target-FD success. The widening follows the
[declared lossy-view contract](23_frozen_model_plugin_contract.md), not an
independently decoded scorer molecule, and does not relabel old rejections.

### Paired source085 validation protocol

[Plan](../outputs/hpc-submissions/source085-charge-repair-plan.json) SHA
`000074348f92aaf61953e8d2fbbc7b7e19be932bf580251eb39017065193a6e1`
pins exact saved requests: neutral prior 0/attempt 0, prior 8/attempts 0–1
and prior 9/attempts 0–1. Each runs under strict and projected policies.
The ten fixed worker requests comprise four expected pre-inference strict
charge rejections and six supported native continuations, each expected to
record 447 transitions on the original 1,000-boundary, t0=0.8 schedule.
All attempts are retained; there is no resampling, Boltz scoring or update.

The [CPU wrapper](../scripts/hpc/defog_charge_cpu_gate.sh) runs seven fixture
files then the [actual native-builder parity](../src/latentmol/defog/charge_policy_diagnostic.py)
on 11 prescribed cases, using no neural inference. That completed,
same-release proof is mandatory for the [repair-only replay](../src/latentmol/defog/charge_repair_diagnostic.py).
Planned GPU allocation is 90 minutes on L40S, four cores/8 GB per core,
one GPU without MPS; the parent remains CPU-only and makes no Boltz calls.
Each worker has 600 seconds, and the diagnostic wall budget is 4,500 seconds.

The neutral pair must preserve native-label parity. Each charged prior must
have at least one eligible projected target within its two fixed attempts;
first eligible selects the reference without ranking or a refill. Target-only
NLL FD then uses the authenticated original decoder/latent on CPU, with
unchanged local_v1 seed 1729, rtol 0.05/atol 1e-6, nine scales, gradient
plus three preselected random directions and two neighboring informative
passing scales per direction. That validates the detached-target gradient
only, not the full binding/combined objective or optimization readiness.
The attempt002 CPU/conversion gate passed, but native candidate inference
and target-only FD results are not yet available.
Historical attempt001 resource/identity snapshot: CPU **29670540** requested
30 minutes/four cores/8 GB per core; native
**29670541** requires `done(29670540)`, with the 90-minute L40S bounds above.
Release is `/work3/s234818/latentmol/releases/source-085-charge-interface-attempt001`;
receipts are under `/work3/s234818/latentmol/submissions/source085-charge-interface-attempt001/receipts`.
Archive/manifest are respectively
`574bcf57f7003ac257c4fbd27a6e2aa06b724618878130a047c3e06b1071d709` /
`d82711eda633981e3b30218839a395021509d00248c7d29932f37cf6fcd84928`.
The ledger preserves exact commands, static environments and earlier failures.
Current attempt002 archive/manifest are
`6e99a276bd89ba9cf2a1d3971987cb5382586ce10d213b11e3766c8c690f03c3` /
`2640110a883f35cfb42c541d239e87049892d9d0fae05e547d2247e5c3f089c8`;
441 safe members authenticated. Only the replay request fixture changed,
as verified by byte-only archive comparison; production source/plan is
identical. Current receipts are under
`/work3/s234818/latentmol/submissions/source085-charge-interface-attempt002`.

Verified [attempt002 evidence](../outputs/hpc-results/source085-charge-interface-attempt002-001.tar.gz)
and [native parity summary](../outputs/hpc-results/source085-charge-interface-attempt002-001/runs/source-085-charge-interface-attempt002-parity/20261009T235405Z_defog-native-charge-rule-parity_faa15623/summary.json)
record 124 tests in 21.14 seconds and 11/11 actual native-builder cases,
model calls 0/sampled repairs 0. The parity proof includes sulfur with
three single bonds staying neutral, no unsupported negative or
hydrogen-dependent charge inference, and preserved aromatic/permuted
labels without invalid-carbon correction or fragment cropping. It uses
RDKit 2024.09.6 and the pinned native builder; exact source/environment
hashes and the failed first attempt remain in the ledger. This conversion
proof does not establish valid targets from real neural continuations.

## Candidate-start native continuation

The provisional policy is **`direct_candidate_no_corruption`**: insert the
supplied hard candidate directly as the categorical state, without applying
another forward-corruption kernel. Requested **`t0=0.8` is actual distorted
model time**, not fraction 0.8 of the grid. Use the pinned native 1,000-step
grid/schedule and start at its first actual boundary at or above 0.8. Record
that effective time, starting grid index and actual suffix-transition count;
with native `polydec` this must not be called an assumed 200-step suffix.
Stochastic parameters and source-specific transition settings remain pinned
backend protocol fields. This declared insertion is a heuristic use of a
candidate at late time, not a claim that it was drawn from the native marginal
distribution there.

The historical `sample_batch` API starts from prior noise at time zero; it
does not itself accept a candidate or partial starting time. Candidate repair
therefore needs an explicit wrapper around the checkpoint's native hard
transition implementation. All such operations are nondifferentiable oracle
work. A successful final valid graph is not proof that it is nearest to its
input, and continuation is not an identity-preserving or validity-guaranteed
projection. This public checkpoint with unmodeled formal charges is not
DBMol's separately trained charge-aware adapted DeFoG.

The audited pinned `polydec` schedule is **f(u) = 2u - u²**. Requested model
time 0.8 starts at uniform-grid index **553**, effective model time
**0.8001910448074341**, and executes **447 native transitions**. Native
receipts confirm this suffix; 0.8 does not mean executing only 200 of the
1,000 uniform-grid steps. The direct-candidate insertion remains a declared
heuristic, not an assertion of the model's native marginal distribution.
See the pinned [native time distorter](https://raw.githubusercontent.com/manuelmlmadeira/DeFoG/365bda9affadd5c2307014a0532ddaa244399441/src/flow_matching/time_distorter.py)
and [molecule postprocessor](https://raw.githubusercontent.com/manuelmlmadeira/DeFoG/365bda9affadd5c2307014a0532ddaa244399441/src/analysis/rdkit_functions.py).
DBMol's separate prior adaptation is described in
[Appendix C](https://arxiv.org/html/2607.19237v1#A3).

Default **`sampling: {}` preserves the checkpoint's actual sampling settings**.
Record original/effective distortion, `eta`, `omega`, RDB design and criterion
rather than assuming universal `polydec`/`eta=300`/`omega=0.1` defaults.
Those values can be declared overrides in a separately pinned protocol;
an empty mapping does not impose them. The actual distorted time grid always
uses the effective native settings recorded in the receipt.
An explicitly selected `checkpoint_forward_corruption` initialization is a
different intervention: it applies the checkpoint's native categorical
corruption at the effective boundary and records candidate-to-initialized
edits before continuing. It is not silently applied by the direct-candidate
default. Node order, masks and count remain fixed throughout either suffix.
Progress receipts count attempted/completed transitions separately and hook
actual denoiser calls; interrupted progress is a lower bound rather than a
fabricated completed cost. Terminal receipts hash frozen model state before
and after and reject changes.

## Scheduled weight versus applied guidance

The source079 telemetry audit found `repair_guidance.weight` and
`effective_repair_weight` record the configured schedule even if no target
is accepted. They must not be read as evidence that a repair gradient was
used. Every saved row in priors 8/9 (60/20 respectively) reports
`no_accepted_target`; all native attempts were rejected for unsupported
formal charge before inference, and inspected first/last total losses
equal binding-only loss. Fallback and gradient/certification calculations
were correct. A positive scheduled lambda or a passed Boltz-only gate does
not certify DeFoG guidance.

Source084 adds explicit scheduled/applied weights,
accepted-target use and actual gradient-measurement status, without changing
losses, fallback or historical data. It is sealed/staged and submitted as
CPU **29670451 DONE**, **82 passed in 24.87 seconds**, in its
30-minute/two-core CPU-only engineering gate. There is no GPU stage.
The verified [final evidence](../outputs/hpc-results/source084-repair-telemetry-002.tar.gz)
records HPC Python 3.11.9 / Torch 2.5.1+cu121 / pytest 8.3.5 and immutable
release identity. These checks validate reporting, not native repair quality
or long-run readiness. The [launcher](../outputs/hpc-submissions/submit-source084-cpu.py)
and [ledger](05_experiment_ledger.md) retain the seven fixture files, immutable
release identity, exact allocated command and receipt paths.
The strict `guidance_evidence` requirements remain: eligible native target,
target-only FD evidence, nonzero measured guidance and accepted guided
updates. These are distinct from the legacy schedule fields.

## Fixed-target guidance and recorded budgets

The provisional default is the **summed negative log-likelihood of the
constant slot-labeled target under the decoder's factorized categorical
outputs**, including atom, charge and unordered bond decisions:

```text
L_reference = -sum over occupied i log p_atom_i(target_atom_i)
              -sum over occupied i log p_charge_i(target_charge_i)
              -sum over occupied i<j log p_bond_ij(target_bond_ij)
L_inner = L_Boltz + weight * L_reference
```

Here `L_Boltz` is the existing minimizing scalar loss for the selected Boltz
profile: negative binary probability for its maximizing control, or the full
minimizing DBMol composite. In the strict policy, neutral target charges are
actual supported labels, not a neutralized rewrite of an unsupported charged
candidate. The optional lossy policy above would instead use explicitly
rule-assigned target charges, while preserving the canonical Boltz graph.
Bond decisions
include NONE once for every occupied unordered pair; diagonal, padding and
duplicate lower-triangle entries are excluded. Conditional atom support must
match the fixed-occupancy shared graph: **real-element softmax excludes
NO_ATOM**, with the same temperature and symmetric bond-logit policy used
by Boltz's shared graph. It is not full ten-way atom CE. There is no target
for absent slots, no absence-channel gradient and no claim of guidance
through atom count; the active-set arm is initially unsupported.

Default reduction `sum` corresponds to that declared factorized target
log-probability. Optional `mean_per_decision` divides by the actual number of
included categorical decisions and is an explicitly different objective.
Larger molecules have more decisions, especially absent bonds; these choices
change gradient scale and cannot be relabeled interchangeable. The provisional
weight is **0.01, illustrative and uncalibrated**, frozen before outcome
inspection. Record temperature, conditional category support, reduction,
decision count, component NLLs and weighted guidance. Do not silently change
the Boltz profile or its term normalization.

Evaluate the NLL using finite logits and stable `log_softmax`, not the log of
materialized probabilities with an arbitrary epsilon floor. A tiny target
probability can underflow when displayed as a probability while its log-NLL
and gradient remain finite. Record such target-probability underflows; do not
turn them into a favorable probability floor or treat them as chemical
impossibility. Missing/nonfinite logits or detached losses are errors.

A fixed reference makes the inner categorical loss differentiable. Its
gradient is the derivative of **that conditional fixed-target loss**, not
DeFoG likelihood, the probability of chemical validity or the derivative of
an expected stochastic repair distance. A target can be valid while the
current argmax graph remains invalid. Chemistry and target eligibility stay
evaluation/refresh operations, outside the decoder's autograd graph.

The provisional refresh interval is **five accepted updates**. At each
declared refresh boundary use a fixed quota, default **one attempt**, with
configurable bounded quota up to sixteen; the final experiment must pin its
quota/seeds before execution. Record every prescribed attempt, even if the
first produces a target. Select the **first eligible valid repaired target
in that prescribed order**, without Boltz-score ranking, favorable redraws
or quota refill. Retain duplicates, invalid/charged/unsupported candidates,
unrepresentable outputs, numerical failures and timeouts against the original
attempted denominator. If refresh yields no eligible target, **clear the old
reference and explicitly fall back to Boltz-only**; do not retain an
undisclosed stale target or coerced chemistry. An ordinary accounted-for
chemical rejection, supported native-continuation failure or bounded timeout
can leave no target. Provenance, request/response identity, adapter drift,
frozen-state mutation or malformed worker contract is a fatal experiment
error, not an excuse for silently falling back to Boltz-only.

The draft [RepairGuidanceConfig](../src/latentmol/repair_guidance/config.py)
pins the provider and these policies. It exposes `weight`, `refresh_interval`,
`attempts_per_refresh`, `seed`, `normalization`, `fallback`, `selection` and
`probability_policy`; unsupported providers/policies are rejected. Only the
official `defog_guacamol` hard provider is currently registered.

An explicit immutable `FrozenRepairTarget` in
[loss.py](../src/latentmol/repair_guidance/loss.py) is supplied to all score,
finite-difference and line-search evaluations for the same current state.
Refresh happens only after an optimization-step boundary, never between the
plus/minus pair or between Armijo trials. Reference-only finite differences
and the combined-total gate are separate diagnostics. Accepted-update
cadence is not attempted-update cadence; stalls and refresh resets must be
recorded rather than manufacturing target changes to improve acceptance.

The target stores detached private CPU copies and verifies its hard-graph
digest and occupied-slot mask before use. `fixed_target_nll` reconstructs the
expected fixed-occupancy graph from the same decoder logits, temperature and
mask and requires exact tensor parity with the supplied shared graph before
computing stable `hard_graph_log_prob_terms`. The decision denominator for
`mean_per_decision` is `2N + N(N-1)/2`. Its record includes per-channel NLLs,
decision counts, probability underflows, target digest, refresh step and
receipt path. These are authored interfaces, not passed numerical evidence.

[DeFoGRepairProvider](../src/latentmol/repair_guidance/provider.py) samples
detached CPU copies of the **actual current shared graph probabilities**,
without CPU re-decoding or temperature changes. Seeds for candidate drawing
and native continuation are distinct SHA-256 derivations of base seed,
accepted-step boundary, attempt index and phase. The provider independently
checks worker source/checkpoint/adapter/codec hashes, exact request/candidate,
native input/output labels, slots, sampler settings, terminal suffix budget,
frozen state and accepted-target chemistry. Raw response/request/progress,
invocation, process exit and timeout receipts remain distinct artifacts.

The opt-in `ActiveSetConfig.repair_guidance` initially requires
`fixed_initial_mask`, `local_v1` and bounded step control. Omitting it preserves
legacy configuration records/defaults; a zero-weight matched control executes
no repair worker. `score_at` takes the explicit frozen reference and retains
binding, raw/weighted reference and combined losses. Separate binding/repair
gradient norms and their alignment diagnose cancellation rather than crediting
only the combined gradient. Decoder atom-logit diagnostics include both
losses; shared atom-probability diagnostics cover binding alone because CE
differentiates stable log-softmax of those same logits.

Refresh occurs at step zero and then each declared accepted-update boundary
while another update is requested. An unchanged `z` can acquire a different
reference and hence a different objective: `repair-refresh-objective-STEP.json`
records old/new target hashes, old/new reference loss, weighted jump and
`is_optimization_improvement: false`. Refresh jumps do not count as accepted
updates or a global monotone optimization trajectory. Each derivative/line
search holds the current target and context fixed.

Record raw/repaired label identities and slot maps, CE targets and their
hashes, changes needed, sampler settings/seeds and the frozen model/source
identities. Preserve reference age, refresh outcome, target-selection reason,
raw/weighted guidance terms, Boltz terms, latent gradients, accepted/rejected
updates and physical oracle costs. DeFoG transitions, batched versus
candidate-level calls and wall time are separate from decoder/CE and Boltz
forward/backward work. Equal latent update count is not equal compute.

The planned native worker runs in a separate Python process/environment and
exits before the next Boltz gradient calculation. This keeps DeFoG sampler
execution outside the attached Boltz graph; it does **not** imply zero peak
memory overlap, because frozen Boltz weights may remain resident while the
worker executes. Record worker/source/environment/checkpoint identities,
device, peak memory, process exit and timeout. Matched config preparation,
HPC engineering fixtures and matched preparation passed in source068;
the native smoke scripts have not run. Source068 preserves the isolated
launcher path instead of dereferencing its environment symlink.
The draft worker CLI `python -m latentmol.defog.candidate_repair --request
REQUEST.json --output UNIQUE_RECEIPT.json` requires a DTU allocation; existing
receipts cannot be overwritten. The optional shell entry
[defog_candidate_repair.sh](../scripts/hpc/defog_candidate_repair.sh) uses the
isolated `env/defog311` interpreter. It declares strict Torch algorithms,
process-local seeded RNG and a supported cuBLAS workspace configuration;
actual device/kernel determinism and native continuation still need HPC
outcomes. A receipt for a chemical rejection is an accounted-for attempt,
while an execution/contract error also yields a nonzero worker exit.

## HPC preparation and evidence protocol

[study.py](../src/latentmol/repair_guidance/study.py) prepares matched configs
from one authenticated saved-start Boltz config. Its base must pin exact
initial `z`, fixed slots, `local_v1` and a bounded optimizer and must have no
preexisting repair branch. Preparation changes the prescribed guidance
coefficient, name/output root and requested update budget while preserving
decoder, target/context, Boltz loss/pocket/pose and update rule. It reuses the
existing anchor pocket; no new outcome-driven pocket or start is selected.
The guidance-only [example YAML](../configs/defog_repair_guidance.example.yaml)
is parsed as `RepairGuidanceConfig`, not as a full `ActiveSetConfig`.

The initial preparation declares **six updates** and coefficients **0 and
0.01**, yielding two serial matched cases. With refresh interval five, a
positive branch can exercise refreshes at accepted boundaries 0 and 5;
an early stall may prevent the latter and remains an incomplete budget.
Weight zero disables all native repair calls by design and cannot certify
guidance. Source068 generated the authenticated config hashes above; source067
configs remain historical and superseded. Prepared configs are not native
molecular results.

The failed initial stage is **CPU engineering/preparation only**.
[boltz_repair_preflight.sh](../scripts/hpc/boltz_repair_preflight.sh) runs
[boltz_repair_tests.sh](../scripts/hpc/boltz_repair_tests.sh), then writes the
metadata-only matched configs/`comparison.json`. The fixture suite covers
fixed-target/gradient, provider, hard candidate continuation with mocks,
sampling/graph policy, bounded Boltz runtime, DBMol gradient and identity
contracts. Mock transitions establish engineering behavior, not actual
pretrained DeFoG repair. CPU request is queue `hpc`, two cores, 4 GB per
core (8 GB host), one host and 30 minutes. Source066 and job 29620487 have
the identities above and observed fixture failure; preparation did not run.
The separately sealed source067 helper/provider retry passed 120 engineering
tests and preparation on HPC. Final inspection found that its provider
Python path had become `/appl9/python/3.11.9/bin/python3.11` through symlink
resolution, instead of the lexical `env/defog311/bin/python` launcher.
Source068 corrected this and passed 121 engineering tests/preparation;
the runtime metadata probe confirms the isolated environment, with native
execution still pending. The ledger retains
the sealed-provider versus later-workspace difference, both immutable releases
and their actual outcomes; old tests do not certify later bytes.

The authored [submit_repair_guidance.py](../scripts/hpc/submit_repair_guidance.py)
prints an authenticated plan without `--submit`. It checks release
archive/manifest, base/guidance config identity and unique receipts and wraps
execution in the immutable release verifier plus `ExperimentRun`. Template:

```text
python scripts/hpc/submit_repair_guidance.py --stage cpu --release IMMUTABLE_RELEASE --archive-sha256 ARCHIVE_SHA256 --base-config AUTHENTICATED_BASE.yaml --base-sha256 BASE_SHA256 --prepared-root UNIQUE_PREPARED_ROOT --receipts UNIQUE_RECEIPTS
```

An executed `--submit` and actual scheduler/receipt evidence are required
before calling this submitted. Local work to date is editing/static parsing
only, without numerical tests or model imports. All fixtures remain HPC work.

A **later, not-yet-submitted native stage** authenticates the actual prepared
comparison SHA/source/base plus a matching CPU receipt and requires CPU DONE
before constructing `done(CPU_JOB_ID)`. Its authored request is queue
`gpua100`, one exclusive-process 80 GB GPU, four cores/16 GB per core
(64 GB host), one host and 6h15m. The serial
[boltz_repair_smoke.sh](../scripts/hpc/boltz_repair_smoke.sh) controller gives
each of the two cases a 10,800-second child cap. The six-hour sum of child
caps leaves fifteen minutes for setup/controller work. This resource plan
does not establish measured runtime, hardware, memory overlap or a queued
native DeFoG campaign.

```text
python scripts/hpc/submit_repair_guidance.py --stage native --release IMMUTABLE_RELEASE --archive-sha256 ARCHIVE_SHA256 --base-config AUTHENTICATED_BASE.yaml --base-sha256 BASE_SHA256 --prepared-root UNIQUE_PREPARED_ROOT --comparison-sha256 ACTUAL_COMPARISON_SHA256 --receipts UNIQUE_RECEIPTS
```

Do not fabricate the comparison digest before CPU preparation. All placeholders
must be replaced only with the corresponding release's actual authenticated
receipts. Source067's prepared identities must not be reused with source068.

`study.guidance_evidence` distinguishes completed orchestration from actual
repair guidance. Positive-weight validation requires all of: an accepted
authenticated native target with actual denoiser calls, at least one accepted
guided update, a passed nonzero reference-only finite-difference gate and a
nonzero weighted reference gradient on an accepted update. The aggregate
`native_guidance_passed` also requires every prescribed case to complete
its requested budget. `engineering_completed` alone, a valid target without
an attached update, or an all-fallback Boltz trajectory is insufficient.
The controller retains stalled/failed cases and reports fresh-binding
validation as required/not run rather than converting a completed process
into chemical or binding success.

Scorer counters record decoder probes, reference backpropagation and worker
attempts/status/wall time. Exact DeFoG denoiser/transition counts come from
the authenticated **raw worker responses** inside refresh receipts, not from
counting one subprocess as one model forward. Timeout progress remains
censored lower-bound evidence; do not add it to complete native-call totals.
Unsupported/chemical rejection counts must follow the actual response reason
codes while preserving their original attempt denominators.

## Validation and claim limits

Before a combined experiment, validate the hard codec both directions,
occupied-slot alignment, genuine candidate-start execution and frozen DeFoG
state. Preserve historical source028/source030 failed soft-auxiliary/edge
derivative gates; they do not pass retroactively because this new path avoids
that derivative. Native prior-generation/transition checks and malformed
schema-compatible input handling need actual new HPC receipts.

Validate that CE backpropagates to `z` through the frozen decoder, that target
labels remain constants across inner updates and that no discrete operation
enters a claimed pathwise gradient. Check a fixed-target local derivative
separately from Boltz's native gradient gates, and reject detached/missing
terms rather than reporting them as useful zero gradients. Combined loss
acceptance must use the same reference/context throughout a step; a new target
or a new pose is a different objective comparison, not global monotonicity.

Matched controls should preserve exact starts, frozen assets, Boltz profile,
occupancy, sampling and native evaluation policies while disclosing additional
repair compute. Compare raw hard validity, sampling validity, edit burden,
diversity, collapse/identity retention and raw-versus-repaired score behavior.
Evaluate actual raw endpoints as such; do not replace invalid endpoints with
a repair and report this as decoder validity. Fresh-native input audits and
prediction remain separate from relaxed or fixed-reference gains.

No new model training, charged-schema support, useful binder, synthesis
feasibility, selectivity, validity guarantee or calibrated molecular prior is
established by this plumbing. Source/config/checkpoint identities, numerical
tests, job submissions and completed/failed outcomes belong in the
[ledger](05_experiment_ledger.md). All numerical/preprocessing/model work
remains on DTU HPC; local editing/static documentation review only is used
for this draft.
