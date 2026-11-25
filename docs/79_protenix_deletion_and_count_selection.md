# Protenix physical-deletion and atom-count-selection study

Design date: 24 October 2026. Immutable identity:
`source139-protenix-deletion-selection-attempt001`.

**Completed diagnostic, 24 October 2026:** **GPU 29803898 DONE**, executed
**02:29:32–02:36:52 DTU-local**. Fresh native/gradient gates passed and all
18 trajectories completed eight updates. **Z q=0 failed physical-deletion
equivalence in all three cases**; charge/name/position/bond ghost routes stayed
active. Gradient optimization reduced soft loss but selected the worse hard
native endpoint in every case; random-sign controls selected the same counts.
This is a completed negative absence/count-selection result, not G validation.
[Exact terminal evidence and interpretation](#12-completed-diagnostic-zero-element-mass-did-not-reproduce-deletion).

The user-authorized design below precedes this execution attempt. This full
identifier is independent of other campaigns that may use the same source
number.

## 1. Questions and starting evidence

The user requested the next study stage: test whether NO_ATOM behaves like
physical atom deletion, then test whether optimization can select atom count.
The completed [C/Z foundation](73_protenix_soft_occupancy_study.md#36-attempt008-completed-cz-native-parity-and-calibrated-gradients)
supports an element-mass gradient mechanism on three fixed contexts. Its
165 CPU fixtures, required loss/TV parity, 36 calibrated derivative rows and
one unresolved Z point remain exactly as recorded. It did not validate
physical absence or variable atom count.

This phase separates two questions:

1. Does the clamped Z input at q=0 agree with a native input from which the
   optional atom/token was physically removed, including independence from
   ghost-atom attributes?
2. Can a prespecified bounded optimization of q select the present/absent
   endpoint that improves the fixed native readout, and does that decision
   transfer to freshly prepared native molecules?

Z remains the element-mass relaxation. It is not renamed G or represented as a
complete occupancy-propagation implementation. After fresh native/gradient
gates pass, the user-authorized **diagnostic Z selection runs even if absence
equivalence or ghost independence fails**. Such failures are negative evidence,
not waived gates or permission to promote an unsupported absence model.

## 2. Fixed cases, model and readout

All three prespecified BRD4 cases from docs73 remain in the study; no case is
selected or dropped using model scores:

| Role | Crystal/ligand | Optional terminal atom |
| --- | --- | --- |
| Development | 7ZA9/HHQ | I1 |
| Confirmation 1 | 7ZAD/IJL | I1 |
| Confirmation 2 | 7ZAJ/HH8 | BR1 |

Use the same authenticated raw crystals and frozen official PyTorch
`protenix_tiny_default_v0.5.0`, pinned source/checkpoint/configuration, one
recycle, FP32 model computation with TF32 disabled, and FP64 normalized
distogram CE reduction. Protein context, retained ligand-core/protein pairs,
target bins and denominator remain fixed and matched across every comparison.
The optional atom is excluded from the readout. The exact artifact identities
must be bound in the immutable release/plan before submission.

Only the optional terminal atom changes chemical occupancy. Both full and
deleted graphs must sanitize and preserve the declared retained-core identity;
deletion recalculates the expected implicit hydrogen. Molecular support,
graph/SMILES hashes, atom correspondence and any unsupported mode are explicit
preparation outputs.

## 3. Native deletion references and preparation controls

Allocated CPU preparation builds four distinguishable native reference inputs:

| Reference | Purpose |
| --- | --- |
| Ordinary full | Original supported full molecule and native context |
| Compact deleted | Physically remove the optional atom/token, preserve retained reference geometry exactly and rebuild dependent caches/indexing |
| Fresh full | Reimport the full molecule through the ordinary FILE/MOL route with seed-17 ETKDG geometry and name-mapping sidecar |
| Fresh deleted | Reimport the deleted molecule through the **same** FILE/MOL route, geometry policy and sidecar mapping as fresh full |

Fresh full is essential: differences between ordinary full and fresh deleted
would confound deletion with importer/conformer changes. Compact deletion
isolates absence while holding the retained geometry fixed. Fresh full/deleted
measure transfer under an actual paired preparation route; geometry drift is
reported separately, not attributed entirely to absence semantics.

All atom/token maps, bond/pair tensors and reference-position-dependent caches
must be consistent with each declared molecule. Fixed readout identities are
mapped through the sidecars and verified rather than silently changing the
target set. Preparation and chemistry checks, fixtures, checkpoint loading and
all model calculations run on allocated DTU HPC. Local work is limited to
editing, static AST/byte review and immutable packaging.

## 4. Fresh native gates and shared execution authority

Rerun required C/Z native-output comparisons and all **36 derivative rows**
under the successful attempt008 calibrated FD policy in the new allocation.
These fresh checks and subsequent Z optimization use the **original ordinary-
full CCD features/reference geometry**. The FILE/MOL fresh-full input is an
independent discrete-transfer reference, not the optimization input.
The loss/TV policies, recorded strict-logit diagnostics, empirical uncertainty
cap, C input invariance/exact-zero gradient and separate resolved direct-q/
NO_ATOM-logit requirements remain as declared in docs73. Old results support
the design but do not replace these new gates.

Use one resident frozen model and a single shared `CountedForward` authority
for these fresh gates and every later stage. The call/time budget never resets
between calibration, deletion comparisons, ghost probes and optimization.
The shared hook validates the same native implementation/identity, zero calls
at entry, the declared global call ceiling and the earlier deadline; the
original standalone gate behavior remains available for earlier protocols.
Source/checkpoint/configuration, frozen state, context and RNG checks remain
mandatory. Failed native/FD checks, nonfinite values, unexpected exceptions or
resource exhaustion genuinely stop the run.

## 5. Absence equivalence and ghost diagnostics

Save ordinary-full, compact-deleted, fresh-full and fresh-deleted native
endpoint references. Compare exact q=0 and q=1 C/Z outputs with the declared
matching endpoints under the existing finite/loss/TV policy; retain original
strict-logit comparisons as diagnostics. In particular, agreement with the full
endpoint is distinct from agreement between Z q=0 and physical compact deletion.

At q=0 independently perturb the optional slot's element identity, formal
charge, atom-name features, reference position and incident token bonds. Each
position intervention rebuilds its dependent caches. Record every intervention
and native-output effect; a response to an absent slot is evidence against
absence equivalence in this relaxation. Capacity/order interventions that are
not implemented must be **not run**, not reported as passing invariance.

Raw-attribute gradients are reported only for completely audited computational
routes. Element, charge, atom-name and token-bond feature routes report finite
gradient norms on the optional row or incident token-bond rows/columns, with
their declared frozen-feature scope. A reference-position gradient that omits dependent cached features is a
partial cached-context derivative and remains **not validated** as the full
position gradient. These diagnostics cannot certify general G propagation,
exact absence or arbitrary variable-size molecules.

Each of the four endpoint references receives **three native forwards**.
Compute separate empirical allowances for the compact pair (ordinary-full CCD,
compact-deleted) and fresh pair (FILE/MOL fresh-full, fresh-deleted):
`U=max(1e-12,4*largest within-endpoint repeat range)`. Mean-loss direction is
resolved only if U is at most **1e-6 normalized loss** and the absolute endpoint
gap exceeds **10U**. Otherwise its ranking is indeterminate. Exceeding the cap
is scientific negative evidence, not permission to increase the allowance
without bound. Nonfinite/invalid endpoint values cannot pass. The allowance
is an empirical diagnostic, not a rigorous noise bound; an ambiguous endpoint
is not presented as an observed preference.

## 6. Prespecified bounded q optimization and controls

Only q is optimized; there is no learned weight, chemistry prior, latent
decoder update or model fine-tuning. Use **q starts 0.2 and 0.8**, **eight
requested updates**, and gradient proposal `delta_q = -0.1 * sign(dL/dq)`.
Clip candidate q to **[0.01, 0.99]**. The fixed readout evaluates every saved
state, with bonds, reference geometry and other non-element features clamped
for the Z arm.

The declared `best_trial_v2` rule tries the full step first and accepts an
actual scalar nonincrease, including equality/no-op. Otherwise it tries up to
16 sizes `1, 1/2, ..., 2^-15`, accepting the first strict improvement. If no
strict improvement exists, apply the lowest-loss finite candidate, breaking
ties by the earliest trial. All-nonfinite candidates are a genuine failure.
Record applied, improving, full-step-nonincreasing, forced-fallback and no-op
counts; do not claim monotone descent or convergence from this policy.

Controls are the unchanged start and a seeded random-sign proposal under Z,
using the same starts, eight opportunities, 0.1 step bound and up-to-16 trial
budget where a proposal is made. The random-sign arm uses the **same
best_trial_v2 acceptance/search** as the gradient arm: it is a random-proposal
control with scored acceptance, not an unscored random walk. Report the actual
forward/trial cost of each arm, including unchanged-start opportunities.
Case seeds are **1729/1730/1731**, with deterministic start-index
handling through a private RNG; model RNG is not repurposed for control
proposal generation. All starting points, controls and budgets are declared
before observing scores.

The complete prespecified comparison has **18 trajectories** (three cases ×
two starts × three arms), with **144 requested update opportunities**. The
unchanged arm records eight opportunities without oracle line-search trials;
gradient and random-sign arms share the same 16-trial ceiling. Actual calls,
accepted changes and no-ops remain separate observations.

## 7. Hard count decisions and native transfer

Harden each saved state as **present if q>0.5, absent otherwise**, including
the absent tie at q=0.5. This selects between two graph-validated native
endpoint identities; it does not invent an intermediate chemical graph. Save
the expected occupancy and hard atom count separately.

Native endpoint outputs are precomputed once and reused for reporting hard
states, rather than rerunning the identical molecule at every q. Keep these
outcomes separate:

- soft Z scalar gain along the optimized trajectory;
- whether the hard atom count changed;
- native gain on ordinary-full/compact-deleted endpoints;
- native gain on the paired fresh-full/fresh-deleted endpoints;
- direction determinate/indeterminate under the declared repeat allowance.

An exact Bernoulli mixture of the two discrete endpoint scores is **analysis
only**. It must not become gradient guidance or replace Z's objective. Soft
gain or a count switch alone is not evidence of native transfer, chemical
utility or improved binding. Preserve failed and invalid outputs and the
complete attempted-state denominators.

## 8. Resources, reporting and execution status

Readiness requests **four CPU slots/45 minutes**. Only sealed-source
authentication, complete CPU gates and fresh quota allow the conditional GPU
request: **one exclusive A10/four CPU slots/one hour**, **55-minute internal
limit**, maximum **3,000 GPU model forwards across the entire allocation**.
At the prior attempt008 observed mean (1,048 forwards/225.749 seconds), 3,000
calls project to roughly **646 seconds**; this is a derived planning estimate,
not a guarantee of runtime for the new inputs/stages. The wall/internal limits
and shared call cap remain enforced. Budget exhaustion is partial/failure
evidence, not fabricated completion or an automatic allocation extension.

Raw JSON/JSONL, manifests, graph mappings, snapshots, native endpoint values,
gradient/FD records and scheduler receipts remain authoritative. A saved-only
offline interactive HTML report may present deletion comparisons and
trajectories; report rendering cannot rerun models, repair graphs or invent
missing states, and its failure cannot change the scientific outcome.

This design section alone establishes no scheduler jobs, parity/absence
decisions or optimization results. Execution entries below retain exact
sealed identities, commands, actual outcomes and scientific limits.
The prospective allocated CPU suite contains **287 cases** by static count:
165 existing occupancy fixtures, 86 deletion-preparation cases and 36 selection
cases. This is not an executed pass; independent final static review and
allocated numerical validation remain pending.
The successful attempt008 foundation, its one unresolved point, every earlier
failure and unrelated VAE/bridge/Mosaic campaign remain preserved.

## 9. Immutable release sealed; submission blocked before remote execution

The [preparation receipt](../outputs/hpc-submissions/source139-protenix-deletion-selection-attempt001-preparation.json)
records **216 sealed members** for
`source139-protenix-deletion-selection-attempt001`, status
`sealed_not_submitted`:

| Immutable binding | SHA256 |
| --- | --- |
| Release archive | `cdca94bcf05f5b3c2fca2cc6587970a561a293ad4afd314fd50e6b4db49bb573` |
| Release manifest | `a2c7018423f86ee916388e2ae9bf9eacf9d10363e8e02789ab55ceb882b7eb11` |
| Execution plan | `6aabc5854c8c3b6dbb753ff3818d6bc542f0e22b19abe9bf277777645168b9e1` |

The [independent static audit](../outputs/hpc-submissions/source139-protenix-deletion-selection-attempt001-static-audit.json)
records **206 unchanged nonmanifest members**, two changed members (docs73 and
the resident-model/shared-forward gate hook), seven added members and
**208 Python AST parses**. Existing model, preparation, occupancy fixtures and
loading code remain byte-identical; the existing study change is limited to
the shared execution hook. This is source/byte evidence, not an allocated CPU
or model pass.

The [actual stage-and-queue operation](../outputs/hpc-submissions/source139-protenix-deletion-selection-attempt001-stage-and-queue-001.json)
returned **255**, with `Permission denied (password)`, before remote execution,
upload or scheduler submission. A read-only SSH diagnostic found the expected
key path and key acceptance, followed by a password requirement that BatchMode
could not satisfy. This is an access failure, not a failed molecular test.
The recorded local observation is **24 October 01:49:53 DTU-local**
(`23 October 23:49:53 UTC`). The coordinator requested restoration of VPN/SSH
access; no unauthenticated workaround or numerical execution occurred.

**No new CPU/GPU job was created by this attempt.** All **287 fixture cases**,
native preparation, strict checkpoint/input gates, deletion/ghost comparisons
and selection trajectories remain unrun. The sealed release/plan stay
immutable, and the failed authentication receipt remains preserved for a
later authenticated staging attempt. Attempt008's completed foundation, all
earlier failures and unrelated campaigns remain unchanged.

## 10. Authenticated staging succeeded; CPU submitted

After the user restored SSH access, the
[second stage-and-queue operation](../outputs/hpc-submissions/source139-protenix-deletion-selection-attempt001-stage-and-queue-002.json)
returned **zero**, staged/authenticated all **216 members**, and submitted
**CPU 29803884** to `hpc`, four slots/45 minutes. Its scope is
`submitted_not_passed`. The archive, manifest and plan remain exactly those in
section 9; no code or policy changed to restore access.

The initial failed authentication/staging receipt remains preserved. This
actual successful submission supersedes the earlier blocked state, not its
historical record. The **287 expected fixtures**, three-case preparation,
strict actual-checkpoint loading and real-input CPU gates remain pending.
Only full readiness and fresh quota permit conditional native GPU dispatch;
no GPU job, deletion/ghost outcome or selection result is established yet.
All earlier scientific evidence and unrelated campaigns remain unchanged.

## 11. Full CPU readiness passed; native GPU queued

The [third collection](../outputs/hpc-submissions/source139-protenix-deletion-selection-attempt001-collect-003.json)
and [CPU readiness](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-001/cpu-ready.json)
record **CPU 29803884 full gate passed in 133.086 seconds**: **287 fixture
passes**, zero failures/errors/skips, **6.62 seconds**, original parent
three-case preparation and new compact/fresh-full/fresh-deleted preparation
passed. The new preparation matched the full parent reconstruction and
retained compact reference positions exactly. Fresh full/deleted molecules
passed sanitization, neutral/connected graph, roundtrip and hydrogen/mapping
checks. Retained readout counts remain **903/1,161/1,290**.

Actual-checkpoint strict loading and the six-forward real-input CPU probe
passed. That probe took **32.381 seconds**, all original strict output
comparisons were exact, its element-leaf gradient was finite/nonzero (norm
**0.057674998699864734**), and model/context remained unchanged. This is CPU
input/readiness evidence, not native absence equivalence or a count-selection
result.

The [native GPU submission](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-001/native-submission.json)
returned zero: **GPU 29803898 submitted 24 October 01:53:41 DTU-local**,
`gpua10`, one exclusive A10/four CPU slots/one hour, 55-minute internal cap and
3,000 shared forwards. The
[01:54:24 queue snapshot](../outputs/hpc-submissions/source139-protenix-deletion-selection-attempt001-gpu-queue-001.json)
records **PEND**, eligible pending time **45 seconds**, ineligible time zero,
four CPU slots reserved on `n-62-18-5` while the physical GPU resource
`ngpus_physical` is unavailable. Reservation is not execution; there is no GPU
controller claim or native result and no reliable start estimate.

The [31-file saved export receipt](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-001/collection-receipt.json)
authenticates SHA256
`5a1bf60949776c5ff823d5f3da9bc48319c3e78ccadd63b3098c1ff58ad81da0`,
including six MOL files and preparation/controller receipts. Key bindings:

| Completed CPU artifact | SHA256 |
| --- | --- |
| CPU readiness | `bb4e171621e646844c21939d7a27c6e402dffb01bf76507ea2ee5c79ea91cf50` |
| Deletion-prepared manifest | `602f2c34d99ad041f7d63c5405ddea9f519cab8d6ae36dba029d5a33d15ec895` |
| Native execution configuration | `3db770c377efd45227204a59c63a8e19b67a1554c522408af5eac072372988d7` |

Fresh allocated C/Z gates, physical-deletion/ghost comparisons, raw-attribute
probes and diagnostic count selection remain pending the GPU allocation.
No absence, NO_ATOM-selection, optimization or binding conclusion follows from
CPU readiness. The authenticated seal, initial access failure, attempt008
success and all historical/unrelated records remain preserved.

## 12. Completed diagnostic: zero element mass did not reproduce deletion

The [fourth collection](../outputs/hpc-submissions/source139-protenix-deletion-selection-attempt001-collect-004.json),
[native result](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-002/native/deletion-selection-result.json)
and [GPU terminal receipt](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-002/gpu-terminal.json)
establish **`deletion_selection_diagnostic_completed`** /
**`completed_diagnostic_not_G_validated`**. **GPU 29803898 DONE**, executed
**24 October 02:29:32–02:36:52 DTU-local** on `n-62-18-5/gpua10`, one exclusive
A10/four CPU slots. It used **1,816 total model forwards**, **427.920 seconds**
in the study/**436.376 seconds** in the controller, peak allocated CUDA memory
**6,091,817,472 bytes**. All **18 trajectories completed eight updates/nine
saved states**, including unchanged controls: **144 applied opportunities**.
All three cases retained unchanged model/context. The prior 287-pass CPU
readiness and initial failed SSH receipt remain preserved.

The fresh [C/Z readiness result](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-002/native/readiness/native-gate-result.json)
passed required output policies and **36/36 calibrated derivative rows,
108/108 scale comparisons**. Original strict FD diagnostics passed **79/108**,
not all. C had 18 exact-zero gradients; all 18 Z gradients were finite/nonzero,
17 resolved under the two-scale requirement. HHQ q=0.9 NO_ATOM-logit remained
unresolved, with other q values satisfying that case's resolution gate.
These fresh functional gates succeeded; they did not force absence/selection
diagnostics to succeed.

**Z q=1 matched the full native input in 3/3 cases under the declared output
policy. Z q=0 failed compact physical-deletion equivalence in 3/3 cases:**

| Case | Absolute scalar-loss difference: Z q=0 versus compact deletion | Maximum pair probability TV |
| --- | --- | --- |
| 7ZA9/HHQ | `0.0009874571142233224` | `0.2861103415489197` |
| 7ZAD/IJL | `0.030773325041033472` | `0.2789417803287506` |
| 7ZAJ/HH8 | `0.012582383029211663` | `0.1383468508720398` |

The tested conditional-element ghost intervention passed in all cases, but
**charge, name, reference-position and incident-bond interventions each failed
in all three cases**. Thus a zero element vector leaves other optional-slot
routes active. Reference-position interventions rebuilt caches; the complete
position gradient itself remains not validated. Capacity/order ghosts remain
not run. These results reject absence equivalence for this clamped Z
implementation on the tested cases; they do not certify or reject an
unimplemented full G propagation mechanism.

Both the compact endpoint pair and the independently imported FILE/MOL fresh
pair had a resolved **full-molecule preference in all three cases** under the
fixed native loss. Gradient optimization reduced soft Z loss in **6/6 runs**,
but all six finished below q=0.5 and hardened to deletion. The q=0.2 starts
were already deleted and had no hard native gain. Each q=0.8 start crossed to
the deleted endpoint, which **increased** both native losses:

| Case | Hard ligand atom count | Soft Z loss change | Compact native loss change | Fresh native loss change |
| --- | --- | --- | --- | --- |
| 7ZA9/HHQ | `8 → 7` | `-0.005276953956886321` | `+0.00010157789626741298` | `+0.0005106427386158963` |
| 7ZAD/IJL | `10 → 9` | `-0.022161824356059068` | `+0.0069022241140199725` | `+0.005472400968491886` |
| 7ZAJ/HH8 | `11 → 10` | `-0.002797315262962674` | `+0.008939273915089752` | `+0.008129330533079404` |

The random-sign proposals with the **same scored acceptance rule** also ended
deleted in all six runs; unchanged controls retained their q/count. There is
no demonstrated gradient advantage in the hard endpoint decision. A count
switch and soft loss reduction occurred, but they selected the worse native
identity under the prespecified readout. This is evidence of relaxation/native
transfer mismatch, not useful atom-count selection or improved binding. The
graphs themselves were supported/validated during preparation; native score
preference is not experimental chemical or affinity ground truth.

Completed case records are saved for
[7ZA9/HHQ](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-002/native/brd4_7za9_hhq-complete.json),
[7ZAD/IJL](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-002/native/brd4_7zad_ijl-complete.json)
and [7ZAJ/HH8](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-002/native/brd4_7zaj_hh8-complete.json).
The [saved-only terminal summary](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/terminal-summary-001.json)
has SHA256 `a5941e81f3ae602e313f644d0dfe4afd4907378d1fdbc73a7b99cb486a6991b2`;
it derives counts from existing artifacts, with no model rerun.

The [57-file artifact receipt](../outputs/hpc-results/source139-protenix-deletion-selection-attempt001/snapshot-002/collection-receipt.json)
authenticates export SHA256
`9b64c8a68e0e57c67c2eda89f79488a1b326b603d32c7f23987d703fccef14ec`;
root native-result SHA256
`3c3e4cb88756cb546be8f55152189e9439399f5a0c2a43004b0e30f36ef2204b`.
The source/manifest/plan remain those in section 9. This status collection
performed no new model run, source/policy change or resubmission. Full G,
useful variable-count optimization and binding improvement remain unvalidated;
attempt008's gradient success, historical failures and unrelated campaigns
remain unchanged.
