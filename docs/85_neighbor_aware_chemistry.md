# Neighbor-aware chemistry guidance: policy, gradients and coverage

Created **29 October 2026**. This document covers the new, opt-in chemistry
guidance being implemented after the [sulfur mechanism audit](84_sulfur_bias_in_validity_guidance.md).
It does not change an existing training run, checkpoint, loss version, sampling
ratio, optimizer, repair teacher, native-model adapter or validation threshold.
The user accepted excluding very rare, specialized sulfur environments in
exchange for a simple policy. The selected version is
`common-organic-neighbor-v1`, described below. **Allocated CPU qualification
passed in source145 attempt004, CPU29851083**, on 29 October 2026: 138 PyTorch
and 26 JAX tests passed with zero failures/errors/skips; six-case FP64
cross-framework scalar/VJP parity passed. All 10,000 audited validation
molecules were retained, including 3,337 sulfur-containing molecules, and the
actual frozen epoch40 VAE had finite nonzero latent gradients. Earlier launch
and fixture failures remain preserved below. The terminal status is
`passed_validation_not_training_adoption`; no active training or optimization
switch, production efficacy or demonstrated sulfur-bias reduction is claimed.

## Why a maximum valence is insufficient

The historical live term in
[`chemistry_regularization.py`](../src/latentmol/chemistry_regularization.py)
penalizes the expected excess above the largest permitted valence for each
element/charge pair. Sulfur can therefore receive less loss than carbon for
the same crowded neighborhood, even if that neighborhood is unlike ordinary
organic sulfur chemistry. This direct substitution shortcut is distinct from
the shared-decoder and refreshed-teacher effects described in the mechanism
audit. Replacing a positive-part penalty with its square does not remove that
ordering.

The correction must inspect the **combination** of central element, formal
charge, neighbor identities and incident bond labels. It must also preserve
legitimate sulfur and phosphorus chemistry. A universal sulfur cap of two,
an oxygen-count requirement, or a rule that every ligand of expanded-valence
sulfur must be electronegative would each reject real molecules.

## Chemistry that the policy must not accidentally forbid

These sources establish counterexamples to simple restrictions, rather than a
complete physical-validity oracle:

* A sulfone has two carbon substituents as well as its two oxo groups.
  [IUPAC definition of sulfones](https://goldbook.iupac.org/terms/view/S06117).
* Sulfoximines contain both an oxo and an imino group. Sulfonediimines replace
  both oxo groups by imino groups, so requiring two oxygen neighbors is wrong.
  [IUPAC sulfoximides](https://goldbook.iupac.org/terms/view/S06125),
  [IUPAC sulfur class definitions](https://iupac.qmul.ac.uk/class/SSeTe.html).
* Both aryl- and alkyl-pentafluorosulfanyl groups have a carbon ligand alongside
  five fluorines. Six single bonds must not be limited to SF6.
  [Beier et al., original aryl-SF5 chemistry](https://pubs.acs.org/doi/10.1021/ol2001478),
  [original aliphatic-SF5 synthesis](https://doi.org/10.1016/j.jfluchem.2006.05.003).
* The neutral sulfur-ylide form CH2=S(CH3)2 is explicitly named in IUPAC
  P-68.4.3.1. A carbon double bond to sulfur is not automatically a defect.
  [IUPAC Blue Book](https://iupac.qmul.ac.uk/BlueBook/PDF/BlueBookV3.pdf).
* More strongly, an all-carbon sulfurane with four carbon-sulfur bonds was
  isolated and characterized crystallographically. An all-carbon
  six-coordinate persulfurane was also isolated and characterized by X-ray
  diffraction. Their constrained ligands matter, and a first-shell ligand
  histogram does not encode those constraints.
  [Ogawa et al., 1992](https://pubs.rsc.org/en/content/articlehtml/1992/c3/c39920001141),
  [Sato et al., 2006, original article abstract](https://pubmed.ncbi.nlm.nih.gov/16719444/).
* Phosphonium ylides have charge-separated and P=C representations. A rule that
  every expanded-valence phosphorus center requires oxygen, nitrogen or fluorine
  would reject this established class.
  [IUPAC phosphonium ylides](https://goldbook.iupac.org/terms/view/P04563).
* Hypervalent iodine is not equivalent to ordinary monovalent organohalides;
  IUPAC gives phenyl- and methyl-substituted lambda3-iodane examples.
  [IUPAC P-68.5.1](https://iupac.qmul.ac.uk/BlueBook/PDF/P6a.pdf).

Consequently, the selected policy is a **declared common-organic local
admissibility policy**, not a proof that every excluded graph is chemically
impossible. The six-carbon persulfurane is deliberately outside its scope;
the four-carbon sulfurane is retained. This tradeoff follows the user's
explicit preference for a simple correction rather than specialized exceptions.

## Selected policy: `common-organic-neighbor-v1`

The new branch combines the existing versioned element/formal-charge valence
caps with a single additional sulfur coordination rule: **at most four occupied
non-fluorine heavy-atom neighbors**. A fluorine neighbor is exempt from this
coordination count, but its bond still consumes valence. The coordination limit
does not replace or relax the existing valence cap.

This admits ordinary sulfones, sulfoximines, sulfonediimines, sulfur ylides and
four-carbon sulfuranes. It also admits SF6 and carbon-substituted SF5
environments. Six-carbon persulfuranes and other sulfur environments with more
than four non-fluorine neighbors are intentionally excluded from this version.
No new requirement forces phosphorus to have oxygen, nitrogen or fluorine
neighbors; five-carbon P(V) environments are not excluded by a new phosphorus
ligand rule.

The branch also checks an incident bond against the **remaining valence
capacity of its other endpoint**. For a focal atom i and neighbor j, all the
other incident bonds of j are held at the prepared hard labels:

\[
 r_{j\mid i}=\max\!\left(c_j-
       \sum_{k\ne i,j}o^H_{jk},\,0\right).
\]

Here c_j is the existing cap selected by j's fixed element/formal charge, and
o^H is the declared hard-context bond order. A live i-j bond label exceeding
r_(j|i) cannot contribute to the admissible event. NONE remains allowed, so
the local search-free event has a repair direction even for crowded contexts.
This is a local, fixed-context capacity check; it does not simultaneously solve
every other bond or certify a globally consistent molecule.

## Existing representation and validity boundaries

[`GraphSpec`](../src/latentmol/representation.py) stores up to 50 slots with
NO_ATOM/C/N/O/F/P/S/Cl/Br/I, formal charges -2 through +2, and
NONE/SINGLE/DOUBLE/TRIPLE/AROMATIC bonds. Hydrogen counts, radicals, isotopes,
stereochemistry, metals and dative bonds are not separate decoder features.
Implicit hydrogen is inferred during graph construction. A lower heavy-atom
bond-order sum is therefore not, by itself, a defect to drive toward the
largest valence.

RDKit supports charge-dependent valences and can normalize some alternative
nitro, azide, phosphorus and halogen-oxo encodings during sanitization. Aromatic
Kekulization and hydrogen handling add separate constraints. A normalization
that changes labels must not be reported as evidence that the chemical species
is impossible.
[RDKit sanitization and valence documentation](https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences).

The existing stages remain separate:

| Stage | Existing code | Meaning |
| --- | --- | --- |
| Local differentiable rule | [`chemistry_regularization.py`](../src/latentmol/chemistry_regularization.py) | A graded surrogate; no RDKit call in its gradient path. |
| Frozen failed-repair preparation | [`repair_failure_guidance.py`](../src/latentmol/repair_failure_guidance.py) | Detached labels, caps, components and optional witness; gradients follow the declared live bond loss. |
| Atom-preserving repair | [`atom_preserving_chemistry.py`](../src/latentmol/atom_preserving_chemistry.py) | Discrete bond edits with atoms, charges, occupancy and slot alignment held fixed. |
| Graph construction/sanitization | [`chemistry.py`](../src/latentmol/chemistry.py) | Hard-graph RDKit checks, connectivity and radical checks. |
| Chemical versus representation classification | [`chemistry_classification.py`](../src/latentmol/chemistry_classification.py) | Raw chemistry, serialized SMILES and strict schema roundtrip have different outcomes. |

Passing a local policy does not establish global connectedness, a valid aromatic
assignment, synthesizability, stability or useful binding. These require the
existing separate checks and downstream evaluation.

## Probability and gradient semantics

The new implementation names its version and serializes its complete policy.
It must leave historical versions and active defaults reproducible. PyTorch and
JAX adapters must use the same category meanings and the same declared event;
matching a label such as “valence loss” is insufficient.

[`prepare_neighbor_environment`](../src/latentmol/chemical_environment.py)
freezes atom labels, charge labels, occupancy
and the outside-bond context. `neighbor_environment_loss` evaluates live bond
probabilities under that prepared environment. `assess_environment` provides
the corresponding hard local-policy assessment in the
[dependency-free canonical policy](../src/latentmol/chemical_environment_policy.py).
The canonical policy and both framework adapters must agree on these meanings.

For each occupied focal slot i, a log-domain dynamic program tracks the joint
distribution of its incident bond-order sum and sulfur's non-fluorine
coordination count. It retains probability mass only where neighbor remaining
capacities, focal valence and the declared sulfur coordination rule all pass.
The local negative log mass is

\[
 m_i=P(A_i\mid i\text{ occupied}),\qquad
 \ell_i=-\log m_i,
\]

where A_i is this combined event, conditional on the prepared labels and
occupied slots. The PyTorch adapter converts symmetric bond logits to FP64
before log-softmax and accumulates the dynamic program in log space. It does
not renormalize away prohibited edge labels or place an epsilon floor on the
admissible event. A safe log-sum-exp handles unreachable states. The all-NONE
assignment keeps the event nonempty for finite logits; a clamp at log mass zero
only prevents a negative risk from numerical roundoff.

The scalar loss is the mean risk over **all active centers with supported
caps across the batch**, not a separate mean for each graph. The required
explicit config is `{"version": "common-organic-neighbor-v1", "weight": w}`
with a finite nonnegative w. Both unweighted and weighted losses are returned.
The result also includes per-center risks, assessed/unassessed center counts
and a receipt binding the prepared policy/context by digest. These numerical
and normalization choices must not silently change in another framework.

**Only live bond probabilities receive direct gradients.** Atom identities,
charges, occupancy, outside bonds and derived capacities are detached constants
for one preparation. There is no direct live carbon-to-sulfur substitution
benefit through this loss. If the latent vector or decoder parameters are
updated, shared computations can still change atom distributions indirectly;
the policy is not a proof against all composition drift. Refreshing the hard
context also changes the next local objective and must be recorded.

The term stays enabled for already-accepted argmax graphs. If their soft bond
distributions still assign mass to prohibited alternatives, moving that mass
into the accepted event can give a useful gradient. It is not limited to rows
where discrete repair previously failed. The dynamic program uses categorical
tails rather than the hinge of a mean bond-order sum.

The implementation review must verify that:

1. The permitted event combines neighbor remaining capacity, focal valence and
   sulfur non-fluorine coordination, rather than merely changing sulfur's cap.
2. Probability on bad alternatives can produce a useful finite gradient even
   when the argmax graph is already accepted. Exact categorical boundary cases
   can have zero loss; a saturated softmax need not have a large gradient.
3. Padding, absent endpoints and diagonal entries do not create spurious
   neighborhood mass. Neighbor occupancy is applied once.
4. The two orientations of an undirected bond agree, and slot permutations do
   not change the graph-level loss.
5. An all-empty or disconnected graph cannot be declared valid by a successful
   local term. Existing size, occupancy and connectivity checks remain separate.
6. Chemical-policy coverage, representation failure and numerical failure are
   reported separately. Invalid categorical input fails explicitly. An
   element/charge combination without a supported cap is recorded as
   unassessed, rather than assigned an invented cap or certified as valid.
   The assessment reports ordinary cap violations and sulfur scope exclusions
   separately; it does not run native chemistry.

A graded local likelihood is not the minimum number of edits to any valid
molecule. The search-based frozen witness remains a different objective. Nor
does this local term eliminate shared-latent composition drift: composition
monitoring and any already-declared composition regularizer remain separate.

## Proposed legitimate-molecule regression panel

**This table is a chemistry coverage catalogue, not a per-input execution
receipt.** Source145 attempt001 passed all 61 positive cases in its actual
sealed fixture list; that result does not assert strict-schema acceptance of
every example listed below. An allocated RDKit/schema check must establish
each actual outcome before it becomes an asserted fixture. In particular,
hydrogen-sensitive or normalized forms may pass raw chemistry while failing
the present strict representation. Charge separation must not be silently
discarded to make a test pass.

| Family | Proposed input or source-defined structure | Pitfall tested |
| --- | --- | --- |
| Thiol, thioether | `CS`, `CSC` | Implicit H and ordinary sulfur valence. |
| Disulfide, thiosulfonate | `CSSC`, `CSS(=O)(=O)C` | S-S bonds are not universally forbidden. |
| Thiocarbonyl | `CC(=S)C` | A sulfur-carbon double bond can be ordinary chemistry. |
| Sulfoxide forms | `CS(C)=O`, `C[S+](C)[O-]` | Neutral expanded-valence and charge-separated encodings. |
| Sulfone | `CS(C)(=O)=O` | Carbon ligands around S(VI). |
| Sulfonamide, sulfamate | `CS(=O)(=O)N`, `COS(=O)(=O)N` | Nitrogen and oxygen single-bond ligands. |
| Sulfonic acid, sulfonate | `CS(=O)(=O)O`, `CS(=O)(=O)[O-]` | Acid/anion states and implicit H. |
| Sulfinic acid, sulfate | `CS(=O)O`, `OS(=O)(=O)O` | Different ligand/bond-order combinations. |
| Sulfimide | `CS(C)=NC` | S=N without an oxo group. |
| Sulfoximine | `CS(C)(=O)=NC` | One S=O and one S=N. |
| Sulfonediimine | `CS(C)(=NC)=NC` | Two S=N groups without oxygen. |
| Sulfonimidate | `COS(=O)(=NC)C` | Mixed S-O, S-C, S=O and S=N. |
| Sulfur fluoride | `FS(F)(F)(F)(F)F` | Six single bonds can be legitimate. |
| Carbon-substituted SF5 | `CS(F)(F)(F)(F)F`, `c1ccccc1S(F)(F)(F)(F)F` | Six singles need not mean SF6. |
| Sulfonium, sulfoxonium | `C[S+](C)C`, `C[S+](C)(C)=O` | Charge-dependent sulfur states. |
| Thiolate | `C[S-]` | An anionic sulfur ligand. |
| Sulfur ylide | `C=S(C)C`, `C[S+](C)[CH2-]` | Neutral and charge-separated forms; check omitted-H behavior. |
| Four-carbon sulfurane | Bis(2,2'-biphenylylene)sulfurane from the original 1992 paper | A genuine all-carbon exception; do not invent its SMILES from a name. |
| Six-carbon persulfurane | Bis(2,2'-biphenylylene)dimethylpersulfurane from the original 2006 paper | Intentionally excluded by the simple common-organic policy, while remaining a legitimate-chemistry coverage exception. |
| Phosphine | `CP(C)C` | Ordinary P(III). |
| Phosphine oxide/sulfide | `CP(C)(C)=O`, `CP(C)(C)=S` | P=S as well as P=O; expanded valence. |
| Charge-separated phosphine oxide | `C[P+](C)(C)[O-]` | Do not conflate normalization with chemical failure. |
| Phosphate, phosphonate | `OP(=O)(O)O`, `CP(=O)(O)O` | Oxygen-rich P(V) families. |
| Phosphonium, ylide | `C[P+](C)(C)C`, `CP(C)(C)=C`, `C[P+](C)(C)[CH2-]` | Distinguish charge support and neutral/ionic resonance forms. |
| Ammonium, oxonium | `C[N+](C)(C)C`, `C[O+](C)C` | Neutral-valence rules must not be applied to charged atoms. |
| Carboxylate | `CC(=O)[O-]` | Legitimate charge and resonance localization. |
| Nitro, azide, diazo | `C[N+](=O)[O-]`, `CN=[N+]=[N-]`, `C=[N+]=[N-]` | Charge-separated normal forms. |
| Pyridine, thiophene | `n1ccccc1`, `c1ccsc1` | Ring aromaticity is not a maximum-valence test. |
| Pyrrole, pyridine N-oxide | `[nH]1cccc1`, `[O-][n+]1ccccc1` | Hydrogen-sensitive and charged aromatic states. |
| Isocyanide, carbon monoxide | `[C-]#[N+]C`, `[C-]#[O+]` | Unusual legitimate charge states are not automatically drug-like or useful. |

Disconnected counterions, isotopes, radicals, metals and dative bonds should be
retained as explicit out-of-schema or out-of-scope controls. Their exclusion
from this project is not a claim that such chemical species do not exist.

## Adversarial, gradient and runtime checks

The allocated test suite should contain paired structures which defeat a
maximum-valence-only shortcut, together with the legitimate panel above:

* Equal bond-order sums with different degrees, neighbor elements or double-bond
  partners. Label common-motif exclusions explicitly; do not call every carbon-
  ligated hypervalent sulfur structure physically impossible.
* A clear cap violation; neutral four-sigma nitrogen versus ammonium; neutral
  three-sigma oxygen versus oxonium; and undercoordinated atoms completed by
  implicit hydrogen.
* Equal means with different categorical violation tails, plus valid argmax
  graphs carrying nonzero probability on prohibited alternatives.
* Soft atoms/charges/occupancy with a frozen prepared context: confirm that
  their direct gradients are absent while the declared live bond gradients
  remain finite and agree across frameworks. Shared-latent indirect effects
  must not be confused with a direct atom-logit derivative.
* Directional AD/finite-difference agreement, finite losses/gradients at extreme
  logits, and probability improvements after a small bounded gradient step.
* PyTorch/JAX value and gradient parity, symmetry, permutation and padding
  invariance, and rejection of unsupported vocabularies.
* Aromatic labels outside a valid ring, unresolvable aromatic assignments,
  disconnected nontrivial components and empty graphs. A locally accepted
  neighborhood must not bypass the existing global validity checks.

Record wall time, peak memory, graph size, batch size, precision and framework
for the new loss and its existing comparator. Do not infer that a differentiable
implementation is fast enough for continuation from a tiny single-graph test.
No training or optimization may adopt a new loss merely because a derivative
test passes; composition, validity, reconstruction and the existing readiness
checks remain required under the applicable experiment protocol.

## Implementation and experiment receipts

Selected design: `common-organic-neighbor-v1`, fixed-context bond-gradient
guidance, at most four non-fluorine sulfur neighbors, with existing valence
caps. The root agent owns the canonical policy and PyTorch implementation;
the JAX adapter implements the same shared policy and passed the allocated
cross-framework qualification recorded below.
The opt-in
[`repair_with_neighbor_environment`](../src/latentmol/chemical_environment.py)
wrapper adds the weighted event loss to the existing failure-aware repair
total without rewriting the historical repair implementation. Preparation
uses original decoder hard labels, never the accepted repair witness. The
context stays fixed throughout trial evaluations and refreshes only explicitly
between accepted optimizer states.

The JAX adapter is in Mosaic's
[`environment.py`](../../mosaic/src/mosaic/small_molecule/chemistry/environment.py).
Its dependency-free `environment_policy.py` must remain byte-identical to this
repository's canonical policy. Each adapter runs in its own existing CPU
environment; the validation runner does not mix their package installations.

### Source145 attempt001: partial validation, interpreter startup failure

The immutable `source145-neighbor-environment-validation-attempt001` was
successfully submitted as **CPU29848286** to the `hpc` queue: four CPU slots,
4 GiB per slot (16 GiB total), one host, 60-minute wall time and 55-minute
internal budget. It requests no GPU. Submission does not establish that the
job has started, that a test passed, or that the new objective is ready for
training or molecule optimization.

The sealed
[`run_neighbor_environment_validation.py`](../scripts/hpc/run_neighbor_environment_validation.py)
first verifies every manifest member and exact portable-policy byte parity.
Its declared stages are:

1. PyTorch/RDKit new motif and gradient fixtures plus existing atom-preserving
   repair and failure-guidance regression fixtures. The new
   [`test_chemical_environment.py`](../tests/test_chemical_environment.py)
   covers charge/resonance/aromatic families, sulfur coordination exclusions,
   neighbor capacity, valid-argmax probability tails, an extreme-logit bounded
   improvement, detached rule choices and context tampering. The independent
   [`reference fixtures`](../tests/test_chemical_environment_reference.py)
   enumerate local events directly, rather than reuse the production DP, and
   compare values/VJPs and permutation/padding/logit-offset invariance.
2. The allocated
   [`PyTorch probe`](../scripts/hpc/neighbor_environment_probe.py) audits
   **10,000 validation-split molecules**, sampled without replacement with
   seed145091 from the authenticated VAE corpus. It requires all rows' local
   caps to be assessed and allowed, and common-organic sulfur scope exclusions
   to be at most **0.1%**. It retains excluded examples and element coverage.
   This is a bounded held-out corpus audit, not a full chemical-space guarantee.
3. The same probe authenticates the frozen completed epoch40 checkpoint,
   decodes **two seeded latent points**, and requires a finite nonzero actual
   decoder-to-latent gradient. It records each latent's gradient norm, checks
   the model state digest is unchanged and confirms that model parameters have
   no gradients. This tests a gradient path, not useful molecular improvement.
4. Mosaic's JAX fixtures run in its separate CPU environment. Six fixed
   PyTorch/JAX cases then compare hard-policy assessments, FP64 scalar losses
   and the complete VJP with respect to the raw bond-logit tensor, using an
   **absolute tolerance of 1e-9** for scalar and maximum gradient error.
   These cases include sulfonamide, six-carbon sulfur scope exclusion, SF6,
   nitro, five-carbon phosphorus and thiophene contexts.

Any fixture failure, error or skip stops promotion. A failed stage remains
recorded, and a successful final result is explicitly named
`passed_validation_not_training_adoption`. No VAE continuation, Boltz binding
evaluation, DeFoG experiment or model training runs in this job.

The frozen checkpoint SHA256 is
`31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e`.
The dataset-manifest SHA256 is
`f246cc93805b73a557cd0c09a78891ac8410c465192db615ce0b6709218c8c0e`.
The release has **656 members**, archive SHA256
`be299bcb78aa47de4cdc4f1b296d4f2d639788df292ee21118f37fee3565c609`
and manifest SHA256
`e83f527a3eb598a268f8c46f698ac8714bba3aed3fb760d427def781d5e2e6be`.
The [seal](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt001/seal.json)
and [successful submission receipt](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt001/stage-submit-1793277772348699500.json)
bind the actual command and resource request.

The [terminal result](../outputs/hpc-results/source145-neighbor-environment-validation-attempt001/snapshot-1793277963521928700/result.json)
records **CPU29848286 EXIT1**, with these completed partial results:

| Stage | Actual result | Scope |
| --- | --- | --- |
| PyTorch/RDKit fixtures | **138 passed**, zero failures/errors/skips; 22.797s stage | Includes all 61 positive motif cases and existing repair regressions. |
| Validation corpus | **10,000/10,000** fully assessed, local-cap accepted and within common-organic scope; 83.214s | Includes **3,337 sulfur-containing molecules**; no excluded example. |
| Actual frozen epoch40 VJP | Loss **0.007848198394372286**; latent gradient norm **0.06656162440776825**; 1.244s | Two latents have gradient norms 0.06482106447219849 and 0.015122141689062119; model unchanged, no parameter gradients. |
| JAX fixtures | Interpreter failed **before tests** | No JAX or cross-framework pass established. |

The full PyTorch probe took 96.271s; the unsuccessful complete driver took
119.115s. The two-latent VJP timing is a small CPU observation, not a training
throughput benchmark or an estimate for a complete epoch. The
[corpus audit](../outputs/hpc-results/source145-neighbor-environment-validation-attempt001/snapshot-1793277963521928700/corpus-audit.json)
and [frozen VAE VJP receipt](../outputs/hpc-results/source145-neighbor-environment-validation-attempt001/snapshot-1793277963521928700/frozen-vae-vjp.json)
retain exact measurements. Coverage is for the sampled existing validation
split; it is not proof that every legitimate molecule is accepted or that
optimization will improve sulfur composition.

The [JAX startup log](../outputs/hpc-results/source145-neighbor-environment-validation-attempt001/snapshot-1793277963521928700/jax-tests.log)
identifies the mechanism: DTU's loaded Python3.11 module exported
`PYTHONHOME=/appl9/python/3.11.9`, but Mosaic's virtual environment uses
Python3.12. Its interpreter searched the wrong standard-library tree and
failed to import `encodings`. This is an interpreter-environment launch error,
not an observed chemical-policy or gradient failure. Attempt001 stays failed;
its successful partial stages do not become a complete qualification.

### Source145 attempt002: interpreter fixed, test-layout packaging failed

The new immutable attempt002 removes inherited `PYTHONHOME` from the child
environment before launching each existing framework interpreter. It does not
change the canonical policy, PyTorch/JAX loss modules, scientific fixtures,
coverage threshold, parity tolerance, checkpoint or dataset. It repeats the
complete qualification rather than promoting attempt001's partial result.

**CPU29848305** was successfully submitted with the same four-slot/16-GiB,
60-minute wall/55-minute internal, CPU-only request. It has 656 members,
archive SHA256
`bf5e5b50b125bef87e29c8f7413314bf2e32101efda1821b0e6702b0c3ece033`
and manifest SHA256
`3e350179306ec48dc64e5f4d6d37d3ee878572d9ef610ac4647effd0f5941216`.
The [attempt002 seal](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt002/seal.json)
and [actual submission receipt](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt002/stage-submit-1793278005064573800.json)
bind this retry.

The [attempt002 terminal result](../outputs/hpc-results/source145-neighbor-environment-validation-attempt002/snapshot-1793278140762905800/result.json)
again records **138 PyTorch fixture passes**, zero failures/errors/skips, plus
the repeated [10,000-row coverage pass](../outputs/hpc-results/source145-neighbor-environment-validation-attempt002/snapshot-1793278140762905800/corpus-audit.json).
All local caps and common-organic scope checks passed; the sample again
contained 3,337 sulfur-containing molecules. The
[frozen decoder VJP](../outputs/hpc-results/source145-neighbor-environment-validation-attempt002/snapshot-1793278140762905800/frozen-vae-vjp.json)
was finite/nonzero, loss 0.007848208385989765 and total latent gradient
norm 0.06656170636415482, with unchanged model state and absent parameter
gradients. These are repeated partial successes, not a full qualification.

JAX's interpreter now initialized, but pytest exited 4 while loading Mosaic's
`tests/conftest.py`: its import of `experiment_layout` could not resolve
because `scripts/layout/experiment_layout.py` was absent from the sealed
bundle. The [actual import log](../outputs/hpc-results/source145-neighbor-environment-validation-attempt002/snapshot-1793278140762905800/jax-tests.log)
records this before test collection. No JAX loss assertion or cross-framework
comparison ran. Attempt002 remains failed and preserved.

### Source145 attempt003: packaging corrected, result retrieval blocked

Attempt003 adds exactly the missing Mosaic test-layout dependencies to the
bundle: four Python files under `scripts/layout/` (`audit_documentation.py`,
`experiment_layout.py`, `verify_layout.py`, `verify_pytest_routing.py`) and
`experiments/folder-layout.json` plus `experiments/migration-map.json`. It
retains the child-environment correction and repeats all qualification stages.

**CPU29848323** was successfully submitted at about 13:50 CET on 29 October 2026,
with the same four-slot/16-GiB, 60-minute wall/55-minute internal CPU-only
request. Its 662-member archive SHA256 is
`ab86bb6346b283428b4d7df904a70014653cc2326f69e7fef010f3fea4b747fe`,
and manifest SHA256 is
`043a0583faa6276e200a0e8088942a7a487d2eaae30209f03b7c75bfd2eca9ce`.
The [attempt003 seal](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt003/seal.json)
and [successful submission receipt](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt003/stage-submit-1793278218960542000.json)
establish submission, not execution or success.

The earlier [collection receipt](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt003/collect-1793299042880108400.json)
has return code 255, `Permission denied (password)` and empty standard output.
It contains no scheduler state or numerical result. At 19:40 CET the only
verified state was submitted with results uncollected; that access error
established neither experiment failure nor a still-pending scheduler state.
The terminal evidence obtained later, described below, supersedes that
temporary information gap without erasing the failed collection receipt.

The [static attempt-history audit](../outputs/implementation-checks/source145-attempt-history-static.json)
confirms that all eight new scientific policy/PyTorch/JAX/test/trajectory files
are byte-identical across all three releases. The changes are limited to the
attempt002 child Python environment, attempt003 test-layout packaging and
documentation. This is a metadata audit, not a numerical pass. All failed
launches and successful partial evidence remain preserved; no attempt changes
an active run or authorizes adoption for training or binding optimization.

### Source145 attempt003 terminal evidence and attempt004 fixture correction

The working unattended connector subsequently retrieved the
[attempt003 terminal result](../outputs/hpc-results/source145-neighbor-environment-validation-attempt003/snapshot-1793305764351307000/result.json).
It confirms 138 PyTorch passes, zero failures/errors/skips; the same 10,000-row
validation audit fully assessed/accepted every molecule, including 3,337 with
sulfur. The actual frozen40 VJP again had finite nonzero gradients and unchanged
model state. JAX ran 26 fixtures: **24 passed, two failed**, zero errors/skips.
The cross-framework probe remained blocked by those failed fixtures.

The [JAX failure log](../outputs/hpc-results/source145-neighbor-environment-validation-attempt003/snapshot-1793305764351307000/jax-tests.log)
shows that both failures were the FP32/FP64 variants of the independent
enumeration/finite-difference fixture. The independently enumerated values and
per-center risks had already matched. Its selected C-S SINGLE-versus-DOUBLE
direction had analytic derivative about 2.48e-17, but the fixture incorrectly
required magnitude above 1e-8. Those two bond categories satisfy the same
prepared events in that fixture: a nearly zero derivative is the correct
answer for that direction, not evidence that all loss gradients vanish.

Attempt004 changes **only the Mosaic fixture**: it now explicitly asserts
that old direction remains flat within 2e-12, then performs the informative
finite-difference check on the S-O DOUBLE-versus-TRIPLE direction. Oxygen's
remaining capacity admits DOUBLE but excludes TRIPLE. The nonzero requirement
of 1e-8 and existing finite-difference tolerances are unchanged. No PyTorch/JAX
loss, canonical policy, updater, scientific default or active run was changed.
The allocated attempt004 result below confirms that the corrected fixture and
the full declared qualification pass.

Immutable attempt004 has 662 members, archive SHA256
`9b1e408a4bcb9794980c6685ee7c1e5511794a7d6765c44c279b754f50370068`
and manifest SHA256
`c00079b4c0c45dcc5c1e49ff405deeb547f0525db7b9692871f4f00f41161fda`.
Its [seal](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt004/seal.json)
and [actual helper submission receipt](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt004/stage-submit-1793306306292703900.json)
confirm **CPU29851083** submitted at about 21:38 CET, with the unchanged
four-slot/16-GiB/60-minute wall/55-minute internal CPU-only request. Submission
alone established no test pass; the actual terminal evidence follows. All
previous failed releases and partial results remain intact.

### Source145 attempt004: complete allocated CPU qualification passed

The [terminal result](../outputs/hpc-results/source145-neighbor-environment-validation-attempt004/snapshot-1793306498640264200/result.json)
records **CPU29851083 DONE**, complete driver elapsed 175.52931658783928s and
status `passed_validation_not_training_adoption`:

| Check | Actual result |
| --- | --- |
| PyTorch fixtures/regressions | 138 passed; zero failures/errors/skips; stage 26.249s. |
| JAX fixtures | 26 passed; zero failures/errors/skips; stage 37.380s. |
| Six-case FP64 scalar parity | Maximum absolute error **4.440892098500626e-16**. |
| Six-case raw-bond VJP parity | Maximum absolute error **8.326672684688674e-17**. |
| Hard-policy assessments | Equal in all six cases. |
| Audited validation corpus | 10,000/10,000 fully assessed, local-cap accepted and in scope; 3,337 sulfur-containing; zero exclusions. |
| Actual frozen40 latent VJP | Loss 0.007848198394372286; gradient norm 0.06656162440776825; model unchanged, parameter gradients absent. |

The [exact parity receipt](../outputs/hpc-results/source145-neighbor-environment-validation-attempt004/snapshot-1793306498640264200/cross-framework-parity.json)
keeps each case's errors; both maxima are below the unchanged absolute 1e-9
threshold. The [corpus audit](../outputs/hpc-results/source145-neighbor-environment-validation-attempt004/snapshot-1793306498640264200/corpus-audit.json)
took 88.765s. The [actual decoder VJP](../outputs/hpc-results/source145-neighbor-environment-validation-attempt004/snapshot-1793306498640264200/frozen-vae-vjp.json)
took 0.697s for two latents, with individual gradient norms 0.06482106447219849
and 0.015122141689062119. These timings are limited CPU measurements, not a
training-throughput benchmark or a complete-epoch estimate.

The [attempt003-to004 byte audit](../outputs/implementation-checks/source145-attempt003-to004-static.json)
shows only the Mosaic fixture, documentation/agent instructions and manifest
changed; no production policy/loss/updater or validation driver changed.
This is a full pass of the **declared allocated CPU qualification**, including
the fixed-context gradient path and sampled corpus coverage. It does not show
that molecular optimization reduces sulfur inflation, that a new training loss
improves generation, or that all rare chemistry is preserved. Those are separate
scientific experiments. Existing runs, scientific defaults and safeguards remain
unchanged; the feature stays explicitly opt-in.

This document will add terminal validation receipts after they exist. All
historical runs/defaults and all failed evidence remain
preserved. This implementation does not switch any active VAE continuation,
Boltz/DeFoG run or Mosaic trajectory to the new objective.
