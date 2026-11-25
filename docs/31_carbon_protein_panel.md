# Carbon-chain elemental response across ten protein contexts

Status, 5 October 2026: **submitted, with numerical fixtures and native
execution pending HPC validation**. Primary job **29604186** and boundary
continuation job **29604187** were accepted and both observed **PEND** at
23:55:21 DTU local time. They run serially after source056 job 29603601,
which was also pending. The user selected ten different
carbon-chain lengths per protein. Source057 prepares **C6–C15 across ten
full canonical human protein sequences**, for 100 declared target/length
cases. No passed numerical fixture, native preparation or molecule result
is established by these receipts. Exact immutable source/config/asset
hashes, receipts and subsequent outcomes are recorded in the
[experiment ledger](05_experiment_ledger.md).

The separate source056 ten-random-latent-start study remains unchanged.
This panel uses the existing frozen stable VAE and Boltz-2; it trains no
new encoder, decoder, chemistry prior or scorer.
The strict study config is
[configs/boltz_protein_panel.yaml](../configs/boltz_protein_panel.yaml),
SHA-256 `41694234e8a5af4c3a5fab8dd57ddaeab0e4b88ef5baea5e3283666802568c54`.

## Question and interpretation

Can gradients from different protein contexts move an encoded carbon-only
chain toward noncarbon atom types, and do those changes appear only as soft
probability shifts, as transient hard labels, or in the final hard graph?
The primary assay deliberately uses **fixed occupancy**, one
`fixed_initial_mask` arm for each target/length. Its number of scored atoms
does not change. The decoder's ordinary occupancy preference is still
logged, but this panel does not establish atom-count adaptation.

Ten lengths are ten starting chemical identities, not ten random seeds
or independent replicates of one chain. At a given length the frozen
encoder mean supplies the start, without redraw, repair or latent fitting.
Different targets can change the prepared pose/context and its gradient.
Varying length also changes starting size and native pose; neither factor
is isolated by pooling all 100 cases. Report results by target and length
as well as the overall original denominator.

## Target manifest and context limits

The local preparation manifest is
`outputs/hpc-submissions/source057-protein-assets/targets.json`. It retains
the official UniProt JSON record, canonical sequence, sequence SHA-256,
entry/sequence versions, download timestamp and one explicit query-only
A3M per target. The raw records and A3Ms have separate hashes. Data come
from the declared UniProt REST URLs under its CC BY 4.0 license; source
attribution is retained in that manifest.

The [asset-fetch helper](../scripts/fetch_protein_panel_assets.py) downloads
official records and materializes these pinned files; HPC jobs use the
staged files without requesting an MSA server. The
[static checker](../scripts/static_protein_panel_check.py) parses source
and checks panel/file identities without running a model. The ledger
distinguishes those local file/static operations from numerical HPC tests.

| Case target | Gene | UniProt accession | Full sequence residues |
| --- | --- | --- | ---: |
| Carbonic anhydrase 2 | CA2 | P00918 | 260 |
| Dihydrofolate reductase | DHFR | P00374 | 187 |
| Peptidyl-prolyl cis-trans isomerase FKBP1A | FKBP1A | P62942 | 108 |
| Mitogen-activated protein kinase 1 | MAPK1 | P28482 | 360 |
| Serine/threonine-protein kinase pim-1 | PIM1 | P11309 | 313 |
| Aldo-keto reductase family 1 member B1 | AKR1B1 | P15121 | 316 |
| Tyrosine-protein phosphatase non-receptor type 1 | PTPN1 | P18031 | 435 |
| Glutathione S-transferase P | GSTP1 | P09211 | 210 |
| NAD(P)H dehydrogenase [quinone] 1 | NQO1 | P15559 | 274 |
| Peptidyl-prolyl cis-trans isomerase A | PPIA | P62937 | 165 |

These are full canonical input sequences, rather than manually selected
domains. The native affinity featurizer may still crop its predicted
context; retain the prepared context's actual native features and maps.
The declared input uses one protein chain and a carbon-chain ligand, with
**no evolutionary MSA, cofactors, metals, templates or pocket constraints**.
Some listed proteins depend on omitted components or native multimeric
contexts. This is a sequence-only engineering assay, not a realistic
target-binding benchmark or a counter-target/selectivity panel. The
single-query policy differs from the earlier official-example C13 MSA,
so cross-experiment score differences do not isolate initialization alone.

## Pipeline and ownership of each step

The [panel controller](../src/latentmol/boltz/protein_panel.py) validates a
strict data-only protocol, pinned baseline config, target manifest, model
assets and executable-source manifest. It constructs the Cartesian panel
with **length outermost and target innermost**: all ten proteins at C6,
then all ten at C7, through C15. This order provides earlier coverage of
all targets if an allocation ends before the panel finishes. All 100 case
rows exist before the first case starts.

For each case, the controller runs separate logged subprocesses:

1. The [generic target-chain input helper](../src/latentmol/boltz/prepare.py)
   verifies the canonical sequence against the pinned official record and
   the A3M query, then writes a native protein/SMILES YAML and input
   manifest. Carbon-chain identity is declared explicitly as `C` repeated
   for that case's heavy-atom count; the relaxed graph is not converted
   into an input SMILES inside the gradient path.
2. Ordinary native Boltz prediction creates a reference pose using seed
   **17**, zero recycling, **20 diffusion steps** and one diffusion sample
   in both declared structure/affinity sampling budgets. This preparation
   is outside autograd. Each target/length receives its own new native
   preparation rather than reusing another target's features.
3. The snapshot helper exports the native tensor context and slot maps,
   recording target sequence, MSA, ligand and reference-pose identities.
   It checks the declared input against the resulting seed graph. These
   provenance checks do not independently prove every processed native
   protein/MSA tensor matches the original files; that existing native
   preprocessing audit limit remains.
4. A child configuration clones the historical source055 optimization
   settings, replaces the target/count context, and requests only
   `fixed_initial_mask` with `element_diagnostics: true`. The
   [active-set runner](../src/latentmol/boltz/active_set.py) encodes the
   native carbon chain and requires its exact decoded mask, atom, charge
   and bond labels. A failed start audit is retained; the case is not
   replaced by a nearby embedding or another chain. This optimization
   gate occurs after the ordinary native preparation above.
5. The child checks frozen-model identity, hard/native affinity parity and
   conditional finite differences before optimizing only molecular `z`.
   The controller retains child logs, partial states and failures. A
   completed case must have an unchanged frozen state, one declared arm,
   100 updates and 101 readable metric states. Terminal cases receive
   independent file-hash seals before they can be skipped on continuation.

Generic input preparation, process supervision, differentiation and
artifact reporting remain separate modules. Neither the controller nor
the report creates a learned reward model or independently decodes a
second chemical identity for another scorer.

## Frozen objective, budgets and gradient scope

The decoder is the source047 stable VAE `best_validity.pt`, selected at
epoch 89; its checkpoint SHA-256 is
`1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.
The frozen Boltz affinity checkpoint retains SHA-256
`dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e`.
The historical source055 config is pinned by file hash; target/pose assets
are replaced explicitly for each new case.

Each case has **100 updates**, learning rate **400**, maximum raw latent
gradient norm **0.000125**, trust radius **3**, temperature **1**, and
ordinary occupancy threshold **0.5**. The proposal displacement is at
most 0.05 before trust projection. The binary affinity output is maximized
with weight/scale 1 and offset 0; this does not optimize the continuous
affinity head. Formal-charge feature loss remains explicitly allowed under
the existing lossy Boltz interface.

Finite-difference probes use `[0.001, 0.003, 0.01, 0.03, 0.1]`, relative
tolerance `0.05`, absolute tolerance `1e-6` and two passing scales required
per direction. Native parity retains relative/absolute tolerances
`1e-4`/`1e-5`. These are the original tolerances with a broader epsilon
grid, not relaxed gates. One argmax and eight categorical draws are saved
at each state: 909 configured hard candidates per completed case. Those
draws are correlated diagnostics, not 909 independently optimized molecules.

The Boltz graph conditions each included slot on real-element categories
and keeps its starting slot mask. There is no direct affinity derivative
through atom selection or `NO_ATOM`; excluded slots do not become scored
atoms. Geometry, native feature context and tokenization remain fixed
during these gradients. Decoder weights and Boltz weights stay frozen.

## Element diagnostics, including invalid hard graphs

The [element diagnostic helper](../src/latentmol/boltz/element_diagnostics.py)
reports before RDKit conversion. It captures the same scalar loss and
backward pass used for `z`; diagnostics do not add a second objective.
Every selected slot retains its conditional real-element probabilities,
hard atomic-number label and carbon-logit margin over its strongest
noncarbon competitor. The vocabulary is C, N, O, F, P, S, Cl, Br and I,
plus `NO_ATOM` in the ordinary decoder distribution.

`conditional_expected_heteroatom_count` sums noncarbon probability over
the fixed selected slots after excluding `NO_ATOM`. The separate
`raw_expected_heteroatom_count_all_slots` and active-slot raw count use the
ordinary decoder distribution, including its occupancy competition. These
quantities answer different questions and must not be substituted for
each other or called measured atom counts.

Gradient-bearing states additionally retain atom-logit and atom-probability
gradient norms, per-slot logit gradients, and probability-gradient
contrasts against carbon. These are derivatives of the **signed minimized
loss**. A negative nitrogen-minus-carbon contrast locally favors shifting
probability from carbon to nitrogen while holding other modeled inputs
fixed. It does not guarantee that the frozen decoder can realize that
shift through `z`, that hard argmax will change, or that the result will
be chemically valid. Gradient-free endpoint records have no such backward
measurement.

The per-case trace counts first, transient and last-observed noncarbon
hard labels even when `argmax_smiles` is null. A nitrogen that appears and
later returns to carbon is retained by `ever_noncarbon`; it is not reported
as a nitrogen-containing final molecule. Last-observed partial states are
distinguished from verified step-100 endpoints. The summary keeps planned,
completed, failed, unstarted and observed-state denominators, including
partial cases; unreadable metrics are reported rather than silently counted
as completed evidence.

## One serial allocation and later boundary continuation

The queue's wall limit is 24 hours; each submitted allocation requests
**23 hours**, with four cores, 8 GB per core and one exclusive `gpua40`
GPU on one host. The controller's **22-hour budget** leaves room
for startup and persistence. Before starting a case it reserves the
configured worst-case child timeouts plus 120 seconds; the planned total
reserve is **4,620 seconds**. The panel may stop at a safe boundary with
unstarted cases. A completed allocation does not mean all 100 cases were
processed or that every processed case succeeded.

The [allocated wrapper](../scripts/hpc/boltz_protein_panel.sh) runs the
selected fixture modules first and only then launches the panel. All
numerical fixtures, native preprocessing and molecule evaluations run on
HPC. Local parsing/packaging alone does not validate the new real-model
workflow. The [submission helper](../scripts/hpc/submit_protein_panel.sh)
requests one long job; it creates no job array, seed workers or automatic
compute-node submissions. The selected modules are `test_protein_panel`,
`test_boltz_active_set`, `test_boltz_active_set_runtime`,
`test_boltz_active_set_rescore`, `test_boltz_chain_prepare` and
`test_random_start_study`; the new panel fixture includes interrupted-case
retry evidence. They are unrun until an HPC allocation executes them.
The following symbolic interfaces explain the submitted helpers; actual
invocations and receipts are in the ledger:

```bash
bash scripts/hpc/submit_protein_panel.sh \
  RELEASE_DIR ARCHIVE_SHA256 PROTOCOL PROTOCOL_SHA256 RECEIPT_DIR [AFTER_JOB] [resume]
```

Inside a verified allocation the controller invocation is:

```bash
python -m latentmol.boltz.protein_panel \
  --config PROTOCOL --sha256 PROTOCOL_SHA256
```

A later continuation adds `--resume` under the **same source archive and
protocol hashes**. It verifies the fixed case order, asset identities and
terminal case seals; completed and ordinary failed cases are not rerun.
An interrupted nonterminal case retains its old attempt and receives a
new uniquely named attempt. This is continuation at case boundaries, not
resuming an individual latent trajectory. Any later allocation needs its
own unique receipt and ledger record; queue dependencies serialize
allocations rather than changing the panel's identity. Source056 remains
a separate sealed study.

The primary 23-hour allocation **29604186** has dependency
`ended(29603601)`. The 23-hour boundary-continuation allocation **29604187**
has dependency `ended(29604186)`. This requests at most two
serial panel allocations (46 hours of combined wall limits), rather than
100 separate jobs or parallel allocations. The panel still contains 100
cases: the second allocation skips verified terminal cases, and exits
without new trajectories if the first already processed them all. The
downloaded 5 October scheduler snapshot verifies those dependency
expressions and pending states. `ended(...)` serializes allocations after
a predecessor exits; it is not evidence the predecessor's tests passed.
The fixtures still run at the start of each panel allocation.

Read `identity.json`, `cases.json`, `summary.json`, allocation experiment
records, terminal seals and each case's attempt artifacts together. A
controller/source/asset integrity failure stops the allocation; ordinary
preparation or optimization failures remain case outcomes and allow later
cases to be attempted. Retain all previous attempts and report total
compute/calls when continuation redoes an interrupted case.

## Claims this panel can and cannot establish

This first panel measures **elemental response at frozen predicted poses**.
There is **no fresh-native endpoint confirmation** in the panel controller,
so relaxed score improvements cannot yet be described as retained native
affinity improvements. Both native heads from preparation remain separate
from the optimized binary objective. Cross-target raw scores are not a
calibrated selectivity estimate; no off-target binding claim follows.

An observed hard heteroatom would establish that this frozen latent path
can cross an element-argmax boundary in at least that recorded context.
Only a separately valid hard graph establishes categorical chemical
validity, and neither observation establishes binding, synthesizability
or usefulness. An all-carbon outcome must distinguish absent element
gradient, a small or unfavorable gradient, soft shifts that never cross
argmax, latent/decoder restrictions, failed gates and unstarted cases.
Final scientific conclusions wait for the retained HPC artifacts.
