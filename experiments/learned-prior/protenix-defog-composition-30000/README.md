# DeFoG Stage 3 ablations at LR 60

This study asks when a frozen, refreshed DeFoG graph prediction improves the
**best valid molecule found during Protenix optimization**. It implements the
Stage 3 plan on branch `codex/defog-stage3-campaign`. The first submission failed before optimization because
Git for Windows changed the archived PDB's line endings. A verified LF retry
has nine reference bundles accepted; the remaining 387 are to follow after a
teacher-on reference records normal update progress. See the
[submission and recovery note](run-20261113.md) for exact scope and evidence.
No completed native optimization result is established by these submissions.
CPU fixtures check software behavior and are labelled mock data.

The initial design has **155 distinct settings × 3 initializations × 3 seeds =
1,395 runs**, each with **500 updates**, for 697,500 updates. Related settings
share this experiment home. The older `30000` folder name identifies the study's
home, not this campaign's update budget.

The three-seed start was selected on 12 November 2026. The
[five-seed configuration](configs/stage3-lr60-500-five-seeds.json) and its earlier
prepared plans are retained for later expansion: 2,325 runs in 660 bundles.
Seeds 71 and 101 are reserved for a later extension or confirmation; the initial
campaign keeps every setting and initialization while reducing the seed budget
by 40%.

## Fixed setup and selection

Every native run uses the frozen Protenix-Tiny BRD4 pocket contact objective,
native coefficient 1, atom and bond learning rates 60, 25 occupied neutral atom
slots, six element categories, and five bond categories. Contact score is a
model proxy, not a binding-affinity measurement. Initial seeds are **17, 23, 42**.
The same seed fixes initialization and the native model key across settings;
the key stays fixed within a run.

For each seed, the three starts are a random valid sparse tree, dense random
bonds using the same atom probabilities, and the sparse tree with four bonds
deleted. This separates bond density from atom composition. Dense and
fragmented starts need not be chemically valid.

The recorded starting policy samples element labels with probabilities
`[0.70, 0.14, 0.08, 0.04, 0.02, 0.02]` in the element order
`[C, N, O, S, F, Cl]`, then mixes each atom row with 0.02 uniform mass.
The sparse tree uses single bonds and available neutral capacity, with
nonterminal atoms added before halogens and a carbon fallback if sampled
capacity cannot support a tree. Its bond rows mix 0.002 uniform mass. Dense
bond rows are independent `Dirichlet([1,1,1,1,1])` samples. The fragmented start
deletes four tree bonds before bond-row mixing. These are initialization
parameters, not empirical molecular frequencies. They appear in the protocol
and every resolved run configuration.

The primary endpoint is the **highest hard Protenix contact score among raw
argmax molecules that sanitize in RDKit and have one connected component**.
The initial state is eligible and is flagged explicitly. The report also keeps
the best valid molecule with a new canonical identity, its improvement over a
valid initial molecule, and final-state diagnostics. No repaired molecule can
win the primary endpoint. Runs with no valid candidate have a missing score;
they are retained in the success-rate denominator.

Each run also retains its **top 20 distinct valid canonical molecules**, ranked
by native hard contact score. If several ordered graphs have the same canonical
SMILES, the highest-scoring observed mapping is selected; score ties keep the
earliest discovery. Its tensors preserve that exact atom order and the stored
soft graph from its first occurrence. `all_candidates.json` keeps every observed
ordered hard graph, including invalid ones, with first/last occurrence and visit
count, so canonical deduplication does not discard the underlying trajectory.
Top-molecule records include RDKit molecular weight, logP, TPSA, QED, donor and
acceptor counts, rotatable bonds, rings, aromatic rings, heavy-atom count, formal
charge, and RDKit version. These describe 2D chemistry, not 3D quality or affinity.
An optional descriptor failure leaves that field unavailable and records a
partial descriptor status/error; it does not discard the valid molecule or fail
the optimization. Ordered-graph/source integrity checks remain required.

Every visited state is retained so this best-so-far result can be computed
without guessing from sparse checkpoints. State zero plus 500 updates gives
501 observations, **not 501 independent replicates**. Seeds are the independent
blocks. Compare starts separately and show valid-run yield alongside score
distributions; a high conditional score among a few successful seeds can hide
failures.

## What the design tests

The [config](configs/stage3-lr60-500.json) is the compact protocol. The
[design generator](scripts/stage3_design.py) expands it into explicit immutable
condition and run records. Identical physical settings share one condition,
even when several comparisons need them. Every teacher treatment records a
matched teacher-off condition with the same cooling, projection and simple
chemistry settings.

| Stage 3 point | Conditions and evidence |
| --- | --- |
| §6.1: checkpoint and graph compatibility | Fixed vocabulary/context checks; soft versus hard teacher input; teacher displacement, changed argmax categories and checkpoint metadata. This is evidence for this checkpoint and schema. |
| §6.2: refreshed one-query teacher | Current versus initial frozen target; identity target; separate teacher query counts and native/teacher gradients. The default uses a stopped target, not backpropagation through DeFoG or a full sampling trajectory. |
| §6.2: useful moderate strength | Weights 0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5; times 0.2, 0.5, 0.8, 0.95; targeted weight/time crossings. |
| §6.2: chemistry after the update | Simplex only, valence, bridge only, valence plus bridge, valence plus shrinkage, and all three. Test caps, damping, bridge strength/retry, fluidity, shrinkage, cadence, delayed projection and final valence reprojection. |
| §6.3: higher raw validity and less fragmentation | Best valid and best new valid candidate; any-valid yield; final and last-100 validity; component count, atoms outside the largest component, hard valence excess, sanitization failures, and a bounded verified endpoint repair witness where found. |
| §6.3: smaller soft-hard gap | Per-state hard/soft score pairs, valid-state gaps, thresholds 0.02/0.05/0.1, entropy, and stored versus cooled-view diagnostics. |
| §6.3: useful exploration without freezing | Hard score, distinct canonical molecules, atom/bond edits, movement, native/teacher gradient norms and ratios, and best new identity. |
| §6.3: revisit strength and compatibility before schedule complexity | Constant baselines plus linear, reverse, delayed and cyclic weights; time schedules; targeted joint schedules. Constants match each schedule's actual 500-update mean. |
| Temperature cooling extension | Shared cooled graph for native and teacher; fixed 0.5 and schedules 1→0.5 or 1→0.25, including delayed cooling; exact derivative versus STE; incremental state cooling; interactions with fluidity and teacher time. |
| Information in the teacher | Atom-only, bond-only, increased bond head, row versus entry normalization, and KL/entry-MSE with a fixed initial-gradient calibration. |
| Practical simple-loss comparison | Explicit `simple_soft_valence` MSE at weights 0.01 and 0.1, alone and with DeFoG. This new baseline does not claim to reproduce a historical custom loss. |
| §8: interpretable matched report | Fixed run IDs, complete planned denominators, paired seed comparisons, tidy CSV tables, raw candidates and probability trajectories. |

This is a staged ablation design, not every Cartesian combination. A positive
interaction can motivate a later targeted confirmation. It does not establish
performance on other targets, atom counts, checkpoints or a second prior model.
Per-state teacher movement is a locality diagnostic, not a certification of
stability under all small perturbations. A detached final-state search tries up
to three bond-label edits, with at most 128 RDKit checks and beam width four.
A verified connected, sanitized witness gives a repair-distance upper bound
under fixed atom labels; failure to find one means unknown/censored. The search
never changes optimization, adds a valid generated candidate, or establishes
the minimum edit distance.

The reference uses weight 0.0625, teacher time 0.8, connected projection,
readout caps, damping 1, bridge probability 0.6, and fluidity threshold 0.95 /
strength 0.05. It does not inherit the published Stage 2 YAML defaults silently.
Projection mode `none` is simplex only. `bridge_only` deliberately omits valence
projection. Neither bridging nor valence projection guarantees a valid connected
argmax molecule.

For probability cooling the transform is
`C_T(p) = p^(1/T) / sum(p^(1/T))`; temperature 1 preserves both the values and
the baseline derivative. This differs from applying a softmax directly to
probabilities. Stored-state cooling uses the incremental exponent
`T_previous/T_next`, then the chosen projection, avoiding repeated compounding
of the full temperature. A projected stored graph does not guarantee that its
cooled view satisfies soft valence constraints, so both views are retained.

## Prepare a plan, then execute bundles

Run commands from the repository root in the supported Mosaic environment, with
the [DeFoG package](../../../models/defog/README.md) installed. Planning and
collection use only the standard library and load no native models.

```bash
python experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_campaign.py plan \
  --wave reference --output results/stage3/lr60-reference-01

python experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_campaign.py plan \
  --wave all --output results/stage3/lr60-full-01
```

Use a **separate output root** for each plan. The reference wave contains only
the teacher-off control and moderate reference: 18 runs in 9 bundles. The full
initial plan contains all 1,395 runs in 396 bundles at the default maximum of four arms.
Creating a plan starts no jobs. The JSON plan lists bundle IDs and their run
IDs; select a listed ID for execution:

```bash
python experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_campaign.py run-bundle \
  --plan results/stage3/lr60-reference-01/plan.json \
  --bundle-id BUNDLE_ID_FROM_PLAN --assets /path/to/native-assets.json
```

Native execution requires an allocated GPU and explicit prepared model assets.
Copy the [assets example](configs/stage3-assets.example.json) and fill the local
paths. Its recorded hashes identify the archived BRD4 assets; another context is
a separately identified study, not a path-only substitution. The JSON identifies
the archived context directory and both context-file hashes, model cache,
model/context digests, DeFoG source, checkpoint and separate teacher interpreter.
[The native adapter](scripts/stage3_native.py) documents the exact fields; no
personal paths are defaults. The teacher stays frozen in a
separate CPU process. Model and context identities are checked before and after
the bundle. Use the approved exclusive GPU resource profile of the selected
queue; this implementation makes no hardware-performance claim.

Bundle **settings together, seeds separately**: each GPU bundle owns one seed,
one initialization and up to four fresh arms. It reuses loaded models and can
cache repeated ordered hard graphs under that seed's model key. Each run has
one writer. Shared controls are referenced by run ID and count only once.
Keep matched comparisons on the same GPU profile where practical and retain
hardware records when placement differs.

A bundle claim and pre-existing run directory prevent accidental overwrites.
Do not relaunch a failed bundle into the same output. Inspect its failure and
create an explicitly named new attempt. Nonfinite computations, incompatible
graphs and frozen-model changes are errors. Invalid chemistry, flat derivatives,
nonmonotone losses and weak scores remain recorded outcomes through the bounded
budget.

For a **CPU-only software fixture**, create a separate plan with `plan --mock`
and run its bundle with `run-bundle --mock --mock-updates 3`. These are synthetic
scores and teacher predictions. Saved records keep the original design budget
and the reduced effective mock budget separately; never mix mock records into
the native campaign.

## Prepare GPU submissions

On Windows, create source archives with `git -c core.autocrlf=false archive`.
The override applies to that command.
The initial default archive changed the pinned PDB from LF to CRLF even though
the Git blob had the correct bytes. Before uploading, inspect the archive's
`tests/data/brd4_3mxf/3MXF.pdb` member and require SHA256
`534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f`.
After extraction, check that hash again and call `derive_definition` with the
archived PDB and fixture. Keep the pinned hash check; repair the archive rather
than accepting different bytes. These are small input checks and load no model.

The [dispatch command](scripts/stage3_dispatch.py) runs on the HPC login host
after the plan and a stable source snapshot have been transferred. It creates
scripts and reviewable submission records; preparing them submits nothing.
For example, using explicit paths appropriate to that snapshot:

```bash
python experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_dispatch.py prepare \
  --plan /work/PREPARED_CAMPAIGN/plan.json \
  --output /work/PREPARED_CAMPAIGN/dispatch-01 \
  --source /work/STABLE_SOURCE_SNAPSHOT \
  --python /work/PREPARED_ENV/bin/python \
  --assets /work/PREPARED_CAMPAIGN/native-assets.json \
  --profiles v100 l40s a10 a100 a40 --hours 2

python experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_dispatch.py submit \
  --dispatch /work/PREPARED_CAMPAIGN/dispatch-01/dispatch.json \
  --bundle-id BUNDLE_ID_FROM_PLAN
```

`submit` is also a preview unless `--execute` is present. Select one or more
bundle IDs with repeated `--bundle-id`; omission selects the dispatch's complete
bundle list. Add `--execute` for actual submission. Every job requests an
exclusive GPU without MPS. The seed/initialization block deterministically
selects a profile so its shared control and treatments retain the same profile;
different independent blocks can use different compatible queues. `--modules`
accepts explicit environment modules when the prepared environment requires
them. Queue and walltime choices remain operational settings to check against
current site availability and measured run times.

Accepted, failed or uncertain submission evidence is retained. If submission
transport fails, reconcile its receipt and scheduler job before retrying; a
connection failure is not evidence that no job was accepted. Never edit the
source snapshot, plan or assets beneath submitted jobs. The current campaign's
accepted submissions are recorded in the [submission note](run-20261113.md).

## Outputs and plotting

Native jobs losslessly gzip the JSONL streams and `all_candidates.json` to
reduce storage. Their filenames end in `.jsonl.gz` and `.json.gz`; the collector
reads either plain or compressed records, including flushed running-job records.
Top-20 JSON/CSV metadata stay plain and probability/gradient arrays use compressed
NPZ. Compression changes no fields, scores, sampling or optimization settings.

```text
results/stage3/CHOSEN_CAMPAIGN/
  plan.json                   explicit settings, requirements, seeds, run IDs, bundles
  bundles/                    writer claims, status, frozen checks, hardware and logs
  runs/RUN_ID/
    config.json               resolved scientific settings and hardware
    status.json               progress, failure, warnings, runtime and query count
    summary.json              best valid/new candidate and secondary endpoints
    states.jsonl              one row per visited state, including state zero
    candidates.jsonl          first occurrence of each ordered hard graph, with labels
    all_candidates.json       every ordered candidate, visit statistics and valid descriptors
    top_valid.json/csv        top 20 distinct valid molecules, ranks, descriptors and source links
    top_valid.npz             ordered hard graphs, exact discovery soft graphs and optional evidence
    gradients.jsonl           native, teacher, simple-loss and combined components
    operators.jsonl           operator sequence, movement, bridging and final components
    teacher.jsonl             prediction movement, changed categories and teacher metadata
    initial.npz               matched starting probabilities and model key
    trajectory.npz            stored/view probabilities for every retained state
    best_valid.npz            selected raw hard graph and its source probabilities, if any
    final.npz                 final available stored graph
    final_assessment.json     detached raw defects and bounded verified repair witness
  tables/                     regenerated tidy plotting tables and schema.json
```

Collect saved results without loading models or changing run records:

```bash
python experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_campaign.py collect \
  --output results/stage3/lr60-reference-01
```

| Table | Row unit and use |
| --- | --- |
| `runs.csv` | One planned run; primary best-valid score, best-new score, yield, final/late diagnostics, and flattened settings. Start most comparisons here. |
| `conditions.csv` | One planned setting; flattened teacher/projection/cooling choices, requirements and comparison memberships for joining plot annotations. |
| `condition_summary.csv` | One setting × initialization; planned-seed denominators (three initially), successful-run yields, conditional score mean and spread. |
| `paired.csv` | One treatment seed paired with its explicit teacher-off reference; score/validity/diversity deltas, all-state score-gap means, and same-update common-valid gap means with counts. Missing reference evidence gives missing deltas. |
| `states.csv` | One run × update; best-so-far curves, score-gap trajectories, entropy, sulfur and fragmentation. |
| `candidates.csv` | One run × ordered hard graph; raw labels, canonical identity, score and first occurrence. |
| `top_valid.csv` | One run × selected distinct valid molecule, rank 1 best; native hard score, descriptors, visits and archive index. |
| `gradients.csv` | One run × gradient update; weighted component size, native/teacher balance and calibration. |
| `operators.csv` | One run × updated state; component changes, bridge survival and operator movements. The raw sequence is preserved in the step details. |
| `operator_steps.csv` | One run × update × zero-based step index; operator name and numeric movement/component/projection fields, for plotting which operation creates or removes bridges. |
| `teacher.csv` | One run × teacher query; target displacement, changed categories and flattened response metadata, including excluded vocabulary mass where reported. |

Tables carry `campaign_id`, `run_id`, `condition_id`, `family`, `seed` and
`initialization`. Settings use dotted column names rather than being inferred
from filenames. `schema.json` records columns and interpretation. Missing
evidence is blank/null, not zero. Collection is repeatable while runs append
records and retains planned runs that have not started; distinguish missing,
running, failed and complete rows before plotting. Canonical SMILES measures
chemical diversity; ordered graph identity is used for exact native cache
reuse and can distinguish atom permutations of the same molecule.

In each run's `top_valid.npz`, `hard_X` / `hard_E_upper` and `source_soft_X` /
`source_soft_E_upper` share the rank order in its metadata. `source_updates`
identifies the exact discovery state; `model_key` records the key when supplied.
Optional discovery evidence contains native, raw teacher, simple-chemistry and
combined gradients with respect to **stored probabilities**, plus the teacher
target and current coefficients, temperature and denoiser time. Every evidence
array has a separate `_available` mask: zero-filled unavailable data are not
measured zero gradients. A molecule first seen at the final state can have no
optimization gradient. The teacher component is retained before its external
weight; the combined gradient includes applied weights. These archives require
no additional native model evaluations.

Collection keeps one run's records in memory, compact gap traces for pairing,
and the small run-summary table. Observation rows pass through temporary disk
snapshots before CSV writing; these snapshots are removed after collection.
This avoids retaining the full campaign as Python dictionaries.

An explicit `scientific_usable: false` records a frozen-model/context failure.
Its raw scores and artifacts remain available, but it contributes no valid
yield or conditional score to aggregate results; paired deltas are blank and
`comparison_eligible` is false. It remains in the planned denominator. Ordinary
failed or partial runs retain their observed evidence and completion flags.

For score-gap comparisons, use the paired common-valid updates as well as
all-state means: each arm's own valid subset can select different states and
hide a gap. These trajectory means are descriptive outcomes of each seed;
their state counts do not increase the number of independent replicates.

Useful first plots are paired best-valid score and best-new score with the
valid-run yield, fraction of seeds finding any new valid molecule, best-so-far
hard score by update, final fragment distributions, teacher/native gradient
ratios, and valid-state score gaps. Keep initial-only successes visibly marked.
The seed-to-seed spread is informative; uncertainty should be computed across seeds,
not by treating successive optimization states as independent samples.

## Implementation checks

The [design tests](tests/test_stage3_design.py) check unique physical settings,
complete planned seed blocks, exact non-teacher control matching, schedule means,
and one bundle owner per run. The math and result tests cover the changed
transformations and selection/collection behavior. These checks do not replace
native GPU evidence, which will be recorded separately after actual execution.

The three-seed revision passed 38 focused CPU checks for design expansion,
reference execution/collection and dispatch preparation. Its
[plan checks](../../../results/implementation-checks/stage3-three-seeds-20261112a/plan-checks.json)
record the current counts; no native runs were started by this revision.

On 12 November 2026, the original five-seed implementation's scoped CPU suite
passed **134 tests**. A separate three-update mock pair exercised execution,
selection, candidate archives and collection.
[The original local software checks](../../../results/implementation-checks/stage3-20261112a/checks.json)
retain that command scope, code identity, counts and runtime versions. Its
prepared [five-seed full design](../../../results/implementation-checks/stage3-20261112a/full-design/plan.json)
and [five-seed reference design](../../../results/implementation-checks/stage3-20261112a/reference-design/plan.json)
remain unchanged.

The current three-seed preparations are the
[initial full design](../../../results/implementation-checks/stage3-three-seeds-20261112a/full-design/plan.json)
and [initial reference design](../../../results/implementation-checks/stage3-three-seeds-20261112a/reference-design/plan.json).
Planning loads no model and submits no job. Software checks and prepared plans
do not establish native derivative accuracy or molecular improvement.
