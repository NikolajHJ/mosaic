# Chemistry guidance

Question: which guidance improves actual validity, connectivity and generation?

For an introduction to the custom loss, see the [one-page group handout](../../outputs/chemistry-guidance/chemical-validity-loss.pdf). The [technical reference for agents](chemical-validity-loss.md) covers the equations, experiments, metric definitions, code paths and proposed motif-retention study.

- The [completed fixed-search Protenix comparison](preservation-protenix.md)
  covers 42 trajectories across two executions of the same three starts, plus a
  99-state repair replay. Atom 2 / structure 2 improved validity on all six
  paired trajectories. Atom 2 / structure 0 kept every endpoint sulfur-free.
  Hard contact rankings varied between executions.

- The [completed mixed-search Protenix comparison](atom-exchange-protenix.md)
  covers three starts × three settings, LR 70 and 500 updates. Current / atom
  0.5 / atom 2 gave 78.9% / 51.1% / 24.9% trajectory validity and mean sulfur
  drift of +0.523 / +0.372 / +0.078. Current guidance had the strongest selected
  hard contact on all three starts.

- [Additive costs with 1-p](atom-exchange-linear.md): explicit revised search
  cost, 65 local tests and completed 48-start/three-seed comparison. Atom/motif
  0.5/0.5 retained 48/48 validity and reduced protected changes 4.646→2.500;
  stronger costs restricted diversity and did not prohibit transient sulfur gains.

- [Atom exchange costs and substructure preservation](atom-exchange.md): opt-in
  mixed repair and persistent reference penalty; completed 16-start/8-arm pilot.
  Atom cost 2 + motif cost 2 retained 16/16 final validity and reduced protected
  changes 3.50→1.125; cost 5 + motifs left 15/16 valid. No defaults adopted.

**Selected 1 November 2026:** use structural proposals for the next new repair
experiment: component joining and connectivity-preserving pruning, with
stochastic samples off and the existing beam as fallback. The saved selected
config is [`repair_structural.json`](repair_structural.json); it is now the
benchmark command's default. Loss weights/normalization stay unchanged. Existing
legacy factory callers, sealed releases and the running VAE continuation remain
unchanged by that selection. A* and sampling remain available
through the explicit comparison configs below. Training-level validity,
reconstruction and diversity effects still require the matched training pilot.
The default CLI smoke repaired all three artificial fixtures in0.55s of measured
benchmark time; [saved result](../../outputs/runs/repair-proposals/selected-default-smoke001/result.json).

**Adoption authorized later on 1 November:** the user requested structural
proposals for the actual epoch-45 continuation from saved epoch 44. The separate
[VAE study](../vae/README.md) records its new source156 release, 32-update native
probe and conditional full-epoch submission. That probe is queued; it is not
yet evidence of training-level improvement. Earlier benchmark evidence and
historical sources remain unchanged.

- Reusable losses/adapters: [latentmol.guidance](../../src/latentmol/guidance/README.md).
- Neighbor/environment rationale and checks: [study85](../../docs/85_neighbor_aware_chemistry.md).
- Local matched pilot: [study88](../../docs/88_local_neighbor_loss_pilot.md).
- Existing pilot command: `python scripts/local_neighbor_pilot.py --help`.
- Failure summary: `scripts/summarize_local_neighbor_failure.py`.
- Broader sulfur/shortcut analysis: [study84](../../docs/84_sulfur_bias_in_validity_guidance.md).

Observed result: neighbor weight0.1 reduced local valence errors but worsened
connectivity and prior validity; failed/not adopted. The new exploratory warning
policy does not change that outcome. Keep another weight/arm in a small config
and a new output directory, and compare the same update indices/endpoints.

## Fixed-atom repair proposals

**Implemented and locally benchmarked 1 November 2026, 09:06 CET.** Question:
can valid-target discovery become faster and handle dense graphs while keeping
the working loss and fixed atom identities? The new opt-in
[`ProposalRepairTeacher`](../../src/latentmol/guidance/repair_proposals.py) extends
the existing target builder; it is not enabled in the running VAE continuation.
No jobs, callbacks, checkpoints, sampling ratios, loss weights or live sources
were changed. No training comparison has run with this teacher yet.

The preferred proposal policy joins components with a greedy degree-bounded
backbone, then removes/downgrades excess bonds while protecting that backbone.
It preserves feasible extra edges instead of making every molecule a tree.
Atomic identities, charges, occupied slots and slot alignment stay fixed. The
valence table is only an optimistic proposal heuristic, not a new sulfur rule:
every target must pass native connected/radical-free sanitization and the same
canonical schema roundtrip, then the inherited teacher verifies it again.
Failure of the greedy backbone is not proof of chemical infeasibility.

Optional conditional bond draws use a point-local NumPy RNG bound to graph,
probabilities and configured seed. Each undirected occupied pair is drawn once;
training/global RNG state is untouched and worker/batch scheduling cannot change
the proposal stream. Draws keep atom/charge/occupancy labels fixed. The earlier
67.8% full-graph sampling rescue result therefore does not transfer directly.

Candidates are deduplicated and ranked by the existing probability-aware edit
cost before native checks. First valid discovery returns immediately; remaining
cases use the existing optimized beam search. Proposals and beam share96
candidate checks in this comparison. The additional raw check and independent
teacher target verification are outside that cap and included in measured
runtime. Macro work is bounded separately by2048 reduction steps and1225 bond
changes by default. These are search bounds, not a closest-repair guarantee.
For a cautious future training test, a smaller `max_bond_changes` can be declared;
large dense witnesses are useful feasibility examples but potentially strong
self-training targets. Unresolved cases remain explicitly censored; no invalid
partial target or previously failed local-valence loss is enabled.

**Two local CPU panels, frozen epoch43:** seeds904017/140041,256 Gaussian
latents each, batch16, four Torch threads,600s cap per benchmark. Total benchmark
times18.41s/15.47s including decoding, four-arm comparison and stress cases.
All arms consume identical checkpoint probabilities and source graphs; no
training or repair substitution into a binding scorer. There are123 invalid
raw prior outputs among512 attempts.

| Target builder | Repaired /123 invalid | Target-discovery seconds over512 attempts | Mean witness edits |
| --- | ---: | ---: | ---: |
| Existing beam |122 |9.889 |1.262 |
| Four bond samples, then beam |122 |6.024 |1.951 |
| Structural proposals, then beam |123 |2.677 |1.301 |
| Structural plus samples, then beam |123 |3.332 |1.423 |

Structural proposals were **3.69x faster in this serial CPU discovery benchmark**;
candidate checks dropped11808→1532. This is not a measured HPC update/epoch
speedup. Among122 commonly repaired cases,119 target label sets were identical;
mean probability-weighted cost increased2.841→2.881 on those122 cases (~1.4%).
Thus the speedup does not establish uniformly better targets. Samples are
implemented but **default off**: their higher edit/cost and extra runtime did
not justify making them the preferred first training comparison.

Three artificial stress graphs are separate from those prior panels, and were
repeated across benchmarks rather than treated as six independent cases. The
old beam repaired four disconnected C2 fragments, but could not repair complete
C8/C16 graphs within three edits. Structural proposals repaired all three;
the dense targets required12 and96 bond edits. RDKit emitted its approximate
ring-finding warning on artificial dense inputs; accepted targets still passed
the native verifier. These examples establish bounded feasibility handling,
not realistic optimization performance or minimal repair distance.

The real frozen decoder smoke has a finite, attached repair gradient through
`z -> decoder -> existing repair_loss`; on its one common target, every arm
returned NLL0.01269474 and latent gradient norm0.14420043. Frozen model weights
and parameter gradients stayed unchanged. Existing target normalization and
all-attempt averaging are retained. The12 initial fixtures passed;138 core
repair/worker/async regressions passed, then83 factory/classification cases
passed after integration (overlapping selections, not221 unique tests). Two
pytest JUnit-property warnings were formatting-only. No training-level
reconstruction/diversity preservation or native-GPU equivalence is established.

Use the existing factory from a **new experiment**, recording the opt-in config:

```python
from latentmol.failure_guidance_training import build_repair_teacher

teacher = build_repair_teacher(
    spec, protocol, execution,
    proposal_config={"sample_draws": 0, "candidate_checks": 96,
                     "max_bond_changes": 8},
)
# Existing build_targets/build_targets_async, repair_loss and frozen-reference
# operations still apply. Keep all other objectives and health checks unchanged.
```

The new optional keyword is separate experiment metadata. Historical protocol
readers stay strict and do not recognize it as authority to resume an existing
campaign. Combining it with the unsuccessful failure-guidance objective is
explicitly rejected. The worker pool gets an explicit task type; no additional
pool or per-point GPU model is created.

Reproduce the frozen comparison using
[`repair_proposals.json`](repair_proposals.json) (four arms in one process):

```sh
python experiments/chemistry-guidance/benchmark_repair_proposals.py --config experiments/chemistry-guidance/repair_proposals.json --checkpoint EPOCH43.pt --output NEW_OUTPUT --count 256 --seed 904017 --seconds 600
python experiments/chemistry-guidance/benchmark_repair_proposals.py --config experiments/chemistry-guidance/repair_proposals.json --checkpoint EPOCH43.pt --output NEW_OUTPUT_2 --count 256 --seed 140041 --seconds 600
```

Add `--gradient-smoke` for the real decoder-to-latent backward check. Omitting
`--checkpoint` runs only the three artificial stress cases. Larger model work
belongs on allocated compute. [Combined measurements](../../outputs/runs/repair-proposals/summary.json),
[matched targets/stress cases](../../outputs/runs/repair-proposals/target-comparison.json),
[seed904017](../../outputs/runs/repair-proposals/epoch43-001/result.json),
[seed140041](../../outputs/runs/repair-proposals/epoch43-002/result.json),
[real-decoder backward smoke](../../outputs/runs/repair-proposals/epoch43-gradient-001/result.json).
Raw graphs, failures and target witnesses are retained in each `attempts.jsonl`.

Next: a bounded matched training comparison using structural proposals and
unchanged losses/anchors. Measure raw validity, connectivity, reconstruction,
size/composition and scaffold diversity before any adoption into a continuation.

## Probability-aware A*

**Implemented and benchmarked 1 November 2026, 10:43 CET.** User authorized an
opt-in A* backend and speed/coverage/cost comparison, explicitly keeping the
loss and running VAE training unchanged. No HPC command, submission, callback,
sealed release, checkpoint or live training setting was changed. Shared code
is [`guidance/repair_astar.py`](../../src/latentmol/guidance/repair_astar.py), using
the existing teacher, ordered worker pool, frozen target checks and repair NLL.

For each search, probabilities and atom/charge/occupancy labels are fixed.
Each occupied unordered bond can change once to any other supported category.
The accumulated cost is the existing edit count plus nonnegative probability
regret, summed in canonical pair order. A state is the vector of upper-triangle
labels; the edited set is recoverable from its differences to the raw graph.
Thus different paths to the same state have the same cost and remaining moves.
This is why deduplication on generation is valid here; allowing repeated edits
to a bond would require revisiting that argument and the implementation.

The heuristic is a lower bound in this declared edit space:

1. Compute `E = sum(max(optimistic_bond_order_sum - allowed_cap, 0))` and the
   number of connected components `K`. Orders are0/1/2/3, with **AROMATIC=1**
   as an optimistic bound;1.5 can falsely flag native-valid aromatic oxygen.
   Caps use the existing explicit element/charge rules. Unknown rules contribute
   zero valence excess rather than an invented cap. Undercoordination is not
   penalized because the schema omits implicit hydrogens.
2. At least `r = max(K-1, ceil(E/6))` remaining bond changes are necessary:
   one category change removes at most three order units at each endpoint.
   The connectivity/valence bounds are combined with max to avoid double counts.
3. For each remaining distinct bond, take its cheapest alternative edit cost.
   `h` is the sum of the `r` cheapest of those costs. These edits need not fix
   chemistry; that relaxation makes the estimate optimistic and probability-aware.

Queue order is `g+h`, then smaller `h`, depth and insertion order. There is no
beam pruning or weighted inflation of the heuristic. Incremental valence and
parent bridge/component information avoid rerunning topology for every child;
native validation is performed on popped candidates. Complete goals must pass
native sanitization, connectivity, radical and canonical schema-roundtrip rules.
If a native-valid goal has positive `h`, execution raises a concrete contradiction.
Costs and priorities reject nonfinite computations.

Each arm has96 candidate native checks (raw and independent final teacher checks
are additional). A* also limits generated states to50,000 and the frontier to
20,000. A memory/generation cap **interrupts** expansion rather than discarding
unpromising states and pretending exact A* ordering remains. The unexpanded
parent's bound is retained when expansion is interrupted. A check cap, edit
radius exclusion, frontier exhaustion and unsupported inputs have separate
receipts. No censored result is labeled globally infeasible or a closest repair;
`minimum_is_proven`/`distance_is_exact` stay false. Mathematical lower-bound
assumptions and tiny-space checks are not independent full-native certification.

The existing benchmark runner now selects arms through
[`repair_astar.json`](repair_astar.json). Two local CPU runs reused frozen43,
256 prior points at each seed904017/140041, batch16, four Torch threads and a
600s cap. Actual complete comparison times were19.59s and16.44s. Every arm
received identical decoder probabilities, and source graphs were preserved.
Three artificial stress graphs were separate from the512 prior attempts.

| Discovery backend | Repaired /123 invalid prior outputs | Discovery time over512 attempts | Speedup versus beam |
| --- | ---: | ---: | ---: |
| Existing beam, radius3 |122 |10.293s |1x |
| Structural proposals plus beam |123 |2.822s |3.65x |
| A*, radius3 |119 |4.056s |2.54x |
| A*, radius128 |114 |4.351s |2.37x |

Use matched successful cases for cost comparisons: on the119 cases repaired
by both radius3 A* and beam, mean weighted cost fell **2.7013→2.5457 (5.76%)**.
Four targets had lower cost,115 tied and none were higher within1e-10. Comparing
unmatched means would hide the harder cases lost to the memory cap. Structural
proposals had slightly higher matched cost2.8410→2.8810 over122 common repairs.

Radius3 A* failures are **three frontier-memory interruptions and one edit-radius
lower-bound exclusion**. It used572 candidate checks versus beam's11808, but
generating/ordering states still costs CPU time and memory. Radius128 A* had
nine memory interruptions; a larger edit space did not improve bounded coverage.
The disconnected stress graph was repaired by all methods. Radius3 A* quickly
recognized that dense C8/C16 need more than three edits. Radius128 reached a
check or memory cap on both dense cases; structural proposals repaired them
in12/96 edits. Coordinated pruning therefore remains the preferred dense path.
The structural path allows large macro edits, so this is explicitly not a
matched three-edit search-space comparison for that arm.

Validation:15 initial A* cases passed, then a100-case A*/proposal/worker/async/
factory regression selection passed. After adding explicit cost-overflow guards,
all16 A* cases passed. Selections overlap. Tiny three-atom spaces exhaustively
check the heuristic against remaining native-valid repair cost and compare A*
with exhaustive minimum cost and a zero-heuristic Dijkstra control. Fixtures
also cover aromatic/sulfur goals, budget honesty, fixed slots, input immutability,
serial/worker/async identity and loss gradients. The real frozen-decoder smoke
on one common target returned the same NLL0.01269474 and finite nonzero latent
gradient norm0.14420043 across all arms. Frozen model state/parameter gradients
were unchanged. No VAE training, reconstruction/diversity gain or HPC throughput
claim follows from these local frozen-checkpoint benchmarks.

Use from a **new experiment**, recording the opt-in settings:

```python
teacher = build_repair_teacher(
    spec, protocol, execution,
    astar_config={"candidate_checks": 96, "max_frontier": 20000},
)
```

`astar_config` and `proposal_config` are mutually exclusive; existing protocol
readers and default callers remain unchanged. A separate untested failure-loss
combination is rejected. To reproduce the comparison:

```sh
python experiments/chemistry-guidance/benchmark_repair_proposals.py --config experiments/chemistry-guidance/repair_astar.json --checkpoint EPOCH43.pt --output NEW_OUTPUT --count 256 --seed 904017 --gradient-smoke --seconds 600
```

Repeat at seed140041 with another output path. [Combined measurements and matched
costs](../../outputs/runs/repair-astar/summary.json),
[seed904017 with backward smoke](../../outputs/runs/repair-astar/epoch43-001/result.json),
[seed140041](../../outputs/runs/repair-astar/epoch43-002/result.json).
Raw graphs, targets, costs, queue statistics and interruption reasons remain in
`attempts.jsonl`. Recommendation: retain structural proposals as the preferred
new discovery option; A* is available for explicit cost-focused comparisons.
No backend has been adopted into the running VAE continuation by this work.
