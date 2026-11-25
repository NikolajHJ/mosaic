# Configurable Protenix restart search

Use `python -m latentmol.design.protenix_search` for cyclic search runs.
For synthesis-aware human BRD4 optimization, start with the [standard three-wave protocol](restart-synthesis-standard.md).
The [long presets](restart-search-lr60-long.json) contain `wide`, `deep`,
and `near-valid`. Each uses four parents and 500 updates per child. Wide
uses six children per parent over 10 waves. Deep and near-valid use three
children per parent over 20 waves. Each has a cap of 120,000 updates and
1,440 fresh-score attempts. The [short presets](restart-search-lr60-short.json) cover screened
restarts, diversity, and adaptive, fixed, and independent parent policies.
The entry point also reads the historical `restarts.json`, `restart-lr70.json`,
six-cycle, and diversity configs directly. The dedicated noise and throughput
studies keep their own runners. Use the dedicated lineage comparison when
all arms must receive the same actual number of fresh attempts each cycle.

New restart searches default to zero optimization noise. The presets still
use their configured perturbations to choose child starting states. A
noise study can opt in with `design.noise_initial` or a paired variant's
`update_noise`.

Validate a preset and print its resolved inputs and budget from the repository root.

```powershell
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-search-lr60-long.json --arm deep --plan-only
```

Repeat `--set SECTION.FIELD=VALUE` for individual overrides. Values accept
JSON numbers, booleans, and `null`, or a bare policy name. For example,
`--set search.batch_size=4 --set search.cpu_workers=2 --set search.overlap=true`
selects up to four children per batch and overlaps CPU repair with scoring.
The long presets use scalar execution. The
[V100 benchmark](restart-batch-throughput.md) is a separate job and the
long searches can start before it finishes.

| Section | Main inputs |
| --- | --- |
| `design` | Updates per child, LR, displacement cap, chemistry, temperature, and update noise schedules |
| `search` | Parents, children per parent, waves, batch size, CPU workers, shortlist policy, and fresh attempts per child |
| `search.initialization` | `archived` loads scored parents from `parent_cohorts`. `bootstrap` runs one initial branch in wave one |
| `search.parent_policy` | `frontier`, `top`, `diverse`, `fixed`, or `independent` |
| `search` frontier limits | `min_step_gap`, `max_similarity`, `ancestor_lookback`, and `max_ancestor_fraction` |
| `search` near-valid controls | `invalid_children` permits one substitution in an `exact_children` slot. `repair_edit_limit` and `invalid_pool_per_branch` bound its source pool |

The frontier gives priority to the latest wave, then to lower fresh loss.
It can draw from older branches and leaves slots empty when candidates fail
the diversity rules. It applies the spacing limit to the source step within
each trajectory. The ancestry limit counts planned child slots. At the start,
the archived presets launch from their designated founders. A near-valid
child replays the saved invalid latent and bounded repair witness. Its slot
uses the designated exact valid start when the invalid pool has no eligible
point. Fresh scoring uses only distinct connected, sanitized candidates.

Set `search.frontier_keep_best=true` to reserve one child when the best
eligible replayable parent would otherwise be omitted. With zero update
noise, `search.duplicate_refill_attempts` enables exact start detection and
bounded retries. It compares the starting latent, occupied mask, and
logit offset across waves. Identical SMILES with different latents remain
eligible.

Put shared defaults in `design` and `search` and overrides under `arms.NAME`.
For a new policy, the optional `hooks` map accepts a trusted Python
`module:callable` for `parent_selector` or `shortlist_selector`. The parent
hook takes `(points, *, wave, options, history)` and returns existing eligible
points for adaptive selection. The shortlist hook takes `(ranked, limit)`
and returns existing candidates. The runner validates membership, uniqueness,
and budgets for both hooks. A custom parent hook owns its selection rules.
The default frontier and branch executor are separate modules.

The existing DTU launcher stages a sealed config. Copy a preset to a new
`restart-*.json` config and edit it for HPC runs. The launcher accepts repeated
`--arm` flags to select presets. This example stages only `deep` on the
32 GB V100 queue configured in the preset.

```powershell
$launcher = 'experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py'
python $launcher stage --name protenix-restart-my-search --config restart-search-lr60-long.json --arm deep --hours 24 --max-updates 120000
python $launcher submit --name protenix-restart-my-search
```

`run.json` records the resolved settings, source config hash, seeds, and
founders. Each wave has a `wave-plan.json` with the exact source checkpoint
and policy for every child. `archive.json` records the completed wave budget,
new and cumulative scores, top-three mean, optional diversity count, and
near-valid recovery. `result.json` aggregates the completed waves. Each child
retains its usual trajectory, raw graph blocks, and summary.

New runs publish a rolling `search-state.pt` at each completed wave. It
contains the eligible and ineligible archive entries, invalid pool,
attempted molecules, selection history, and prior seed state. To continue,
copy the last complete checkpoint to a fixed path and record its SHA-256.
Use the same resolved design and search settings and add an arm-level
`resume` object with `path` and `sha256` to the staged config. The
launcher verifies the checkpoint and passes it to the runner. Use a new
campaign name for the output segment. The new segment retains the earlier
wave summaries and continues with the next wave number.

Run `restart-operations/compare_search_budgets.py` with the saved run
directories to compare the latest completed archives at common ceilings
for updates, fresh attempts, and successful scores. It reports the actual
budget used at each selected wave and identifies ceilings matched exactly.

The long presets use a 23-hour forecast within the 24-hour allocation.
Before opening a new wave, the runner compares the remaining time with
recent full wave durations, including startup and fresh scoring. It closes
the run at the last complete wave if the next one is unlikely to fit. The
update cap is the planned maximum. Compare actual results at shared update
or fresh-score checkpoints if an arm stops early.

DTU accepted the three scalar searches on 9 November 2026 at 15:47 CET:
`wide` **29940513** on A100, `deep` **29940516** on V100, and
`near-valid` **29940520** on L40S. All were pending at verification.
Keep the different GPU models in mind when comparing wall times.
The [submission records](../../../outputs/protenix-restart-lr60-long-20261109a/)
include the sealed config, budgets, and receipts. That submission used
0.05 optimization noise, fading to zero by update 100. The current presets
use zero for future launches.

At 16:47 CET on 9 November, DTU confirmed all three originals had exited
and released their replacements with optimization noise disabled. The
replacements start from the configured parents with the original child
initialization policies, budgets, and queues.

| Search | Queue | Original job | Replacement job |
| --- | --- | ---: | ---: |
| Wide | gpua100 | 29940513 | 29943696 |
| Deep | gpuv100 | 29940516 | 29943705 |
| Near-valid | gpul40s | 29940520 | 29943712 |

The [replacement records](../../../outputs/protenix-restart-lr60-long-zero-noise-20261109a/)
include the sealed config and the cancellation and release receipts.

At 19:42 CET on 9 November, the near-valid replacement was in wave four
with about 23,450 updates recorded. The first three completed waves
lowered the best fresh loss from 2.21284 to 2.11829 to 2.06081. Waves two
and three used the invalid-start slot and recovered valid graphs at
updates three and one, respectively. Wide and deep were still queued.

At 20:34 CET on 9 November, the near-valid search had five completed
waves and 30,000 updates, with 333 fresh attempts and 315 successful
scores. Its best loss was 2.06081, its top-three mean was 2.06990, and
it had 101 distinct low-loss candidates at the configured similarity
threshold. Wide and deep still had no completed wave for comparison.

## Completed results on 10 November 2026

The near-valid arm of `python -m latentmol.design.protenix_search` finished
all 20 waves and 120,000 updates in 13 hours 2 minutes. It attempted 1,036
fresh scores and succeeded on 978. Its best fresh loss fell from 2.06081
at wave 3 to 2.05132 at wave 7. The remaining 78,000 updates added 244
diverse candidates at loss ≤2.30 and Morgan similarity ≤0.60, bringing the
total to 398. The top-three mean reached 2.05801 in wave 9 and held there.
All 19 near-valid starts recovered validity within 18 updates, with a
median of one. Their best direct fresh score was 2.12706. The final four
elites trace to screened latent starts from the lowest-ranked founder.
[Near-valid result](../../../outputs/restart-finished-20261110/29943712-result.json).

The three [lineage panels](restart-lineage-feedback.md) each gave adaptive,
fixed, and independent starts 2,880 updates and 90 fresh attempts per
policy. Adaptive beat fixed in every panel on best and mean top-three
fresh loss. Independent found the best single molecule in two panels.
Mean best loss was 2.22593, 2.25552, and 2.21533 for adaptive, fixed, and
independent, respectively. Their summed per-panel diverse low-loss counts
were 32, 7, and 10. [Panel A](../../../outputs/restart-finished-20261110/29931143-result.json),
[panel B](../../../outputs/restart-finished-20261110/29931145-result.json), and
[panel C](../../../outputs/restart-finished-20261110/29931146-result.json).

The [initialization pilot](restart-step-cap.md) used 1,500 updates and 18
fresh attempts per arm with update noise off. Mean parent-to-best gains
across three parents were 0.17724 for exact, 0.17473 for screened latent,
0.13627 for protected sparse graph mixing, and 0.08617 for independent
Gaussian starts. [Initialization result](../../../outputs/restart-finished-20261110/29943695-result.json).

The first [noise timing panel](restart-noise.md) used 768 updates and 24
successful fresh scores per arm across two parents. Parent-only radius
0.003 retained the parent molecule at all six starts and separated all
six jointly valid sibling endpoint pairs. All six unperturbed pairs had
identical endpoints. Radius 0.03 plus 1% update noise gave the largest mean
gain (0.29656) and lowest trajectory validity (77.3%) in this panel.
[Timing result](../../../outputs/restart-finished-20261110/29948071-result.json).

The [V100 throughput pilot](restart-batch-throughput.md) completed 5,760
timed updates and scored all 12 fresh probes. One 32 GB V100 reached 2.66
updates/s with eight children and four overlapping CPU workers, versus
1.82 for four-thread scalar execution. Four children with four overlapping
workers reached 2.51 updates/s with 7,012 MiB peak sampled GPU memory,
versus 15,208 MiB for eight. Initial scores and gradients closely agreed,
while trajectories differed at 40 updates.
[Throughput result](../../../outputs/restart-finished-20261110/29936883-result.json).

## Eight additional results on 10 November 2026

The [500-update persistence panel](../../../outputs/restart-finished-20261110/29948086-result.json)
matched 2,000 updates and 24 successful fresh scores per treatment across
two parents. Each parent had two sibling branches per treatment.

| Treatment | Mean gain in best loss per parent | Lowest new loss | Scores below 2.40 | Valid saved states |
| --- | ---: | ---: | ---: | ---: |
| No noise | 0.24246 | 2.36272 | 2 | 80.2% |
| Parent kick, radius 0.03 | 0.27808 | 2.30287 | 6 | 86.5% |
| Update noise, 1% fading to zero at update 100 | 0.25050 | 2.29133 | 3 | 88.8% |
| Parent kick plus update noise | 0.21700 | 2.38033 | 2 | 81.0% |

Untouched sibling branches replayed all recorded hard states. Perturbed
pairs first differed by update 6. All four jointly valid perturbed
pairs had distinct endpoints and two more pairs had an invalid endpoint.
Radius 0.03 retained the parent molecule in three of four starts. The
peak hard-bond count during updates 0–100 was 30–32 across the arms.
Update noise alone led on one parent and the untouched control led on the
other. The combined arm had the lowest mean across all 24 scores, 2.47521,
and the smallest average gain in each parent's best. [Paths](../../../outputs/restart-finished-20261110/29948086-path-analysis.json)
and [bond counts](../../../outputs/restart-finished-20261110/29948086-early-bond-density.json).

The [graph-edit panel](../../../outputs/restart-finished-20261110/29948378-result.json)
matched eight successful fresh scores per arm and parent across three
parents. The direct arm checked 768 atom or bond proposals, accepted 523,
and fresh-scored four valid candidates of each edit type per parent. It
improved all three baselines and had the lower best score on two parents,
one by 0.00130. Mean per-parent best was 2.45124 for direct edits and
2.45228 for latent restarts. Latent found the panel's best molecule at
2.31108 versus 2.39919 for direct edits, and put four candidates below
2.40 versus one. Fresh scoring took about 108 seconds per arm. Proposals
took 0.42 seconds for graph edits and 643 seconds for 1,500 latent updates,
including native startup. We shelved this pilot and canceled the queued
second panel to focus on gradient-guided hallucination.

In the [near-valid replay](../../../outputs/restart-finished-20261110/29948379-result.json),
exact valid continuations produced the lower best descendant loss in all
four paired slots, averaging 2.15841 versus 2.47342. The arms matched 2,000
updates and 24 fresh attempts. Near-valid scored 24 and exact valid scored
21. All four near-valid starts recovered in 1–18 updates. Exact starts
already had lower baseline scores than the repair witnesses, so this
comparison measures the actual choice of slot in these four waves.

The [recency frontier](../../../outputs/restart-finished-20261110/29948377-result.json)
and [reserved-slot frontier](../../../outputs/restart-finished-20261110/29948382-result.json)
both completed 3,000 updates and 70 successful scores, reaching 2.27707.
They selected the same sources and seeds in all four waves, with no slot
reservation or duplicate refill triggered.

Across all four LR 70 cap panels, including the previously completed seed
C, cap 0.5 won on five of eight matched parents. Mean per-parent gain was
0.18212 at 0.5 and 0.12173 at 0.25. Each cap used 8,000 updates and 96
fresh attempts, with 94 and 91 successful scores, respectively.
[Seed A](../../../outputs/restart-finished-20261110/29943697-result.json),
[seed B](../../../outputs/restart-finished-20261110/29943704-result.json),
[seed C](../../../outputs/restart-finished-20261110/29943710-result.json),
and [seed D](../../../outputs/restart-finished-20261110/29943716-result.json).

## Long searches completed on 11 November 2026

Wide (**29943696**) finished at 02:49 CET and deep (**29943705**) at
04:40 CET. Both completed all 120,000 updates.

| Search | Best fresh loss | Top-three mean | Diverse low-loss count | Fresh attempts / successful | Runtime |
| --- | ---: | ---: | ---: | ---: | --- |
| Wide | 2.09213 | 2.10295 | 177 | 1,043 / 1,041 | 14 h 43 min |
| Deep | 2.08135 | 2.08397 | 389 | 1,265 / 1,159 | 19 h 02 min |

Wide found its best at wave 8 and 96,000 updates. Deep found its best at
wave 15 and 90,000 updates. Near-valid remains the lowest at 2.05132.
The configurations used different GPU families and actual fresh-score
counts, so this comparison is descriptive. [Wide result](../../../outputs/restart-finished-20261111/29943696-result.json)
and [deep result](../../../outputs/restart-finished-20261111/29943705-result.json).
