# Configurable Protenix restart search

Use `python -m latentmol.design.protenix_search` for cyclic search runs.
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
include the sealed config, budgets, and receipts.
