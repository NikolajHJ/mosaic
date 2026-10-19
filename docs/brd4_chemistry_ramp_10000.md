# BRD4: delayed chemistry, two fresh 10,000-update runs

19 October 2026. User requested one 50% uniform mix and one random
initialization, with a tiny chemistry weight that starts growing around update
5,000 and becomes large at the endpoint. Status: implemented, 274 allocated CPU tests passed; both GPU arms submitted
and queued. Native gates, optimization and final report remain pending.

## Scientific choices

Both runs start fresh, independently, on the same six occupied neutral slots,
frozen Protenix-Tiny, BRD4 3MXF context, whole-target contact objective, key 17,
FP32/highest matmul precision and learning rate 0.03. No new ligand coordinates,
affinity predictor or binding-site objective is introduced. Their GPU jobs have
no dependency on each other. Preserve every earlier experiment unchanged.

The first initialization is the exact 50% carbon-chain + 50% uniform
categorical mixture used before. The second uses independent Dirichlet(1)
rows over C/N/O/S/F/Cl and NONE/SINGLE/DOUBLE/TRIPLE/AROMATIC for each unordered
pair. It uses NumPy PCG64 seed **20261019**, independent of the native model
key. This means random probability vectors, not equal probabilities or random
hard molecules. Save the exact draws, logits, native key and artifact SHA;
never clip, repair, redraw or select a more convenient random initial state.
The existing shared graph supplies all branches; charge and occupancy stay fixed.

**Pivotal choice: delayed geometric ramp.** For one-based applied update u:

```
lambda(u) = 0.001                                      for 1 <= u <= 5000
lambda(u) = 0.001 * (50 / 0.001)^((u - 5000) / 5000)   for 5000 < u <= 10000
```

The final applied update uses exactly 50. The midpoint of the ramp at update
7,500 is about 0.2236. A saved state s carries the coefficient for update s+1;
the terminal state keeps the last coefficient. State 5,000 therefore prepares
the first ramp update. The 50,000-fold range is an exploratory choice to
make chemistry weak early and strong late, without an abrupt jump at 5,000.
This may trade away contact quality and does not guarantee a valid molecule.

The objective at each state is native contact + lambda × failure-aware repair
+ 0.1 × empirical training-element JS. The original internal repair weights,
atom-preserving search depths/budgets, empirical reference population and
frozen model/context remain unchanged. Source-equivalent repair is zero when
the hard readout already passes validity, including the entire previous 5,000-
update hexane trajectory; multiplying zero adds no chemistry gradient. The
random arm tests a different initial graph, with no promise of chemical efficacy.

**Pivotal choice: coefficient frozen within an update.** Freeze the coefficient,
repair references/caps/cuts, composition reference and native context/key for
all ten halving trials. Prefer the first strict AD + independent-primal decrease;
otherwise apply the lowest finite independent-primal candidate, earliest tie,
even if equal or worse. Ordinary lack of descent must not end the budget.
After acceptance, refresh the detached targets and next coefficient and
recompute the chemical scalar/gradient. Only the exact raw native observation
may be reused for byte-identical graph/key inputs. Never reuse a weighted
combined gradient after a coefficient change. Nonfinite/native/FD/parity/identity
failures remain real failures; preserve their partial trajectory.

Combined totals change with targets and coefficients. Do not rank them as one
objective. Record coefficient in state/trial/snapshot/check evidence, plot raw
and weighted repair, coefficient, fixed-contact overlays and actual within-
update deltas. Rank best contact only under the authenticated fixed contact
identity. Keep invalid/disconnected graphs and fallback/no-op counts visible.

## Implementation and validation

- [x] Reusable update-indexed schedule and seeded shared-graph initialization.
- [x] Optional schedule in the existing logit optimizer; v1 constant behavior retained.
- [x] Driver/configs for two fresh 10,000-update arms, with authenticated initial draws.
- [x] Initial and final coefficient native derivative assays on the same controlled
  invalid internal-F graph; contact-only order-null gates remain separate.
- [x] Disposable early and high-weight pilots, then reset exact requested initial
  graph/key for the full run. Runtime gate uses 1.5 × the slower measured mean
  × 10,000 + 1,800 seconds, plus elapsed setup, within a 23-hour allocation.
  This empirical estimate is not a worst-case line-search bound. Higher later
  backtracking/search cost or resource exhaustion can still leave a partial run.
- [x] Tests for schedule indexing/endpoints, invalid configurations, deterministic
  random schema, changed-weight native reuse/AD/FD, fixed trial coefficients,
  no-op budget consumption, saved-only two-arm reports and tampered native gates.
- [x] Independent GPU submitter and dependent allocated CPU comparison/evidence job.
- [x] Local static checks passed; no local numerical work.
- [x] Allocated CPU readiness passed, with actual XML/oracle/reference identities.
- [x] Two native GPU submissions authenticated; queued/running is not passed.
- [ ] Both 10,000-update trajectories and terminal native gates authenticated.
- [ ] Combined report generated on allocated CPU, downloaded, verified and opened.

`examples/brd4_chemistry.py` is shared by v1 and v2. Configurations are
`examples/configs/brd4_chemistry_ramp_{mix50,random}_10000.json`. Core schedule
and initialization live in `src/mosaic/small_molecule/chemistry/`. The new
`scripts/render_chemistry_ramp.py` reuses saved-frame validation and the existing
probability-aware viewer. It never runs models, searches or chemistry readouts.

CPU readiness reruns legacy engineering tests and required source-oracle JAX
parity tests, verifies exact campaign deltas and original repair/model/data pins,
and exports an authenticated training-only composition reference. Native jobs
verify the assessment SHA, source manifest, all proof files and selected config.
Scientific tests, draws, pilots, inference, optimization and rendering are
allocated DTU HPC only. Local work is editing/static/packaging/byte verification.

Submit each GPU arm to real queue `gpua10`, one exclusive GPU, four slots,
8 GiB per slot, 23 hours; report to `hpc`, two slots, 4 GiB per slot, one hour,
after both native jobs end. No concurrent copies of one arm. Before submissions
authenticate CPU DONE and assessment, current quota and immutable source bytes.
Each job has a quota gate; quota is not a reservation. Do not delete prior data.

## Actual receipts and outcomes

Initial access check succeeded. `gpua10` is Open:Active, max runlimit 1,440
minutes, so the 1,380-minute requests exist within queue limits. Home quota
reports 11.19/30 GiB (cached update 09:04 Europe/Paris); conservative free 18 GiB.
Work3 pool 6 reports 92.45/100 GiB. New model assets are unnecessary; only a
small immutable source release goes to work3 and numerical evidence goes to
the dedicated home Mosaic run directories. Fresh per-job quota gates remain.

This access/queue observation is not a passed numerical check or submission.


Source01 manifest `dc54acf72b906da9f5e2ceb0f6458314064e48408414496d1717a81117a91c80`;
362 files, archive39,187,476bytes SHA
`d1735e168c5c6396a58aea341107ee09c58b7daee510a79c8d22bba0f6dedd9c`.
Remote base `/work3/s234818/mosaic/releases/20261019-brd4-chemistry-ramp-01/source`.
Actual readiness **29753654 RUN**, started09:35:12 Europe/Paris, n-62-30-25,
hpc4slots/4GiB per slot/1hour. Native submissions remain gated on its outcome.


Readiness **29753654 DONE**, 217 legacy + 57 chemistry/schedule/report tests
= **274 passed**, no failures/errors/skips (three intentionally deselected
model/slow tests in the legacy marker selection). Actual XML and scheduler
success downloaded and authenticated with local stdlib byte/XML checks only.
Source-oracle generation produced12 pinned original-source fixtures. Assessment
SHA `35dac42a903ee397e77ff06f2b7a35cde5a191a475ebde007f844cc44912a895`.
New exported reference SHA
`37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb`.

Actual independent native submissions, 19October09:39:23 Europe/Paris:

| Arm | GPU job | Initial scheduler state |
| --- | --- | --- |
| Fresh50% chain/uniform |29753667|PEND, waiting for scheduling|
| Fresh random Dirichlet(1) |29753668|PEND, waiting for scheduling|

Both have only `done(29753654)` readiness dependency, no dependency on each
other. Scheduler records confirm gpua10/one exclusive GPU/four slots/8GiB
per slot/1,380minutes and the correct sealed source, assessment and individual
config arguments. Queued is not started or a native pass.

Report/evidence **29753669 PEND**, submitted09:39:24, dependency
`ended(29753667) && ended(29753668)`, hpc2slots/4GiB per slot/60minutes.
Output will be `/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-ramp-01-report/payload/report/`;
archive sibling `20261019-brd4-chemistry-ramp-01-report.tar.gz` and its receipt.
It preserves both studies and terminal failures; it cannot certify a native
pass from a scheduler state alone. Download/open remain future work after
actual completion.

Tracked compact receipt: `scripts/hpc/receipts/20261019-brd4-chemistry-ramp/campaign.json`.
Full local readiness proof: `results/hpc/20261019-brd4-chemistry-ramp/readiness-01/`.
Post-release documentation records actual outcomes; it is not retroactively
claimed part of the immutable executed source01 bytes.
