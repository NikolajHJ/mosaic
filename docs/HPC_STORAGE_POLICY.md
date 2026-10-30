# Shared HPC storage accounting policy

This is the default for new Mosaic launchers and continuations. Use
`scripts/hpc/storage_accounting.py`; an experiment adapter may authenticate its
specific writers and inputs, but must not invent another counting policy.
Older sealed releases and failed receipts remain immutable. Adopt changes in a
new release with portable tests, allocated integration checks and actual startup
commands before scientific execution.

* Query live quota in exact bytes from every finite storage pool. WORK3 uses
  BeeGFS `--getquota --uid ... --csv` against its authenticated client config.
  Record raw commands, user/pool identity and observation time. Require a complete
  response within 60 seconds. Human-readable or cached HOME reports are not live
  exact-byte WORK3 evidence and must retain their own freshness handling.
* Existing files are already represented by live quota. Do not subtract all
  historical files, all current experiment trees, or another job's current output
  a second time. Current inventories remain useful for cap enforcement and
  forecasting. Count each future output scope once; require disjoint ownership.
* Budget remaining writes, `cap-current`, not a new copy of the full lifetime
  budget. Completed updates are not future updates. A terminal job has no new
  execution budget once its terminal proof and lack of successors are verified;
  retained files continue to consume reported quota.
* Forecast growing data separately from fixed setup. Prefer sustained measured
  growth and the actual save cadence. For the embedding study use each case's
  remaining updates times max(pilot rate, observed trajectory growth), with ONE
  25% margin and one 1 GiB campaign-control allowance. Recompute as progress grows;
  don't apply stacked multipliers or repeatedly amortize startup over20steps.
  This forecasts foreign writes; it does not change the other study or impose a
  new hard cap on it.
* Keep 4 GiB and 100,000-file safety reserves per filesystem. Add one 1 GiB quota/write
  burst allowance, rather than every historical byte. Finite quota pools and
  positive physical free-space/inode evidence still gate; an all-zero inode
  tuple is explicitly unreported and cannot prove filesystem file headroom.
* Account for known concurrent writers using authenticated remaining budgets.
  An unprofiled other-account allocation in RUN/SSUSP/USUSP gets an explicit
  temporary 2 GiB / 10,000-file forecast and remains identified in the receipt.
  Its actual state is never changed/fabricated, its job is never controlled,
  and live account quota is rechecked periodically. This is a forecast, not a
  whole-account reservation or a guarantee against arbitrary external writes.
  Unknown OWN/Mosaic allocation provenance, partial inventories, malformed
  quota, exceeded writer caps and ambiguous submission outcomes still block.
* Repeat admission at dispatch, entry and periodic checkpoints. Report quota,
  current usage, remaining writes, individual margins and reserve separately.
  A pass is a timestamped observation, never a physical reservation.

Keep all scientific gradient, precision, frozen-state, initialization, pilot,
optimizer/runtime and completion checks. Changing storage accounting does not
turn a failed scientific result into a pass. Sparse matrix output does not omit
scalar losses. Preserve exact saved studies and sources; storage policy changes
never authorize cleanup or foreign job changes by themselves.

Authorized 30 October 2026: the user requested correction of double counting and
inflated forecasts and made this the common policy. DBMol's new target is30,000
TOTAL updates at unchanged LR 0.0025; its saved 7,500 endpoint remains intact and
22,500 updates remain, subject to fresh gradient/runtime qualification.
