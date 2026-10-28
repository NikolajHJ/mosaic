# Reusable report selections and connection profiles

`current-runs.json` lists the six existing saved trajectories and their ordered
segments. It contains paths and presentation choices, not copied scientific data.
Add a segment or select different run paths without editing report code.

Use `python scripts/reporting/report.py --runs my-runs.json`, or repeat `--run`
with a name from this preset or an absolute HPC saved-run path. One to24 runs are
accepted. Only terminal JQ1 N25 fixed-step/noisy saved schemas are supported by
this engine. Scalar history is complete; matrices are shown only where saved.
Optional `status_detail`/`continuation_status` notes describe their capture date,
not a fresh scheduler observation. Remove/update them when selecting newer runs.

`dtu-current.json` is an explicitly account-bound connection/capacity profile for
the current installation. Other users must supply their own profile, runtime,
authorized roots and authenticated capacity guard. Generic code reads these
values rather than hard-coding a contributor's account.
