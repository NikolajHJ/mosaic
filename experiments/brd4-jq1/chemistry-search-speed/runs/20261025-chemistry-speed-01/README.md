# 20261025-chemistry-speed-01

Status: completed and independently verified. All79allocated fixtures and the
six-input comparison passed; warm exact reuse reached52–55×preparation speedup
on difficult random states. See [results and limits](RESULTS.md). Existing runs
and scientific results were unchanged. The following initial plan is retained
as historical preparation context.
User authorization: test promising chemistry preparation accelerations, preserve
all existing runs/results. Original search budgets and loss/scientific settings
are unchanged. Benchmark-only adapters are isolated from production sources.

Planned execution: one allocated hpc CPU job, eight slots/4 GiB per slot/two
hours, native threads limited to one. Persistent spawn pools use at most four
workers. CPU fixtures must pass before saved-input benchmarks. No GPU/native
model execution or optimizer updates. Persistent output/source allowance is
512 MiB, separately added to the unchanged whole selected-campaign storage gate,
with its 4 GiB and 100,000-file reserves. Check source/input hashes and capacity
before submission and again at job entry. Exclusive submission markers prohibit
blind duplicate recovery. All numerical work stays on allocated DTU HPC.

Inputs: original random Protenix and Boltz saved pilot snapshots, exact path/
length/SHA recorded by the package preflight and rechecked at entry. Original
chemistry source pinned to the 643-member source03 manifest. Profile serial,
cache, cost, combined reuse and 2/4-worker ordered parallel modes. Compare exact
reference identity/receipts/arrays and frozen loss/components/gradients, with
first-call and warm timings reported separately. No whole-run speed projection
or scientific readiness claim follows from microbenchmarks.

Local commands use the bundled Python with `-X utf8`, working directory Mosaic.
`scripts/prepare.py` packages stdlib-only; `scripts/launch.py` records one-shot
SSH/stage/submission; `scripts/observe.py` collects actual scheduler/log/output
evidence. Exact commands and source/job receipts will be added when executed.
Existing jobs, released source, stored molecules, delayed schedules and paused
monitor are preserved.

Actual submission25October00:49:47Paris: CPU29809404, eight slots/4GiB each/two
hours/hpc, no GPU. Source9files a30fd3937ea79935d0c449c600d2ea42f5f26da31906508a31e306f81d1c13b0;
archive17604bytes SHA5ec552bcdd5be9a98cba26b00a9db9720098214af66786eee09b7e9da9ce6b31.
Whole capacity plus separate512MiB passed32499679864required/40594731268available
bytes. First actual observation00:49:55Paris PEND, no fixture/benchmark outcome.
All six saved input path/length/SHA and exact productionsearch policy are pinned
in campaign.json. Local layout command retains preexisting discovery/historical
README errors; all newdirectories have README and all new Python ASTs passed.

At00:51:59Paris the allocated fixture stage logged79passed/no failures, and
the worker's exact79rawXML gate admitted the saved-input benchmark. Full job/
LSF/artifact independent delivery verification remains pending. Pytest's cache
write warning is expected for the read-only sealed source, not a skippedcase.
Benchmark source_update is a fixed per-input index for reproducible preparation
receipts, not a claim of rewriting the original pilot's global update metadata.
Original savedNPZ paths retain their real source step in their labels/filenames.

A separate local lazy-neighbor draft, now archived in ignored results,
explores exact lazy neighbor ordering. Its41 planned fixtures are unexecuted
and excluded from the sealed9files/source01/79-case claim; no speedup/adoption
is claimed for that optional draft. Current tested methods remain the three
cache/cost/ordered-parallel approaches.

Terminal25October01:00:56Paris: actual29809404DONE/hpc/same-job LSFsuccess.
All79unique raw fixtures passed, no errors/failures/skips. All108preparations
(6inputs×6modes×3repeats), targetarrays/complete receipts/identities and36frozen
loss/component/gradient comparisons matched exactly. Full643source and six
original input hashes rechecked after. Output798325bytes; no production changes.
Independent35file source/archive/XML/schema/LSF/input-byte audit passed.
Verification receipt and exact hashes are in campaign.json/delivery-verification.json.
