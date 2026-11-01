# Structural target discovery: matched comparison

**Current execution, 1 November20:45UTC:** reuse the completed coefficient1
controls `constant1-s17/29862957` and `constant1-s23/29862958` at20,000 updates.
Following the user's request, duplicate controls29870659/29870661 were held,
rechecked as never started with no outputs, then intentionally retired. The
structural jobs29870660/29870664 remain unchanged/PEND toward30k. Compare
validity/contact at the common20k checkpoint; the additional10k is unmatched
exploration. Archived controls usedV100 and the structural jobs useA100/A40,
so this is an exploratory cross-device comparison, not a clean same-hardware
speedup measurement. [Actual retirement/pairing proof](baseline-reuse.json).
The original four-arm source/plan/receipts below remain immutable history;
no saved results or experimental settings were changed or restarted.

Question: does constructing a connected repair backbone before the existing
bounded beam search improve chemical validity, native contact optimization or
end-to-end runtime? User authorized implementation and fresh HPC comparison on
1 November 2026. Existing runs, saved results and sealed releases are untouched.

The initial submission requested four fresh 30,000-update Protenix trajectories: beam and structural discovery,
each with initialization seeds 17 and 23. Within each seed the initialization,
GPU family, loss, optimizer and output measurements are matched. Seed 17 uses
A100; seed 23 uses A40, distributing queue exposure while preserving within-pair
runtime comparisons. Scientific model key remains 17 for all four original arms.
The reuse decision above supersedes execution of the two fresh beam controls.

Shared settings match the previous constant1 comparison: 25 occupied neutral
atoms, the same sparse random tree-plus-extra-edges initialization, frozen
Protenix tiny/context/JQ1 pocket, FP32, temperature 1, LR 0.3, chemistry
coefficient 1 throughout, composition and confidence margin off, no extra noise.
Exactly one finite fixed update; no backtracking, clipping or normalization.
The existing complete-target NLL, partial/local-valence/connectivity fallbacks,
and their weights are unchanged. The loss file is byte-identical to the prior
comparison snapshot (SHA256
`9ed3d8a5b609f7f1ecff432ed8186e452071ebfb8e3ed32ac903088427463c4b`).

The structural arm constructs probability-ranked, degree-cap-compatible
backbones, then removes/downgrades excessive bonds while retaining connectivity.
It considers bounded trim, aromatic-to-single and single-backbone proposals,
preserving atom labels, formal charge and occupied slots. Each accepted target
passes the existing native connected sanitization and canonical schema
roundtrip. At most six constructed candidates, 1,024 bond operations per
candidate, plus the initial graph check. A miss falls through to the original
beam and progress search. Degree caps are proposal heuristics, not acceptance
certificates. The method does not prove a globally closest repair or promise a
repair for infeasible fixed atom compositions. Targets remain detached and fixed
across every trial/gradient evaluation in a step.

Measurements: every scalar loss/update, argmax connected validity, atom/bond
identity changes, best contact and best valid-argmax contact, preparation route
and target edit count, preparation/full-step runtime and component gradient
norms. Save matrices and complete target receipts every 20 steps, plus endpoints,
latest and best. At step 0, every 1,000 and the endpoint retain 128 independent
categorical samples, including invalid samples, with connected validity and
unique valid SMILES. Samples are neither repaired nor rejected. Native contact
is a relaxed fixed-pose model objective, not experimental binding evidence;
hard argmax molecules and sampled molecules are not secretly rescored.

The A100 reservation is 10h per arm; A40 is 8h45. Budgets use same-device
completed 20k comparisons' late warm p95 (0.90846s/A100, 0.79054s/A40) multiplied
by 30k and 1.25, plus 30min setup/cleanup, rounded to five minutes. They are
estimates; the saved prefix remains recoverable if the new method is slower.
Own storage: 1GiB/4,500 files per arm plus 256MiB controls, using common
remaining-write accounting and periodic checks. No cleanup or global forecast
stacking. One allocated focused CPU check and actual CLI metadata startups
precede submission; ordinary production startup/first updates check the native
path. No new native/FD qualification chain.

Use the existing scripts with the new config variant:

```powershell
.venv\Scripts\python.exe experiments/brd4-jq1/protenix-random-chem/scripts/prepare_chemistry_screen.py --repo . --tag 20261101-structural-repair-01 --version source01 --comparison structural
```

This command has already prepared source01; do not repeat it or its remote
staging. Actual jobs/source/receipts are in [campaign.json](campaign.json).
For recovery read the remote `jobs.json` and each owned submission/intent/release
before any retry. Leave other running/pending experiments alone. Completed data
uses the canonical saved-result report command and existing engine/cache.

Verified 1 November 20:16 UTC: source01 is read-only. Allocated CPU 29870653
finished successfully with 42 cases (22 existing screen, 20 structural), zero
failures/errors/skips, and all four actual metadata startup commands passed.
All four long producers were accepted, persisted while held, and released:

| Discovery | Init seed | GPU | Actual job | Observed state |
| --- | --- | --- | --- | --- |
| Existing beam | 17 | A100 | 29870659 | PEND |
| Structural + beam fallback | 17 | A100 | 29870660 | PEND |
| Existing beam | 23 | A40 | 29870661 | PEND |
| Structural + beam fallback | 23 | A40 | 29870664 | PEND |

Independent byte/XML/LSF/command/held-release audit passed; see
[verification.json](verification.json). Native GPU execution, efficacy and
speedup are not yet established. `beam_search_performed` records invocation
of the old search routine, including its already-valid fast exit. The global
layout checker could not finish because a documentation file is not UTF-8;
owned Python syntax, UTF-8 notes and the focused allocated controls were checked.
