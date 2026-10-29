# Valid-source confidence margin comparison

Experiment ID: `brd4/chemistry-confidence`. Parent: the
[compatibility probe](../chemistry-compatibility/README.md).

## Question and scope

Does weak confidence guidance improve robustness of an already-valid argmax
without excessive sharpening or atom/bond identity locking? This is an opt-in
experimental objective, not a change to current production configurations,
saved runs, VAE training, native models or their validation protocols.

Current repair is the unchanged `repair-failure-guidance-v1`. The comparison
adds only `valid-confidence-huber-v1`, weight0.01, log-odds margin0.2 and
slope cap1.0. For deficit `t=max(0, margin+logsumexp(rivals)-target)`, the
unweighted penalty is `t*t/2` below the cap and `cap*(t-cap/2)` above it.
Its derivative is bounded by the cap; its value is not globally bounded.
The loss and gradient are zero once the target probability exceeds sigmoid0.2.
Mean atom and mean unordered-bond terms receive equal weight. Inactive slots,
charges and edges touching inactive slots receive no margin guidance.

Only independently valid SOURCE argmax rows (authenticated preparation route0)
are eligible. Repaired witnesses are not eligible. All labels, validity masks,
occupancy masks and coefficients are detached/fixed within each step and its
trials. Targets refresh only between updates. Atom choices remain trainable;
no live atom-dependent valence cap or sulfur preference enters the margin.
Composition weight is exactly zero; counts and JS remain diagnostics. Margin
weight is independent of the chemistry repair ramp.

## First controlled comparison

Allocated CPU fixtures check independent primals/atom and bond FD at predeclared
0.01/0.003/0.001 scales, two adjacent agreements, bounded slopes, inactive and
invalid routes, tamper detection, zero-weight baseline parity, independent
native/repair/margin gradient addition and actual fixed-step target lifecycle.
The native term in these integration fixtures is explicitly a polynomial test
function, not a Protenix/Boltz binding model.

Matched chemistry-only pilots compare baseline repair against repair+margin:
four predefined starts (uncertain valid CCC, confident valid CCC, valid15atoms
including3sulfurs, invalid F with two bonds), seeds17/23/29,128fixed updates,
LR0.3, one finite candidate, repair coefficient1.0, and identical search policy.
This isolates chemistry behavior; it is not a native binding optimization or
a claim of preserved/improved binding. Start jitter is fixed before arm choice.
Both arms use the same categorical random draws at checkpoints0/64/128.
Raw argmax and256 sampled graphs/checkpoint are inspected without repair or
rejection conditioning; all sampled matrices, including invalid ones, are saved.
Report validity/failure categories, identities/changes, sulfur counts, confidence,
valid-sample canonical/fingerprint/scaffold diversity, composition diagnostics,
gradient norms and warm loss/gradient/preparation/end-to-end runtimes. A
separate identity-change fixture proves this weak term is not a hard atom freeze.
No improvement assertion or post-hoc coefficient tuning promotes the hypothesis.

Native binding efficacy, long-horizon exploration and useful molecular diversity
still require a later matched full-model comparison using unchanged source/input
pins and all native/gradient/frozen/runtime/storage gates. CPU chemistry-only
measurements must not be presented as those model results. Targets change between
steps, so refreshed total surrogates are not one fixed objective over a run.

The CPU comparison's JS diagnostic reference is its own matched initial soft
composition, not a training dataset. The library's existing caller-supplied
empirical reference/diagnostics are unchanged. A coefficient0.01 is a small
predeclared value, not proof of a small gradient relative to each native model.

Explicit library opt-in for a NEW qualified experiment:

```python
from mosaic.small_molecule.chemistry.confidence import ConfidenceMarginConfig
options["element_js_weight"] = 0.0
options["confidence_margin"] = ConfidenceMarginConfig(
    weight=0.01, margin=0.2, slope_cap=1.0,
).record()
```

Existing sealed drivers/configuration schemas are not repinned or extended by
this example. Absent the new key, old values/gradients/component schema remain
unchanged. Current production runs and VAE training do not enable this term.

## Ownership and execution

- Shared implementation: `src/mosaic/small_molecule/chemistry/confidence.py`;
  explicit optional integration in `chemistry/trajectory.py`.
- Shared allocated fixtures: `tests/small_molecule/test_chemistry_confidence.py`.
- `scripts/`: controlled comparison and stdlib package/allocated execution.
- `runs/`: immutable attempt manifests, scheduler/test/integrity receipts.
- Large raw outputs: ignored `results/hpc/<attempt>/`, transferred to laptop.

All numerical work runs on allocated DTU HPC CPU. No local model/array tests.
The operator packages a byte-pinned source and uses an interactive batch CPU
allocation with node-local source/output, streaming the resulting archive back
to the laptop. It writes no shared WORK3/HOME study data or persistent remote
source. Source is sealed read-only after node-local verification. For future
allocations, inspect active admission controllers: a conservative account-wide
guard may still classify any new allocation as external even without shared
writes. Node-local storage alone is not a scheduler exemption.
Each hpcint allocation has4CPU/2GiB per slot and an enforced840-second process
deadline (TERM then KILL after15seconds), one-thread numerical libraries,
256MiB output cap and at least1GiB node-local free space. Never invent LSB_JOBID.

From the Mosaic root, local stdlib preparation/execution:

```powershell
py -3.13 experiments/brd4/chemistry-confidence/scripts/confidence_operator.py prepare
py -3.13 experiments/brd4/chemistry-confidence/scripts/confidence_operator.py run
```

The exact allocated pytest and comparison commands, archive hashes, actual job
ID, runtime and scope are saved per attempt. `run` refuses existing transfer or
unknown outcome markers; inspect receipts before recovery. No automatic retries,
cleanup, monitor changes or scientific continuation submissions.

## Current status

29October2026: final CPU job29848105 actualDONE on n-62-29-6,12:50:51--12:53:18Paris.
All35fixtures passed in21.14seconds, zero failures/errors/skips. All12matched
pairs (24trajectories,128updates each) completed. Independent146source-file,
82evidence-member,72sample-archive, rawXML/LSF/scheduler/schema audit passed.
See [comparison results](runs/20261029-chemistry-confidence-05/results.md).
Only modest/zero sampled-validity changes occurred; native binding improvement
and long-run exploration remain untested. Keep the margin opt-in/experimental.

Earlier attempt001 rejected before submission because batch queue hpc
does not accept interactive jobs. Its source/transport receipts remain intact.
Read-only site queue inspection identifies hpcint as the allocated interactive
CPU queue. Attempt002 changes only this resource declaration/guard; same loss,
tests and comparison.
Attempt002 was also rejected before submission: the site's hpcint esub rejects
the `-W` option. Attempt003 uses hpcint's normal interface without `-W` and an
explicit1700-second process timeout, unchanged scientific checks/settings.
Attempt003 job29848043 EXIT1 before any numerical import or source extraction:
inherited TMPDIR was on shared storage and the required guard rejected it.
Attempt004 explicitly uses compute-node `/tmp` and an840-second process deadline
inside the observed15-minute site limit; no guard or scientific check removed.
Attempt004 passed numerically. Attempt005 repeats its exact scientific/test bytes
after explicitly setting/proving child TMPDIR/TEMP/TMP beneath node-local scratch.
It also rejects existing execution markers independently of transfer folders.
See [run005](runs/20261029-chemistry-confidence-05/README.md)
and the [global ledger](../../../docs/experiment_ledger.md).
