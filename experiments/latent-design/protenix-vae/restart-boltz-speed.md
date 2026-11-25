# Boltz2 branch throughput

Question: can independent children share one GPU efficiently while retaining the
full fixed-pose DBMol objective, LR 1, parent perturbations and gradient updates?
The seven full searches remain unsubmitted; their current baseline is still
200 updates/child, six generations, three parents and four children.

The existing runner now accepts an optional resident scorer pool:
`scorer.backend="vjp_pool"`, `scorer.parallel_workers=2`,
`scorer.worker_cpu_threads=2`, `search.batch_size=2`. Start with
`search.cpu_workers=1`, `search.overlap=false`. The launcher requests an exclusive
GPU with LSF-managed MPS and retains its MPS environment. Each native process
owns the two frozen Boltz trunks and evaluates one child. A CPU autograd bridge
returns each child's probability gradients separately, without averaging; the
CPU VAE receives its own latent gradient. Selection/archive updates still occur
after the completed generation. Defaults remain direct FP32, update noise off.

Implementation: `latentmol.guidance.boltz_vjp` and the existing
`latentmol.design.boltz_restart`/`protenix_search_branches` runner. Workers have
distinct PIDs, authenticated inputs, per-worker timing/memory, frozen-state
checks and cooperative shutdown. Failed/forced CUDA cleanup ends the allocation
before another MPS case. No daemon is started manually and no CUDA model/tensor
is shared between processes. Batch size three is now supported too.

Other opt-in `scorer.execution` settings test TF32, BF16 inside native neural
blocks, checkpoint cadence and checkpoint RNG preservation. TF32/BF16 change
arithmetic; compare measured losses/gradients/trajectories before adoption.
Block BF16 returns outputs to FP32; parent geometry/binning, heads and reductions
remain outside autocast. Broad whole-scorer BF16 is excluded here because it
also covers geometry/binning. Checkpoint cadence keeps block mathematics while
exchanging memory for recomputation. RNG replay opt-out is limited to frozen
eval/fixed-MSA execution and records its RNG-stream effect.

## Bounded native tests, 13 November 2026

Each case uses the same six seeded latent perturbations of the best valid ERK2
timing-pilot parent, with two updates per child. Initial latent/mask hashes must
match. Compare initial full loss, binding/combined latent gradients, terminal
latent/graph and aggregate updates/second. GPU UUIDs identify the physical
allocation. Warmup/loading/frozen checks are separate from measured optimization;
NVML samples record whole-GPU usage. Direct-worker cases capture geometry bins;
proxy bin equivalence is unmeasured (`null`). This measures throughput and local
numerical differences, not sustained search quality.

| Job | GPU | Question | Maximum updates / warmup gradient probes |
| --- | --- | --- | ---: |
| 30046016 | A10 | Precision and memory-feasible two-child execution | 72 / 36 |
| 30046017 | A40 | One/two/three workers, precision and checkpoints | 132 / 66 |
| 30046018 | A100 80 GB | One/two/three workers, precision and checkpoints | 132 / 66 |
| 30046054 | L40S | Actual VJP pool versus direct scorer, FP32 and block BF16 | 48 / 24 |

The first three have two-hour limits and the scout one hour. Total ceilings:
384 updates plus 192 initial gradient probes, no generated hard scoring calls.
Skip concurrency if same-policy serial reserved memory predicts more than 90%
of capacity. Retain OOM/incompatibility/nonfinite/detached failures; safe cleanup
is required before continuing. Flat gradients or changed molecules are findings.

Configs: [main grid](restart-boltz-speed.json),
[production-pool scout](restart-boltz-speed-scout.json). They share
`latentmol.design.boltz_speed` and the existing campaign submitter:

```powershell
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py stage --name boltz-restart-speed-scout-20261113a --config restart-boltz-speed-scout.json --hours 1 --max-updates 48
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py submit --name boltz-restart-speed-scout-20261113a
```

These names are already submitted; use new names for new work.
[Main commands/receipts](../../../outputs/boltz-restart-speed-20261113a/commands.json),
[scout receipt](../../../outputs/boltz-restart-speed-scout-20261113a/l40s-submission.json).

At **21:03 CET all four were PEND**. No measured GPU speedup is available yet.
A40 reported a recent-start throttle, A10 a host-slot limit and A100 a queue
CPU-slot cap; active L40S GPUs were occupied. Counts are CPU/job slots, not an ETA.
Existing jobs/releases/checkpoints were not changed.
[Latest status](../../../outputs/boltz-restart-speed-scout-20261113a/status-001.json).

Thirty-five scoped CPU tests passed: exact unaveraged mock gradients for all
three graph channels, real overlapping worker dispatch, failures/timeouts,
cleanup, execution modes, matched-start/device guards, search and resume wiring.
Ten scoped search/resume/benchmark checks passed again after adding pooled-worker
memory/timing logs. These establish wiring only; native GPU equivalence and
speedup are pending. Choose worker/precision settings from the matched evidence
before enabling them in full searches.

At **21:49 CET on 13 November**, all four jobs were still **PEND**. None had a
launch receipt or output directory, so there are no native throughput, parallel
speedup or memory results yet. Scheduler reasons: host job-slot limits for A10,
A40 and L40S; queue CPU-slot resource limit for A100. No reliable start ETA was
provided. No jobs were changed.
[Fresh scheduler and filesystem evidence](../../../outputs/boltz-speed-status-20261113T214909/snapshot.json).

Latest collection: **16 November, 16:50 CET**. All four speed controllers have
finished. A100 **30046018** ran 09:35–10:03 CET with zero launcher return code;
its results are summarized below. A10 ran 00:09–00:29, A40 06:38–07:15 CET;
A40 includes two failed experimental checkpoint cases. No jobs changed.
[Fresh A100 evidence](../../../outputs/boltz-single-target-plan-20261116a/summary.json).
Earlier collection:
[Scheduler/history](../../../outputs/boltz-speed-scheduler-20261116T091618_856115.json),
[collected results](../../../outputs/boltz-speed-status-20261116T091611/snapshot.json).

## First native results: L40S

Job **30046054** ran **18:19–18:33 CET on 14 November**, on `n-62-13-18`, with
managed MPS and zero launcher return code. All four cases completed; 48 optimizer
updates plus 24 warmup gradient probes executed. Each timed case contains six
identical seeded starts and two updates per child, hence 12 updates total.

| Arithmetic / concurrent children | Optimizer seconds | Aggregate seconds/update | Throughput vs FP32 serial |
| --- | ---: | ---: | ---: |
| FP32 / 1 | 85.60 | 7.13 | 1.000x |
| FP32 / 2, VJP pool | 91.17 | 7.60 | 0.939x |
| Neural-block BF16 / 1 | 62.45 | 5.20 | 1.371x |
| Neural-block BF16 / 2, VJP pool | 57.22 | 4.77 | 1.496x |

Parallel FP32 took **6.5% longer**. Within BF16, parallelism improved aggregate
throughput **9.1%**, reducing optimizer time **8.4%**. The combined best option
reduced optimizer time **33.2%** versus FP32 serial; most of that gain is precision,
not concurrency. This is one short, ordered measurement, without repeated timing
confidence bounds. Including final worker verification/shutdown, group times were
90.39/103.62/67.27/69.91 seconds respectively: BF16 parallel is slightly slower
than BF16 serial on that short inclusive measure. Loading and initial probes are
outside both timing measures; long-run amortization remains unmeasured.

At a fixed arithmetic setting, serial and pooled execution had identical initial
losses, binding/combined latent-gradient hashes and terminal latents/graphs for
all six children. All 24 terminal observations were valid. All cases reported
the same physical GPU UUID; frozen checks and cooperative shutdown succeeded.

BF16 versus FP32 changed initial losses by up to **0.03291**, with binding-gradient
cosines at least **0.9998628** and relative L2 differences up to **4.49%**. The initial
native geometry bins matched in the direct-worker probes. Despite close gradient
directions, **two of six terminal molecules differed after two updates**. This is
an approximate numerical variant, without evidence yet of equivalent long-search
quality. Serial/pool agreement at the same precision does not resolve that issue.

Summed per-worker reserved peaks were 15.56/27.83/14.80/25.83 GiB respectively;
these are sums of individual maxima, not simultaneous whole-GPU usage. No NVML
samples were captured, so no physical utilization/peak claim is supported. Logs
retain a Lightning checkpoint minor-version warning; there were no failed cases.

Current assessment: FP32 parallelism has not earned adoption on L40S. BF16 is the
promising speed change, with only a modest added parallel benefit. A paired
200-update quality check under common FP32 hard scoring, plus the other GPU
benchmarks, should precede adopting it for the seven-protein baseline. Defaults
and the unsubmitted full-search configs remain unchanged.
[Derived timings](../../../outputs/boltz-speed-status-20261114T211250/analysis.json),
[per-child checks](../../../outputs/boltz-speed-status-20261114T211250/child-summary.json),
[terminal history](../../../outputs/boltz-speed-history-30046054-20261114T211318_241954.json).

## A10 and A40 results, 16 November

A10 **30046016** completed four cases (48 timed optimizer updates, 24 warmup
probes); both two-worker cases were skipped by the reserved-memory estimate.
A40 **30046017** completed eight cases (96 updates, 48 warmup probes), skipped
three-worker BF16 on its memory estimate, and recorded two OOM failures while
testing half checkpoint coverage. These are successful controller completions,
not a claim that every tested configuration succeeded. Failed native work is
additional to the successful update/probe counts.

| GPU | Setting | Seconds / same 12 updates | Throughput vs own FP32 serial |
| --- | --- | ---: | ---: |
| A10 | FP32 / 1 | 175.01 | 1.000x |
| A10 | TF32 / 1 | 149.97 | 1.167x |
| A10 | Neural-block BF16 / 1 | 115.31 | 1.518x |
| A40 | FP32 / 1 | 136.63 | 1.000x |
| A40 | FP32 / 2 | 132.92 | 1.028x |
| A40 | TF32 / 1 | 126.19 | 1.083x |
| A40 | TF32 / 2 | 123.62 | 1.105x |
| A40 | Neural-block BF16 / 1 | 93.10 | 1.468x |
| A40 | Neural-block BF16 / 2 | 89.89 | 1.520x |

At the same precision, A40 parallelism adds **2.8% FP32 / 2.1% TF32 / 3.6% BF16**
throughput, while roughly doubling reserved memory. These are independent-worker
measurements; the earlier L40S parallel case used the VJP pool, so cross-card
differences also include backend/CPU differences. No repeat-timing uncertainty
was measured. Disabling checkpoint RNG preservation changed serial BF16 time
by only −0.1% on A10 and +0.5% on A40: no meaningful benefit is established.

TF32 retained all six two-update terminal molecules on each card. Its initial
binding-gradient cosine was at least 0.9999944, relative L2 difference at most
0.72%, and absolute initial loss change at most 0.00603. BF16 changed three/six
terminal molecules on A10 and two/six on A40; maximum relative gradient L2
differences were 8.59% and 7.30%, with cosine minima 0.9997688 and 0.9998456.
All completed terminal observations were valid. Initial geometry bins matched
for all completed direct-worker cases. Short endpoint agreement does not
establish long-search quality or precise derivative equivalence.

Halving checkpoint coverage exhausted A40 GPU memory with both one and two
workers, including about 44 GiB usage in the single-worker failure. Workers
synchronized, checked frozen state and exited without forced termination; no
group required reuse of an unhealthy allocation. No NVML samples were captured.

Assessment: precision remains the main measured speed lever. Single-worker TF32
with full checkpointing is the more conservative candidate for a longer paired
test; BF16 delivers the larger throughput gain with more numerical/trajectory
change. There is little evidence yet for defaulting to multiple workers. A100
results were pending at this collection; no defaults were adopted. The subsequent
[six-pair 200-update quality comparison](restart-boltz-precision.md) was submitted
on 16 November with common exhaustive FP32 scoring of both trajectories.
[Derived comparisons](../../../outputs/boltz-speed-status-20261116T091611/analysis.json),
[per-child observations](../../../outputs/boltz-speed-status-20261116T091611/child-summary.json).

## A100 80 GB results, 16 November

Job **30046018** completed ten cases and skipped one predicted-memory case.
For the same twelve timed ERK2 updates, FP32 serial/two/three-worker aggregate
times per update were **7.038 / 6.455 / 6.268 seconds**. Neural-block BF16 serial
and three-worker times were **4.652 / 3.855 seconds**: **20.7%** extra throughput
from three workers within BF16, and **1.826x** FP32-serial throughput combined.
This is the clearest parallel benefit measured so far, on an 80 GB GPU.

BF16 changed one/six terminal graphs after two updates; maximum initial loss
change was 0.0412, gradient relative L2 difference 7.28%, and cosine minimum
0.999794. TF32 serial took 4.946 seconds/update and retained six/six terminal
graphs, but its geometry-bin probes changed in all six cases. Half-checkpoint
BF16 serial took 3.907 seconds/update while reserving **77.54 GiB**, leaving very
little headroom; its three-worker variant was skipped. These are short timing
and numerical observations, not long-search quality or V100 performance evidence.
[Native results and readiness investigation](../../../outputs/boltz-single-target-plan-20261116a/summary.json).
