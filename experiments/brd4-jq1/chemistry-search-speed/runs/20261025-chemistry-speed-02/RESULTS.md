# Allocated CPU comparison results

CPU 29809477 completed with 143 fixtures and exact six-input comparisons.
The byte/XML/schema audit is verification.json; original callback failure
and every original source/results file remain preserved. GPU results are now verified in GPU_RESULTS.md.

All timings below are preparation seconds on the allocated CPU host.
First encounter includes cold native facts and pool startup; warm is the
median of the next two calls. Each mode retains its cache across saved inputs,
so input1/input2 first encounters measure successive saved graphs, not fresh
independent cold replicas.

| Input | Serial first/warm | Cache+cost first/warm | Hybrid4 first/warm | Hybrid8 first/warm |
| --- | ---: | ---: | ---: | ---: |
| Random early0 | 16.833/16.072 | 10.576/.345 | 8.674/.288 | 7.633/.282 |
| Random early1 | 15.881/15.889 | .308/.302 | .372/.285 | .283/.278 |
| Random early2 | 15.827/15.820 | 1.469/.303 | .639/.288 | .503/.282 |
| Random high10 | 1.101/1.097 | .787/.343 | .470/.277 | .444/.276 |
| Boltz valid | .013/.011 | .015/.008 | .024/.009 | .026/.009 |
| Boltz invalid | .258/.255 | .281/.082 | .183/.084 | .174/.084 |

Warm early-random preparation improved 56–57 times with hybrid8 versus
serial. Eight workers improved warm times only 2–3 percent over four, while
using twice as many workers; four is the more economical candidate to validate
in a future native pilot. The first cold input improved only 1.94/2.21 times
with four/eight workers. This is not a whole-optimization speed estimate.

The separate cold native profile identifies DetectChemistryProblems as the
main cost: 290 baseline calls took 14.848s, versus 773 SanitizeMol calls at
.054s. Cache+cost reduced detection to 193 calls/9.752s. This supports central
native-fact caching plus CPU workers; GPU bookkeeping cannot replace RDKit
chemistry diagnosis. Profiling overhead is included and these profiles cover
only input0 baseline/combined, not worker internals.

All six inputs retained original checks, targets, full receipts/identities,
scalar/components and gradient hashes. Difficult random preparations still
charged 96 original plus 192 progress checks. Worker bundles are not individual
RDKit calls. Nested cost_reuse counters in saved rows are final-context totals
because of shallow copying; top-level primitive cache/task counters remain
per-row snapshots. No long-run cache hit rate or native-model timing is proven.

The GPU matrix features and scan/log-semiring loss candidates remain separate
experiments. Their 19 GPU-only checks require actual L40S execution, original
2e-5 parity tolerance and .01/.003/.001 FD ladder/15 percent/two adjacent.
No existing run, model, objective, saved result or paused monitor was changed.

Historical queue observation: GPU29809554 was actually submitted after admission03 passed19 allocated
metadata regressions and actual CPU/callback terminal authentication. At
01:29:38Paris it is PEND on queue physical GPU limits. No GPU performance
result or production installation is claimed; the monitor remains paused.

Completed GPU outcome25October: all19 checks passed; matrix local-risk
value+gradient6.16x faster atbatch1/3.77x at32,24percent slower at256.
Scan warm execution slower throughout. See GPU_RESULTS.md; these are component
gains and not overall optimization speedups.
