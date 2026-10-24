# Verified chemistry preparation timings

**Exact native-fact reuse is the strongest tested approach.** Allocated DTU CPU
job29809404 completed25October2026 at01:00:56Paris with79fixture passes and zero
failures/errors/skips. The original source643files and all six saved input NPZs
were unchanged before/after. An independent35-file byte/XML/schema/LSF audit
passed; [verification](delivery-verification.json) and [campaign](campaign.json)
bind the complete evidence.

Warm preparation times on three difficult random N25 saved states:

| Method | Seconds per preparation | Relative to original |
| --- | ---: | ---: |
| Original sequential | 16.87–18.17 | 1× |
| Cost/validation reuse alone | 18.16–18.49 | No material gain |
| Exact hard-graph native-fact cache | 0.453–0.461 | 37–40× faster |
| Cache plus cost reuse | 0.322–0.331 | 52–55× faster |
| Two persistent workers | 8.74–8.80 | About2× faster |
| Four persistent workers | 4.97–5.12 | 3.4–3.6× faster |

Each warm value is the median of two repetitions after a first encounter. All
methods used the same8-slot host, one native thread per process and the same
search budgets. Four workers plus coordinator fit that allocation. Pool startup
is included separately in first-encounter timings, not hidden in warm numbers.

Cold cache+cost preparation of the first random state was12.73seconds. With that
persistent cache, the first encounters of the next two *actual saved states*
took0.322 and1.683seconds. Those states have different soft probabilities; the
search is rerun with their exact probabilities. Only probability-independent
chemistry facts for exact hard graphs/schema are reused. No whole target is
reused under an argmax-only key. Caches are bounded and return private copies.

Four-worker first encounter was7.63seconds including pool startup, then about
5seconds per difficult search. It is useful for new graphs that lack cached
facts. Cache and process-pool implementations were tested separately; a combined
parallel-cache implementation has not been benchmarked.

The invalid Boltz assay was already much cheaper:0.270seconds original,
0.089seconds warm cache+cost,0.165seconds warm four-worker. Its first cache+cost
encounter was0.302seconds, slightly slower than original0.271. The valid Boltz
state was already about0.013seconds. Benefits depend strongly on the workload.

All108prepared batches (6inputs×6modes×3repetitions) matched the original target
arrays, full scientific receipt, status/check counts and identity exactly.
All36frozen loss/component/gradient comparisons matched, including exact gradient
array hashes. The difficult cases still report96base+192progress logicalchecks;
cache hits reuse previously authenticated facts without pretending to perform a
new native call. Original targets, loss scaling, candidate/tie order, budgets,
occupancy/charge rules and numerical checks were retained. See the complete
[benchmark data](benchmark-results.json) and [raw fixtures](cpu-tests.xml).

The [baseline profile](baseline-profile.txt) attributes most cold preparation
time to hard-graph diagnosis. Probability cost bookkeeping is a small component,
which explains why optimizing it alone did little. The optional local lazy
neighbor-generation draft was archived without promotion or included in the79pass
claim; its41planned fixtures remain unexecuted.

This is a detached-preparation benchmark on saved pilot inputs. It does not
establish a52×speedup of the entire native-model optimization, convergence,
long-run cache hit rate, or a passed future GPU runtime gate. The first cold
search remains costly. The recommended next validation is a new immutable
full native low/high pilot with bounded reuse and the original scientific gates,
before any deferred chemistry production run. Existing saved results, current
jobs and the paused monitor were not changed.
