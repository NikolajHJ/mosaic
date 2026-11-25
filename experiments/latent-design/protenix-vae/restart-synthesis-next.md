# Synthesis guidance follow-ups — 13 November 2026

The 17 November audit supersedes the historical low-SA hit counts and the interpretation of several individual molecules below. [Audited results](restart-synthesis-followup.md#audited-synthesis-scores-and-validity-17-november-2026).

Question: can controlled synthesis gradients simplify the archived winners
without losing their fresh Protenix contact scores? These are small exploratory
tests on `codex/protenix-restart-search`, using frozen models and no new training.
Each hypothesis gets **one seed, 1313113**, and the same three paired founders
from the near-valid, deep and wide searches. They are selected difficult
molecules, not a representative molecular-generation benchmark.

Every child has **500 updates, LR 60, displacement cap 0.25, 25 occupied atoms**,
and temperature 1. Chemistry and valence weights are 1 unless the wave profile
below explicitly changes chemistry. **Optimization noise is zero.** Matched
parent kicks remain child initialization; they are distinct from update noise.

## Short comparisons

[The shared config](restart-synthesis-next.json) selects related variants by
task. A ratio refers to a synthesis/contact latent-gradient norm ratio, not a
raw loss coefficient. Fixed guidance calibrates once; controlled guidance also
caps its combined contribution against an EMA of the contact-gradient norm
(decay 0.9). After more than 20 consecutive invalid observations, its validity
controller scales guidance to 0.2. Logs retain the actual contributions.

| Task | Hypothesis and variants | Updates | Configured queue |
| --- | --- | ---: | --- |
| `weights` | Baseline, fixed SynFrag 0.1, controlled 0.1 and controlled 0.3: does fixed calibration become too strong as binding gradients change? | 6,000 | A10 |
| `phases` | Controlled 0.3; binding-only first 100 updates, full guidance for 300, then 0.1× guidance for 100; constant 0.186 nominal exposure control; binding-conflict projection. | 6,000 | V100 |
| `views` | Controlled SynFrag 0.1 versus the worst difficulty across guide-only temperature views 1 and 0.5. Binding keeps its normal temperature. | 3,000 | A40 |
| `gasa` | Baseline, controlled SynFrag 0.1, GASA 0.1, and SynFrag/GASA 0.05 each. Does a different topology critic help? | 6,000 | A10 |
| `defog` | Baseline, controlled SynFrag 0.1, DeFoG 0.1, and SynFrag/DeFoG 0.05 each. Does a molecular structural prior help where synthesis predictors are unreliable? | 6,000 | L40S |
| `anchors` | Baseline, controlled SynFrag 0.1, low-SA latent attraction 0.1, SynFrag/low-SA attraction 0.05 each, and the same combination with an unfiltered real-molecule bank. | 7,500 | A40 |

The projection removes a synthesis-gradient component that opposes the contact
gradient; this is a first-order control, not guaranteed binding preservation.
Exposure controls match nominal scheduled ratios, not realized gradient work.

These six tasks total **34,500 updates / 69 children**, with up to six fresh
attempts per founder and variant: **414 attempts** overall. Short panels use independent equal allowances after per-arm deduplication;
actual attempts, successes and shortfalls remain separate. A failed or
zero-valid arm cannot shrink another arm's allowance. Founder rescoring is charged to the real budget
and distinguished from genuinely new identities. DFRscore is also retained as
an independent hard-molecule readout rather than optimized in these variants.

## Four-wave search and recovery

[The wave config](restart-synthesis-wave-focus.json) compares four policies:
constant controlled SynFrag 0.1; constant 0.1625; wave ratios
**0.1 → 0.1 → 0.4 → 0.05**; and the same schedule with Pareto parent selection
using fresh contact loss and detached hard SA. The 0.1625 arm matches the
schedule's nominal mean ratio.

Each policy uses three children per wave, **6,000 updates and up to 72 fresh
attempts**, totaling **24,000 updates / 48 children / 288 attempts**. Fresh
allowances are independent, so an empty shortlist in one arm cannot prevent
the others from updating their archives. Compare actual scoring costs too.

Archive selection reserves the best fresh-scored parent. Other choices use a
maximum Morgan similarity of 0.85 and a 100-update source-step gap. The Pareto
arm also targets fresh loss ≤2.35. Budget-preserving relaxations and exemptions
for the protected best parent are logged. Wave four replays observed wave-three
latents **without a new kick**, selecting replayable terminals or documented
fallbacks, including observed near-valid states with a small repair witness.
The witness is evidence; its edited graph is not substituted for the latent.
All policies use chemistry weights **1 → 1 → 1 → 2** for this recovery test.

## Scope of the new guides

- **GASA:** released frozen easy/hard classifier, with a pure-Torch adapter for
  expected degree and probability-weighted attention. Binary layer algebra was
  checked against an independent sparse reference. Ring/conjugation features
  remain explicit approximations; native DGL runtime parity is unmeasured.
  [Weights, source and GPL-2.0 receipt](../../../outputs/synthesis-next-models-20261113/gasa/assets.json).
- **DeFoG:** deterministic consistency at model time 0.95, using the existing
  GuacaMol checkpoint in an isolated CPU worker. Its continuous auxiliary
  features and explicit charge-null projection are experimental. This is a
  structural prior, neither a synthesis classifier nor a likelihood; no sampler
  or random optimization noise is introduced.
- **Latent anchors:** posterior means of up to 32 low-SA (≤3), 25-heavy-atom
  ChEMBL training molecules, paired with a disjoint unfiltered bank. At most
  2,048 seeded cached rows are examined. Controls receive one recorded scalar
  to match mean latent norm; both original and transformed decode diagnostics
  are retained. Low source SA does not guarantee low decoded SA or a route.
- **SynFormer:** public graph-encoder code exists, but compatible pretrained
  graph weights are unverified; the public default config uses SMILES. Expected
  token embeddings do not supply a differentiable soft-graph-to-SMILES bridge.
  No SynFormer GPU arm is claimed in this batch and its 2.76 GB checkpoint was
  not downloaded. [Feasibility audit](../../../outputs/synthesis-next-models-20261113/synformer-feasibility/audit.json).

## Preparation and current status

Allocated CPU library preparation **30042663 completed** after initial job
**30042655** failed on command quoting. The corrected job used the bounded
[library preparer](../../../src/latentmol/design/protenix_synthesis_library.py);
its [submission receipt](../../../outputs/synthesis-next-models-20261113/library-submission-r2.json)
and [local checks](../../../outputs/synthesis-next-models-20261113/library-preparer-checks.json)
are retained. Invalid reconstructions are recorded findings, not a selection
filter. Both 32-entry bank files now have verified hashes in the sealed config.
Source-molecule mean SA is 2.430 for the low-SA bank and 2.833 for its disjoint
unfiltered control; fixed decodes were valid in 29/32 and 25/32 respectively.
This is a modest source-difficulty contrast, not an easy/impossible split.

Allocated CPU smoke **30043136 completed**: SynFrag, DFRscore, GASA, DeFoG and
low-SA latent attraction had finite attached latent VJPs and unchanged frozen
weights. DeFoG took 19.18s for its first load/evaluation and 0.244s for the next
call on this fixture. This is a backward usability check, not derivative or
native-parity qualification. Earlier smoke 30043102 failed on a harness panel
key; 30043112 exposed a protocol-negotiation mismatch in the new DeFoG worker.
The worker was fixed and its actual init/evaluate/close protocol now has a
model-free regression. No original optimization campaign was changed.
[CPU smoke result](../../../outputs/synthesis-next-models-20261113/model-smoke-result-r3.json).
The final focused local selection passed **68 tests**.

All seven GPU jobs were submitted and independently observed PEND:

| Task | Job | Queue | State at submission |
| --- | ---: | --- | --- |
| `weights` | **30043380** | gpua10 | PEND |
| `phases` | **30043381** | gpuv100 | PEND |
| `views` | **30043383** | gpua40 | PEND |
| `gasa` | **30043384** | gpua10 | PEND |
| `defog` | **30043385** | gpul40s | PEND |
| `anchors` | **30043386** | gpua40 | PEND |
| `seed-a` | **30043389** | gpuv100 | PEND |

Short tasks have six-hour wall limits; the four-wave panel has eight hours.
These are upper bounds, not runtime predictions. Total optimization budget is
**58,500 updates**, with at most **702 fresh calls**; equal allowances do not
imply equal consumption or wall time. No GPU optimization outcomes are claimed.
[Submission and scheduler summary](../../../outputs/synthesis-next-models-20261113/submission-summary.json),
[short source/commands](../../../outputs/protenix-restart-synthesis-next-20261113a/commands.json),
[wave source/commands](../../../outputs/protenix-restart-synthesis-waves-20261113a/commands.json).

Both source snapshots were checked against the final guidance and runners.
The new jobs use separate output/release paths. Existing pending/running jobs,
checkpoints and caches were preserved.

Use the central runner to inspect either plan:

```powershell
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-synthesis-next.json --arm weights --plan-only
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-synthesis-wave-focus.json --arm seed-a --plan-only
```

Judge fresh contact loss jointly with hard SA, validity, novel identities,
top-three quality and diverse low-loss yield at recorded updates, scoring
attempts and wall time. Surrogate improvement alone is not success. Hard
chemistry and any route assessment remain outside the differentiable path.


## All seven pilots complete, 16 November 2026

All seven jobs above have completed result records and zero launcher return
codes: **58,500 optimizer updates** completed. Short panels took about 0.6–1.3
hours each; the four-wave panel took 4.16 hours. No new jobs or job changes.
[Full results and launch evidence](../../../outputs/restart-status-20261116T100852/remote.jsonl),
[compact metrics](../../../outputs/restart-status-20261116T100852/summary.json).

Within their paired panels, the clearest signals are:

- Low-SA latent attraction reduced selected mean SA **7.513→7.086** and raised
  QED **0.456→0.527**, with trajectory validity **100%→92.9%**. SynFrag plus
  low-SA attraction gave SA **7.266**, QED **0.504**, validity **99.7%**; the
  matched unfiltered-bank combination gave **7.556**, **0.494**, **93.9%**.
  These are selected-molecule means, not route feasibility or all-state quality.
- DeFoG reduced selected mean SA **7.444→7.184**, improved QED **0.484→0.564**,
  and retained essentially 100% validity. Best novel binding loss worsened
  **2.06081→2.13525**; SynFrag+DeFoG reached **2.11538**. The trade-off matters.
- GASA kept validity at 100% but did not improve selected mean SA over its
  unguided baseline (**7.524 versus 7.194**). It is not an established win.
- Within-child phased guidance improved validity versus strong constant
  guidance (**96.2% versus 82.3%**) but worsened selected mean SA
  (**7.779 versus 7.156**); its exposure-matched constant arm was 100% valid
  with mean SA 7.372. Fixed weak calibration had 76.7% validity versus 100%
  with the weak online controller, supporting bounded guidance over that
  fixed-calibration setting.
- Whole-wave focus did not improve best binding or top-three binding quality
  after wave one, and did not beat the matched constant control on minimum
  archive SA (**5.683 versus 4.655**). Adding Pareto parent selection found
  **SA 4.217 at fresh binding loss 2.2028**, versus **SA 4.655/loss 2.2434** for
  constant matched guidance. This is one useful joint-quality point, not proof
  that wave scheduling itself is better. Pareto used 53 fresh attempts versus
  constant matched 44; allowances and updates were equal, actual costs differed.

No pilot materially beats the best starting founder's binding loss **2.051323**.
Tiny founder-rescore differences are numerical ties. Results vary across GPU
families even for nominally identical controls, so compare arms within a task;
one seed does not establish a robust cross-task ranking. Evaluate lower-SA
molecules at acceptable binding, not just the binding-best candidate.

Sixteen of 23 short arms share the same binding-best novel molecule
(SA 8.124, fresh loss approximately 2.060812). This is a recurring attractive
basin, not sixteen independent improvements. Selected mean SA/QED include
selected valid graphs whose fresh scoring failed; successful-score chemistry
threshold counts are separate.
[Paired metrics and collapse audit](../../../outputs/restart-status-20261116T100852/short-results-summary.json).


## Detailed attraction and parent-selection audit, 16 November 2026

The attraction result is heterogeneous. Anchor-only improves selected mean SA
for two founders and worsens one; most aggregate gain comes from founder 1
(7.808→6.358), while founder 0 worsens 7.868→8.228. SynFrag+low-SA attraction
beats the matched SynFrag+unfiltered-bank SA means for all three founders,
but its best fresh loss is slightly worse in all three. It generates more
branchwise distinct valid identities (77/178/43 versus 30/87/33); these counts
are correlated within trajectories, not independent replicates.

At SA≤5 and fresh loss≤2.3, successful selected novel counts in the anchor
panel are baseline 0, SynFrag 0, attraction-only 1, combined-low 0,
combined-unfiltered 0. Thus the combination's higher validity and lower mean SA
have not yet produced more candidates meeting this joint cutoff. Attraction-only
loses validity at both lower-SA endpoints, making its intermediate archive
important. Combined-low has a valid SA 4.363 endpoint absent from the fresh
shortlist; its binding remains unknown. Means include selected valid graphs
whose fresh scoring failed. [Per-founder audit](../../../outputs/synthesis-analysis-20261116/anchor-analysis.json).

A new allocated CPU frozen-decoder diagnostic **30077712 completed** in about
11s; it performed no optimization, training or fresh Protenix scoring. Original
bank diagnostics use occupied slots 0–24, while all search founders use 0–23,25.
Low-SA bank decodes are valid in **29/32** under the original mask but only
**1/32** under the actual search mask; the unfiltered bank changes **25/32→3/32**.
The earlier bank validity figures therefore do not apply to the search's view.
The decoder has fixed slot outputs; changing existing founder masks at unchanged
latents would change molecules. Re-encoding arbitrary graph permutations is not
a guaranteed latent-permutation correction. Align and verify a new bank under
the actual search mask, or use a newly scored matched founder panel sharing
canonical contiguous occupancy; do not silently alter old founder identities.
Initial diagnostic 30077685 failed on the harness's obsolete two-argument
hard_from_graph call; its corrected retry and outputs are preserved.
[Diagnostic result](../../../outputs/synthesis-analysis-20261116/bank-mask-result-r2.json).

At the original founders, the effective low-SA anchor counts (exp entropy of
soft distance weights) are **30.79/30.46/30.14 out of 32**, with each strongest
anchor carrying only 5.4–6.4% weight. The present temperature 0.5 objective is
therefore close to broad attraction toward the bank centroid, not local
attraction to a single easily synthesized molecule. Observed gains survive as
optimization findings, but their chemical-guidance interpretation is weaker.
A centroid control and a more local softminimum are high-priority controls.

The strongest selection evidence is preservation of an alternate lineage:
binding-only policies use only founder 0 descendants after wave 1; the Pareto
policy keeps founder 1 in one slot. It re-expands a molecule present in the other
arms' archives (SA 6.127, fresh loss 2.16734, 8 rings), obtaining an intermediate at
wave 3/update 278 with **SA 4.217, loss 2.20284, 6 rings**. That is 1.91 SA points for
0.0355 worse loss, without any discrete graph edits. It appeared at fresh call 42,
so the result survives the common 44-call prefix despite final costs 53 versus 44.
But Pareto and constant matched each produce only one SA≤5/loss≤2.3 candidate;
this is one improved trade-off point, not a broad yield victory. Independent
hard DFR moves in the opposite direction (5.93→6.42), so route feasibility is
not established by the SA gain.

The same branch's step 500 endpoint regresses to SA 6.758/9 rings; recovery then
continues this endpoint, overlooking the saved step 278 joint-quality candidate.
Its recovery best SA is 6.232. This supports a cheap test of replaying the best
eligible archived joint-quality intermediate versus continuing the terminal.
[Selection audit and lineage](../../../outputs/synthesis-analysis-20261116/selection-analysis.json),
[molecule sequence](../../../outputs/synthesis-analysis-20261116/pareto-lineage.png).

Attribution remains incomplete: the Pareto policy changes the nondominated
pool, diversity priority and binding ceiling together, and constant+Pareto is
missing. Proposed next tests, all with gradient-based child optimization,
no update noise and frozen weights:

1. Fix/verify bank occupancy, then three-seed, 500-update paired attraction
   controls: common SynFrag 0.05, attraction off versus unfiltered-bank,
   low-SA-bank and centroid attraction at a common 0.05 budget. Choose/locality-
   diagnose the softminimum offline; report anchor entropy and copying.
   Four arms×three founders×500×three seeds=18,000 updates. A later local-versus-
   broad comparison should not be confused with this bank-specificity test.
2. Complete constant/wave-focus×binding/synthesis-aware selection across three
   seeds (four waves, 500 updates,three children; 24k updates/seed). Hold diversity
   ordering, ceiling and shortlist rule fixed while changing SA eligibility;
   report both equal allowances and common actual-call prefixes.
3. First score the already-saved SA 4.363 endpoint and validate the promising
   joint-quality candidates with independent synthesis readouts. Pair continuation
   from a selected useful intermediate versus its worse endpoint before combining
   all mechanisms in another long run.

Primary comparison endpoints should be distinct successfully scored candidates
meeting several predeclared SA/contact thresholds, diversity and the full
SA/contact Pareto front, plus all-state validity. The illustrative SA≤5/loss≤2.3
cutoffs are study comparisons, not synthesis or affinity certification. The
[original SA method](https://pmc.ncbi.nlm.nih.gov/articles/PMC3225829/) combines
fragment prevalence and complexity penalties; it does not construct a route.
[Saved candidate scatter](../../../outputs/synthesis-analysis-20261116/chemistry-binding.png)
uses successful scores only and makes the sparse joint-quality region visible.
No new GPU optimization panel was submitted during this analysis.
