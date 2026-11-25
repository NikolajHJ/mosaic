# Does the embedding mapping improve useful gradient flow?

Question: does bypassing the decoder's final categorical boundary improve latent
optimization, or mainly allow Protenix to score inputs that no decoded molecule
represents? This is a diagnostic study, not another admission gate for the running
10k experiment.

## Evidence from the existing paired starts

The relaxed50k run and mapping10k run use the same frozen45 checkpoint, initial
latent/graph, occupied slots, native model/context/key, SGD, chemistry, noise and
temperature schedules through their shared prefix. Endpoint budgets and snapshot
cadences differ. Their latents diverge after their first updates, so later gradient
norms alone do not compare the two paths at an identical point.

Saved observations collected 3 November at11:57CET:

| Update | Direct relaxed trajectory: contact | Mapping trajectory: mapped contact | Mapping trajectory: decoded relaxed contact |
| ---: | ---: | ---: | ---: |
| 0 | 0.1513 | 0.1454 | 0.1513 |
| 100 | 0.2931 | 0.5914 | 0.3032 |
| 1,000 | 0.2856 | 0.8167 | 0.2239 |
| 2,000 | 0.2887 | 0.8640 | 0.2088 |
| 2,450 | 0.2886 | 0.8631 | 0.2122 |

Higher contact is better. Raw validity over0--2450 was892/2451 (36.39%) for
mapping versus113/2451 (4.61%) for direct relaxed. Mapping thus has a useful
validity signal in this single pair, while most of its self-score improvement
does not transfer to its decoded graph. Neither is a binding-affinity result.
At the shared initial latent, gradient norms were0.1006 mapped and0.06637 relaxed.

[Saved comparison and configuration audit](../../../outputs/hpc-results/protenix-embedding-comparison-20261103/saved-comparison-1793703447271869900.json).

## Why the intuition is plausible, and what freedom is added

The relaxed path contains the categorical Jacobian
`(diag(p) - p p^T) / temperature`. Near one-hot outputs it can strongly attenuate
logit changes. The learned path replaces the final probability/projection chain
with a learned Jacobian, so it can provide different, less saturated directions.
Large gradient norms alone are not a conditioning or optimization benefit.

The current relaxed path already uses fixed occupancy: both paths omit atom-count
gating and keep pair symmetry/zero diagonal. The hidden mapping still receives
internal decoder softmax feedback through charge/bond inputs and refinement
rounds. It bypasses the final categorical boundary, not every softmax in the VAE.

There is also extra nonchemical freedom. Native charge and bond-presence
projections lie on one-dimensional chemical manifolds; their learned heads emit
arbitrary128-dimensional vectors. Element projections must normally lie in the
convex hull of the element prototypes. A learned output can leave these sets.
Optimizing those directions may improve its score while leaving the decoded
graph unchanged or worsening it. The score gap above is evidence motivating this
diagnosis, not yet a measurement of which directions cause it.

Simply projecting the same probabilities algebraically preserves the objective
and gradient. The learned bridge changes the function. An exact charge control
must use `arcsinh(E[charge])`, matching native preprocessing, rather than a mixture
of the hard-charge prototypes `E[arcsinh(charge)]`.

## Matched latent probe

[Configuration](embedding-comparison-probe.json) declares mapped steps0,100,1000,
2000 plus three fresh prior seeds7001--7003. At each point it compares mapped,
direct relaxed, and exact-projected relaxed objectives on the same resident native
model/context/key. It records latent-gradient norms/cosines, probability saturation,
native-gradient components outside chemical prototype spans, and warmed wall time.
It then takes one radius0.1 binding-only step along each normalized gradient and
a random control; every candidate is evaluated through both paths and the hard
fixed-context graph. Invalid hard scores remain labeled invalid. There is no
step-size search or descent/quality gate.

Job **29877460** was submitted to one32GB V100, fourCPUslots/32GiB host memory,
one-hour maximum. At12:01CET it was PEND. Six focused local diagnostic/transport
tests passed; native observations are pending. The running mapping experiment
was not changed. Source snapshot SHA256
`02c81c5cadc7a759110ab911ad0111c23820e2a8d9a00108c8451c5382300d49`.
Outputs: `/work3/s234818/latentmol/runs/protenix-embedding45-comparison-20261103a`.

A useful advantage means better movement on a common decoded-graph objective
and valid molecules per unit compute. More freedom or a larger mapped self-score
alone does not establish that advantage. This small probe and one trajectory pair
cannot establish performance across molecules or independent seeds.

## Native probe completed, 3 November 2026 12:28 CET collection

Job29877460 completed all seven points successfully. The exact algebraic
projection control reproduced the probability path closely: maximum scalar
difference2.38e-7, maximum relative latent-gradient difference7.64e-6, and
gradient cosines above0.99999999997. This shows that projecting the same
probabilities alone does not change gradient flow in the tested implementation.

The learned mapping's gradient was larger at four of seven points, smaller at
three. Its cosine with the relaxed gradient ranged from-0.354 to0.763; two
directions opposed the decoded relaxed objective locally. At the one fixed0.1
radius, mapped steps improved the relaxed objective at five of seven points;
relaxed steps improved it at all seven. Mapped steps gave larger finite relaxed
improvements at three points; relaxed steps did so at four. Invalid hard graphs
remain invalid score observations, not candidate gains.

| Probe point | Mapped/relaxed gradient norm | Gradient cosine | Relaxed loss change: mapped step | Relaxed loss change: relaxed step |
| --- | ---: | ---: | ---: | ---: |
| Initial | 1.52 | 0.403 | -0.00656 | -0.01253 |
| Mapping100 | 7.30 | 0.717 | -0.05777 | -0.04775 |
| Mapping1000 | 2.02 | -0.354 | +0.00202 | -0.04268 |
| Mapping2000 | 0.19 | 0.110 | -0.01425 | -0.03995 |
| Fresh7001 | 1.12 | 0.763 | -0.03540 | -0.03497 |
| Fresh7002 | 0.31 | 0.343 | -0.01895 | -0.00205 |
| Fresh7003 | 0.27 | -0.203 | +0.00292 | -0.00940 |

Lower loss is better. The original relaxed outputs are indeed saturated at
these points:72--88% of element rows,82--94% of bond rows, and all charge rows
have maximum categorical probability at least0.99. Nevertheless, their latent
gradients were finite and sometimes considerably larger than the mapping's.

The summed latent-gradient contribution from native directions orthogonal to
the prototype spans was only about1--3.2% of total mapped gradient norm in these
probes. Thus the present measurements do **not** establish orthogonal embedding
escape as the dominant mechanism. Freedom/calibration along the spans and outside
convex bounds remain possible; affine-span diagnostics alone do not resolve them.
The near-neutral charge head's large relative residual must be read with its
small absolute magnitude rather than as a large chemical effect.

Median warm decoder+scorer+VJP wall time was0.591s mapped versus0.382s relaxed
on this same device/protocol. The current mapped implementation is about55%
slower, including its larger host/native tensor transport; this is not a claim
about an optimized in-process implementation.

The evidence supports changed gradients and a validity benefit in the existing
trajectory, but **no consistent useful-gradient or runtime advantage** for the
current mapping. The main10k run continues; no model settings were changed.
[Native results and current run snapshot](../../../outputs/hpc-results/protenix-embedding-comparison-20261103/news-1793705310088590700.json).

## Completed10k trajectory comparison

Both prefixes0--9999 are now available. Raw-valid source counts are5576/10000
(55.76%) for mapping and202/10000 (2.02%) for the direct relaxed run. At the10k
endpoint, direct relaxed contact is0.325734 and its argmax is valid. Mapping's
self-contact is0.885266, its decoded relaxed contact0.258247, and its argmax is
invalid. Thus the larger mapped endpoint score does not represent endpoint
superiority on the common decoded-graph objective.

There is a more encouraging candidate-selection result: the best *valid decoded
state with a saved native relaxed-graph score* in the mapping run is step180,
contact0.390198. The direct run's best valid score over all saved source states
is0.345866 at step27. Mapping actual-graph scores are observed every10 updates;
direct scores are available every update. On a matched every10 sample grid,
the direct best valid score is0.325734 at10000. These cadence differences are
explicit; none of these values is a fresh discrete hard-graph rescore.

Mapped step180 SMILES:
`NN1CC23C4C(O)CCC(=O)NC1CCNCC1=C(Cl)CC2(CC1)N43`.

The completed experiment therefore shows promising validity and best-observed
candidate signals in this single trajectory pair, while its final state and
the local gradient probe still do not establish general gradient superiority.
[Full saved10k comparison](../../../outputs/hpc-results/protenix-embedding-comparison-20261103/saved-final-comparison-1793710627282510800.json).

### Did longer optimization improve the result?

The saved common-score record for valid states was0.1513 at0,0.1550 at10,
0.3127 at80,0.3445 at140 and0.3902 at180. There were no further record breakers
among the every10 native observations. The best valid observed state after180
was0.3613 at400, below the early peak. From1000 onward the highest observed valid
score was0.2590 at9900. Average decoded relaxed contact fell from roughly0.30
in180--499 to0.21--0.22 in most later windows, while mapped self-contact kept
rising. Later validity was often higher but did not translate to improved best
decoded-graph score.

The best observed mapping state improved the initial contact surrogate by
0.2389 (+157.9% relative to0.1513), and exceeded the control's best by0.0443
(12.8%). These are changes in the native score on *soft probabilities attached
to valid decoded states*. The earlier shorthand "best molecule" must not imply
a discrete hard-graph rescore or an affinity/binding improvement. The evidence
supports early useful surrogate search followed by proxy drift, not sustained
candidate improvement through10k.
[Detailed progress audit](../../../outputs/hpc-results/protenix-embedding-comparison-20261103/progress-audit-1793713771450162400.json).

### Learning rate and actual step size

Saved-update audit3November: LR0.3 produced median latent displacements0.241
in0--99,0.229 in100--179 and0.164 in180--249. Median requested noise norms in
the first two windows were0.088 and0.083. The step leaving180 was0.2815, with
requested noise0.1031. Such updates can contribute to early oscillation or
overshooting a narrow peak; this run does not isolate their causal contribution.

Only80/10000updates hit the0.5 displacement bound. From2000 onward the median
step was0.076 and only3/8000hit the bound. This does not prove that uncapped
steps are sufficiently small, but cap saturation is not a persistent explanation.
The stronger evidence against an LR-only explanation is the matched derivative:
at mapping-step1000, cosine-0.354 means infinitesimal pure mapped descent points
toward *worse* decoded relaxed loss. A normalized0.1step improved mapped loss by
0.02434 but worsened decoded relaxed loss by0.00202. Reducing LR cannot reverse
that local direction, though it could alter the future trajectory and slow drift.

A minimal next intervention would compare one fixed lower LR0.03 against0.3
from the same step180 latent/global schedule with shared noise draws, rather than
a sweep. Noise removal should be a separate change if its effect is to be
identified. No new run or setting change was made for this audit.
