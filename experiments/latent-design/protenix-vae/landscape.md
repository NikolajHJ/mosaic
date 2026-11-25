# Is the epoch-45 latent landscape too sharp to optimize?

The [runner](../../../src/latentmol/design/latent_landscape.py) separates hard
graph changes, saturation of the categorical decoder, and the usefulness of
the score gradient. It uses a fixed direction through each center at latent
L2 radii 0.03, 0.1, 0.5 and 1.0. At each radius it compares the gradient at
the center and a neighbor, checks the sign of the central secant against
the predicted slope, and compares a descent step with both signs of the
random direction. Loss differences below the repeat/FP32 floor are unresolved.

- **Decoder panel** uses the frozen epoch-45 checkpoint, six prior draws and
  posterior means of six illustrative molecules. Membership in training is
  unknown. Frozen random projections of the logits and probabilities probe
  the local gradient field. The probability arms use temperatures 1, 0.7 and
  0.4. Atom, charge, present-bond and no-bond saturation are reported
  separately. Valid graphs use canonical SMILES for change detection, with
  raw graph signatures for invalid graphs. Argmax itself is temperature
  invariant here.
- **Native panel** uses the existing fixed-25-slot Protenix contact scorer,
  three saved latent states and two fresh prior draws. Same radii and
  temperatures. Repeats and comparisons between centers provide reference
  points for neighboring gradient cosines. The saved centers come from
  one trajectory. This
  uses one protein context and the relaxed contact objective. The chemistry
  teacher, hidden bridge, occupancy pilot and actual binding are separate
  questions. At most 210 native calls are made.

Sharp categorical probabilities can coexist with an informative latent
gradient. Nearly uncorrelated neighboring gradients at small radii, together
with weak descent performance against random steps, would be stronger
evidence of the concern described by [Balduzzi et al.](https://proceedings.mlr.press/v70/balduzzi17b.html).
Interpolation training is a separate intervention and will need matched
evaluation when its checkpoint is available.

Config and entry point

```sh
python -m latentmol.design.latent_landscape \
  --checkpoint EPOCH_45 --config experiments/latent-design/protenix-vae/landscape.json \
  --output NEW_OUTPUT
```

For native scoring, use the existing `scripts/hpc/protenix_latent_run.sh`
allocation wrapper and add `--mosaic-config` and `--mosaic-python`.

DTU job **29890699** was submitted at 13:46 CET on 4 November 2026 to
`gpuv100`, with four CPU slots and a one-hour allocation. The scheduler
reported PEND with GPUs unavailable for reservation. The frozen source
archive SHA-256 is
`1f71481f7f4844a16c28336c135f1a7ec4695f99e2dd575675d4f096468519e9`.
Native output is under
`/work3/s234818/latentmol/runs/protenix-latent-landscape-20261104a`.

The first local decoder attempt completed ten centers before one illustrative
molecule failed the schema's encoding roundtrip. The CPU panel replaces that
example with phenacetin and reruns in a new directory. This list of example
molecules is unused by the queued native panel.

## Decoder results

The corrected local CPU panel completed all 48 arms in 206 seconds with
the authenticated epoch-45 checkpoint. A separate analytic scorer smoke
exercised the fixed-25 adapter and backward path. Native results are pending.
[Summary](../../../outputs/latent-landscape/protenix-latent-landscape-20261104a/summary.json).

| Center family at T = 0.7 | Random hard change at radius 0.1 | Random hard change at radius 1.0 | Median gradient cosine at 0.1 | Median gradient cosine at 1.0 | Descent beats both random steps at 0.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Encoded examples | 0/6 | 0/6 | 0.9992 | 0.9738 | 5/6 |
| Prior draws | 3/6 | 6/6 | 0.9872 | 0.5930 | 6/6 |

The encoded median probability-gradient norm falls from 5.01e-5 at 0.7 to
5.16e-8 at 0.4. Only two of six encoded descent steps resolve an improvement
at radius 0.1 and T = 0.4. The same encoded centers have strong logit gradients
under the matched projections. All nine resolved probability secants at
radius 0.1 and T = 0.7 agree with the AD direction; three encoded secants
are unresolved. These projections show strong local directional coherence
with saturation near encoded examples. They do not measure the contact or
adaptive repair loss.

## Native Protenix results

DTU job **29890699 DONE** on 4 November 2026. The five-center native panel
completed in 98 seconds. Each row below pools the same three saved and two
fresh-prior centers at radius 0.1. Step lengths are fixed by the probe.

| Temperature | Median neighboring gradient cosine | Median gradient norm | Descent improves relaxed loss | Changes raw hard decode | Changes fixed-25 hard graph |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 0.969 | 0.671 | 4/5 | 2/5 | 4/5 |
| 0.7 | 0.985 | 0.422 | 5/5 | 4/5 | 5/5 |
| 1.0 | 0.995 | 0.487 | 5/5 | 4/5 | 5/5 |

At radius 0.03 the median cosine is about 0.997 for each temperature.
At radius 0.5 it falls to 0.199 at 0.4, versus 0.658 at 0.7. At radius
1.0 no descent step beat both random controls. Cooling preserved local
directional signal in this panel, with fewer raw hard-graph changes and
less reliable directions over larger steps. Only one of five radius-0.1
fixed-slot outputs was chemically valid at each temperature. These are
relaxed contact scores under one fixed context.
[Native summary](../../../outputs/hpc-results/protenix-latent-landscape-20261104a/completed-summary.json)
· [Pooled readouts](../../../outputs/hpc-results/protenix-latent-landscape-20261104a/native-pooled-readouts.json).

## Seven-round epoch-48 checkpoint at update 597604

On 22 November 2026 we repeated the frozen decoder probability-projection probe on the first epoch-48 leg. With seed 2026112289 we drew 12 molecules without replacement from the saved 512-row PubChem validation panel, encoded their posterior means, and compared them with 12 Gaussian prior vectors. Each center had one random unit direction, a frozen mean-centered Gaussian contrast over its categorical rows, and normalized descent steps on that contrast. We used latent L2 distances (256 dimensions) and temperatures 1.0 and 0.4. The raw argmax graph uses the production occupancy mask and canonical chemical identity. Different valid molecules below are relative to their own starting molecule. Novelty against the training corpus and protein scores were outside this replay.

| Center, temperature 1.0 | L2 radius | Random direction changed hard graph | Descent changed hard graph | Descent reached a different valid molecule | Median neighboring gradient cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Encoded | 0.1 | 1/12 | 0/12 | 0/12 | 0.9992 |
| Encoded | 1.0 | 1/12 | 1/12 | 1/12 | 0.9257 |
| Encoded | 2.0 | 1/12 | 5/12 | 4/12 | 0.7580 |
| Prior | 0.1 | 2/12 | 10/12 | 9/12 | 0.9978 |
| Prior | 1.0 | 11/12 | 12/12 | 10/12 | 0.8960 |
| Prior | 2.0 | 12/12 | 12/12 | 9/12 | 0.6808 |

All 12 held-out starts reconstructed exactly. At temperature 1.0 the median fraction of occupied atom rows and present-bond rows with winning probability at least 0.99 was 100% for both heads at encoded starts. It was 77.8% and 85.2% respectively at prior starts. At radius 0.1, normalized descent improved the continuous projection more than either sign of the random direction for all 12 encoded starts, even as all 12 hard molecules stayed fixed. Cooling to 0.4 reduced the encoded median projection-gradient norm from 0.000303 to 0.000000000614, and only one of those 12 descent effects cleared the same comparison and FP32 resolution rule. Prior median gradient norms were 2.24 and 2.67 respectively. These norms depend on the scalar projection and temperature. Frozen weights matched before and after. [Detailed observations](../../../outputs/runs/vae-e48-597604-latent-landscape-20261122a/results.json) · [Replay script](../../../outputs/runs/vae-e48-597604-latent-landscape-20261122a/probe.py)

### Effect of the three appended residual rounds at update 600116

On 23 November 2026, we measured how these rounds change the gradient back to the 256-dimensional latent. A CPU probe used 12 standard Gaussian starts, temperature 1, evaluation mode, and the same checkpoint in both arms. The comparison set all six extra-round gates to zero in memory. Each readout was a fixed category-centered Gaussian contrast normalized to unit Frobenius norm, shared across both arms and every start. The atom contrast covered all 50 slots. The bond contrast covered the 1,225 undirected pairs after soft endpoint-occupancy gating.

All 48 latent gradients were finite and nonzero. The median paired ratio of trained to bypassed gradient norms was 0.994 for atoms and 0.874 for bonds. The corresponding median direction cosines were 0.953 and 0.573. Thus the three rounds preserved substantial sensitivity for these readouts and altered the bond direction more strongly. Binding gradients and the effects on discrete edits require separate measurements. With `src` on `PYTHONPATH`, run `python outputs/runs/e48-s02-early-stop-600116/gate-gradient-probe.py`. [Paired results](../../../outputs/runs/e48-s02-early-stop-600116/gate-gradient-probe.json) · [Probe](../../../outputs/runs/e48-s02-early-stop-600116/gate-gradient-probe.py)

### Locating the output softmax bottleneck at update 600116

On 23 November 2026 we drew 24 molecules from the 512-row held-out PubChem packet, one posterior sample per molecule and 24 unfiltered standard Gaussian vectors. The three panels used a frozen checkpoint. The output temperatures were 0.4, 0.7, 1.0, 1.5 and 2.0. Changing this temperature affects the final readout only. Atom, charge and conditional bond rows were selected once at each center using the hard graph. Occupied and padding slots, present and absent bonds have separate summaries. Each undirected bond enters once. [Reproducible diagnostic](softmax_bottleneck.py)

For each categorical row we calculated the squared Frobenius norm of the softmax Jacobian with respect to its unscaled logits. This measures sensitivity across all output directions for equally scaled perturbations to individual logit coordinates. Dividing by the sum across rows gives each row's share. The reciprocal sum of squared shares gives the effective number of sensitive rows. This CPU calculation used double precision for the softmax and its Jacobian, with the frozen decoder evaluated in FP32. Reported values are medians across molecules; ratios across temperatures are paired within molecule. [Per-point concentration results](../../../outputs/runs/vae-e48-600116-softmax-20261123a/concentration.json)

| Center | Output rows | Median row count | Combined Jacobian norm, T=1 | Effective rows, T=1 | Top row's share, T=1 | Norm ratio, T=1.5 vs 1 |
|---|---|---:|---:|---:|---:|---:|
| Encoded mean | Occupied atoms | 20.5 | 0.000100 | 2.00 | 66.5% | 25.7 |
| Encoded mean | Present bonds | 21.5 | 0.0000223 | 1.32 | 86.8% | 33.2 |
| Posterior sample | Occupied atoms | 20.5 | 0.000573 | 1.34 | 85.7% | 12.1 |
| Posterior sample | Present bonds | 21.5 | 0.000185 | 1.35 | 85.7% | 17.2 |
| Gaussian prior | Occupied atoms | 23 | 0.620 | 2.78 | 45.8% | 0.807 |
| Gaussian prior | Present bonds | 27 | 0.637 | 2.54 | 53.2% | 0.798 |

Among currently absent pairs of occupied atoms, the top pair accounted for a median 89.7% of softmax sensitivity at an encoded mean, over a median 178 eligible pairs. At T=1.5, present-bond sensitivity at encoded means spread to an effective 1.78 rows, and at T=2.0 to 2.25. The warm readout therefore recovers some flexibility while most pair rows remain insensitive relative to the most responsive ones. Relative shares by themselves omit the large differences in absolute sensitivity between families.

The local RTX 4060 sweep measured a unit-norm, category-centered random contrast for each head and center, held fixed across temperatures. We differentiated it first on the unscaled logits and then on the head probabilities with respect to both the unscaled logits and the latent vector. The conditional bond readout isolates the bond softmax before occupancy gating. All calculations used the same frozen decoder weights, and evaluation verified their identity afterward. These contrasts describe local sensitivity on this checkpoint. The scorer's choice of direction, validity of an edit and fresh protein score need separate measurements. [All temperatures, heads and per-point gradients](../../../outputs/runs/vae-e48-600116-softmax-20261123a/bottleneck.json)

| Center | Head | Median latent gradient on raw logits | Median latent gradient on probabilities at T=1 | Median paired latent gradient norm ratio, T=1.5 vs 1 | Median direction cosine, T=1.5 vs 1 |
|---|---|---:|---:|---:|---:|
| Encoded mean | Occupied atom | 2.38 | 0.0000172 | 27.9 | 0.967 |
| Encoded mean | Present bond | 4.94 | 0.00000631 | 38.5 | 0.981 |
| Posterior sample | Occupied atom | 3.02 | 0.000310 | 11.2 | 0.992 |
| Posterior sample | Present bond | 6.72 | 0.0000721 | 18.1 | 0.984 |
| Gaussian prior | Occupied atom | 5.68 | 0.480 | 0.853 | 0.977 |
| Gaussian prior | Present bond | 10.5 | 0.681 | 0.898 | 0.984 |

At encoded means, a unit probability-space readout produced a median logit-space gradient of only 0.00000554 for occupied atoms and 0.00000167 for present bonds at T=1. This localizes a major part of the attenuation to the final softmax. Cooling from 1 to 0.4 reduced the corresponding median paired latent norm ratios to 2.4 × 10⁻⁷ and 9.1 × 10⁻⁸. Priors had substantially stronger gradients for atom and bond edits, although the prior charge gradient was still only 0.0000144 at T=1. Temperature should be evaluated separately by start and output head.

To reproduce, set `PYTHONPATH` to the repository root and `src`. Run `python experiments/latent-design/protenix-vae/softmax_bottleneck.py OUT/bottleneck.json --device cuda`, then `python experiments/latent-design/protenix-vae/softmax_bottleneck.py OUT/concentration.json --device cpu --concentration`, with fresh file paths. The default 24 centers per family and seed 2026112289 reproduce this panel. Row masks and contrasts are fixed across temperatures within a center, and current hard argmax identities are independent of the final temperature. The CPU row calculation used PyTorch 2.6.0 and the GPU backward run used 2.5.1+cu121.
