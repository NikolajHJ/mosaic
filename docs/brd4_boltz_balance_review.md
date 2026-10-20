# Boltz mix50 learning-rate and chemistry-balance review

20 October 2026. Saved-only allocated analysis **29760613 DONE** supports
an earlier, gentler chemistry ramp and a persistent validity-stability
signal. LR 0.1 is a reasonable maximum to test with existing backtracking,
but the run does not establish that it is better than 0.03. Retain 10,000
updates for the next comparison. These are recommendations only: no new
scientific configuration is implemented or molecular run submitted.

## Authenticated analysis

Job 29760613 was submitted 09:34:00, ran 09:34:02–09:34:06 Europe/Paris on
n-62-11-61, and actually completed successfully in 4 seconds. It used one
CPU slot, 2 GiB, hpc, 10 minutes, zero GPUs. LSF Successfully completed was
authenticated. The script read source02 run 29759212 metadata and selected
saved snapshots; no inference, chemistry evaluation or optimization reran.

Immutable source:
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-balance-01/source`.
Archive SHA
`d01c08b88c9f7d0abdd2b80069db9c95722ad24a4ed47922fe26ab168c05669b`;
analyzer SHA
`3a1edf8959b0bd7cdfa0dd142c1a6c51a55b300c0577feff1e9a7850d22594dc`;
wrapper SHA
`95c80c0abab4d3382acf79d380e4c585ef25e5f13079d8ad8cb10a29009226fd`.
All five downloaded proof files passed local stdlib byte/schema checks at
`results/hpc/20261020-brd4-boltz-balance-01/verification.json`.
Analysis JSON SHA
`5e93dabedf857d359492e94d532564dbc4665774f72dd26a1612a6e7c9d2a750`.

## What the saved trajectory shows

The fixed-pose relaxed affinity improved until its best saved value at
state 8294, then worsened as chemistry guidance became stronger. The hard
readout at that best relaxed state was invalid. Selected saved states:

| State | Relaxed affinity, lower better | Chemistry coefficient | Raw chemistry | Weighted chemistry | Saved hard validity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1.131608 | 0.001 | 0 | 0 | Valid, `CCCCCC` |
| 5000 | 0.009444 | 0.001002 | 0.541128 | 0.000542 | Invalid |
| 8294 | -0.103593 | 1.249169 | 0.548083 | 0.684649 | Invalid |
| 9000 | -0.020505 | 5.755934 | 0.449405 | 2.586745 | Invalid |
| 9500 | 0.339036 | 16.982937 | 0.270738 | 4.597928 | Invalid |
| 9639 | 0.546533 | 22.942681 | 0 | 0 | Valid, `SCCCCCl` |
| 9999 | 0.630336 | 50 | 0 | 0 | Valid, `SCCCCCl` |
| 10000 | 0.622722 | 50 | 0.112377 | 5.618865 | Invalid, chlorine valence 2 |

The best connected/sanitized saved hard readout by its associated relaxed
score was state 9639, `SCCCCCl`, at 0.5465328693. That scalar still scores
the relaxed graph in the frozen context, not a fresh prediction of the
hardened ligand. Across all 10,001 states, 1,686 met the saved validity
criterion. States 9500..10000 inclusive contained 501 states, 218 valid and
280 validity switches over the last 500 updates. This is evidence of boundary chattering, rather
than stable feasibility at the end.

All 10,000 applied updates were recorded strict improving with zero fallback
or no-op. Those comparisons hold each state's repaired target/coefficient
fixed; refreshed totals need not decrease across the whole trajectory.
Accepted step sizes were:

| Effective rate | Applied updates |
| ---: | ---: |
| 0.03 | 2431 |
| 0.015 | 1439 |
| 0.0075 | 4883 |
| 0.00375 | 1175 |
| 0.001875 | 72 |

Most frequent acceptance was 0.0075, so raising the requested maximum to 0.1
does not demonstrate a larger useful step or faster progress. A matched
comparison is needed.

## Why coefficients alone do not solve the last-step failure

The declared route 0 `already_valid: skip_without_self_target` sets chemistry
loss and gradient to zero on a valid hard graph. Complete witnesses on
route 1 use full atom-plus-bond NLL; local valence-risk guidance is active
only for fallback routes 3/4. There is no always-on feasibility signal.
At valid state 9999 the chemistry term is zero even though its coefficient
is 50, so the next accepted affinity-driven step can cross the hard
chlorine-edge boundary. State 10000 then has chlorine valence 2 and a restored
repair term. This is protocol behavior consistent with the saved endpoint,
not a proven derivative bug.

Selected logit gradients show a shift in balance. The ratio
`norm(total_gradient - native_gradient) / norm(native_gradient)` was 0.0284
at 5000, 1.1800 at 8294 and 3.5324 at 10000, with final residual/native cosine
-0.8692. At valid 9999 the ratio was only 0.002499 because repair guidance was
zero. **This residual includes weighted chemistry plus composition; it is
not a pure chemistry gradient.** Scalar loss sizes alone do not establish
which direction controls the update, and these selected snapshots do not
represent an independent causal intervention.

The priority is an explicitly versioned margin/hysteresis policy or a
continuous feasibility signal that remains informative near the valid/
invalid boundary. It must keep the same shared molecule, define fixed
references within trials and pass allocated engineering/native gradient,
parity, finite-value and identity checks. Do not silently remove the complete-
witness NLL, change its atom/bond scope or repair the displayed output.
A new margin is a changed protocol and not a global validity guarantee.

## Proposed next comparison

Retain N=6, the same 50% initialization, seeds, frozen model/context, affinity
coefficient 1, element-JS 0.1, ten-trial best-primal acceptance and 10,000 updates.
The main schedule proposal is **linear chemistry weight 0.01 to 5 over
applied updates 1..8000, then hold 5 over 8001..10000**. This makes chemistry
active earlier and removes the steep exponential finish, while lowering the
final coefficient from the actual old 50. It is stronger early, not a smaller
penalty everywhere, and remains an untested candidate. The earlier 0.001-to-5
geometric proposal is only an optional conservative comparator: like the old
prefix, it may leave early guidance negligible. Specify/test exact state and
applied-update indexing before sealing any new protocol.

Keep the explicitly declared validity-stability term as the priority. A
schedule change alone still supplies zero repair gradient on the existing
valid route and therefore may not prevent boundary switching.

Test **maximum LR 0.1 with backtracking**, and preferably compare against
LR 0.03 under the same new chemistry/feasibility protocol. Changing both
schedule and rate in one arm cannot isolate the learning-rate effect.
Do not extend beyond 10,000 merely to chase the endpoint; the observed issue
is unstable validity and competing guidance, not evidence that an extra
budget resolves it. None of these settings is selected, implemented or
submitted by this review.

## Interpreting the Boltz scores

Saved binder probability rose from 0.0903825 to 0.1377155. It is a separate
readout and does not establish useful binding. Official Boltz documentation
assigns the binary output to distinguishing binders from decoys, while the
quantitative value compares affinity among active molecules. See
[Boltz prediction documentation](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md).
The current frozen-pose relaxed-graph optimization is exploratory and does
not show that these hard molecules are active, nor provide fresh-pose or
experimental affinity. Keep both metrics separate and preserve all invalid
states and historical failures.
