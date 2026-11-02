# Structural repair with stronger guidance and annealed noise

Latest completion check, 2 November at17:00UTC: sparse structural10 plus noise
(29873081) completed all30,000 updates. Same-job LSF success, all30,001 committed
states/30,000 updates and unchanged frozen identities passed the
[metadata audit](completion-20261102T170007Z.json). It recorded zero valid connected
argmax states and final relaxed contact loss-0.23515722155570984. The other three
arms remain PEND. Canonical report refresh request20261102T170052Z-9d264ce8 is
in progress; publication is not yet established. The initial table below is dated history.

Question: can structural repair plus constant chemistry weight10 produce a valid,
optimized molecule from either a sparse random start or a diffuse carbon chain?
Does binding-gradient-relative noise help exploration before a noiseless settling phase?

All four NEW runs request30,000 updates. LR0.3,25 occupied neutral atoms,
model key17,T1, original frozen Protenix/JQ1 context, composition0 and margin0.
The existing structural builder, repair loss, beam/progress fallbacks and one
finite candidate are unchanged. Targets remain detached and fixed within each
step's trial. Saved studies and running/queued predecessors are untouched.

| Start | Treatment | GPU queue | Actual job | State at10:15UTC |
|---|---|---|---|---|
| Sparse random | Structural + constant10 | A100 |29873078|PEND|
| Sparse random | Same + annealed noise | V100 |29873081|PEND|
| Chain/uniform + small initial noise | Structural + constant10 | A40 |29873082|PEND|
| Chain/uniform + small initial noise | Same + annealed noise | L40S |29873084|PEND|

Sparse starts retain the seed17 Dirichlet atoms and connected Pruefer-tree
plus2% extra-edge recipe. The second initializer mixes50% carbon-chain one-hot
probabilities with50% uniform probabilities, then adds independent row-centered
Gaussian logit perturbations with standard deviation0.05 using PCG64 seed17.
No filtering, redraw, clipping or validity-dependent initialization selection.
The chain argmax starts valid; its probability distribution can still sample
invalid alternatives. Sampling and argmax validity are therefore both measured.

The noisy treatment uses the existing noise-direction implementation. Each
atom/bond block has independent row-centered directions, norm
`ratio(update) * LR * norm(Protenix-only logit gradient)`, where
`ratio(u)=0.4*max(0,1-u/20000)`. Chemistry gradients never set the noise scale.
The final10,000 updates are exactly noiseless, with no unused RNG draws.
Separate PCG64 seed230017 and the complete stream state are saved in every
committed state; a resumed run restores that stream and GLOBAL schedule.
Zero binding-gradient blocks receive zero noise. Model key17 is unchanged.

The four compatible families spread queue delays. At submission, A100 and A40
had one remaining account GPU slot each; two existing A10 jobs were already
queued. Wall requests are10h/A100,11h05/V100,8h45/A40 and10h/L40S, based on
prior compatible measurements with setup/cleanup allowance. Requests use
`num=1:mode=exclusive_process:j_exclusive=yes:mps=no`,8CPU/4GiB per slot.
Queue position is not a reliable start-time guarantee. Common seeded draws are
shared between treatments; cross-device FP32 rounding is not claimed bit-identical.
Cross-device runtime differences must not be attributed to the treatment.

Allocated CPU29873064 actually DONE:39/39 cases passed (24 existing screen,
15 new), zero failures/errors/skips, plus all four actual metadata CLI startups.
Controls cover the mixture, binding-only scaling, zero-gradient/noise-off cases,
fixed within-step teachers and an uninterrupted/resumed loop through annealing.
No native-model or derivative certification is inferred from these small tests.
Source01 CPU29873061 remains EXIT:38passed and one overly tight FP32
log/softmax roundtrip assertion. Source02 changes only that test's explicit
8-epsilon FP32 bound and owner paths; producer/loss/noise math is byte-identical.

[Actual configurations](configurations.json), [campaign/jobs](campaign.json),
[byte/XML/submission audit](verification.json). Source02 is immutable under
`/work3/s234818/mosaic/runs/20261102-structural-strong-noise-01/source02/source`.
All four accepted receipts were persisted before exact owned held-job releases.
Live quota and bounded remaining-output checks passed; no files were deleted.

Assess every-state valid connected argmax,128 unfiltered categorical samples
every1000 updates and at endpoints, valid identity/diversity, atom/bond changes,
contact-loss trajectories, gradient balance and preparation/update runtime.
All losses are retained; matrices every20 plus endpoints/latest/best. Contact
on the saved relaxed graph is a proxy, not measured binding, and samples are
not rescored. Compare the sparse constant10 arm to the existing structural
constant1 seed17 trajectory rather than launching another baseline.

The canonical report selection now has four conditional completion candidates.
The existing report/cache stays unchanged until actual completed results exist.
No renderer fork or report-driven optimizer rerun.

Reproduce preparation (fresh tag/version only; never replay current submission):

```text
python experiments/brd4-jq1/protenix-random-chem/scripts/prepare_chemistry_screen.py --repo . --tag NEW-TAG --version source01 --comparison structural-strong-noise
```

Stage once through the documented connector, run the capsule's focused
`chemistry_screen_ops.submit(plan,'controls',wall_minutes=10)`, then submit
its four `run` arms using their explicit budgets after actual controls pass.
Use current queue observations for future placements. Do not change live plans.
