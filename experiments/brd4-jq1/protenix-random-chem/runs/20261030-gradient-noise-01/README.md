# Gradient-relative noise with delayed chemistry,30 October2026

Question: does fresh per-step logit noise improve the random Protenix + chemistry trajectory? User explicitly authorized choosing the noise rule/scale and submitting without more input.

Actual GPU job **29857358**, `gpuv100`, one32GBV100,8CPU/4GiB per slot,18h allocation; actual PEND at18:57 UTC. Fresh50,000 updates, LR0.3, same original Dirichlet(1) seed17 step-zero snapshot, model key17, original frozen Protenix/context/JQ1 pocket, N25/T1. Chemistry stays0.001 through5000 then rises geometrically to100 at50000; composition0/no confidence margin. Same accelerated repair math and one finite candidate; no backtracking, clipping or gradient normalization. Existing runs/results are unchanged.

For each atom/bond logit block separately:

```text
base = logits - 0.3 * (g_binding + weighted_g_chemistry)
noise = 0.4 * norm(0.3*g_binding) * normalized_row_centered_Gaussian_direction
next_logits = base + noise
```

Only the random direction is centered/normalized, removing the softmax constant-shift direction. The binding-only logit pullback sets the noise scale; the chemistry gradient/weight never directly sets it. PCG64 seed230017 is separate from model key17. Targets/coefficient stay fixed and detached throughout that step's candidate. No ratio annealing or positive floor: a zero native-gradient block receives zero noise. Every scalar and noise/RNG measurement is retained; matrices every20 plus initial/end/latest/best. FP32 rounding can reduce the realized perturbation; both requested and realized norms are logged.

0.4 is a declared exploratory hypothesis, not an established optimum. Gradient noise has precedent in [neural-network optimization](https://arxiv.org/abs/1511.06807); that paper does not certify this molecule-optimization rule or coefficient. Constant ratio can slow late settling; gradient-relative scaling can also lose exploration when a binding gradient becomes flat. One noisy trajectory does not establish a multi-seed advantage or chemical/binding validity.

Validation: five portable metadata/AST controls, actual original scientific proposal AST and actual packaged startup passed. Allocated CPU29857352 passed all13 focused cases/zero fail/error/skip, with same-job LSF success, covering norm/dimension calibration, row gauge, seeds, zero-noise control, chemistry-weight independence, finite inputs and live retention/teacher helpers. These are small proposal controls, not native/FD certification. GPU smoke/optimization awaits allocation. Full own-output storage passed2,362,232,013required versus55,031,668,736available bytes (2GiB output cap/10,000 files plus policy reserve); no global double counting or cleanup.

The initial source01 preflight rejected the older saved `type(evaluation.logits)` constructor before any submission. Source02 supported both exact constructors; its new unstarted job29857341 was intentionally retired at zero execution when a source-inspection test-fixture issue was found. Source03 fixes that fixture; noise math is identical. All failed/prepared sources, launch syntax-error evidence and retirement receipts are preserved. Nothing was edited in a sealed source or existing saved run.

Use the existing exploratory launcher/stage with this capsule's plan, not a new controller. Canonical noise implementation is ../../scripts/gradient_scaled_noise.py and policy ../../configs/gradient-noise-ratio04.json. Actual command/resources/source/CPU/held-release proof are in `campaign.json` and `submission-verification.json`; raw artifacts are under results/hpc/20261030-protenix-chemistry-noise-01. Do not replay any stage/launch intent or resume the retired predecessor. Current outputs are ROOT/retry01/protenix_chemistry/payload/study. The existing monitor now includes this sixth arm; report only actual completed saved results with the canonical report command/cache.
