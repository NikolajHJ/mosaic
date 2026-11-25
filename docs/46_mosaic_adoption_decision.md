# Mosaic: adopt, extend, or reimplement?

Decision note, 8 October 2026. Static source review only: no code was run, no job
was submitted, and nothing here is an experimental result. Every statement about
`escalante-bio/mosaic` and `escalante-bio/protenij` comes from reading their
default branches on 8 October 2026 (mosaic HEAD dated 2026-10-06, protenij HEAD
dated 2026-09-28) and `bytedance/Protenix` as vendored in protenij. Cost figures
are order-of-magnitude estimates, labelled as such. This note is a
recommendation to the project owner, not an implemented change.

**Scope of the comparison.** The question was posed as a greenfield decision:
this repository's existing experimental code is deliberately excluded from the
cost accounting, because it is not expected to survive into the final codebase.
"What Mosaic saves" therefore means "versus writing the Stage 0-2 functionality
from scratch".

## Conclusion

**Do not extend Mosaic. Port four small algorithms from it, with attribution, and
build the rest independently.** Mosaic is MIT-licensed, so copying is permitted.

What Mosaic is, in one line: a loop that makes a design variable differentiable
(per-position probabilities), scores it with frozen models, steps downhill,
projects back onto the simplex, and takes an argmax at the end. That loop is what
protocol v12 Stage 1 also needs. It is also the cheap part of the build.

| Port from Mosaic (about 300 lines of new code) | Source | Why |
|---|---|---|
| Simplex projection | `optimizers.projection_simplex` (20 lines, itself from Blondel's public gist) | Protocol 4.3 A.1: project every atom and bond vector after each step |
| Accelerated projected-gradient loop | `optimizers.simplex_APGM` (about 110 lines) | Protocol 4.3 A.2. Needs a rewrite to a PyTree, because `X` and `E` have different category counts. Its best-iterate tracking is self-described as inexact ("we evaluated loss at v, not x") |
| Straight-through hardening ramp | `optimizers._colabdesign_transform`, `colabdesign_stage` (about 10 lines of actual math) | Protocol 5.5 asks to reuse "the existing Mosaic hardening logic". This is it |
| Weighted loss-term algebra | `common.LossTerm`, `LinearCombination` (75 lines) | Composable objective: `1.0 * a + 0.3 * b` |

Copy the sign conventions and ramp schedules exactly rather than re-deriving them.

## What Mosaic does not provide

Roughly 10% of a Stage 0-2 build, and none of the three hard parts.

| Component | Estimated new LOC | Mosaic contribution |
|---|---|---|
| Ligand featurization (atom slots, `ref_pos = 0`, atom-to-token map) | 300-600 | none |
| Relaxed-graph state (upper-triangle `E`, mirroring, fixed diagonal, simplex invariants) | 150-250 | none |
| Soft injection into native features (`X` into `ref_element`, `1 - P(NO_BOND)` into `token_bonds`) | 150-250 | none |
| Differentiable frozen trunk with manual recycling | 100-200 | about 45 (`get_trunk_state`) |
| Objective extraction, Stage 1 objective | 100-250 | about 40 |
| Projected-simplex optimizer over `(X, E)` | 150-250 | about 130 |
| Hardening ramp | 50-100 | about 40 |
| Loss algebra | 50-100 | 75 |
| Diagnostics (finite differences, gradient norms, fluidity, valence, fragmentation, soft-to-hard gap) | 300-600 | none |
| Experiment plumbing (run directories, checkpoints, RNG state, resume) | 300-600 | none: Mosaic has per-iteration `print` and an optional in-memory `trajectory_fn` only |
| Stage 2: valence projection, connectivity correction, sparsity prox | 400-800 | none: proteins have no valence constraint |
| **Total** | **about 2.2k-4.1k** | **about 330 lines (about 10%)** |

Evidence behind the "none" entries, all from source:

- `Protenix.target_only_features` is `assert False` for any non-protein chain.
  There is no SMILES, CCD, `is_ligand` or ligand featurization anywhere in
  `src/mosaic` outside the vendored AlphaFold2 tree.
- `set_binder_sequence` writes an `N x 20` array into `restype` and rank-one-updates
  the MSA `profile`. The ligand variables live in `ref_element` (128 columns),
  `ref_charge` and `token_bonds`. This is a new function, not an edit.
- `StructureModelOutput` requires `full_sequence (N,20)`, `backbone_coordinates
  (N,4,3)` and `atom37_coords/mask`. The Protenix forward derives backbone
  coordinates as `first_atom_idx + 0..3` (meaningless for one-atom ligand tokens)
  and unconditionally runs `sample_structures` and `confidence_metrics`. Widening
  the struct touches all fifteen model wrappers; bypassing it forfeits the loss
  library, which was the reuse argument.
- 37.7k LOC in total, of which 17.8k is vendored AlphaFold2 and 5.2k is examples.
  The transferable core is a few hundred lines.

## Cost of adopting it anyway

- **protenij becomes the model of record**, not `bytedance/Protenix` (protocol
  section 9's reference). It is a 2.4k-line JAX reimplementation plus a 337-line
  torch-to-JAX weight translator.
- **protenij's ligand path appears never to have been exercised.** Its tests are
  `test_atom_padding.py` and `test_token_padding.py`; `ChainInput` takes amino-acid
  sequences only; no test or script builds a ligand. Using it would require a
  numerical parity check against native torch Protenix on a ligand input before any
  gradient is trusted. This was established by reading the repository, not by
  running it.
- **Environment pins:** `python == 3.12.*`, `torch == 2.11.0`, about fifteen git
  dependencies on Escalante forks, `required-environments` restricted to Linux
  x86_64. Compatibility with the DTU HPC environment is unchecked.
- **A fork would not remain mergeable** once the output struct is changed, against
  an upstream that committed two days before this review.

## Is JAX itself worth it? (separate from Mosaic)

Short answer: it is a throughput question, a small one, and not a reason to adopt
Mosaic.

- This is input optimization through a frozen model, at batch size one. It is not
  training, so published training-throughput comparisons do not apply.
- Protenix-Tiny is `pairformer.n_blocks = 8`, `N_cycle = 4`: about 32 pairformer
  block executions per forward. Order-of-magnitude estimate (not measured): each
  block is dozens of small kernel launches, so launch and dispatch overhead is
  comparable to or larger than arithmetic. For the ligand-only phase (N = 25) it is
  almost entirely dispatch-bound, which is where a compiled step loop helps most.
- The fair comparator for JAX is `torch.compile`, not eager PyTorch. The residual
  JAX advantage is plausibly 1.2-1.5x, unmeasured. XLA compile time (minutes,
  re-paid on shape or loss changes) is a real cost during Stage 0/1 debugging.
- Two Stage 1 choices dominate framework choice by an order of magnitude:
  whether gradients pass through diffusion (they will not, see below) and how many
  recycles carry gradients (all four attached costs roughly 3x the backward pass
  and memory of stop-gradient on the first three).
- If the framework question needs settling, it is one half-day benchmark: eager
  torch vs `torch.compile` vs jitted JAX, forward+backward per step, at N = 25 and
  N around 135. Not yet done.

## Finding with larger consequences: ligand PAE has no gradient path

While checking whether protocol 4.1 (mean intra-ligand PAE as the first signal) is
reachable, the confidence head was read. In native Protenix
(`protenix/model/modules/confidence.py` lines 179-182, with `stop_gradient: True`
in the base config) `s_inputs`, `s_trunk` and `z_trunk` are `.detach()`ed on entry,
and `protenix/model/protenix.py` lines 584-586 detach them again at the call site
and `detach_()` the rollout coordinates. PAE depends on the relaxed ligand only via
the coordinates (diffusion, which the project will not differentiate through) and
the trunk (explicitly detached). By source reading, **the derivative of PAE with
respect to `X` and `E` is identically zero in native Protenix.**

Two consequences:

1. **A silent-wrong-gradient trap in protenij.** Its only `stop_gradient` is the
   recycling state at `protenij.py:2231`. The JAX confidence head receives
   `initial_embedding.s_inputs`, `trunk_embedding.s` and `trunk_embedding.z`
   undetached, so it would return a finite, smooth, nonzero PAE gradient through a
   path the native model severs. A finite-difference check against protenij alone
   would pass. Only a check against native torch exposes it.
2. **The protocol's stated PAE failure mode is not the real one.** Section 4.1.1
   anticipates invalid ligand frames at `ref_pos = 0`. In Protenix, frames
   (`has_frame`, `frame_atom_index`) appear only in the PAE training target; the
   inference head uses `z_trunk` plus a distance encoding of predicted coordinates.
   The AF3-documentation caveat does not transfer.

Not verified by execution: the zero-gradient claim (read from source, not measured)
and whether a differently configured checkpoint changes `stop_gradient`.

## Recommended protocol v12 edits (not applied)

1. Reframe Stage 0 from "extend Mosaic" to "build the relaxed-graph Protenix-Tiny
   branch (featurization, soft injection, trunk-only differentiable forward)".
2. Replace 4.1 (ligand PAE as first signal, with its 4.1.1 feasibility gate) with a
   one-line static finding that the confidence head is gradient-isolated; promote
   4.1.2 (BRD4-BD1 whole-target distogram contact) to the primary Stage 1 objective.
   This also removes the contradiction between 3.3 (avoid diffusion and the
   confidence head) and 4.1 (an objective that needs both).
3. In 5.5, replace "reuse the existing Mosaic hardening logic" with "port
   ColabDesign-style straight-through hardening, as in Mosaic's `colabdesign_stage`".
4. Note that the contact objective needs only the trunk and one linear distogram
   head, so diffusion and confidence-head weights need not be loaded or validated.
5. Add a native-torch finite-difference gate for whichever framework is chosen,
   because a self-consistent reimplementation cannot detect a detach it lacks.

## Open items

- Choose the framework (torch, torch.compile, or JAX) by the benchmark above.
- Note that `BinderTargetContact` in Mosaic averages the top-3 contacts per
  position; the protocol's `S_contact = mean_i max_j c_ij` is a different
  objective, and a top-k variant is a design choice to decide, not a port.
- Stage 3's frozen DeFoG teacher is stop-gradient by protocol 6.2, so it does not
  constrain the framework choice.
