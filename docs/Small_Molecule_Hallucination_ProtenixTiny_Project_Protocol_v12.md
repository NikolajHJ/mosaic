# Small molecule hallucination with Protenix-Tiny

This bachelor project extends Mosaic's sequence hallucination workflow to optimize a relaxed 2D ligand graph using gradients from frozen Protenix-Tiny. The long-term goal is de novo, target-specific small-molecule design through an AF3-class model. The immediate goal is one reproducible optimizer whose gradients, update dynamics, and soft-to-hard behavior are understood before adding chemistry.

This brief describes the planned implementation from protocol v12. Check the repository and experiment records to establish which pieces already exist. Use the stage gates below to scope coding work.

## Roadmap

| Stage | Work | Exit criterion |
| --- | --- | --- |
| 0 | Add the Mosaic graph state and differentiable Protenix-Tiny wrapper | Reproducible forward and backward passes for ligand-only and protein + ligand modes |
| 1 | Choose one native objective and stabilize unconstrained optimization | One canonical trajectory with reliable gradients, reproducible improvement, and measurable hard-readout failures |
| 2 | Add explicit chemistry operators to the frozen Stage 1 baseline | Better raw molecular feasibility with continued atom and topology movement |
| 3 | Add a frozen pretrained 2D molecular prior | Matched evidence on validity, soft-to-hard gap, and preservation of objective progress |

## Stage 0 — Graph state and model interface

Reuse Mosaic's initialization, forward pass, scalar objective, backward pass, optimizer step, logging, and checkpointing where possible. The molecular branch replaces the sequence variables and their featurization.

| State | V0 contract |
| --- | --- |
| Atom probabilities `X[i, a]` | One simplex per atom slot over `C, N, O, S, F, Cl`. All slots are present. Use fixed `N` within a trajectory, initially `N = 25`. |
| Bond probabilities `E[i, j, b]` | One trainable simplex per unordered pair `i < j`, with categories `NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC`. Mirror the upper triangle and fix each diagonal vector to one-hot `NO_BOND`. |
| Model bond input `B[i, j]` | Feed bond existence `1 - E[i, j, NO_BOND]` through the differentiable model path. Retain full `E` for chemistry operators and the prior. |
| Formal charge | Fixed at zero. |
| Ligand `ref_pos` | All-zero placeholder with ligand reference-geometry-derived contributions disabled. The zeros represent absent geometry. Protein features stay native. |

Load Protenix-Tiny once and keep it resident in evaluation mode with model weights frozen. Preserve autograd from the objective to `X` and `E`. Start with the native input, trunk, pair representation, and distogram/contact path. Enable coordinate diffusion or the confidence head only when the chosen objective requires it. Use the cheapest stable recycle setting if supported. The loop operates on continuous probabilities. Use discrete graph conversion and RDKit evaluation on readouts outside the differentiable path.

Validate the wrapper with one ordinary hard ligand and its relaxed counterpart. Check finite forward and backward results, nonzero gradients to atom variables, and an objective response to a bond-probability perturbation through `B`. Compare representative autograd derivatives with finite differences. Once both forward modes are reproducible, proceed to Stage 1.

For each experiment, save raw `X/E` probabilities, optimizer and RNG state, objective values, separate atom and bond gradient norms, and periodic hard readouts.

## Stage 1 — Unconstrained optimizer

### Choose one native objective

1. **Try ligand-only PAE once.** On a hard ligand with its native reference conformer, verify finite `token_pair_pae`. Run the same ligand through the differentiable geometry-off branch. Require finite ligand-pair PAE, valid ligand frames, and nonzero, numerically stable gradients to both `X` and `E`. Minimize mean `PAE[i, j]` over valid off-diagonal ligand token pairs, including both directions because PAE is asymmetric. Ligand PAE depends on reference-derived frames, so the geometry-off branch may fail this gate. Ligand pLDDT measures ligand-to-polymer local accuracy and is unsuitable as the ligand-only objective.
2. **If any check fails, use BRD4-BD1 whole-target contacts.** Use human BRD4-BD1 (PDB `3MXF`, with JQ1 removed) and keep all target residue tokens eligible. From the native Protenix distogram, use `c[i, j] = P(distance < 8 Å)`. Maximize `mean_i(max_j(c[i, j]))` over ligand atoms `i` and target tokens `j`. This debugging score rewards each ligand atom's strongest predicted contact. Use the whole domain without pocket selection. Skip ligand-only proxy development, confidence-head changes, and affinity heads.

Freeze the selected objective for optimizer development and subsequent matched comparisons.

### Initialize one canonical trajectory

- Start `X` from a broad, carbon-heavy prior. Add small independent probability-space perturbations and project each atom vector onto the simplex.
- Initialize bond-existence mass near `p0 = d0 / (N - 1)` with `d0 ≈ 2.0–2.5` (roughly `0.08–0.10` for `N = 25`). Perturb one bond vector per upper-triangle pair, project it onto the simplex, and mirror it. Keep the diagonal fixed.
- Keep zero charge and the geometry-off ligand path. If the canonical graph cannot move despite healthy gradients, use one soft-chain or soft-tree control to diagnose topology formation.

### Update and diagnose

Use Mosaic-style projected gradient updates directly on probabilities. After each step, project each atom and upper-triangle bond vector onto its own simplex and reconstruct the symmetric tensor. Allow separate step sizes `eta_X` and `eta_E` and verify both variable classes move. Begin with a single optimizer and initialization. Reserve entropy or other fluidity controls for persistent early saturation after reducing step size or projection aggressiveness.

Track objective and gradient stability, atom/bond saturation, whether early strong edges can weaken and weak edges can take over, expected and argmax valence violations, fragmentation, total bond mass, and expected degree. Periodically evaluate raw `argmax(X)` and `argmax(E)` graphs and measure the soft-to-hard objective gap. Keep valence, connectivity, sparsity, hardening, repair, and molecular priors out of this baseline.

Run a short trajectory before extending it. Advance when both gradient paths are reliable, the extended run avoids NaN/Inf, exploding gradients and immediate saturation, the objective improves reproducibly from the canonical start, and the hard-readout gap and feasibility failures are measurable.

## Stage 2 — Explicit molecular feasibility

Keep the Stage 1 objective, optimizer, and graph invariants. Compare against the frozen baseline.

1. **A — Valence projection after each update.** Run the AF3 gradient step and simplex projection, then project `E` to the closest feasible bond-probability state. With bond-order vector `o = [0, 1, 2, 3, 1.5]` in category order, compute `v[i] = sum_{j != i, b} o[b] * E[i, j, b]`. Require `v[i] <= sum_a X[i, a] * V[a]`, starting with neutral ceilings `C=4, N=3, O=2, S=2, F=1, Cl=1`. Preserve edge simplexes and symmetry. Remove excess valence without forcing saturation. Monitor topology movement after projection.
2. **B — Connectivity only if fragmentation persists.** Find components in the raw argmax graph. Use soft bond probabilities to choose the minimum number of intercomponent links needed to connect them. Raise the selected edges toward bonded states and reapply simplex and valence projection. Leave the target topology unspecified.
3. **C — Sparsity only if diffuse edge mass persists.** If many weak edges remain despite acceptable valence, shrink low-confidence bonded mass toward `NO_BOND` and reapply simplex and valence projection. Use one shrinkage mechanism.
4. **D — Separate hardening experiment.** Compare a relaxed continuation and a Mosaic-style hardening continuation from the same checkpoint. Keep probabilities as the optimized state and progressively discretize the forward representation. Measure the immediate score drop, recovery with further optimization, raw validity, and topology locking. Keep raw argmax as the evaluation readout for all runs.

Stop adding explicit rules once hard graphs are substantially more molecule-like and atom identities and topology can still change.

## Stage 3 — Frozen molecular prior

Choose a pretrained 2D atom/bond denoiser or flow model after checking vocabulary compatibility, fixed-size or padding support, availability of a drug-like checkpoint, inference cost, and access to per-atom and per-bond predictions. DeFoG and DiGress are candidates. The exact model and checkpoint remain open. Keep the models frozen and consider training only if compatibility proves a concrete blocker.

At each iteration, query the prior once on the current `(X, E)`. For a denoiser, use one prediction at a chosen noise level. Treat the cleaner distributions `(X_hat, E_hat)` as stop-gradient targets and add a discrepancy penalty with one moderate initial weight. Recompute the targets after the graph changes. Follow the combined objective's gradient step with simplex and valence projection, retaining any conditional Stage 2 operators.

Compare against Stage 2 under matched settings. Look for higher raw validity, fewer severe valence or fragmentation failures, a smaller soft-to-hard gap, and fewer fractional states that exploit the objective. Check that AF3 objective progress and graph movement persist. If the prior underperforms, check model compatibility and weight before changing the denoising schedule.

## Deferred work and final output

After the fixed-size 2D optimizer is stable, extend toward target-specific design, native 3D geometry, richer property objectives, charge, variable size, expanded vocabularies, property-conditioned priors, and independent evaluation with larger AF3-class models. To test native geometry, first harden/project the graph, generate a conformer, enable ligand geometry, and measure the 2D-to-3D score gap. For property objectives, start with quantities defined on relaxed graphs and prefer target ranges where appropriate.

The final report should compare the canonical unconstrained trajectory, matched feasibility runs, separate hardening test, and matched prior run. Identify which mechanisms improve raw hard-molecule feasibility while preserving gradient-based exploration.

## Source and implementation references

Source protocol: `Small_Molecule_Hallucination_ProtenixTiny_Project_Protocol_v12.docx`.

- [Protenix implementation](https://github.com/bytedance/Protenix)
- [AF3 confidence and output documentation](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md)
- [AF3 ligand input and reference conformer documentation](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)
- [RCSB PDB 3MXF](https://www.rcsb.org/structure/3MXF)
- [DeFoG implementation](https://github.com/manuelmlmadeira/DeFoG)
- [DiGress paper](https://arxiv.org/abs/2209.14734)
- [DBMol paper](https://arxiv.org/abs/2607.19237)
