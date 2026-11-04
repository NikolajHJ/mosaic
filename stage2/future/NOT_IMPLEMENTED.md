# Deliberately not implemented in Stage 2

The five improvements the plans suggested for Stage 2 itself are implemented and
toggleable; see the README. This file is the other list: work that is defined
well enough to pick up, and deliberately left out, with the reason.

Nothing here is a missing feature. Each entry is either out of Stage 2's scope
by the protocol, or gated behind evidence that does not exist yet.

## Gated behind evidence we do not have

**Sparsity as a default.** `--mode sparse` exists, but protocol 5.4 says not to
add a sparsity loss automatically, and nothing so far shows bond probability
staying spread across many weak edges once valence is satisfied. Promote it to
the default arm only if `diagnostics.py` flags `sparsity` on a real run.

**A second sparsification mechanism.** Explicitly ruled out by 5.4: one
interpretable mechanism only, no stacked edge-count, degree, L1 and bond-mass
penalties. If weak-edge shrinkage proves insufficient, replace it rather than
adding to it.

**Bridge protection inside the valence projection.** Today a bridge can be
removed by the valence projection that follows it, and `--bridge-retry` answers
that with one stronger attempt. The stronger version — giving bridge edges
priority on valence capacity inside the dual — means a constrained projection
with per-edge weights, which is a real change to the solver. Worth it only if
`bridge_survival` keeps flagging.

**Entropy control as a loss.** Protocol 4.3 C.10 asks for update size and
projection aggressiveness to be reduced first, which is what `--valence-damping`
and `--fluidity-threshold/--fluidity-strength` do in projection space. A proper
entropy term in the objective is the next escalation and should not be reached
for until those two are shown to be insufficient.

## Out of Stage 2 by the protocol

**Formal charge.** Fixed at zero in V0. Protocol 4.3 B.7: charge adds another
easy continuous exploitation direction, so it waits until atom identity and
topology optimization are stable. The valence table in `operators.py` is the
neutral one; charge would make the ceiling itself a variable.

**Variable molecule size and atom existence.** `N` is fixed within a
trajectory (4.3 B.9). Comparing a small set of sizes — 15, 20, 25, 30 — is a
Stage 1 question, not a Stage 2 operator.

**A larger element vocabulary.** `{C, N, O, S, F, Cl}` until there is a clear
chemical reason to extend it (4.3 B.8). Extending it means extending
`NEUTRAL_VALENCE` in the same order, and nothing else in Stage 2.

**Bond-order pressure from the objective on Protenix-Tiny.** Its adapter
passes only bond existence, so bond order is a null gradient direction there.
Fixing that means changing the Protenix feature path, which is Stage 0 work.
Resolved differently in the end: the Boltz-2 backend differentiates bond order
natively, so this is no longer a blocker for Stage 2 — it is a known limitation
of the legacy backend only.

**Hardening inside the baseline loop.** Protocol 5.5 requires it as a separate
experiment from a matched relaxed checkpoint, which is what `--hardening` plus
the `D-relaxed`/`D-hardening` pair gives. Folding it into the default arm would
destroy the comparison.

## Later stages

**Stage 3, the frozen molecular prior.** A DeFoG- or DiGress-type 2D denoiser
queried once per iteration as a stop-gradient moving teacher. There is prior
local work on this (`src/mosaic/small_molecule/defog_prior.py`,
`teacher.py`), and it is intentionally not wired into `stage2/`: Stage 3 is the
comparison *against* constraint-only Stage 2, so the two must stay separable.

**3D and reference geometry.** Only after the graph is discrete enough: harden,
embed a conformer, enable the ligand geometry path, then quantify the 2D-to-native
score gap. Ligand `ref_pos` stays zero here.

**Property objectives.** Molecular weight, composition and net charge are
well defined on a relaxed graph; HBD/HBA, TPSA, logP and ADMET are not, and the
protocol prefers target ranges over unconstrained maximization. The October runs
produced valid molecules with logP near 8 and QED near 0.3, so this matters —
but it is a future-extensions item, not Stage 2.

## Known rough edge

**Slot-order robustness.** The 21 October work found that a canonical
graph-to-slot remapping changed one diagnostic's native score from 0.859 to
0.653. Nothing in Stage 2 depends on slot order, and `diagnostics.py` is
slot-order dependent only where it says so (per-edge reversal tracking uses
fixed slots within one run, which is valid). Flagged here because it affects how
far any cross-run comparison can be pushed.
