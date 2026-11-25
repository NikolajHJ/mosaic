# Is the last continuation better spent on an independent start?

Compare the [synthesis-aware BRD4 standard](restart-synthesis-standard.md) against replacing its third child in wave three. In the earlier 48,000-update comparison, independent 500-update starts supplied one strict discovery hit per seed. The three adaptive hits all came from the same child. That comparison used binding-only guidance. This pilot tests the independent start within the current synthesis-aware search. [Original molecules and per-seed counts](../../../outputs/restart-joint-retrospective-20261119/audit.json)

Both arms replay the same three near-miss founders, search three waves of three 500-update children at LR 60, use the same weak SynFrag guidance, and have zero update noise. The standard continues all three branches in wave three using terminal recovery. The experimental arm continues branches one and two. Its third child uses the historical independent initializer, one unfiltered standard-normal latent with the 25 most occupied slots selected by its decoder. The replaced slot and the unscreened draw are recorded in `cycle-03/parent-plan-independent-slot.json`. The fresh native scorer reads the resulting hard molecule's SMILES. The relaxed native scorer and SynFrag both consume the independent child's own occupied slots.

| Job | Perturbation seed | Independent draw seed | Submission status, 19 November at 00:24 CET |
| --- | ---: | ---: | --- |
| 30111017 | 1911261 | 1911269 | Pending on V100 |
| 30111018 | 1911273 | 1911281 | Pending on V100 |
| 30111020 | 1911287 | 1911295 | Pending on V100 |

Each arm receives 4,500 updates, up to 57 discovery scores and up to six held-out geometry scores. Each pair shares the three founder scores, limiting physical discovery calls to 111. The three-seed total is 27,000 updates, up to 333 physical discovery calls and up to 36 confirmation calls. The frozen history lists 59,211 valid hard identities and retains the standard's founder latents. All three independent starts were invalid at step zero in the local CPU replay and had different masks from the founders. The frozen SynFrag model produced finite connected gradients on all three. The CPU check covered native input assembly; the GPU jobs perform native scoring. [Input and budgets](restart-synthesis-independent-slot.json) · [CPU replay](../../../outputs/synthesis-independent-slot-preflight-20261119a/cpu.json) · [HPC submission receipts](../../../outputs/protenix-restart-synth-independent-slot-20261119a/)

Read `arms.<arm>.final_slot` in each `result.json` for the chosen start, branch validity, successful fresh scores, and strict/distinct/diverse hits added by that slot. Compare it to the displaced continuation on that seed plus both whole-arm archives. The primary readout is new-to-history, audited hard SA ≤5 and fresh loss ≤2.3. Check `confirmation/result.json` for the prespecified held-out geometry results of the best three strict hits per arm. If more than three qualify, additional strict hits retain their discovery status. Also report SA ≤6 yield, actual fresh attempts, time to first valid state, and the recurring N–C(OH)₂–N motif among qualifying structures. One independent draw per seed limits what a negative result can establish.

Advance the independent slot to a larger paired comparison if it adds a new structurally diverse strict hit over the continuation in at least two of three seeds and the independent arm preserves or improves pooled distinct held-out-confirmed yield. Compute per-slot and whole-arm distinct and greedy-diverse counts on the same frozen history; use the existing Morgan radius-two fingerprint and 0.60 Tanimoto cap. Pool confirmed identities once across seeds for each arm. Sparse or zero outcomes are inconclusive, and the continuation stays the default pending stronger evidence.

DTU CPU job **30111066** waits for all three GPU jobs to end. It reads completed search and confirmation receipts, checks their counts against the frozen reference, and writes `/work3/s234818/latentmol/assets/synth-independent-slot-readout-20261119a/decision.json`. To count as a structurally new addition for a seed, the tested slot's strict molecule must have similarity ≤0.60 to every strict molecule from the entire control arm and all other branches of its own arm. The advancement rule also requires at least one pooled confirmed molecule in the independent arm. Its receipt preserves each held-out score or failure. [Readout script](restart-operations/compare_independent_slot.py) · [Sealed inputs and scheduler dependency](../../../outputs/protenix-restart-synth-independent-slot-20261119a/slot-readout-submission.json)

Inspect any of the three paired plans from the repository root.

```powershell
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-synthesis-independent-slot.json --arm seed-1 --plan-only
```
