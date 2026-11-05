# Validity-constrained seed 41 hard-contact continuation

DTU job **29898493** completed on L40S at 13:42 CET on 5 November 2026. It started from the exact saved hard argmax at global update **52,000** of the [atom-exchange arm](../20261105-seed41-atom-exchange-pair-01/README.md), with hard contact **0.1866476** and one sulfur atom. [Campaign and pins](campaign.json)

The continuation optimizes the discrete graph directly. At each current graph, rank local atom substitutions, bond insertions/removals, and connected single-bond reattachments by the frozen scorer's graph gradient. Score the valid proposals as hard one-hot graphs with the same Protenix Tiny 8 Å pocket objective, context, and key 17. A score improvement above 0.000001 is required to accept a move. The initial graph stays eligible throughout. Save accepted labels, scores, canonical SMILES and the final hard snapshot.

Every scored candidate must pass native sanitization, connectivity, schema round trip and the same common-organic sulfur scope. The neutral occupied slots stay fixed. A new sulfur must also fit the previous atom's bond-order cap. The scope policy and frozen scorer are bound to the completed paired run.

The search is bounded to 24 ranking rounds and 768 candidate forwards. Each round tries up to 32 valid candidates, widening once when the first batch finds no improvement. It stops at the budget or when the ranked neighborhood yields no improving move. Compare against the valid start and use the older hard scores **0.2030956** and **0.2150352** as context, since those earlier graphs fell outside the sulfur scope. The score measures model contact on a discrete graph.

## Result

The first accepted move reached **0.2242611**, above the earlier 0.2150352 comparison. The saved final graph scored **0.6522143** on replay, with one sulfur atom. All 22 accepted moves and the final graph passed native chemistry, connectivity, the schema round trip and the shared sulfur rule. The 22 accepted moves comprised 15 atom substitutions, three single bond edits and four bond reattachments.

The job used all 768 candidate forwards, 22 ranking VJPs and two separate reference forwards. It stopped at the candidate cap. LSF recorded 100 seconds of run time. The [terminal verification](terminal-verification.json) binds the exact endpoint, saved result, frozen scorer, accepted score sequence and final hard snapshot. The improvement is on the frozen contact scorer used to select the moves. Binding and pose quality would require separate evaluation.
