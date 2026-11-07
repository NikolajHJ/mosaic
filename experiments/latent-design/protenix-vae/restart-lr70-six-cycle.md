# Which changes improve the six-cycle Protenix restart search?

7 November 2026. The [campaign config](restart-lr70-six-cycle.json) uses LR 70,
1,000 updates per branch, and six cycles. Start from one bootstrap branch.
In each subsequent cycle, start three branches from each of up to three
fresh-scored parents. That gives a maximum of 46 branches and 46,000 updates
per arm. Each branch has up to 16 fresh Protenix FILE scoring slots after
canonical SMILES deduplication. Six full arms can use 276,000 updates and
4,416 fresh scoring slots.

All arms share the frozen epoch-45 VAE, BRD4 pocket, 25 fixed atom slots,
bootstrap seed 17, temperature 1, and repair weight 1. The conformer seed is
1201. Independent latent update noise begins at 5% of the binding step norm
and fades to zero by update 100 in every branch and arm. That lets three
unchanged-parent branches start at the same molecule and follow different
trajectories. Each arm uses the same seeds by branch slot. Parents can
diverge as the policies change which molecules are found and selected.

| Arm | Difference from the combined run |
| --- | --- |
| combined | Screened latent kicks, step cap 0.25, valence excess weight 1, mixed shortlist, diverse restart parents |
| cap050 | Step cap 0.5 |
| repair-only | Valence excess weight 0 |
| unchanged-parent | Exact archived latent at step zero |
| relaxed-shortlist | All shortlist slots allocated by relaxed contact rank |
| top-parents | Restart from the three best fresh scores |

The mixed shortlist uses up to eight slots by relaxed rank and up to eight
distinct molecules spread by source update. Diverse parent selection keeps
the fresh winner. It prefers parents from other branches among molecules
within 0.10 fresh loss of the winner and uses radius-2 Morgan similarity to
favor candidates distinct from the parents already chosen. It fills empty
slots by fresh rank. Each cycle records the three best fresh scores and the
actual restart parents separately, with their source branches and updates.

The valence term computes expected local bond-order excess on the shared
relaxed graph and supplies a gradient even when repair finds no witness.
At weight 1 its median gradient norm was 17.4% of the binding gradient
norm across 25 archived states. The runs log both norms separately.
Fresh scoring records conformer failures. A flag marks molecules with four
or more three-membered rings in the RDKit ring set for review. The flag
does not change scores or parent selection.

Compare best fresh loss by cycle and by actual fresh calls, parent
improvements, number of distinct high-scoring molecules, source-branch
diversity, validity, ring flags, and cap hits. The arms use a single matched
bootstrap seed in one prepared pocket. The first completed duration panel
split four wins apiece between short branches and a single longer branch.
This campaign fixes branch length at 1,000.

## Run command

The allocated entry point is
`python -m latentmol.design.protenix_restarts --config <config> --arm <arm>`.
Use the [DTU submitter](restart-operations/submit_campaign.py) to stage
and submit the config. The [campaign directory](../../../outputs/protenix-restart-lr70-six-20261107a/)
holds the source archive, capacity check, exact commands, and job receipts.

DTU accepted all six jobs on `gpul40s` at 16:42 CET. Each requests one
exclusive L40S, four CPU slots, and a 24-hour wall limit. The wall limit
allows for the added CPU backward pass and fresh scoring. Using one GPU
type across the comparison holds the GPU architecture constant.
All six were pending at the first scheduler check due to host slot limits.

| Arm | DTU job |
| --- | ---: |
| combined | 29916515 |
| cap050 | 29916522 |
| repair-only | 29916530 |
| unchanged-parent | 29916537 |
| relaxed-shortlist | 29916544 |
| top-parents | 29916552 |

The staged source SHA-256 is
`f37057b6697b33930bd3728885c25928aaec13076e5dd720de5916d481dab363`.
Fifty focused tests passed. Input hashes and the combined storage allowance
were verified before submission. A two-cycle CPU smoke with the frozen
decoder and mock scorers exercised the combined policies. The unchanged-parent
control shared the exact archived starting latent across branches and
diverged on the first update. Scientific results are pending.
