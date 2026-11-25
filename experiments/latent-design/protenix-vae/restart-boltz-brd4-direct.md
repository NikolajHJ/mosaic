# BRD4 direct relaxed-graph pilot

The baseline for new small-molecule optimization studies optimizes categorical element, formal-charge and undirected bond logits directly. On this branch, VAE latent optimization requires an explicit request. The [direct preset](restart-boltz-brd4-direct.json) runs without a VAE checkpoint. It reuses the FP32 fixed-pose Boltz2 DBMol scorer and 31 occupied ligand positions of the [historical BRD4 trial](restart-boltz-brd4.md).

## Paired comparison

Can a carbonyl reward form a recognizable carbonyl on a valid molecule, and can a separate ring penalty prevent excessive three-membered rings? Each of three seeds generates a distinct, connected tree of 25 carbons, three nitrogens and three oxygens on the native JQ1 coordinates in 3MXF chain A. Short distances guide the tree's construction, and heteroatoms occupy nonadjacent sites. These founders begin with all single bonds, neutral atoms, zero rings and zero carbonyls. This tests motif acquisition and prevention of excess rings. Recovery from a pre-existing cage is a separate question.

Each seed runs four arms from identical logits. All use DBMol, the existing repair-target loss and valence-excess penalty. The control adds neither experimental term. The carbonyl arm adds a loss with weight 0.3 for the strongest neutral carbon–oxygen double-bond motif with terminal oxygen. The reward saturates at a soft motif probability of 0.5. The ring arm adds a penalty with weight 0.2 for the expected number of triangles above one. The fourth arm uses both terms. The [optimizer](../../../src/latentmol/design/boltz_direct_graph.py) uses Adam at LR 0.1 with an L2 step cap of 0.25 per categorical row and a single seeded perturbation at initialization.

Each arm receives 75 optimizer updates and up to eight new valid molecules freshly scored as discrete graphs in the fixed pose. The full study totals 900 updates and up to 96 hard scores. Selection deduplicates by SMILES, excludes the founder, and prioritizes the best relaxed DBMol score seen for each identity. Every observed valid identity is available for the chemistry summary.

A separate CPU job per seed independently embeds and redocks the founder and up to two distinct new molecules per arm selected by hard DBMol loss. It records Vina score relative to the founder, discrete carbonyl counts, triangle counts, and audited SA. Duplicate structures share a docking evaluation within the seed, and empty arms and failed dockings stay visible. This quick Vina protocol uses seed 1701 and exhaustiveness 4. Compare scores within this protocol. The preceding cross-method BRD4 report used a different docking protocol.

## Execution and receipts

On 19 November 2026, the CPU smoke replayed the actual three BRD4 founder geometries. With a flat binding objective over 75 updates, distinct new valid molecules containing a carbonyl numbered 0, 0 and 0 in the controls, and 0, 2 and 5 with the carbonyl reward. Seed A reached a high soft motif probability without producing a new valid molecule in the motif arm. This checks that the direct optimizer and scoring distinction are exercised. Binding and docking outcomes come from the allocated runs. [Smoke metrics](../../../outputs/direct-pilot-smoke-0uyxpcyw/)

| Founder seed | GPU job | Queue | Dependent CPU docking job |
| --- | --- | --- | --- |
| 517111 | 30113968 | gpuv100, 32 GB | 30113972 |
| 517133 | 30113970 | gpua40 | 30113973 |
| 517147 | 30113971 | gpuv100, 32 GB | 30113974 |

All six jobs were pending at 12:38 CET. GPU allocations have four-hour limits, and CPU docking allocations have one-hour limits. [GPU plan](../../../outputs/boltz-restart-brd4-direct-20261119a/gpu-plan.json) · [Docking submissions](../../../outputs/boltz-restart-brd4-direct-20261119a/vina-submissions.json)

```powershell
$runner = 'experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py'
python $runner stage --name boltz-restart-brd4-direct-20261119a --config restart-boltz-brd4-direct.json --max-updates 300 --max-fresh-attempts 32 --hours 4 --queue gpuv100
python $runner submit --name boltz-restart-brd4-direct-20261119a
```

The [CPU readout](restart-operations/evaluate_boltz_direct.py) takes `--run SEED_OUTPUT --output NEW_DIRECTORY`. Its submission receipts preserve the successful-completion dependencies and the exact source snapshot used for both stages.

## First completed BRD4 seed

On 19 November at 22:34 CET, LSF confirmed seed A (30113968) completed. Each of the four LR 0.1 variants spent 75 updates, with 14 valid observations among its 76 states. Every valid observation reproduced the founder. All four variants therefore spent zero fresh scores on new molecules. The remaining two GPU seeds were pending. [Seed A result](../../../outputs/boltz-restart-brd4-direct-20261119a/seed-a-result.json).

## Second completed BRD4 seed

20 November 2026. Seed C (30113971, LR 0.1) returned the same one new molecule in each of the four 75-update arms. Each arm recorded 15 valid states in 76 observations and one fresh score, −0.10288. The new molecule retained the founder connectivity and changed one nitrogen formal charge from 0 to −2. Redocking gave −6.842 kcal/mol compared with −6.852 for the founder. [GPU result](../../../outputs/boltz-restart-brd4-direct-20261119a/seed-c-result.json) and [redocking result](../../../outputs/boltz-restart-brd4-direct-20261119a/seed-c-vina-result.json).
