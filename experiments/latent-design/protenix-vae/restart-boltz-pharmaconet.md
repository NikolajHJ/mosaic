# Does PharmacoNet guidance create BRD4 interactions that survive redocking?

19 November 2026. One seed compares direct relaxed-graph optimization with the pocket-map weight set to 0 or 0.3. Both arms run in one 32 GB V100 allocation from the same 31-slot sparse founder and initial logits (seed 517111). FP32 Boltz-2 DBMol, repair and valence losses, fixed pose, Adam LR 1 and the per-row step cap of 0.25 are shared. Each arm has 75 updates and up to eight fresh fixed-pose DBMol scores on distinct valid descendants. The trainable variables are element, formal-charge and undirected-bond logits. [Preset](restart-boltz-pharmaconet.json) · [Budget and output locations](../../../outputs/boltz-restart-brd4-pharmaconet-lr1-20261119a/gpu-plan.json)

The pretrained PharmacoNet model found eight hotspots from the 3MXF chain A receptor and the JQ1 pocket box. These comprise an acceptor facing Asn140, a T-shaped aromatic interaction with Trp81, and six hydrophobic sites. Each of the three feature families receives one third of the auxiliary weight. Sites within a family are weighted by the model scores. The ligand-side loss differentiates local atom and bond probabilities at fixed atom positions and candidate ring centroids. In this pose, two five-membered cycles and one six-membered cycle fall within the ring edge distance range. The prepared Boltz slots registered to the deposited coordinates within 0.000002 Å. [Pretrained model](https://github.com/SeonghwanSeo/PharmacoNet) · [Map and source hashes](../../../outputs/brd4-pharmaconet-3mxf-20261119a/all-hotspots.json) · [Guidance](../../../src/latentmol/guidance/pharmaconet.py)

The CPU readout independently embeds and Vina-redocks the founder and every freshly scored descendant, up to 17 unique molecules. Selection uses the same DBMol ranking in each arm. Identical SMILES share one redocking. Vina uses seed 1701 and exhaustiveness 4. For each molecule and site, compare the designed graph on its fixed coordinates to the best redocked pose in the receptor frame. Report hotspot overlap, overlap with compatible partner geometry, and contact with that same protein partner regardless of hotspot radius. Record failures and distinct candidate denominators separately in each arm. [Readout](../../../src/latentmol/evaluation/pharmaconet_redock.py)

## Reference and smoke checks

Independent redocking of neutral (+)-JQ1 retained the Asn140 hydrogen-bond geometry using the same triazole nitrogen. The N···N distance changed from 3.15 to 2.87 Å and the donor–H–acceptor angle from 152° to 149°. The pose scored −8.439 kcal/mol with a crystal-frame heavy-atom RMSD of 0.75 Å. The deposited CCD +1 representation has a proton on this nitrogen. It retained three of its four hydrophobic partner contacts and zero strict hotspot contacts. These controls are outside the discovery budgets. [Neutral reference](../../../outputs/brd4-pharmaconet-3mxf-20261119a/neutral-jq1-calibration/result.json) · [CCD +1 control](../../../outputs/brd4-pharmaconet-3mxf-20261119a/jq1-ccd-calibration.json)

A paired 75-update CPU smoke at the original LR 0.1 with a flat binding objective reproduced identical start logits. The final map losses were 2.50 for control and 1.18 for guided, with four and one distinct new valid molecules, respectively. These values only test the implementation. [CPU trajectories](../../../outputs/pharmaconet-map-paired-smoke-_uro1aru/)

## Submission and results

GPU job **30115179** completed both LR 1 arms on 19 November at 22:53 CET. CPU redocking job **30115180** finished at 22:55 CET. The LR 0.1 pair (30115064 and 30115065) was cancelled before it started when the requested learning rate changed. The two sealed archives differ only in the learning rate and the associated run plan. [GPU receipt](../../../outputs/boltz-restart-brd4-pharmaconet-lr1-20261119a/seed-a-submission.json) · [Redocking receipt](../../../outputs/boltz-restart-brd4-pharmaconet-lr1-20261119a/seed-a-vina-submission.json)


### Completed LR 1 comparison

| Seed 517111 | Valid observations / 76 | Distinct valid molecules including founder | New valid | Freshly scored / 8 | Descendants docked | Missing poses | Final relaxed map loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 11 | 1 | 0 | 0 | 0 | 0 | 2.958 |
| PharmacoNet | 11 | 1 | 0 | 0 | 0 | 0 | 2.441 |

Both arms decoded to the founder at initialization and steps 1–10. Every hard graph at steps 11–75 failed sanitization. The common starting map loss was 2.688. Guidance improved the relaxed score, but neither arm supplied a new valid molecule for scoring. Candidate retention is therefore undefined for Asn140, Trp81 aromatic stacking and the six hydrophobic sites under both the strict hotspot and partner-geometry criteria. This was a single paired seed. [Optimization result](../../../outputs/boltz-restart-brd4-pharmaconet-lr1-20261119a/gpu-result.json) · [Control trajectory](../../../outputs/boltz-restart-brd4-pharmaconet-lr1-20261119a/control-metrics.jsonl) · [Guided trajectory](../../../outputs/boltz-restart-brd4-pharmaconet-lr1-20261119a/guided-metrics.jsonl)

The shared founder redocked successfully at −5.987 kcal/mol. It retained zero strict contacts. Its original Val87 strict contact was lost and an Ile146 strict contact appeared. By protein-partner geometry alone, two of three original hydrophobic contacts survived (Val87 and Ile146). The Asn140 interaction appeared only after docking and outside the strict hotspot. There was no retained aromatic interaction. The separate neutral (+)-JQ1 control above recovered the strict Asn140 interaction. [Redocking result](../../../outputs/boltz-restart-brd4-pharmaconet-lr1-20261119a/redocking-result.json) · [Founder pose](../../../outputs/boltz-restart-brd4-pharmaconet-lr1-20261119a/founder-redocked.sdf)

```powershell
$runner = 'experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py'
$study = 'boltz-restart-brd4-pharmaconet-lr1-20261119a'
python $runner stage --name $study --config restart-boltz-pharmaconet.json --max-updates 150 --max-fresh-attempts 16 --hours 2 --queue gpuv100
python $runner submit --name $study
python $runner redock --name $study
```
