# Does Nesso-1 affinity improve after direct ligand optimization?

The pilot uses the 127-residue BRD4 BD1 sequence from the 3MXF construct and the three 25-atom founders used in the [Protenix repair comparison](repair-search.md). Nesso predicts binding affinity from a protein sequence and ligand. Its scalar is predicted log₁₀(IC₅₀ in µM), where lower is better. Nesso selects its own protein pocket. Protenix in the companion experiment scores contact with the JQ1-defined pocket. The two scores measure different properties. [Native prediction protocol](https://github.com/recursionpharma/nesso/blob/main/docs/prediction.md)

Each seed gets two 100-update trajectories from identical probabilities. One follows the gradient toward stronger predicted binding. The directional control reverses only the affinity gradient. Both minimize the same weight-1 repair loss with LR 60, a global step cap of 0.5 and a probability floor of 0.0001. The only design variables are 25 neutral atom rows and 300 unordered bond rows. [Inputs and settings](nesso-pilot.json)

During an update, the frozen model uses the founder’s reference conformer and affinity-head crop. The bridge makes the atom features and typed-bond embeddings differentiable while retaining the original affinity heads. At the founder, the runner checks that a one-hot graph gives the same scalar as the native model with the same settings, to within 0.001 log units. It first tries a single trunk pass and falls back to one additional recycling pass if the founder crop contains no protein. Both directions use the same protocol for that seed.

The main outcome is a **new valid molecule whose freshly predicted affinity improves over its own founder**. Each direction nominates up to eight distinct connected molecules actually visited during optimization, sorted by their lowest relaxed affinity. For each nominated identity the runner builds a fresh conformer, recomputes the protein crop, and calls the released model in FP32 with five recycles. It records the value and interface entropy for each candidate and the founder. A zero interface entropy is flagged and excluded from the hit count. A result from this test measures the behavior of a frozen predictor. Validation of binding requires separate evidence.

The seed output includes synchronized warm model-forward timing on the hard founder, warm value-and-gradient timing, complete optimizer-update time, peak GPU memory and the trajectory. The publisher’s H100 inference benchmark used optimized kernels and excluded fixed preprocessing costs. This pilot uses the standard PyTorch kernels for value and gradient comparison. [Technical report](https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf)

On an allocated GPU with prepared official model assets, run `python experiments/latent-design/direct-jump/nesso_pilot.py --seed 111210 --assets ASSET_DIR --output NEW_DIR`. Seeds 111310 and 111410 use the same entry point. The GPU environment must have this checkout’s `src` on `PYTHONPATH`.

## First native measurement, 20 November 2026

Job **30131386** used an A100 PCIe 40 GB and the published weights on founder 111210. The native and differentiable one-hot founder scores matched at 0.461488 under the zero-recycle protocol, with 51 protein tokens in the founder affinity crop. The native five-recycle founder score was 0.523669. After three optimization updates, the relaxed scalar changed from 0.456605 to 0.125842 downhill and to 0.651044 in the reverse control. Both trajectories retained the founder as their only valid hard identity.

| BRD4 measurement in FP32 | Observations | Median |
|---|---:|---:|
| Warm full-protocol native forward on prepared hard features, five recycles | 3 | 1.700 s |
| Warm fixed-crop value and gradient, one trunk pass | 2 downhill updates | 0.853 s |
| Warm complete update including argmax inspection and projection | 2 downhill updates | 0.854 s |

The preparation and reference checks took 22.84 seconds in total. Peak allocated device memory was 32.08 GiB. The smoke test visited only valid states, so the optimizer timing excludes the cost of finding a repair on an invalid state. [Measured timings, validity, identities and frozen-state audit](../../../outputs/nesso1-brd4-pilot-20261120a/a100-three-update-timing.json)

Job **30131395** replayed the downhill run on seed 111210 and stopped at the first distinct valid graph, at **update 8**. All nine observations, including step zero, were valid. The relaxed affinity went from 0.456605 to 0.095104. Fresh five-recycle native rescoring gave **0.523669 → 0.470817** (change −0.052851). The separate binary binder probability decreased from 0.152 to 0.089. This single new molecule demonstrates transfer of a small part of the relaxed gain to the native affinity scalar. It leaves open how often the gain repeats across founders. [First new SMILES and fresh native readouts](../../../outputs/nesso1-brd4-pilot-20261120a/first-new-hard-rescore.json) · [Stepwise observations](../../../outputs/nesso1-brd4-pilot-20261120a/first-new-trajectory.jsonl)

The three longer jobs were pending on submission on 20 November. Each has a one-hour wall limit and reserves one GPU with at least 40 GB. The two directions run in the same allocation. [GPU plan and output locations](../../../outputs/nesso1-brd4-pilot-20261120a/gpu-plan.json)

| Founder seed | GPU queue | Job |
|---:|---|---:|
| 111210 | A100 | 30131388 |
| 111310 | L40S | 30131390 |
| 111410 | A40 | 30131391 |

Job 30131390 completed both 100-update trajectories on the L40S in 154 seconds including preparation and fresh scoring. The downhill arm averaged 0.618 s per warm update over 99 updates, with a median of 0.620 s. It produced three distinct new valid molecules. Two improved on the founder under fresh native affinity scoring, with the best moving from 1.3524 to 0.7841 at step 45. The reverse arm produced no new valid molecule. [Seed 111310 result](../../../outputs/nesso1-brd4-pilot-20261120a/111310-result.json)

For 25-atom BRD4 affinity optimization on an L40S, historical Boltz-2 job 29769525 averaged 1.743 s over 326 updates and 1.741 s over its first 100. This gives Nesso 2.8 times the measured throughput. Boltz-2 used fixed-pose affinity and checked its objective again after each proposed update. Nesso used a fixed protein crop for gradients and included chemistry repair. For the older 31-atom VAE search with the full Boltz-2 DBMol loss, three 80 GB A100 runs averaged 3.99–4.11 s per warm update. Relative to the short 40 GB A100 Nesso measurement (0.854 s), this gives a 4.7–4.8-fold difference. Both ratios compare optimizer workloads. [Boltz-2 L40S timings](../../../outputs/nesso1-brd4-pilot-20261120a/boltz2-brd4-l40s-timings.json) · [Boltz-2 A100 timings](../../../outputs/nesso1-brd4-pilot-20261120a/boltz2-brd4-a100-timings.json)

By 21 November all three paired 100-update pilots had completed. Across the downhill arms, seven of eight new valid molecules had better fresh predicted affinity than their founder. Seed 111210 contributed five improvements, seed 111310 contributed two, and seed 111410 yielded zero new valid molecules. The reverse-gradient arms produced two new valid molecules and one improvement, both molecules from seed 111410. [Seed 111210](../../../outputs/nesso1-brd4-pilot-20261120a/111210-result.json) · [Seed 111410](../../../outputs/nesso1-brd4-pilot-20261120a/111410-result.json)
