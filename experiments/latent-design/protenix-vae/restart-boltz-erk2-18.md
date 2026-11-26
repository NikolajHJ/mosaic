# Eighteen-wave ERK2 search

Can short children keep finding better valid molecules in the human ERK2 kinase pocket over 18 waves? The crystal ligand CQ8 occupies chain A, residue 1355 in [PDB 4ZZN](https://www.rcsb.org/structure/4ZZN). The prepared model uses that fixed ligand pose, 25 occupied graph slots, and the eight previously specified pocket residues.

The [run config](restart-boltz-erk2-18.json) uses full FP32 Boltz2 DBMol gradients, LR 1, a 0.25 step cap, constant chemistry weight 1, and zero noise during updates. Each wave spends 75 updates on each of eight perturbed children, two per parent. Parent selection keeps the three best replayable molecules by fresh loss and the best remaining molecule with Morgan Tanimoto similarity at most 0.85 to each of the three. If none passes, the fourth-best molecule fills the slot. The full search budget is 10,800 updates and up to 144 fresh scores.

The [founder panel](../../../outputs/boltz-erk2-founders-20261119a/panel.json) contains six distinct saved latents from the FP32 ERK2 precision pilots, one from each seed. These pilots share an original founder. All six reproduced their recorded canonical molecule and exact scored pose in a CPU replay. Wave one took two distinct starts apiece from seeds D, C, E and A. The strongest founder had a fresh loss of −4.1308. The output ranks newly found molecules and excludes the 377 identities successfully scored in the source pilots. Each completed wave records best and tenth-best new losses, distinct scored molecules, and its parent panel.

Submitted job **30112966** on 19 November 2026. LSF showed it pending at 11:21 CET for one 80 GB A100 with a 24-hour limit. Two prior 200-update FP32 ERK2 trajectories measured 7.07–7.08 seconds per update on this GPU. That gives about 21.2 hours for the updates alone. Setup, child restarts and fresh scoring add wall time. [Submission receipt](../../../outputs/boltz-restart-erk2-18x75-20261119a/erk2-18-submission.json).

```powershell
$runner = 'experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py'
python $runner stage --name boltz-restart-erk2-18x75-20261119a --config restart-boltz-erk2-18.json --queue gpua100 --max-updates 10800 --max-fresh-attempts 144 --hours 24
python $runner submit --name boltz-restart-erk2-18x75-20261119a
```

## Matched search in relaxed graph space

Does optimizing the molecular graph directly improve the yield or ranking of new valid molecules at the same compute budget? The [graph-space config](restart-boltz-erk2-18-graph.json) differs from the sealed ERK2 control config only in its optimization space. Both use seed 4919011, the same six founder molecules, historical exclusions, 18 waves, 10,800 updates and up to 144 fresh scoring attempts.

The frozen VAE supplies the founder graphs and reproduces the control's first eight perturbed starts. Each optimization step then takes gradients of the same Boltz2 and chemistry losses with respect to free logits for elements, formal charges, and unordered atom pairs. Softmax turns these logits into the probabilities received by both losses. The 25 fixed slots give 1,850 trainable values. Each update uses LR 1 and the same global 0.25 norm cap as the control, measured in its own coordinate space. All subsequent restarts perturb the saved graph logits directly, using the same candidate radii and direction count as the control. The logs record the resulting change in each probability distribution. If bounded retries exhaust exact-duplicate proposals, the graph variant can perturb the parent's winning logit margins while preserving its hard graph.

CPU replay matched every element, charge and bond probability for all eight first-wave starts and all six founders. The eight start distributions corresponded to six distinct hard poses. A two-wave integration test with a differentiable toy scorer selected a new valid parent, reloaded its graph coordinates, and verified that the optimizer and later restarts made zero VAE calls. A separate CPU screen from the actual ERK2 parent panel produced eight distinct graph-space restarts, all with their parent's original hard labels. The relative size of these continuous perturbations is a material difference between the two arms.

Compare the best and tenth-best novel hard losses and number of distinct valid scored molecules at shared updates and actual fresh scores. Branch logs distinguish validity, gradient norms, and bounded steps. Raw trajectory blocks and wave checkpoints store the new coordinates under `graph_logits` for replay.

Submitted the graph-space run as **30114008** on 19 November 2026, using the campaign name `boltz-restart-erk2-graph-18x75-20261119a` and the same 24-hour limit on one 80 GB A100. LSF showed both this run and latent control 30112966 pending at 12:41 CET. [Graph-space submission receipt](../../../outputs/boltz-restart-erk2-graph-18x75-20261119a/erk2-18-submission.json).

Job **30114008** completed all 18 waves and 10,800 updates in 20 h 43 min. Wave one produced five distinct newly scored molecules, wave two added two and wave three added two. Waves 4–18 added none, leaving nine successful fresh scores against a cap of 144. The best new molecule appeared in wave one, child 1, step 6. Its hard Boltz2 DBMol loss was −3.7950, compared with −4.1308 for the strongest starting founder (lower is better). On this seed, the remaining 17 waves never improved the best novel score. [Completed graph-space result and wave history](../../../outputs/boltz-restart-erk2-graph-18x75-20261119a/result.json).

### Matched results through wave six

Through the last wave completed by both jobs, each search spent 3,600 updates and about seven hours on one 80 GB A100. The strongest founder scored −4.1308 on the discrete Boltz2 DBMol objective (lower is better). [VAE wave-six receipt](../../../outputs/boltz-restart-erk2-18x75-20261119a/generation-06-result.json) · [Relaxed-graph results](../../../outputs/boltz-restart-erk2-graph-18x75-20261119a/result.json)

| Optimized variables | Distinct new molecules freshly scored | Best new loss | Tenth-best new loss | New molecules beating strongest founder | Valid trajectory observations |
|---|---:|---:|---:|---:|---:|
| VAE latent vector | 48 | −6.2721 | −4.8989 | 14 | 896 / 3,648 (24.6%) |
| Relaxed graph logits | 9 | −3.7950 | — | 0 | 1,911 / 3,648 (52.4%) |

The VAE used all eight fresh scores per wave. Its first eight candidates already included a molecule scoring −4.8989. The graph search exhausted its supply of new candidates in wave three. The VAE job exited during wave seven when fresh rescoring found a mismatch between a saved hard graph and its declared canonical molecule. Only the six completed waves enter this comparison. The study uses one seed and a founder panel drawn from earlier VAE trajectories.

## ERK2 with graph-space founders throughout

Can 75-update children generate good new ERK2 molecules when the entire search starts in graph space? The [preset](restart-boltz-erk2-18-native.json) starts with crystal CQ8 and three distinct valid molecules built as sparse trees on the 25 crystal positions. All four are encoded directly as relaxed atom, charge and bond logits. This run uses zero VAE checkpoints or latent parents. It excludes the four founders and uses the same 377 historical molecule identities solely to filter results.

Job **30115233** uses LR 1, seed 4919011, four parents with two children each and 18 waves of 75 updates per child. The 144 fresh-score budget covers four initial FP32 founder scores and up to 140 child scores, with 10,800 optimizer updates. Scores for the crystal reference and other founders are reported separately from novel candidates. The original CPU checks on the prepared ERK2 pocket recovered the four valid founders and eight distinct relaxed starts. An integration test with a toy scorer exercised the search entry point, wave-one scoring budget and graph checkpoints with all VAE entry points disabled.

Staged and submitted as `boltz-restart-erk2-native-18x75-20261119a` with a 24-hour limit on one 80 GB A100. At 13:31 CET on 19 November, all three ERK2 jobs were pending. [Submission receipt](../../../outputs/boltz-restart-erk2-native-18x75-20261119a/erk2-18-submission.json).

## Epoch-48 ERK2 searches from newly drawn founders

Can two independent epoch-48 VAE searches produce strong ERK2 molecules from newly drawn founders? Each uses the full frozen encoder/decoder checkpoint at update 623116 and optimizes only the latent vector. It draws a Gaussian prior and takes the first eight distinct, connected, round-trip-valid molecules with 25 naturally occupied slots, excluding 377 earlier pilot identities. It hard-scores each founder with fixed-pose FP32 Boltz2 DBMol. Four founders seed the first wave. Every following wave selects from the scored archive using the same top-three-plus-diversity rule as the previous ERK2 search. A generated molecule must survive hard graph and canonical identity checks to enter the archive. Its SMILES, branch and step accompany the discrete score.

Both [runs](restart-boltz-erk2-e48.json) use LR 1, a global step cap of 0.25, chemistry weight 1, no update noise and six waves of four parents × two children × 75 updates. Each run has 3,600 updates and up to 96 hard-score attempts, including its eight founders. Wave one reserves eight attempts for child molecules. Each later wave has 16 attempts for children. The earlier epoch-45 panel came from preoptimized parents, so score differences also reflect founder selection.

```powershell
$runner = 'experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py'
python $runner stage --name boltz-restart-erk2-e48-20261124a --config restart-boltz-erk2-e48.json --max-updates 3600 --max-fresh-attempts 96 --hours 24
python $runner submit --name boltz-restart-erk2-e48-20261124a
```

At 20:59 CET on 24 November, both submissions were pending with 24-hour limits. For seed 6248011, eight eligible founder molecules appeared within 139 draws. For seed 6248053, eight appeared within 128 draws. A CPU replay prepared eight distinct child starts per panel using placeholder founder rankings; only the real GPU founder scores determine production ranking. A decoder bond-probability probe produced finite nonzero latent gradients for both seeds. Eight generation and founder regression tests passed. Fresh molecular scores will come from the GPU jobs. [A100 job 30359611](../../../outputs/boltz-restart-erk2-e48-20261124a/erk2-a-submission.json) · [32 GB V100 job 30359612](../../../outputs/boltz-restart-erk2-e48-20261124a/erk2-b-submission.json)

Both epoch-48 jobs completed six waves and 3,600 updates each. Each used all 96 fresh attempts, including eight founders. Seed A improved from the strongest founder at 0.57184 to -4.53166, and Seed B from 0.83295 to -2.67227. Both winners appeared in wave six. [Mosaic backup](https://github.com/NikolajHJ/mosaic/releases/tag/dbmol-research-backup-20261125).
