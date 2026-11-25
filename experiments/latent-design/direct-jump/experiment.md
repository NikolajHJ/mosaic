# Do learned jumps improve direct molecule search on BRD4?

## Direct chemistry-loss pilot, 20 November 2026

On interactive V100 job **30129301**, nine trajectories completed 48 updates each from three random sparse, valid 25-atom founders (111210, 111310, 111410). The variables were the 25 neutral element rows and 300 unordered bond rows. The frozen BRD4 Protenix contact weight was 1. Projected probability SGD used LR 60, a 0.5 displacement cap and a simplex floor of 0.0001. The three arms used identical starts and the same native scorer key (111210). Each active chemistry term used the existing fixed-atom search and verified-target negative log likelihood divided by 350. Atom and motif preservation were zero. This short pilot used one continuous path, with no VAE, connectivity operator, noise, Vina or learned jump.

| Repair weight | Valid source states / 144 | Valid endpoints / 3 | Distinct new valid molecules beating their founder | Mean best hard loss change | Mean sulfur drift | Mean Morgan distance among new valid molecules |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 105 (72.9%) | 2 | 5 | −0.07145 | −0.076 | 0.745 |
| 0.1 | 109 (75.7%) | 2 | 7 | −0.07145 | −0.042 | 0.713 |
| 1 | 109 (75.7%) | 2 | 7 | −0.07145 | −0.035 | 0.685 |

Every arm found the same best hard molecule for each founder. All three founders improved in every arm. Across all arms there were eight distinct new, connected molecules that passed RDKit reparsing and improved same-context hard contact. The largest improvement, on seed 111310, went from −0.16849 to −0.32898. Scoring used one-hot molecules in the geometry-off context, without a fresh pose. Validity uses steps 0–47 and molecule discovery includes step 48. All 432 requested updates hit the cap. Each invalid source supplied a verified repair witness. Seed 111410 still finished invalid in all three arms, with only 12/48, 14/48 and 14/48 valid source states. On seed 111210 all source states were valid and the repair gradient stayed zero, yet late hard readouts differed. The small yield difference therefore mixes repair effects with numerical trajectory variation.

The driver reused the frozen direct search archive from the separately queued November 20 study (SHA-256 `de575fb2c94f719fd132d166703760a2e6569800281bf6cda91fcd9f5c3759d1`). Run `bash run.sh` in the staged interactive allocation with that archive extracted as `source/` beside the pilot driver. The final frozen model/context check passed, as did seven local tests of the archived direct repair and trajectory code. CUDA emitted a kernel timing warning during warm-up. The downloaded result archive matched its remote SHA-256. [Pilot driver](../../../outputs/direct-chem-pilot-20261120/pilot.py) · [Run settings and receipts](../../../outputs/direct-chem-pilot-20261120/results/summary.json) · [Analysis](../../../outputs/direct-chem-pilot-20261120/analysis.json) · [Candidate SMILES and scores](../../../outputs/direct-chem-pilot-20261120/improved-molecules.json) · [First differing steps](../../../outputs/direct-chem-pilot-20261120/divergence.json)

## Original learned-jump experiment

Submitted on 19 November 2026. The experiment optimizes 25 occupied neutral atom rows and 300 unordered bond rows directly. Each run uses one frozen scorer. Protenix-Tiny uses LR 60 on its pocket contact loss. Boltz2 uses LR 1 on its fixed-pose affinity output. Both take an ordinary gradient step on probabilities, project each row onto the simplex, and apply Mosaic's connected Stage 2 operator after each gradient update. Direct probability SGD and logit SGD can move different distances at the same nominal rate.

## Held-out comparison

The two cohorts each have 24 training and 32 held-out founders. The 112 sparse-tree founder molecules have distinct canonical identities across all splits. Protenix and Boltz2 use the same founders. Initial soft atom rows mix a uniformly drawn fraction from 0.01 to 0.20 with the uniform distribution. Bond rows use 0.001 to 0.04. Each fraction is drawn once per head per start. These ranges give the restart a new continuous state with the same hard molecule. The pocket, scorer key and all subsequent refinement rules are shared within each comparison. [Founder panel](../../../outputs/direct-jump-20261119a/founders.json)

Each scorer and cohort fits its own ridge regression to ordered atom- and bond-probability displacements. Each training founder supplies a 16-update trajectory. For targets, use its best eligible fresh-scored state if the gain exceeds 0.005. Otherwise use the ordinary eight-update state as an imitation target. Each evaluation saves the count and identity of both target types. Fit, scaling and displacement caps use training data only.

| Held-out arm | Starting state for eight gradient calls |
| --- | --- |
| Learned jump | Add the learned residual to the founder probabilities and project to the simplex. |
| Same-size random jump | Shuffle the learned proposal across rows and alternative labels, preserving feasibility and realized L2 displacement separately for atoms and bonds. |
| Restart | Independently reinitialize the probabilities of the same hard founder under the original softening distribution. |

For each arm, inspect the initial state and all eight updates. Reserve one shared founder score and up to four fresh scores for distinct connected, RDKit-sanitized edits. If there are more than four, select evenly across their first appearances without seeing their scores. Unused slots expire. Shared molecules receive the same score within the cohort's evaluation batch. Training calls and fresh scoring are accounted for separately.

The primary contrasts are the paired differences in mean best fresh loss reduction for learned versus random and learned versus restart. An arm can retain the founder. A missing founder score contributes zero verified gain in all three arms. Report the scorable-founder subset and the fraction improving by more than 0.005 as well. The 95% bootstrap intervals resample matched starts within each fitted cohort. They describe variation across these starts in one pocket.

Protenix rebuilds the ligand conformer and FILE features with geometry seed 1201 and measures negative mean log pocket-contact probability. Boltz2 builds a fresh protein–ligand cofold from SMILES and measures native `affinity_pred_value` using structure and affinity diffusion. Lower is better within each model. The report presents their scales separately.

## Execution

| Stage | DTU job | Requested allocation |
| --- | ---: | --- |
| Protenix, cohort 0 | 30113049 | V100 32 GB, 4 h |
| Protenix, cohort 1 | 30113050 | A10, 4 h |
| Boltz2, cohort 0 | 30113051 | A100, 12 h |
| Boltz2, cohort 1 | 30113052 | A100, 12 h |
| Paired report | 30113053 | CPU, after all four GPU jobs succeed |

Each GPU job owns its training, separate fresh training scores, held-out evaluation and fresh evaluation scores. The budget is 1,152 gradient calls per GPU job, including 384 for training and 768 for the held-out arms. Each cohort allows at most 416 unique fresh evaluation scores before deduplication (32 founders and up to four edits per arm per founder). The wall limits are scheduler reservations.

The remote experiment root is `/work3/s234818/latentmol/runs/direct-jump-20261119a`. Reproduce a cohort from the frozen source and founders with `bash source/experiments/latent-design/direct-jump/run.sh ROOT MODEL COHORT` inside a new GPU allocation and output root with the staged dependencies. Combine complete cohorts with `python report.py --root ROOT --output NEW_JSON`. The worker checks founder identity, finite inputs and derivatives, displacement matching and frozen model state. Local smoke checks exercised six real founders with the production projection and mock model scores. At submission all four native GPU jobs were pending. [Source snapshot](../../../outputs/direct-jump-20261119a/source.tar.gz) · [Scheduler and submission receipts](../../../outputs/direct-jump-20261119a/submitted.json)

## Startup recovery on 25 November

Protenix job 30113049 and Boltz2 jobs 30113051–52 received GPUs and exited after two seconds. Each wrote a GPU allocation receipt and zero training trajectories. The original A10 cohort 30113050 was still pending. Its report 30113053 depended on three unsuccessful jobs. No fresh scores were produced.

Sourcing DTU’s shell profile with strict error handling reproduced exit code 1 and empty stderr on the login host. Both corrected launchers completed the same shell and Python-module initialization check. The native GPU checks run inside each allocation. The recovery snapshot changes only the two shell launchers and submission script. It reuses the original founders, study code, rates, scoring and budgets.

DTU accepted recovery jobs 30363468 and 30363469 for Protenix (V100), 30363470 and 30363471 for Boltz2 (A100 and L40S), and CPU report 30363472. All four GPU jobs were pending at 10:17 CET. The recovery writes to `/work3/s234818/latentmol/runs/direct-jump-20261125a`. The earlier A10 submission and its report still point to the original experiment. [Startup checks and scheduler receipts](../../../outputs/direct-jump-20261119a/recovery-20261125a/submitted.json)
