# Does updating the parent archive improve cyclic search?

8 November 2026. The [comparison config](restart-lineage-feedback.json) uses
three disjoint panels of three previously scored parents from the BRD4 flow
cohorts. Each panel runs all three policies on one GPU. All policies start
with the same scored archive within a panel and use their own cumulative
archive to report outcomes.

| Policy | Starting states each cycle |
| --- | --- |
| Adaptive elite | Screened latent kicks from the three best fresh-scored molecules found by this arm so far |
| Fixed parents | Screened latent kicks from the original three parents in every cycle |
| Independent starts | Distinct valid draws from the frozen decoder prior |

Adaptive and fixed share perturbation seeds by cycle, parent rank, and
repeat. They start from the same latents in cycle one. Independent starts
draw from a separate seed stream. Their starting molecules are eligible for
fresh scoring under the same selection rule. Prior draws and rejected
starts are recorded separately from optimizer updates.

Five cycles of two 96-update branches per parent give 2,880 updates per
policy and 8,640 per seed. The LR is 70 with a 0.25 latent displacement cap,
repair and valence excess weights of 1, and 25 fixed occupied slots. This
uses the epoch-45 VAE and the same Protenix BRD4 pocket as the noise study.

Each policy gets up to 18 new distinct fresh-score attempts per cycle and
90 per seed. Each branch contributes a mixed shortlist by relaxed loss and
source step, with deeper candidates available to fill slots after
deduplication. The arms use separate caches and scoring batches. If one
arm cannot supply the target, all three use the same lower count for that
cycle. Embedding and representation failures count as attempts and are
reported separately from successful scores.

The readout includes best and mean top-three fresh loss by cycle, actual
updates and fresh attempts, starting and elite provenance, and a count of
new molecules at fresh loss 2.30 or lower selected with a maximum pairwise
Morgan Tanimoto similarity of 0.60. The shared initial archive is excluded
from the diversity count. The primary comparison is adaptive versus fixed
within each seed.

The allocated entry point is `python -m latentmol.design.protenix_restart_study`.
The [submission directory](../../../outputs/protenix-restart-lineage-feedback-20261108a/)
contains the sealed source, input and storage checks, commands, and receipts.

DTU accepted all three jobs at 22:41 CET. They were pending at the first
check. Each has a six-hour wall limit and a 45-minute startup allowance.

| Seed panel | Perturbation seed | Prior seed | Queue | Job |
| --- | ---: | ---: | --- | ---: |
| A | 1811901 | 1816901 | `gpua100` | 29931143 |
| B | 2111901 | 2116901 | `gpua100` | 29931145 |
| C | 2411901 | 2416901 | `gpuv100` | 29931146 |

Fourteen focused tests passed. A two-cycle CPU smoke used the verified
frozen decoder and mock scorers. Adaptive and fixed saved identical metrics
in cycle one, and all arms used one fresh attempt per cycle. The native
comparison is pending.
