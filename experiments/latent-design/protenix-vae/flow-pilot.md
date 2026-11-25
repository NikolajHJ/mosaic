# Can learned latent moves reduce the cost of hallucination?

Exploratory branch `codex/flow-guided-hallucination`, 5 November 2026. The
pilot separates early stopping, learned proposals, and detached graph repair.
All native arms use the frozen epoch-45 VAE and BRD4 pocket at 25 occupied
slots. Fresh hard-graph scores use the existing Protenix FILE workflow with
regenerated ligand conformers, geometry seed 1201, and trunk contact loss.
Lower is better. These are model-based contact measurements.

## Saved replay completed

Using the completed matched temperature study 29897058, inspect fresh-scored
unique edits available by updates 7, 16, and 32. Two arms lacked a valid scored
starting graph. Of the ten comparable arms, three had an improvement among the
preselected candidates. The two early improving arms found stronger edits later.

| Prior seed and temperature arm | First observed gain | Best delta by 7 | Best delta by 32 |
| --- | ---: | ---: | ---: |
| 191411 constant | 2 | −0.00738 | −0.13154 |
| 191521 warm | 5 | −0.12222 | −0.13280 |
| 191521 constant | 14 | — | −0.12222 |

These three arms share two starts. Candidate selection and conformer failures
limit coverage, so the first *observed* improvement can be later than the first
actual one. The replay uses existing scores and consumes no new native calls.

```powershell
$env:PYTHONPATH = 'src'
python experiments/latent-design/protenix-vae/flow_pilot.py replay --fresh outputs/hpc-results/protenix-latent-editability-20261105a/temperature-warm-29897058/fresh.json --output NEW_JSON
```

[Replay](../../../outputs/flow-guided-hallucination-20261105a/saved-replay.json)
· [16 screened starts](../../../outputs/flow-guided-hallucination-20261105a/screen.json).
Local checks ran both students on a synthetic displacement and passed 45
existing candidate-selection and DeFoG-worker cases.

The [CPU repair preflight](../../../outputs/flow-guided-hallucination-20261105a/repair-local-preflight.json)
replayed all six eligible hard graphs from the saved latent and verified the
exact candidate requests. The chronological selection filled both quotas from
seed 191411, so all six graphs represent one trajectory. Structural repair
found a connected witness on each of the three invalid cases and kept the three
valid inputs. The preflight consumed no DeFoG or Protenix model calls.

## Native pilot design

| Job | Queue and allocation | Question and budget |
| --- | --- | --- |
| 29903888 | 32 GB V100, 4 CPU slots, 90 min | Screen distinct connected starts from at most 256 seeded prior draws. Train on up to 12 independent starts and evaluate on four held-out starts. The 24-update teacher also supplies the 7-update baseline prefix. Compare one residual jump and four Euler steps, each followed by seven identical bounded updates. Up to 464 relaxed calls total, including step zero. |
| 29903887 | L40S, 4 CPU slots, 45 min | Up to three valid and three invalid distinct hard graphs from the completed constant-temperature prior runs. Compare the structural repair policy with strict-neutral GuacaMol DeFoG started directly at model time 0.8 on a 128-point grid. Count eligibility, connected outputs, edits, and the fresh score change where the input can be scored. |

The student uses one endpoint per training start, selected by the best fresh
loss decrease above 0.005 among the capped candidate panel. Four improving
starts are required to fit. Fewer will be reported as insufficient pairs, with
the completed short-versus-long baseline still available. The residual and flow
networks have the same size and fit budget. Proposal displacement is capped at
the maximum training displacement. The pocket and fixed active slots are shared
within each comparison. The hard-scoring cap is three distinct edits per arm
and the same valid starting graph. Report gains, unique molecules, scorer calls,
and timings on held-out starts. The small panel measures an early signal for one
pocket, and the reported fit cost matters when estimating the savings from reuse.

Repair comparisons use the same schema-compatible inputs for both methods and
record unsupported inputs encountered during selection. Structural repair keeps
an already-valid input as-is. For invalid starts, report recovery and compare
the repaired candidates' fresh scores when both can be scored. DeFoG sampling
and RDKit conversion run outside the differentiable objective.

[Pilot config](flow-pilot.json) · [Fresh score config](flow-fresh-dtu.json) ·
[Allocated launcher](run_flow_pilot.sh) ·
[Latent submission](../../../outputs/flow-guided-hallucination-20261105a/latent-submission.json) ·
[Repair submission](../../../outputs/flow-guided-hallucination-20261105a/repair-submission.json).
The submission receipts contain the exact commands, output paths, and snapshot
SHA-256. The existing workers check frozen models, saved inputs, and output
ownership.

## Completed results, 6 November 2026

Both jobs exited successfully. The latent job took 23.9 minutes and the repair
job took 3.5 minutes of worker wall time. The results below use the fresh
hard-graph Protenix contact loss.

### Learned proposals and search budget

Six of the 12 training starts supplied an improving endpoint above the
predeclared 0.005 threshold. All four held-out starts completed each arm. One
held-out starting molecule failed conformer embedding across all arms, leaving
three starts for the paired loss comparison. Its valid edits with scores remain
in the fresh results without a start-relative delta.

| Held-out seed | 7-update SGD | 24-update SGD | Residual + 7 | Flow + 7 |
| --- | ---: | ---: | ---: | ---: |
| 281089 | −0.14644 | −0.19307 | −0.13616 | −0.13616 |
| 281090 | −0.11321 | −0.14283 | +0.03449 | +0.03449 |
| 281091 | +0.23382 | +0.03764 | −0.10738 | −0.10738 |
| 281095 | — | — | — | — |

Entries show the best fresh loss delta among selected edits. At the same eight
relaxed scorer calls per start, including step zero, plain SGD found five
distinct improving molecules among nine scored edits. Each learned method found
the same three distinct improvements among ten scored edits. Both learned
proposals improved seed 281091 immediately at step zero, before refinement.
Both SGD budgets found gains on the same two starts, and the longer budget
found stronger edits on both. The four flow steps tied the single residual jump
on best observed score for all three comparable starts. This is a small
one-pocket pilot with six training pairs and a capped fresh candidate panel.

[Baseline scores](../../../outputs/flow-guided-hallucination-20261105a/results-20261106a/latent-fresh.json)
· [Training pairs and fit](../../../outputs/flow-guided-hallucination-20261105a/results-20261106a/student-fit.json)
· [Student scores](../../../outputs/flow-guided-hallucination-20261105a/results-20261106a/student-fresh.json).

### DeFoG repair

Both methods yielded connected molecules for all three invalid inputs.
Structural search used 1–3 label changes. DeFoG used 12–13 atom label changes
and 48–54 bond label changes across the six inputs, with 57 forwards per case.
Structural search took 6–15 ms per case on the CPU. The DeFoG worker took
17.5–18.5 s per case including setup and checkpoint loading. Fresh scoring was
separate from those timings.

DeFoG worsened the fresh contact loss on all three already-valid inputs by
0.112–0.421. On the invalid inputs, which have no start-relative fresh loss,
structural repair had the better output score on two of three cases. DeFoG won
at saved step 14 with loss 2.493 versus 2.603. This comparison used one draw per
graph, one denoising schedule, and six states from one trajectory.

[Repair results](../../../outputs/flow-guided-hallucination-20261105a/results-20261106a/repair.json)
· [Fresh scores](../../../outputs/flow-guided-hallucination-20261105a/results-20261106a/repair-fresh.json)
· [Download checksums](../../../outputs/flow-guided-hallucination-20261105a/results-20261106a/remote-sha256.txt).
