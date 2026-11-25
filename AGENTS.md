# Active agent instructions

The user's current preference is **fast exploratory research**. Apply the
[common exploratory policy](docs/EXPLORATORY_RESEARCH_POLICY.md) and
[gradient policy](docs/GRADIENT_VALIDATION_POLICY.md) to new work. Use exploratory
defaults; make rigorous qualification explicit. Historical campaign rules do not
become universal prerequisites. Small local CPU/unit/smoke tests are allowed.
Keep tests and documentation proportional to the changed behavior.

Keep checks for nonfinite computations, incompatible inputs/shapes, detached
required paths, unintended frozen-model writes, output collisions and duplicate
writers. Quality drops, flat derivatives, inconclusive FD and nonmonotone loss
are usually findings to record, not reasons to abort a bounded pilot. Do not
claim numerical/native correctness beyond the evidence actually collected.

Do not modify running/pending jobs, callbacks, sealed releases, live states,
caches or checkpoints while changing defaults. Preserve other working-tree edits.
Keep a short experiment note (question, command/config, budget, result/warnings);
update the [ledger](docs/05_experiment_ledger.md) for actual experiment outcomes. Do not require
full manifests, package inventories, repeated native gates, a README in every
generated directory, or separate reviewer/documentation agents for small work.
Read only the relevant current experiment/config/run record before changing it.

Keep AE/VAE comparisons on the shared graph schema and report differences in
sampling, loss, initialization and compute. Keep chemistry conversion outside
the differentiable scorer path. External frozen-model branches should consume
the same intended graph; small bridges must not silently change the objective.
For an existing VAE/mapping continuation, read its actual current run/config;
[docs80](docs/80_epoch40_to45_continuation.md),
[docs87](docs/87_v100_continuation_handoff.md) and
[docs88](docs/88_local_neighbor_loss_pilot.md) describe specific campaigns.
The failed neighbor-weight0.1 pilot is not adopted by this policy change.

## Keep organization simple

Start at the [README](README.md), [workflow guide](docs/workflows.md),
[package map](src/latentmol/README.md) and [experiment index](experiments/README.md).
Reusable code belongs in the existing model/training/dataset/evaluation/guidance/
design packages; new code imports the grouped paths. Old flat modules that forward
to grouped modules are compatibility aliases: edit the single implementation.
Existing campaign modules/launchers keep their recorded paths; do not move their
sealed sources, live dependencies or assets as part of routine cleanup.

Group shared tests by subsystem and experiment-only tests with their question.
Related arms/seeds/budgets share an experiment home and config variants, not
copied runners or a new controller per retry. Keep a short current question/
command/result note; numbered docs and the long ledger are dated context.
Put generated artifacts in ignored outputs and expose small useful result links
from the experiment note. Do not add a layout framework, extra README per output
folder, or migration aliases for new files. Update current navigation when adding
a workflow; preserve historical paths/results instead of rewriting them.

## HPC transport on Nikolajs_laptop

Before HPC operations read the [current connection guide](docs/HPC_CONNECTION.md).
Use the installed unattended connector, account `s234818`, pinned login1 host:

```powershell
$dtu = 'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
& $dtu exec "bash -lc 'bjobs -u s234818'"
```

`exec` takes one remote-command string; `exec --timeout SECONDS` controls timeout.
Use its `upload`/`download` commands for transfers. Do not routinely ask for VPN,
passwords or reconnection before trying it. Never expose credentials. Reconcile
scheduler/receipt evidence before retrying a lost submission; connection failure
does not prove job failure. Other operators use their own authorized setup.
Use allocated resources for substantive HPC model work, not the login node.

## Historical context

The [previous agent file](docs/history/AGENTS-20261030-before-exploration.md) is an exact
archive of dated instructions/status. Consult a campaign's relevant history when
needed; do not load it as mandatory current policy. Explicit frozen contracts
retain their original meaning. This edit starts no experiment and authorizes
no restart, cancellation or adoption into a live campaign.
