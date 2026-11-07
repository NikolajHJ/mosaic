# Exploratory research policy

Policy ID: `exploratory-research-v1`. Adopted by the user on 30 October 2026
for DBMol and Mosaic. Keep this policy synchronized in both repositories.
This is the default for new work; it supersedes blanket historical requirements
for qualification, exact replay and process overhead during early exploration.

## Default: get a useful early signal

Start with a small bounded experiment that answers one question. Use
`research_profile="exploratory"` (CLI `--research-profile exploratory`) in the
shared entry points that expose it. A short run does not need a qualification
campaign first. Small local CPU/unit/smoke tests are allowed. Use an appropriate
allocated device for substantive model runs; never run training on a login node.

Keep a short note with the question, command/config, inputs/model, code reference
or snapshot, budget, result and material warnings. Record the seed when useful;
exact replay, complete package inventories, per-file manifests, repeated native
predictions and a new release controller are optional for this phase. Snapshot
code used by a queued job so later edits do not change that job. Do not overwrite
another run's output. Reuse existing commands and make directories only as needed.
One experiment note is enough; generated folders need no README of their own.

| Observation | Exploratory behavior |
| --- | --- |
| Nonfinite objective/gradient, incompatible shapes, corrupt inputs, detached required live path | Stop with the concrete error. |
| Unexpected writes to a frozen model/input, output collision or duplicate writer | Stop the affected operation. |
| Flat derivative, noisy/inconclusive FD, floating replay variation | Record the scope; continue if the finite required live path is present. |
| Worse validity/reconstruction, no improvement, nonmonotone loss, surrogate/chemistry disagreement | Record the negative result and continue to the bounded endpoint; an explicit study stopping rule may override this. |
| Unrelated job lacks provenance or a global quota inventory is incomplete | Warn; assess this run's own bounded output against available space. |
| Measured insufficient capacity for this run, exceeded own output/resource budget | Stop or reduce the proposed budget before launch. |

Do not silently convert a measured material derivative contradiction into a pass.
Use the common [gradient policy](GRADIENT_VALIDATION_POLICY.md). A smoke pass says
only that the tested computation is finite and connected; it does not establish
native parity, precise derivatives, chemical quality or reproducibility.

## Qualified work is explicit

Select `research_profile="qualified"` when the task calls for stronger evidence,
reproduction, promotion, publication or a specific numerical diagnosis. Run the
checks relevant to that claim: independent mapping/native parity, AD/FD evidence,
frozen-state identity, repeatability, endpoint evaluation and regression controls.
Do not require every unrelated historical campaign's checks. Explicit legacy
contracts retain their meanings and their recorded outcomes.

## Existing work and collaboration

Leave running/pending jobs, callbacks, sealed source, live state, caches and
checkpoints unchanged when improving defaults. Adopting new code into an existing
campaign is a separate decision. Do not rerun, cancel or restart jobs for a policy
edit. Preserve failed evidence and distinguish partial/unknown outcomes from
completed experiments. A later changed policy does not make an old failure pass.

Read the current experiment note/config and relevant latest run record when
working on that experiment. Historical logbooks and archived agent notes are
context, not automatic requirements for every new task. Use reviewers and
parallel work when useful; a separate reviewer and documentation agent are not
mandatory for every small edit. Test the affected behavior, then move on.
