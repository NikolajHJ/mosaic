# <Experiment title>

- Experiment ID: `<family>/<experiment-id>`
- Status: `prepared | submitted | running | completed | failed | cancelled | partial | blocked | unknown`
- Last observed: `<ISO 8601 timestamp with UTC/offset>`
- Owner / most recent agent: `<name or chat/task identifier>`

## Scientific question and lineage

<Question/hypothesis, comparison arms, parent experiments and what this study
can or cannot establish. Link the parent protocol and relevant ledger entries.>

## Protocol and authorized scope

<Models/versions/hashes, data/protein/pocket/context, occupancy/representation,
initialization and seeds, objective/weights/schedule, optimizer/acceptance rule,
update budget, frozen/trainable components and controls.>

<User-authorized actions and limits, execution location, environment/interpreter,
allocation/runtime/quota requirements, readiness/native/scientific acceptance
criteria and conditions that stop the experiment. Preparation is not execution.>

## File map

| Owned file or subtree | Purpose and entry point |
| --- | --- |
| `README.md` | Experiment protocol, status and navigation |
| `<relative path>` | `<Role; enumerate files or link the subtree README>` |

| Shared, historical or external dependency | Exact location and identity | Why outside this folder |
| --- | --- | --- |
| `<name>` | `<Path/link and version/hash>` | `<Single shared owner, immutable history or large artifact storage>` |

<Include configs, fixtures, tests, launchers, collectors, analysis and templates.
Record compatibility exceptions and old-to-new paths here when migrating.>

## Commands and execution location

- Working directory: `<explicit path>`
- Interpreter/environment: `<exact executable/environment identity>`
- Run only on: `<authorized host/allocation; state local static-only limits>`

```text
<exact preparation command>
<exact validation/readiness command>
<exact launch command using a fresh run ID>
<exact collection and report commands>
```

<Identify dependencies, required arguments/inputs, source sealing and expected
outputs. Label commands not yet executed and historical commands explicitly.>

## Runs and current findings

| Run ID / record | Stage/arm | Submitted / started / ended | Execution state | Scientific acceptance | Evidence |
| --- | --- | --- | --- | --- | --- |
| `<link to runs/<run-id>/README.md>` | `<stage>` | `<actual timestamps or not observed>` | `<state>` | `<passed/failed/not run/pending>` | `<receipts and artifacts>` |

<Dated finding with evidence; failures, partial/unreached work, limitations and
next action. Link the repository-wide `docs/experiment_ledger.md` using the
correct relative path. A fresh scaffold has no executed runs or results.>
