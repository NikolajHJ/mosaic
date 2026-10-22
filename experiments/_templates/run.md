# <Run ID>: <stage / purpose>

- Experiment: `<ID and link to experiment README>`
- Actor / agent: `<name or chat/task identifier>`
- Parent run / campaign / arm: `<link or not applicable>`
- Authorized scope: `<reference and limits>`

## Timeline and state

| Event | Actual timestamp (UTC or explicit offset) | Observation time / evidence |
| --- | --- | --- |
| Prepared | `<timestamp>` | `<record>` |
| Submitted | `<timestamp or not submitted>` | `<submission receipt>` |
| Started | `<timestamp or not observed>` | `<scheduler/log evidence>` |
| Ended | `<timestamp or not observed>` | `<terminal evidence>` |

- Execution state: `<prepared/submitted/running/completed/failed/cancelled/partial/blocked/unknown>`
- Scheduler state / exit code: `<actual or unknown>`
- Scientific acceptance / required gates: `<passed/failed/pending/not run; evidence>`
- Report / delivery status: `<independent state or not applicable>`

## Source, input and environment identity

- Commit and working-tree/source changes: `<reference; commit alone is insufficient>`.
- Immutable source archive and manifest paths/hashes: `<identities or not sealed>`.
- Config paths/hashes, input/model/context identities: `<identities>`.
- Initialization, seed/RNG and parent checkpoint/optimizer: `<identities or not applicable>`.
- Interpreter, environment/lock, host and resources: `<identities and requested/actual allocation>`.
- Job IDs, dependency chain and child arms: `<IDs or not submitted>`.

## Exact commands

- Working directory: `<path>`
- Execution location: `<authorized allocation or local static-only work>`

```text
<exact command/argv, including configuration and output paths>
```

<Distinguish planned commands from actually executed commands. Link saved
stdout/stderr and submission/terminal receipts; never include credentials.>

## Evidence and outcome

| Artifact / receipt | Local or external location | Integrity identity / scope |
| --- | --- | --- |
| `<source, submission, log, validation, output or collection receipt>` | `<path/link>` | `<hash and what was checked>` |

<What ran, observed outcome, actual checks, failures/cancellations/unknowns,
skipped and unreached work, interpretation limits and next action. Preserve
earlier attempts. State "no experiment executed" for documentation-only work.>

## Global ledger event skeleton

Append a dated event to `docs/experiment_ledger.md` at each required transition:

```markdown
## <timestamp with offset> — <experiment ID> — <run ID> — <event/state>

Actor: <agent>. Experiment folder: <link>. Run record: <link>.
Action and exact command/working directory: <command or link>.
Source/config/input identity: <hashes or linked receipt>.
Jobs, resources and actual submitted/start/end times: <facts or unknown>.
Outcome and required-gate status: <execution vs scientific vs report state>.
Evidence: <receipt/artifact links>. Limitations / next action: <facts>.
```
