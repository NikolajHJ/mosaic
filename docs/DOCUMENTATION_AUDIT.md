# Documentation accuracy and portability review

Reviewed 22 October 2026. The initial scan read **584 documentation files and
attachments**, including dated receipt text and the imported OneNote material.
The [per-file audit](documentation-audit.json) records the final file inventory,
read scope, local references and account assumptions. This is a source/document
review, not a new model experiment or a live permission/scheduler assessment.

## Findings and corrections

| Finding | Resolution |
| --- | --- |
| Setup guides presented a contributor's private work, home, runtime and worker locations as project defaults. | Added [an account-independent guide](HPC_GUIDE.md), rewrote the maintained Codex guide and made the older named guide a compatibility pointer. Private historical paths remain clearly scoped evidence. |
| The two Codex/HPC guides duplicated the same account-specific recipe. | One maintained integration guide now serves all readers; the original workstation description is preserved under [history/](history/README.md). |
| Account portability was implied by copied commands, but many source files still pin account paths. | Inventoried **104 canonical source files** with concrete account references and added explicit portability notes to all 24 experiment homes. Environment-variable examples do not override controller invariants. |
| The TeX runbook used Python 3.10 examples despite the repository requiring Python 3.12.x. | Corrected examples to require an operator-selected available Python 3.12 module. Distinguished full project dependency resolution from pinned experiment environments. |
| The runbook assumed a frozen lock and showed an obsolete repository tree. | Documented that `uv.lock` is ignored and replaced the tree with canonical experiment/shared-tool homes. |
| Fixed queue/hardware counts, queue-wait preferences and generic CUDA minimums looked like current guarantees. | Replaced them with actual-system discovery and the selected model/runtime contract; no live queue or module claim was made. |
| Generic cleanup/reset and in-place normalization recipes could discard scientific evidence. | Replaced broad destructive examples with inspection and scoped-change guidance; immutable releases and authenticated bytes stay unchanged. |
| Stage 0 future commands assumed the original operator's interpreter/cache/output roots and omitted the new source directory from verifier coverage. | Made caller settings explicit, linked canonical entry points and documented `experiments/` authentication plus the remaining hard-coded launcher root. |
| The quota helper's docstring described a pool default, while its CLI requires `--storage-pool`. | Corrected that docstring only; the required argument, quota arithmetic and failure behavior are unchanged. |
| The original PDF and dated implementation paragraphs could be mistaken for current implementation status. | Added a [brief-to-source scope map](project_reference_status.md), date/scoping notes on protocol documents, and clarified the old Stage 0 static receipt. |
| Imported lab notes include named-user paths and suggested storage allowances. | Kept original notes/attachments intact and labeled the export as a reference snapshot, not an account-independent Mosaic runbook or externally validated policy. |

## Source checks and limits

The review checked `pyproject.toml`, canonical script ownership, launcher/runtime
constants, CLI declarations, Stage 0 config/fixture values, source-verifier
coverage and the source tree against the guidance. It checked local Markdown
targets and distinguished real repository files from relative experiment paths
and explicitly illustrative templates. Older references to compatibility entry
points remain valid; they do not change the canonical owner.

The original project PDF and supported imported PDFs/Office attachments were
text-extracted. Extraction does not establish visual fidelity or validate lab,
HR, legal, grant, financial or other institutional policy. Images, ink and
embedded/OCR-only objects retain the export report's limitations. External
citations were checked selectively for the maintained access/runtime guidance,
not exhaustively revalidated across historical evidence.

Official sources used for maintained guidance include [DTU storage](https://www.hpc.dtu.dk/?page_id=927),
[DTU access](https://www.hpc.dtu.dk/?page_id=2501),
[DTU's OS migration notice](https://www.hpc.dtu.dk/?page_id=5209), and official
OpenAI [environment](https://learn.chatgpt.com/docs/environments/modes) and
[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) guidance.
The old DTU access page's OS description conflicts with its newer migration
notice; the guide states this and asks readers to inspect the actual host.

## Remaining engineering work

The **104 account-referencing canonical source files are not fully portable**.
Preparing them for another operator is a separate source/configuration effort:
some paths can be supplied through real CLI options, while controllers pin roots,
proofs and runtime identities. A new account needs authorized inputs/assets and
fresh required validation. This documentation change neither edits those
controllers nor claims their current certificates apply elsewhere.

No HPC connection, remote ACL change, permission widening, allocation,
submission, cancellation, model import or numerical test was performed.
Historical receipts, scientific configs, model/optimizer code and current sealed
releases were not modified by this review. The shared quota helper has a docstring-only edit;
its executable AST is checked unchanged. Earlier records and imported source
notes remain preserved.

## Repeat the mechanical review

From the repository root:

```powershell
py -3.13 scripts/layout/audit_documentation.py --root . --output results/static/documentation-audit.json
```

The command reads text/attachments and checks local references. It does not
execute documented commands or connect to remote systems. Current reusable
guides fail on concrete account defaults; historical/imported findings remain
explicit inventory entries. Detailed before/after audit and edit receipts are
under ignored `results/static/20261022-documentation-audit/`. Actual checks and
compilation outcomes are appended to the [global logbook](experiment_ledger.md).
