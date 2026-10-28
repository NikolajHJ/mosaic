# Default reporting guidance, 28 October2026

The user requested that this agent and future agents treat the reusable report
command as the default. This is documentation-only work: no experiment, report
render, test suite, HPC job, cleanup or monitor action was run.

The root [agent instructions](../../../../../AGENTS.md) now require agents to
start with `python scripts/reporting/report.py` for supported saved trajectory
reports. Run-list/data-only changes reuse the existing qualified engine and cache;
agents should not create a renderer/controller/release or repeat fixture suites
for each refresh. Code/template/runtime-profile changes and unsupported schemas
still require the canonical owner and applicable validation. Original results,
scientific failures, source/proof identities, resource gates and old editions
retain their existing meaning.

The root README, documentation index, reporting family index, public command guide,
canonical owner and run index expose the same default. Named/path `--run` options
and small `--runs` JSON selections are documented. The current terminal JQ1 N25
fixed-step/noisy/chemistry, stride20 schema and account-bound profile limitations
remain explicit; other operators must use an authenticated profile for their own
account. Existing historical manual launch instructions are retained as history.

Validation is local standard-library documentation/link/byte checking only. The
existing [engine qualification and delivery](../20261028-reusable-report-01/README.md)
provide the186-case allocated CPU evidence. This guidance does not change the
qualified engine, runtime code, template, selection/profile or published report.
Exact validation outcome is recorded in `verification.json`; the repository-wide
ledger records the documentation preparation and completion. Unrelated working
changes remain outside this task.
