"""Record an existing readonly scheduler/prefix observation; stdlib metadata only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARTIFACTS = REPO / "results/hpc" / HERE.name


def main():
    scheduler_path = ARTIFACTS / "handoff-20261023T091848Z.json"
    prefix_path = ARTIFACTS / "chemistry-entry-observation-01.json"
    scheduler = json.loads(scheduler_path.read_bytes())
    prefix = json.loads(prefix_path.read_bytes())
    assert scheduler["returncode"] == 0
    assert "29792672 RUN gpul40s" in scheduler["scheduler"]
    assert prefix["submission.json"]["job_id"] == "29792672"
    assert prefix["entry-quota/lrreset-storage-gate.json"]["status"] == "passed"
    assert prefix["states.jsonl"]["committed_rows"] == 62
    assert prefix["updates.jsonl"]["committed_rows"] == 61
    evidence = {
        "status": "scientific_updates_observed",
        "job_id": "29792672",
        "scheduler_observation_utc": scheduler["utc"],
        "prefix_local_receipt_mtime_utc": datetime.fromtimestamp(
            prefix_path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "prefix_remote_capture_utc": None,
        "scheduler": scheduler,
        "saved_metadata": prefix,
        "evidence_bytes": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (scheduler_path, prefix_path)
        },
        "scope": (
            "Readonly metadata: actual allocation, passed entry gate, driver invocation "
            "and 61 newline-committed update rows. No terminal pass, full trajectory "
            "audit, scientific calculation or original-output change by this recorder."
        ),
        "published_edition_unchanged": "edition-0001",
        "future_started_chemistry_report_requires_new_immutable_entry_pins": True,
        "diagnostic_findings": None,
        "random_protenix_native_status": "PEND",
        "random_protenix_long_id": None,
    }
    target = HERE / "chemistry-start-observation.json"
    assert not target.exists(), "Do not overwrite an existing observation"
    target.write_bytes((json.dumps(evidence, indent=2) + "\n").encode())
    campaign_path = HERE / "campaign.json"
    campaign = json.loads(campaign_path.read_bytes())
    campaign["latest_chemistry_observation"] = {
        "job_id": "29792672",
        "scheduler_utc": scheduler["utc"],
        "status": "RUN",
        "entry_quota": "passed",
        "committed_update_rows": 61,
        "committed_state_rows": 62,
        "observation_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "published_capture_unchanged": True,
    }
    campaign_path.write_bytes((json.dumps(campaign, indent=2) + "\n").encode())
    text = """

## Original chemistry started, 23 October 11:18–11:19 Paris

Readonly scheduler observation at09:18:48.777054UTC shows original chemistry
29792672 RUN/gpul40s. Its separate saved metadata receipt, locally collected at
09:19UTC, authenticates the same submission, passed entry quota, driver invocation,
62 newline-committed state rows and61 update rows. This is actual optimization
progress, not a terminal validation or a new numerical analysis. Exact path/length/
SHA bindings are preserved in chemistry-start-observation.json. Original run,
settings and saved results were not changed by the analysis.

Published edition0001 remains the truthful earlier08:48UTC capture with chemistry
unstarted; it is immutable. Future chemistry publication requires a new immutable
source containing its actual entry/driver pins. Existing diagnostic29796551 and
capture29796552 remain PEND, with no cause established. New random Pro native
29796780/callback29796781 remain PEND; no optimization long ID exists yet. No
duplicate submission, cleanup, original-run restart or heartbeat unpause occurred.
"""
    with (HERE / "README.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    with (REPO / "docs/experiment_ledger.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace("## Original chemistry started,", "### Original chemistry started,"))
    print("Recorded existing chemistry entry/progress evidence; no SSH or numerical execution.")


if __name__ == "__main__":
    main()
