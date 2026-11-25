"""Assemble the offline Protenix report from a saved scheduler and run snapshot."""
import argparse
import base64
from functools import lru_cache
import gzip
from html import escape
import json
from pathlib import Path

from rdkit import Chem, rdBase

from structures import _molecule_structure, _raw_structure, build_structures


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parents[3] / "outputs" / "protenix-restart-report-20261107"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_payload(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def script_json(value):
    return json_payload(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def tree_index(trees):
    jobs = []
    for job in trees["jobs"]:
        available = [branch for branch in job["branches"] if branch.get("states")]
        jobs.append({key: job.get(key) for key in ("id", "task", "campaign", "label", "status", "reason")}
                    | dict(has_states=bool(available), has_fork=any(branch.get("source") for branch in available)))
    return dict(captured_at=trees.get("captured_at"), jobs=jobs)


def tree_chunks(trees, playback_structures):
    @lru_cache(maxsize=4096)
    def get_structure(identity):
        try:
            return _molecule_structure(identity)
        except (RuntimeError, ValueError):
            return None

    raw_graphs = trees.get("raw_graphs", {})
    table = Chem.GetPeriodicTable()
    with rdBase.BlockLogs():
        for job in trees["jobs"]:
            smiles, raw_keys = set(), set()
            for branch in job["branches"]:
                for row in branch["states"]:
                    identity = row[2] if row[1] else row[3]
                    if identity and identity not in playback_structures:
                        (smiles if row[1] else raw_keys).add(identity)
            if not any(branch.get("states") for branch in job["branches"]):
                continue
            payload = dict(job=job, structures={key: layout for key in sorted(smiles)
                                                if (layout := get_structure(key))},
                           raw_graphs={key: raw_graphs[key] for key in sorted(raw_keys) if key in raw_graphs})
            if trees.get("layout_invalid_states"):
                payload["structures"].update((key, _raw_structure(graph, table))
                                             for key, graph in payload["raw_graphs"].items())
            packed = gzip.compress(json_payload(payload).encode("utf-8"), compresslevel=5, mtime=0)
            yield job["id"], base64.b64encode(packed).decode("ascii")


def build(playback, tree_snapshot, output, *, include=(), exclude=()):
    assemble(read(playback), read(tree_snapshot), output, include=include, exclude=exclude)


def assemble(snapshot, trees, output, *, include=(), exclude=()):
    playback_jobs, tree_jobs = ({str(job["id"]) for job in data["jobs"]} for data in (snapshot, trees))
    if len(playback_jobs) != len(snapshot["jobs"]) or len(tree_jobs) != len(trees["jobs"]):
        raise ValueError("A snapshot contains duplicate job IDs")
    available = playback_jobs | tree_jobs
    unknown = (set(include) | set(exclude)) - available
    if unknown:
        raise ValueError(f"Job IDs absent from both snapshots: {sorted(unknown)}")
    selected = (set(include) or available) - set(exclude)
    if not selected:
        raise ValueError("Select at least one report run")
    unmatched = selected - (playback_jobs & tree_jobs)
    if unmatched:
        raise ValueError(f"Collect both snapshots for job IDs: {sorted(unmatched)}")
    for data in (snapshot, trees):
        data["jobs"] = [job for job in data["jobs"] if str(job["id"]) in selected]

    snapshot["structures"] = build_structures(snapshot)
    snapshot.pop("raw_graphs", None)
    frames = drawn = 0
    for job in snapshot["jobs"]:
        for branch in job["branches"]:
            for row in branch["points"]:
                frames += 1
                drawn += (row[5] if row[4] else row[7]) in snapshot["structures"]
    snapshot["structure_coverage"] = dict(frames=frames, drawn=drawn)
    script = "\n".join((ROOT / name).read_text(encoding="utf-8")
                       for name in ("tree-layout.js", "tree-chart.js", "tree-ui.js", "report.js"))
    template = (ROOT / "report-template.html").read_text(encoding="utf-8")
    before_snapshot, rest = template.split("__SNAPSHOT_JSON__", 1)
    before_chunks, after_chunks = rest.split("__TREE_CHUNKS__", 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    stream = temporary.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(before_snapshot)
            stream.write(script_json(snapshot))
            stream.write(before_chunks.replace("__TREE_INDEX__", script_json(tree_index(trees))))
            for job_id, payload in tree_chunks(trees, snapshot["structures"]):
                stream.write(f'<script type="application/octet-stream" id="tree-payload-{escape(str(job_id))}">')
                stream.write(payload)
                stream.write("</script>\n")
            stream.write(after_chunks.replace("__REPORT_JAVASCRIPT__", script))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"{output} · {output.stat().st_size:,} bytes · {len(selected)} runs")
    print(f"Playback coverage {snapshot['structure_coverage']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--playback", type=Path, help="Playback snapshot file")
    parser.add_argument("--trees", type=Path, help="Tree snapshot file")
    parser.add_argument("--output", type=Path, help="Output HTML file")
    parser.add_argument("--job", action="append", default=[], help="Include this job ID, repeat for more jobs")
    parser.add_argument("--exclude-job", action="append", default=[], help="Omit this job ID")
    args = parser.parse_args()
    build(args.playback or args.data_dir / "playback-snapshot.json",
          args.trees or args.data_dir / "tree-snapshot.json",
          args.output or args.data_dir / "protenix-restart-report.html",
          include=args.job, exclude=args.exclude_job)


if __name__ == "__main__":
    main()
