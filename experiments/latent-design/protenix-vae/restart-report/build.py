"""Assemble the offline Protenix report from a saved scheduler and run snapshot."""
import argparse
import base64
import gzip
import json
from pathlib import Path

from rdkit import rdBase

from structures import _molecule_structure, build_structures


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parents[3] / "outputs" / "protenix-restart-report-20261107"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_payload(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def build(playback, tree_snapshot, output, *, include=(), exclude=()):
    snapshot, trees = read(playback), read(tree_snapshot)
    playback_jobs, tree_jobs = ({str(job["id"]) for job in data["jobs"]} for data in (snapshot, trees))
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
    frames = [row for job in snapshot["jobs"] for branch in job["branches"] for row in branch["points"]]
    snapshot["structure_coverage"] = {
        "frames": len(frames),
        "drawn": sum((row[5] if row[4] else row[7]) in snapshot["structures"] for row in frames),
    }
    data = json_payload(snapshot).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    tree_states = [row for job in trees["jobs"] for branch in job["branches"] for row in branch["states"]]
    smiles = {row[2] for row in tree_states if row[1] and row[2] and row[2] not in snapshot["structures"]}
    raw_keys = {row[3] for row in tree_states if not row[1] and row[3]}
    trees["raw_graphs"] = {key: graph for key, graph in trees.get("raw_graphs", {}).items() if key in raw_keys}
    trees["structures"] = {}
    with rdBase.BlockLogs():
        for identity in sorted(smiles):
            try:
                structure = _molecule_structure(identity)
            except (RuntimeError, ValueError):
                continue
            if structure:
                trees["structures"][identity] = structure
    tree_bytes = json_payload(trees).encode("utf-8")
    tree_payload = base64.b64encode(gzip.compress(tree_bytes, compresslevel=5, mtime=0)).decode("ascii")
    script = "\n".join((ROOT / name).read_text(encoding="utf-8")
                       for name in ("tree-layout.js", "tree-chart.js", "tree-ui.js", "report.js"))
    template = (ROOT / "report-template.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    stream = temporary.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(template.replace("__SNAPSHOT_JSON__", data)
                         .replace("__TREE_PAYLOAD__", tree_payload)
                         .replace("__REPORT_JAVASCRIPT__", script))
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
