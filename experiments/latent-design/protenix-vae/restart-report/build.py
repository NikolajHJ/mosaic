"""Assemble the offline Protenix report from a saved scheduler and run snapshot."""
import argparse
import base64
import gzip
import json
from pathlib import Path

from rdkit import rdBase

from structures import _molecule_structure, build_structures


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=root.parents[3] / "outputs" / "protenix-restart-report-20261107")
data_dir = parser.parse_args().data_dir
snapshot = json.loads((data_dir / "playback-snapshot.json").read_text(encoding="utf-8"))
snapshot["structures"] = build_structures(snapshot)
snapshot.pop("raw_graphs", None)
frames = [row for job in snapshot["jobs"] for branch in job["branches"] for row in branch["points"]]
snapshot["structure_coverage"] = {
    "frames": len(frames),
    "drawn": sum((row[5] if row[4] else row[7]) in snapshot["structures"] for row in frames),
}
data = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
trees = json.loads((data_dir / "tree-snapshot.json").read_text(encoding="utf-8"))
smiles = {
    row[2] for job in trees["jobs"] for branch in job["branches"] for row in branch["states"]
    if row[1] and row[2] and row[2] not in snapshot["structures"]
}
trees["structures"] = {}
with rdBase.BlockLogs():
    for identity in sorted(smiles):
        try:
            structure = _molecule_structure(identity)
        except (RuntimeError, ValueError):
            continue
        if structure:
            trees["structures"][identity] = structure
tree_json = json.dumps(trees, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
tree_payload = base64.b64encode(gzip.compress(tree_json, compresslevel=5, mtime=0)).decode("ascii")
script = "\n".join((root / name).read_text(encoding="utf-8") for name in ("tree-layout.js", "tree-ui.js", "report.js"))
template = (root / "report-template.html").read_text(encoding="utf-8")
output = data_dir / "protenix-restart-report.html"
temporary = output.with_suffix(".tmp")
temporary.write_text(template.replace("__SNAPSHOT_JSON__", data)
                     .replace("__TREE_PAYLOAD__", tree_payload)
                     .replace("__REPORT_JAVASCRIPT__", script), encoding="utf-8")
temporary.replace(output)
print(f"{output} · {output.stat().st_size:,} bytes · {len(snapshot['structures'])} structures")
print(f"Playback coverage {snapshot['structure_coverage']}")
