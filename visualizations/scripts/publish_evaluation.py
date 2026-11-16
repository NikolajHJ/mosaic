"""Publish a saved scoring comparison and the maintained viewer assets."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def publish(report, library):
    data = json.loads(report.read_text())
    if (
        data.get("schema_version") != 1
        or not data.get("records")
        or not any(r["id"] == "C25" for r in data["records"])
    ):
        raise ValueError("Expected a saved scoring comparison with C25")
    if not (library / "index.json").exists():
        raise ValueError("Use an existing run library")
    tmp = library / "evaluation.json.tmp"
    tmp.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")))
    tmp.replace(library / "evaluation.json")
    # Molecule entries are direct shortcuts to saved results, not trajectories.
    index = json.loads((library / "index.json").read_text())
    curation = (
        json.loads((library / "curation.json").read_text())
        if (library / "curation.json").exists()
        else {}
    )
    entries = {row["id"]: row for row in index["runs"]}
    for molecule in data["records"]:
        rid = hashlib.sha256(
            f"molecule:{report.resolve()}:{molecule['id']}".encode()
        ).hexdigest()[:20]
        g = molecule["gnina_summary"]
        description = f"Saved scores and poses for {molecule['id']}. GNINA mean Vina {g['minimizedAffinity']['mean']}, CNN affinity {g['CNNaffinity']['mean']}. Source optimization steps are linked in the comparison."
        meta = {
            "id": rid,
            "kind": "molecule",
            "molecule_id": molecule["id"],
            "name": molecule["id"].replace("mol-", "Mol-"),
            "source": str(report.resolve()),
            "frames": 0,
            "description": description,
            "backend": "GNINA + native Boltz",
            "mode": "scored_molecule",
            "complete": bool(data["complete"] and data.get("guided_complete")),
        }
        entries[rid] = meta
        if not curation.get(rid, {}).get("deleted"):
            folder = library / "runs" / rid
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "meta.json").write_text(json.dumps(meta, allow_nan=False))
    index["runs"] = list(entries.values())
    pending = library / "index.json.tmp"
    pending.write_text(json.dumps(index, allow_nan=False))
    pending.replace(library / "index.json")
    shutil.copy2(HERE / "template.html", library / "index.html")
    for name in ("viewer.js", "style.css"):
        shutil.copy2(HERE / "assets" / name, library / name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("library", type=Path)
    a = p.parse_args()
    publish(a.report, a.library)
