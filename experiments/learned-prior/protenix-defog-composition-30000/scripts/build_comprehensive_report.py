"""Build the self-contained DeFoG report from saved records and sibling checkouts."""
import argparse
import base64
import gzip
import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D


def pack(data):
    return base64.b64encode(gzip.compress(data, compresslevel=6, mtime=0)).decode("ascii")


def source_notes(mosaic, dbmol, experiment):
    files = {
        "DBMol · stochastic decoding and projection protocol": dbmol / "docs/18_stochastic_defog_projection_protocol.md",
        "DBMol · sampled repair targets and charge compatibility": dbmol / "docs/37_defog_repair_guidance.md",
        "DBMol · integrated six-update smoke": dbmol / "docs/49_defog_minimal_integration_smoke.md",
        "DBMol · why sulfur accumulated": dbmol / "docs/84_sulfur_bias_in_validity_guidance.md",
        "DBMol · native repair and integration": dbmol / "docs/48_defog_comprehensive_diagnosis.md",
        "DBMol · random starts": dbmol / "docs/47_defog_long_run_diagnostics.md",
        "DBMol · matched pilot": dbmol / "docs/51_defog_matched_pilot.md",
        "DBMol · longer Boltz comparison": dbmol / "docs/60_long_boltz_defog_comparison.md",
        "Checkpoint and locality tests": mosaic / "experiments/learned-prior/defog-soft-teacher/RESULTS.md",
        "First 1,000-update pilot": mosaic / "experiments/learned-prior/protenix-defog-1000/RESULTS.md",
        "Fixed-time 20k attempt": mosaic / "experiments/learned-prior/protenix-defog-20000/runs/20261028-protenix-defog-20000-01/README.md",
        "Dense 30k time ramp": mosaic / "experiments/learned-prior/protenix-defog-30000-time-ramp/runs/20261028-protenix-defog-30000-time-ramp-01/README.md",
        "Balanced gradients and 30k outcome": mosaic / "experiments/learned-prior/protenix-defog-balanced-30000/runs/20261029-protenix-defog-balanced-30000-01/README.md",
        "Sulfur gradient attribution": mosaic / "experiments/learned-prior/protenix-defog-balanced-30000/runs/20261029-sulfur-attribution-01/README.md",
        "Attached teacher attempt": mosaic / "experiments/learned-prior/protenix-defog-attached-30000/runs/20261030-attached-02/README.md",
        "Composition and solver checks": mosaic / "experiments/learned-prior/protenix-defog-composition-30000/README.md",
        "Five sulfur pilots and 30k follow-up": mosaic / "experiments/learned-prior/protenix-defog-composition-30000/runs/20261101-sulfur-exploratory-02/README.md",
        "Boltz, Protenix, schedules and projection": mosaic / "experiments/learned-prior/protenix-defog-composition-30000/stage2-pair.md",
        "Extracted DeFoG package benchmarks": experiment.parents[2] / "models/defog/BENCHMARKS.md",
        "Stage 3 design": experiment / "README.md",
        "Stage 3 observations and recovery": experiment / "run-20261113.md",
    }
    return {label: {"path": str(path.relative_to(dbmol if dbmol in path.parents else mosaic if mosaic in path.parents else experiment.parents[2])).replace(chr(92), "/"), "text": path.read_text(encoding="utf-8-sig")} for label, path in files.items()}


def history(mosaic):
    hpc = mosaic / "results/hpc"
    datasets = {}
    for path in sorted(hpc.glob("*defog*/comparison.json")):
        datasets[str(path.relative_to(mosaic)).replace(chr(92), "/")] = json.loads(path.read_text(encoding="utf-8-sig"))
    for pattern in ("*lr70-starts*/recap*.json", "*sulfur-exploratory-02/pilot-outcomes*.json", "*defog*/completed*summary*.json"):
        for path in sorted(hpc.glob(pattern)):
            datasets[str(path.relative_to(mosaic)).replace(chr(92), "/")] = json.loads(path.read_text(encoding="utf-8-sig"))
    return datasets


def structure_library(runs):
    smiles = sorted({m[0] for r in runs for m in r["molecules"]})
    library = {}
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        drawer = rdMolDraw2D.MolDraw2DSVG(470, 205)
        drawer.drawOptions().clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        library[s] = pack(svg[svg.index("<svg"):].encode())
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "mosaic", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--dbmol", type=Path,
                        help="Optional historical source override. The archived notes are used by default.")
    parser.add_argument("--structure-cache", type=Path)
    args = parser.parse_args()
    exp = Path(__file__).resolve().parents[1]
    data = json.loads(gzip.decompress(args.data.read_bytes()))
    data["history"] = history(args.mosaic)
    data["sources"] = source_notes(args.mosaic, args.dbmol or exp / "historical-dbmol", exp)
    data["benchmark"] = json.loads((exp.parents[2]/"models/defog/benchmarks.json").read_text())
    previous = ("DeFoG-results-Mosaic-2026-11-05", "DeFoG-cyclic-10k-Mosaic-2026-11-06",
                "DeFoG-high-lr-Mosaic-2026-11-07", "DeFoG-cyclic-10k-LR150-Mosaic-2026-11-07")
    data["archive"] = {name: pack((args.mosaic/"results/reports"/(name+".html")).read_bytes())
                       for name in previous}
    data["depiction_version"] = rdBase.rdkitVersion
    structures = json.loads(args.structure_cache.read_text()) if args.structure_cache else structure_library(data["runs"])
    assert set(structures) == {m[0] for r in data["runs"] for m in r["molecules"]}
    template = Path(__file__).with_name("comprehensive_report.html").read_text(encoding="utf-8")
    scripts = ["comprehensive_report.js", "comprehensive_report_runs.js",
               "comprehensive_report_history.js", "comprehensive_report_app.js"]
    template = template.replace("/* REPORT_SCRIPT */", "\n".join(
        Path(__file__).with_name(name).read_text(encoding="utf-8") for name in scripts))
    payload = pack(json.dumps(data, separators=(",", ":"), allow_nan=False).encode())
    output = template.replace("__DATA_GZIP__",payload).replace("__MOLECULES_JSON__",json.dumps(structures,separators=(",", ":")).replace("<", "\\u003c"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")
    print(f"{len(data['runs'])} runs, {len(structures)} distinct 2D structures, {args.out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
