"""Build an offline overview next to the epoch-48 search-tree explorer."""
import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


HERE = Path(__file__).resolve().parent


def drawing(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol, canonOrient=True)
    img = Draw.MolToImage(mol, size=(680, 330))
    buf = BytesIO()
    img.save(buf, "WEBP", lossless=True, method=3)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build(data_path, output):
    data = json.loads(data_path.read_text(encoding="utf-8"))
    molecules = list(data["m02"]["molecules"])
    for arm in data["broad"]["arms"]:
        molecules.extend(arm.get("top", []))
    for arm in data["erk2"]["arms"]:
        molecules.extend(arm["founders"])
        molecules.extend(arm["observations"])
    smiles = sorted({row["smiles"] for row in molecules if row.get("smiles")})
    data["drawings"] = {s: picture for s in smiles if (picture := drawing(s))}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    html = (HERE / "epoch48-overview.html").read_text(encoding="utf-8")
    script = (HERE / "epoch48-overview.js").read_text(encoding="utf-8")
    html = html.replace("/*REPORT_SCRIPT*/", script).replace("/*REPORT_DATA*/", payload.replace("<", "\\u003c"))
    output.write_text(html, encoding="utf-8")
    print(f"{output}: {len(molecules)} molecule records, {len(data['drawings'])}/{len(smiles)} drawings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.data, args.output)
