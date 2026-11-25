"""Render a standalone interpolation report from saved graphs only; no models."""
import argparse
import base64
import gzip
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def _image(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("A saved valid identity cannot be depicted")
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 180)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return "data:image/svg+xml;base64," + base64.b64encode(drawer.GetDrawingText().encode()).decode()


def render_interpolation_report(root):
    root = Path(root)
    summary = json.loads((root/"summary.json").read_text())
    pairs = json.loads((root/"pairs.json").read_text())["pairs"]
    data = dict(summary=summary, pairs=pairs, arms={})
    images = {}
    for arm in summary["arms"]:
        rows = []
        with gzip.open(root/arm/"raw-graphs.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["method"] != "argmax":
                    continue
                if row["valid"]:
                    if row["smiles"] not in images:
                        images[row["smiles"]] = _image(row["smiles"])
                    row["image"] = images[row["smiles"]]
                rows.append(row)
        data["arms"][arm] = rows
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    document = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Epoch 45 molecular interpolation</title><style>
body{font:15px system-ui;margin:24px;background:#f4f6f8;color:#172235}h1{font-size:26px}
table{border-collapse:collapse;background:white}td,th{padding:9px 14px;border:1px solid #dde2e8;text-align:left}
.points{display:grid;grid-template-columns:repeat(5,minmax(180px,1fr));gap:10px;overflow:auto}
.card{background:white;padding:12px;border-radius:8px;border-top:4px solid #36836b}.bad{border-color:#b55242}
img{width:100%;height:180px;object-fit:contain}pre{font-size:11px;max-height:200px;overflow:auto;white-space:pre-wrap}
small{color:#526071}select{padding:8px;max-width:90vw}.notice{max-width:1000px;line-height:1.5}
</style><h1>Frozen VAE interpolation: epoch 45</h1>
<p class="notice">Same-size chemical neighbors with explicit atom maps. Fractions are the weight of B:
25% means 75% A + 25% B. No training or repair. Canonical and matched inputs are encoded separately;
poor reordered-endpoint reconstruction can confound the matched path. Raw invalid graphs remain available below.</p>
<div id="totals"></div><h2>Inspect a pair</h2><select id="pair"></select><p id="info"></p><div id="paths"></div>
<p><small>Different from endpoints does not establish dataset novelty, synthesis feasibility or binding.
Summary JSON contains endpoint-conditioned results and uncertainty. No model was rerun to render this page.</small></p>
<script type="application/json" id="data">PAYLOAD</script><script>
const d=JSON.parse(document.getElementById('data').textContent), sel=document.getElementById('pair');
const table=document.createElement('table');
table.innerHTML='<tr><th>Encoding</th><th>B weight</th><th>Argmax valid</th><th>Categorical valid</th><th>Exact graph endpoints</th></tr>';
for(const [arm,s] of Object.entries(d.summary.arms)) for(const p of s.points.filter(x=>x.fraction>0&&x.fraction<1)){
 const tr=document.createElement('tr'); const a=p.views.all_pairs;
 for(const text of [arm,p.fraction,`${a.argmax.valid_count}/${a.argmax.attempts}`,
 `${a.categorical.valid_count}/${a.categorical.attempts}`,`${s.cohort_pair_counts.both_endpoints_graph_exact}/${d.pairs.length}`]){
 const td=document.createElement('td');td.textContent=text;tr.append(td)}table.append(tr)}
document.getElementById('totals').append(table);
for(const p of d.pairs){const opt=document.createElement('option');opt.value=p.pair_id;
opt.textContent=`Pair ${p.pair_id+1} — similarity ${p.tanimoto.toFixed(3)}, ${p.atoms} atoms`;sel.append(opt)}
function render(){const id=Number(sel.value),p=d.pairs[id];
document.getElementById('info').textContent=`A: ${p.a_smiles} | B: ${p.b_smiles} | Core: ${p.mapping.common_core_atoms}/${p.atoms} atoms | Identity map: ${p.mapping.identity_permutation}`;
const area=document.getElementById('paths');area.replaceChildren();
for(const [arm,rows] of Object.entries(d.arms)){
 const title=document.createElement('h2');title.textContent=arm+' encodings';area.append(title);
 const grid=document.createElement('div');grid.className='points';
 for(const r of rows.filter(x=>x.pair_id===id)){
 const card=document.createElement('section');card.className='card'+(r.valid?'':' bad');
 const label=document.createElement('strong');label.textContent=`${Math.round(r.fraction*100)}% B`;card.append(label);
 if(r.image){const img=document.createElement('img');img.src=r.image;img.alt=r.smiles;card.append(img)}
 const caption=document.createElement('p');caption.textContent=r.valid?r.smiles:`Invalid: ${r.classification.failure_kind} / ${r.classification.reason}`;card.append(caption);
 const details=document.createElement('details'),summary=document.createElement('summary'),raw=document.createElement('pre');
 summary.textContent='Raw labels and chemistry assessment';raw.textContent=JSON.stringify({graph:r.graph,classification:r.classification},null,2);
 details.append(summary,raw);card.append(details);grid.append(card)}area.append(grid)}}
sel.addEventListener('change',render);render();
</script></html>'''.replace("PAYLOAD", payload)
    with (root/"report.html").open("x", encoding="utf-8") as stream:
        stream.write(document)
    return root/"report.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saved_run", type=Path)
    print(render_interpolation_report(parser.parse_args().saved_run))
