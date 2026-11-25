const explanatory=[
 ["What the loss does","Each update starts from atom and bond probabilities on the same 25 graph slots. In the default Stage 3 arm, the frozen DeFoG checkpoint makes one clean-graph prediction at the selected teacher time. Hard argmax auxiliary features and soft transformer inputs follow the checkpoint compatibility pilot. The optimizer minimizes negative Protenix contact plus the weighted mean-squared discrepancy in atom and bond probabilities. The teacher output is detached and refreshed on the next update. Projection operates on the graph after the optimizer step. Teacher time, guidance weight, and softmax temperature can be changed independently."],
 ["How a candidate qualifies","The primary result is the highest hard Protenix-Tiny BRD4 pocket contact among raw argmax graphs that pass RDKit sanitization and have one connected component. The initial graph is eligible. A new identity has a canonical SMILES different from the starting graph. Higher contact is better on this model proxy. No binding assay was run. A run can find and later lose a valid molecule. The optional endpoint repair searches have no role in this ranking."],
 ["What was matched","Stage 3 used native coefficient 1, atom and bond learning rates 60, 500 updates, and seeds 17, 23 and 42. Each seed produced a sparse valid tree, a dense bond initialization with the same atom probabilities, and the tree with four bonds removed. The Protenix key stayed fixed within seed. The control lookup uses the teacher-off setting specified for each treatment. Several conditions reuse a control. GPU family differed in some pairs after queue migration. The three seed blocks limit generalization and the many comparisons are exploratory."],
 ["How to read missing results","Of 1,395 original runs, 1,390 completed and five stopped early on the projection solver check. The six recovery runs comprise those five settings plus a repeat of the matched weight-0.5 treatment. All completed, five found a new valid molecule, and none ended valid. Every planned seed/start/setting now has a complete attempt. Default condition rankings use original full runs; score means and wins require observed scores. Missing scores remain missing. The recovery comparison uses both rerun arms for the ninth pair. No recovery attempt triggered the solver fallback, leaving its behavior on the original failing states unverified."],
 ["How to read the score gap","Soft minus hard contact is signed. A negative mean indicates higher hard contact and can mask large deviations of either sign. For the eight original complete weight-0.5 pairs, only 39 updates were valid in both arms at the same update. Averaging those common-valid gaps within each pair and then across pairs gives −0.062 guided and −0.045 control. The LR 0.3 matched 2k arm also increased the positive mean gap. These experiments provide mixed evidence for closer relaxed and discrete scores."],
 ["What is archived","Run-level counts, best scores and rankings use every saved update. The offline graph traces retain all updates for the main control, weight 0.5, and the six recovery attempts. Other traces retain every 25th update plus the best and last observed states. Trace values are rounded to six decimal places. Candidate scores in the run summaries retain their recorded precision. Each shortlist keeps up to 20 canonical identities per attempt. Their 2D drawings share one depiction per canonical SMILES, irrespective of the atom mapping used for hard scoring."]
];
function setupMethods(){
 const cards=explanatory.map(o=>'<article class="box"><h3>'+o[0]+'</h3><p>'+o[1]+'</p></article>').join("");
 set("methodsBody",'<div class="two">'+cards+'</div><div class="card separator"><h3>Download tables for analysis</h3><p>Attempt identifiers, settings, starting graphs and seeds stay attached to each record. The JSON export includes the displayed trajectory samples, ranked molecules, benchmark metadata and earlier structured results.</p><div class="actions">'+btn("All run summaries · CSV","allCSV")+btn("All ranked molecules · CSV","allMol")+btn("Analysis data · JSON","allJSON")+'</div></div><div class="card separator"><h3>Original research notes</h3><p>Read or save a dated record. Earlier queue statuses appear in their original chronology.</p><div class="controls">'+label("Source record","sourceChoice",options(Object.keys(data.sources)))+btn("Download note","noteDL")+'</div><p class="mono" id="notePath"></p><details class="note"><summary>Read selected note</summary><pre id="noteText"></pre></details></div><div class="card separator"><h3>Previous interactive reports</h3><p>These four earlier HTML reports preserve the plots and snapshots available on their publication dates. Download one to open it offline.</p><div class="archive-list">'+Object.keys(data.archive).map(n=>'<button type="button" class="btn" data-archive="'+esc(n)+'">'+esc(n.replace(/^DeFoG-/,"").replace(/-Mosaic-/," · ").replace(/^results/,"DeFoG results"))+'</button>').join("")+'</div></div>');
 function note(){const n=data.sources[el("sourceChoice").value];el("notePath").textContent=n.path;el("noteText").textContent=n.text}
 el("sourceChoice").onchange=note;note();el("noteDL").onclick=()=>{const s=data.sources[el("sourceChoice").value];download(s.path.split("/").pop(),s.text,"text/markdown;charset=utf-8")};
 el("allCSV").onclick=()=>csv("defog-stage3-attempts.csv",data.runs.map(({trace,molecules,...r})=>r));
 el("allMol").onclick=()=>csv("defog-stage3-ranked-molecules.csv",data.runs.flatMap(r=>records(r.molecules,data.molecule_fields).map((m,i)=>({attempt:r.attempt,run_id:r.run_id,condition:r.condition_id,seed:r.seed,initialization:r.initialization,rank:i+1,...m}))));
 el("allJSON").onclick=()=>{const {archive,sources,...analysis}=data;download("defog-analysis-data.json",JSON.stringify(analysis),"application/json")};
}
let globalCandidates=[],candidatePage=0;
function setupTimeComparison(){
 const box=document.createElement("article");box.className="card separator";
 box.innerHTML='<div class="kicker">One-query teacher · fixed-time comparison</div><h2>How did teacher time interact with weight?</h2><p>Each cell summarizes the original three seeds and three initializations at constant time and weight, using the same connected projection. Open a cell to see its underlying runs.</p><div class="controls">'+label("Cell value","timeMeasure",options([["mean_best","Mean best valid contact"],["mean_distinct","Mean distinct valid identities"],["final","Valid endpoints"],["signed_gap","Mean signed soft − hard gap"]]))+'</div><div id="timeTable" class="scroll"></div><p class="caption">Color shows the numerical value relative to the other cells in this view. A larger signed gap need not indicate closer agreement.</p>';
 el("overview").appendChild(box);
 function render(){const wt=["0p03125","0p0625","0p125"],ts=["0p2","0p5","0p8","0p95"],m=el("timeMeasure").value;
 const matrix=ts.map(t=>wt.map(w=>{const id=t==="0p8"?(w==="0p0625"?"reference":"weight-"+w):"time-"+t+"-w-"+w;const cell=stats(conditions.get(id),"all","all");return {id,value:cell[m],n:m==="mean_best"?cell.scored:cell.complete}}));
 const values=matrix.flat().map(x=>x.value).filter(Number.isFinite),min=Math.min(...values),max=Math.max(...values);
 const rows=matrix.map((row,j)=>[ts[j].replace("p","."),...row.map(c=>{const f=max>min?(c.value-min)/(max-min):0,text=m==="final"?c.value+"/"+c.n:fmt(c.value,m==="mean_distinct"?1:3);return '<button class="btn" data-cond="'+c.id+'" aria-label="'+esc(c.id+" "+text)+'" title="'+c.n+' contributing runs" style="width:100%;min-width:78px;background:rgba(105,174,113,'+(.10+.55*f).toFixed(3)+')">'+text+'</button>'})]);
 set("timeTable",gridTable(["Teacher time",...wt.map(x=>"Weight "+x.replace("p","."))],rows))}
 el("timeMeasure").onchange=render;render();
}
function setupRecovery(){
 const box=document.createElement("article");box.className="card separator";
 const rs=data.runs.filter(r=>r.attempt==="Recovery");
 box.innerHTML='<div class="kicker">Projection solver reruns · 18 November</div><h2>All six reruns finished</h2><p>Five found a new valid molecule. All six ended disconnected after 500 updates. The treatment and control for fragmented seed 17 were both repeated, allowing the ninth weight-0.5 pair to be scored.</p><div class="scroll">'+gridTable(["Setting","Start / seed","Original updates","Recovery best valid","Best at","Valid states / observed","Endpoint fragments"],rs.map(r=>[buttonLink(r.run_id,"recovery",r.condition_id),esc(r.initialization)+" · "+r.seed,getRun("Original",r.condition_id,r.seed,r.initialization).updates_observed,fmt(r.best_valid_hard_score,4),r.best_valid_update??"—",r.valid_state_count+"/"+r.states_observed,r.final_components]))+'</div>';el("overview").appendChild(box);
}
function setupGallery(){
 const box=document.createElement("article");box.className="card separator";box.innerHTML='<h3>Highest-scoring molecules across the campaign</h3><p>One row per distinct shortlisted SMILES, retaining its highest observed score across attempts. Search a SMILES substring or open the run that found it. Scores here reflect selection across many settings.</p><div class="controls"><label>Search canonical SMILES<input type="search" id="molSearch" placeholder="e.g. Cl"></label></div><div class="pages">'+btn("Previous","galleryPrev")+'<span id="gRange"></span>'+btn("Next","galleryNext")+'</div><div id="gTable" class="scroll"></div>';el("runs").appendChild(box);
 const best=new Map();for(const r of data.runs)r.molecules.forEach((m,i)=>{const prev=best.get(m[0]);if(!prev||prev.score<m[1])best.set(m[0],{smiles:m[0],score:m[1],index:i,run:r,update:m[2]})});globalCandidates=[...best.values()].sort((a,b)=>b.score-a.score);
 el("molSearch").oninput=()=>{candidatePage=0;renderGallery()};el("galleryPrev").onclick=()=>{candidatePage--;renderGallery()};el("galleryNext").onclick=()=>{candidatePage++;renderGallery()};renderGallery();
}
function renderGallery(){const input=el("molSearch").value.trim(),list=globalCandidates.filter(c=>c.smiles.includes(input)),size=8,offset=candidatePage*size;
 el("gRange").textContent=list.length?(offset+1)+"–"+Math.min(offset+size,list.length)+" of "+number(list.length):"0 matches";
 el("galleryPrev").disabled=candidatePage===0;el("galleryNext").disabled=offset+size>=list.length;
 set("gTable",gridTable(["Hard contact","Setting","Start / seed","Update","Open candidate"],list.slice(offset,offset+size).map(c=>[fmt(c.score,5),esc(c.run.condition_id),esc(c.run.initialization)+" · "+c.run.seed,c.update,buttonLink(c.smiles,"gallery",c.smiles)])));
}
async function init(){
 data=JSON.parse(await unpack(el("packed").textContent));lib=JSON.parse(el("structures").textContent);conditions=new Map(data.conditions.map(c=>[c.id,c]));original=data.runs.filter(r=>r.attempt==="Original");lookup=new Map(data.runs.map(r=>[key(r.attempt,r.condition_id,r.seed,r.initialization),r]));
 set("nav",Object.entries(pages).map(([id,t])=>'<button type="button" id="tab-'+id+'" data-page="'+id+'">'+t+'</button>').join(""));
 setupOverview();setupTimeComparison();setupRecovery();setupConditions();setupRuns();setupGallery();setupHistory();setupMethods();
 document.addEventListener("click",async ev=>{const b=ev.target.closest("button");if(!b)return;const d=b.dataset;
 if(d.page)show(d.page);
 if(d.cond)chooseCondition(d.cond);
 if(d.select){condChoice=d.select;renderCondDetail();el("condDetails").scrollIntoView({block:"start"})}
 if(d.run){const r=original.find(x=>x.run_id===d.run);openRun(r.attempt,r.condition_id,r.seed,r.initialization)}
 if(d.recovery){const r=data.runs.find(x=>x.attempt==="Recovery"&&x.run_id===d.recovery);openRun(r.attempt,r.condition_id,r.seed,r.initialization)}
 if(d.candidate!==undefined)await pickMolecule(+d.candidate);
 if(d.gallery){const c=globalCandidates.find(x=>x.smiles===d.gallery),r=c.run;openRun(r.attempt,r.condition_id,r.seed,r.initialization);await pickMolecule(c.index);el("molDetail").scrollIntoView({block:"center"})}
 if(d.archive)download(d.archive+".html",await unpack(data.archive[d.archive]),"text/html;charset=utf-8");
 });
 window.addEventListener("hashchange",()=>show(location.hash.slice(1)));
 el("loading").hidden=true;show(location.hash.slice(1));
}
init().catch(error=>{el("loading").textContent="The saved report could not be opened. "+String(error);console.error(error)});
