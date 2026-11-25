const chapters=[
 ["DBMol and Boltz","Sampled repairs as targets","In DBMol, DeFoG continued a discrete sampling trajectory. Eligible repaired graphs became fixed decoder targets for latent optimization. The six-update integration reached the same latent variable from both losses and selected two targets from eight eligible outputs. All seven raw argmax states were invalid. Ten longer random-start cases stopped, four on wall time and six on numerical checks.","The matched 25-update trial produced 0/26 valid raw argmax states in each arm. Hard samples yielded 1/208 valid in the control and 0/208 with guidance. At a budget of 500, the guided arm completed 500 updates and had 0/9 valid endpoint candidates. The control stalled after 225 updates, leaving the full-budget comparison incomplete."],
 ["Checkpoint and locality","One query can still change the target sharply","The frozen GuacaMol checkpoint takes a graph and a time t and predicts clean atom and bond probabilities. On fractional test graphs, native auxiliary features worked for 4/20 inputs. Recomputing auxiliary features from the hard argmax while passing soft probabilities into the transformer worked for 20/20. Repeated identical queries gave identical probabilities.","Ordinary small perturbations moved the adapted teacher by at most 0.0056 maximum-cell total variation. Crossing a near tie with input change 0.0002 could move an output by 0.499. Fully hardening the input increased the largest observed jump to 0.994. The ten-step teacher-only probe also left the overvalent test graph invalid."],
 ["KL, composition and sulfur","Initialization changed the chemistry","The initial 1,000-update stopped-teacher KL arm ramped weight from 0.001 to 0.1. Every readout was a valid C25 chain, with one identity across 1,001 states. The fixed-time random-start 20k attempt was cancelled after 1,006 accepted updates. A completed dense time-ramp 30k arm improved relaxed contact from 0.188 to 0.581 and found zero valid molecules in 30,001 states.","Matching total gradient norms at fixed DeFoG weight 0.53642 again yielded zero valid molecules, 20 sulfur atoms at the endpoint and a failed terminal combined atom derivative check. The native contact gradient initially favored sulfur and the conditional teacher reinforced it later. Adding a composition JS term reduced the observed hard sulfur count from 11 to 5 around update 7,000 in separate runs. That attempt stopped at 7,000 accepted updates with zero valid readouts. The attached-gradient variant passed 109 CPU checks and stopped before its GPU pilots because a storage check encountered another writer."],
 ["Long sparse runs","Valence projection improved sanitization more than connectivity","In the five 1,000-update sulfur pilots, the explicit sulfur penalty lowered dense-start endpoint sulfur from four to zero. Both dense arms remained invalid. All three sparse arms had 1,001/1,001 valid connected readouts, zero hard sulfur, and the same single new identity. The sulfur penalty was inactive on the sparse starts.","Over 30,000 updates, sparse initialization with JS gave 10,155 valid connected readouts and three distinct valid identities. Adding valence projection raised sanitizing readouts from 10,611 to 22,656, while connected valid readouts fell slightly to 10,036. The unprojected endpoint failed sanitization on a four-valent nitrogen. The projected endpoint sanitized in three fragments. Both had zero hard sulfur throughout."],
 ["Projection and model choice","The sign of the benefit depended on the scorer","In the matched, single-seed Boltz-2 trial, projection alone produced 400/501 valid connected readouts and best valid native loss −1.392. Adding DeFoG yielded 188/501 and −0.824, a worse loss. The projection-only endpoint was valid. Boltz native loss is reported in its own units.","With Protenix at LR 0.3 and 2,000 updates, weight 0.0625 gave 1,998/2,001 valid connected readouts, 68 distinct identities and best hard contact 0.385. Projection alone gave 512/2,001, 23 identities and 0.311. The guided mean relaxed-minus-hard gap nevertheless grew from 0.035 to 0.135. At 500 updates, stronger weights preserved validity but reduced diversity and best score on this start."],
 ["Weights, cycles and learning rate","Repeated cycles preserved validity only at the lower step size","With a 0–0.5 raised cosine of period 1,000, the 10k LR 0.3 run gave 9,958/10,001 valid connected readouts, 278 identities and best score 0.379. At LR 150 the corresponding counts were 39/10,001 and 39 identities, with best score 0.261 and 11 endpoint fragments.","At LR 150 we also tested fixed and linearly ramped 500-update weights and six 1,500-update cycles peaking at 0.1. The flat-floor and final-cycle raised-floor trajectories yielded 1 and 5 valid readouts out of 9,001. The extra valid states occurred before the floor changed, so they do not establish a floor effect. Both runs ended in 11 fragments. Superseded LR 15 submissions do not supply independent outcomes."],
 ["Connectivity and random starts","A local bond incentive did not keep the final graph in one piece","A 500-update LR 150 connectivity pilot yielded 2/501 valid connected readouts versus 1/501 in its control. With a row movement cap of 0.05, adding the connectivity term reduced the count from 74 to 22. A further four-arm test crossed connectivity guidance with valence projection on and off. Every arm ended invalid or disconnected.","At LR 70, weight 0.0625 beat projection alone on best valid score in seven of nine matched seed/start pairs. Their mean best scores were 0.268 and 0.223. The guided arm yielded 64 valid connected readouts across 4,509 states versus 31 in control. All guided valid states occurred by update 20. None of the 18 endpoints was valid and connected."]
];
const knownNames=[
 ["20261104-defog-sparse-js-30000","30k · sparse + JS"],
 ["20261106-defog-sparse-js-valence","30k · sparse + JS + valence"],
 ["20261101-defog-sulfur-exploratory","1k · five sulfur pilots"],
 ["lr70-starts","LR 70 · three seeds × three starts"],
 ["stage2-defog-pair","Boltz-2 · projection pair"],
 ["stage2-protenix-defog-pair","Protenix · 500-update projection pair"],
 ["stage2-protenix-defog-weights","LR 0.3 · fixed weights, 500 updates"],
 ["stage2-protenix-defog-2000","LR 0.3 · fixed weights, 2k"],
 ["defog-cyclic-10000","LR 0.3 · cyclic 10k"],
 ["defog-cyclic-lr150-10000","LR 150 · cyclic 10k"],
 ["peak01-floor005","LR 150 · six cycles, final floor 0.05"],
 ["peak01-9000","LR 150 · six cycles, floor 0"],
 ["linear-lr150-500","LR 150 · linear weight ramps"],
 ["defog-lr150-500","LR 150 · fixed weights"],
 ["connectivity-rowtv005","LR 150 · connectivity and row-TV limit"],
 ["defog-connectivity-lr150","LR 150 · connectivity pilot"],
 ["valence-ablation","LR 150 · connectivity × valence"]
];
const historicalName=k=>knownNames.find(x=>k.includes(x[0]))?.[1]??k.split("/").slice(-2,-1)[0];
let historicalRows=[];
const earlyRecords={
 "DBMol · repair and latent optimization":{source:"DBMol · native repair and integration",rows:[
 ["Source079 random starts","Ten longer attempts","4 timeouts, 6 numerical exits","No complete cohort"],
 ["Source085 charged-view replay","Four projected outputs","3 eligible outputs, 2 targets","Repair-only check"],
 ["Source089 integration","Six guided updates","8 eligible outputs, 2 targets","0/7 valid raw argmax states"],
 ["Source109 matched control","25 updates","0/26 valid raw argmax","1/208 valid hard draws"],
 ["Source109 matched guided","25 updates","0/26 valid raw argmax","0/208 valid hard draws"],
 ["Source120 control","225 accepted of 500 planned","Line search stalled","Partial trajectory"],
 ["Source120 guided","500 accepted of 500 planned","0/9 valid endpoint candidates","No valid endpoint argmax"]]},
 "Checkpoint · fractional inputs and local stability":{source:"Checkpoint and locality tests",rows:[
 ["Native soft auxiliary features","4/20 fractional baseline inputs succeeded","0.004740 max output TV on ordinary probes","0.421241 max output TV across an argmax tie"],
 ["Hard input, soft prediction","20/20 after hardening","0 ordinary probe response","0.994151 max output TV across an argmax tie"],
 ["Hard auxiliary features, soft core","20/20 fractional inputs succeeded","0.005573 max output TV on ordinary probes","0.498548 max output TV across an argmax tie"]]},
 "KL · first pilot, dense starts and attached teacher":{source:"Composition and solver checks",rows:[
 ["1k chain · weight ramp","1,000 updates","1,001/1,001 valid connected, 1 identity","Relaxed contact 0.07795 → 0.08778"],
 ["20k random · fixed time","1,006 accepted, last saved update 1,005","User cancelled","Partial trajectory"],
 ["30k random · time ramp","30,000 updates","0/30,001 valid","Final relaxed contact 0.58054, terminal combined atom derivative check failed"],
 ["30k random · balanced constant weight","30,000 updates","0/30,001 valid, endpoint 20/25 sulfur","Final relaxed contact 0.32896, terminal combined atom derivative check failed"],
 ["30k planned · dense + composition JS","7,000 accepted, 7,000 saved states","0/7,000 valid, last observed 5 sulfur","Storage check interrupted the trajectory"],
 ["30k planned · attached teacher","109 CPU checks passed, zero GPU pilot updates","No production trajectory","Concurrent-writer storage check stopped launch"]]}
};
const chapterRecord=["DBMol · repair and latent optimization","Checkpoint · fractional inputs and local stability","KL · first pilot, dense starts and attached teacher","20261104-defog-sparse-js-30000","defog-2000-","defog-cyclic-lr150-10000","lr70-starts"];
function historyStory(){const v=chapters[+el("hchapter").value];set("htext",'<h2>'+v[1]+'</h2><p>'+v[2]+'</p><p>'+v[3]+'</p>')}
function setupHistory(){
 set("histStory",'<div class="card highlight"><div class="controls">'+label("Study chapter","hchapter",options(chapters.map((c,i)=>[i,c[0]])))+'</div><div id="htext"></div></div>');el("hchapter").onchange=historyStory;historyStory();
 const keys=Object.keys(data.history).sort((a,b)=>historicalName(a).localeCompare(historicalName(b)));
 set("histControls",label("Comparison record","hrecord",options([...Object.keys(earlyRecords),...keys.map(k=>[k,historicalName(k)])]))+btn("Export arm CSV","hcsv"));
 el("hchapter").onchange=()=>{historyStory();const s=chapterRecord[+el("hchapter").value];el("hrecord").value=s in earlyRecords?s:keys.find(k=>k.includes(s));historyTable()};
 el("hrecord").onchange=historyTable;el("hcsv").onclick=()=>csv("defog-historical-arms.csv",historicalRows);historyTable();
}
function historyTable(){
 const k=el("hrecord").value;
 if(k in earlyRecords){const d=earlyRecords[k];historicalRows=d.rows.map(([arm,budget,observation,detail])=>({arm,budget,observation,detail}));set("histTable",gridTable(["Arm","Budget or coverage","Observed result","Further detail"],d.rows.map(r=>r.map(esc))));set("histDesc","Dated research notes and saved run records. Related notes are available under Methods &amp; sources.");el("histRaw").textContent=data.sources[d.source].text;return}
 const d=data.history[k],boltz=k.includes("stage2-defog-pair"),isLong=!!d.connected_valid_states, rows=[];
 const add=(arm,v)=>rows.push({arm,readouts:v.readouts,valid_connected:v.sanitized_connected_readouts,distinct:v.distinct_valid_smiles??null,best_valid:typeof v.best_valid_hard_native_loss==="number"?(boltz?1:-1)*v.best_valid_hard_native_loss:null,final_valid:v.final_sanitized_connected,final_sulfur:v.sulfur_final,gap:v.mean_hard_minus_soft_native_loss??null,fragments:v.final_fragments??null});
 function walk(obj,path){if(!obj||typeof obj!=="object")return;if(Number.isFinite(obj.readouts)&&Number.isFinite(obj.sanitized_connected_readouts)){add(path.join(" / "),obj);return}for(const [n,v] of Object.entries(obj))walk(v,[...path,n])}
 if(isLong){const end=d.final??d.endpoint??{};rows.push({arm:d.projection?"Sparse + JS + valence":"Sparse + JS",readouts:d.saved_states,valid_connected:d.connected_valid_states,distinct:d.valid_smiles.length,best_valid:null,final_valid:false,final_sulfur:end.hard_S??0,gap:null,fragments:d.projection?3:1})}
 else if(Array.isArray(d.arms)){for(const a of d.arms){if("readouts" in a)add([a.arm,a.initialization,"seed "+a.seed].join(" / "),a);else rows.push({arm:a.name,readouts:a.saved_states,valid_connected:a.valid_connected,distinct:a.distinct_valid,best_valid:null,final_valid:a.valid_connected===a.saved_states?true:a.valid_connected===0?false:null,final_sulfur:a.last_hard_S,gap:null,fragments:null})}}
 else walk(d,[]);
 historicalRows=rows;set("histDesc",esc(k)+(k.includes("defog-weights-")?" · Weight 0 and 0.5 reuse the matched pair.":"")+(isLong?" · Sanitized "+number(d.valid_states)+"/"+number(d.saved_states)+".":"")+" · A dash indicates an unreported metric.");
 set("histTable",gridTable(["Arm","Valid connected / observed","Distinct valid",boltz?"Best valid Boltz loss ↓":"Best valid hard contact ↑","Final valid","Final fragments","Final sulfur",boltz?"Hard − soft loss":"Soft − hard contact"],rows.map(r=>[esc(r.arm),number(r.valid_connected)+"/"+number(r.readouts),r.distinct??"—",fmt(r.best_valid,4),yes(r.final_valid),r.fragments??"—",r.final_sulfur??"—",fmt(r.gap)])));
 el("histRaw").textContent=JSON.stringify(d,null,2);
}
