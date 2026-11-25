let figureURL;
function setupRuns(){
 set("runControls",label("Attempt","rat",options(["Original","Recovery"]))+label("Setting","rcid","")+label("Initialization and seed","rid","")+btn("Export trace CSV","trCSV")+btn("Export ranked CSV","molCSV"));
 el("rat").onchange=()=>runOptions();el("rcid").onchange=()=>runOptions(el("rcid").value);el("rid").onchange=()=>pickRun();
 el("scrub").oninput=renderTrace;el("traceMetric").onchange=renderTrace;
 el("trCSV").onclick=()=>csv(current.run_id+"-trace.csv",records(current.trace,data.trace_fields));
 el("molCSV").onclick=()=>csv(current.run_id+"-ranked.csv",records(current.molecules,data.molecule_fields));
 runOptions("weight-0p5",17,"fragmented");
}
const records=(rows,fields)=>rows.map(row=>Object.fromEntries(fields.map((f,i)=>[f,row[i]])));
function runOptions(cid,seed,start){
 const rs=data.runs.filter(r=>r.attempt===el("rat").value), ids=[...new Set(rs.map(r=>r.condition_id))].sort();
 set("rcid",options(ids));el("rcid").value=ids.includes(cid)?cid:ids[0];
 const group=rs.filter(r=>r.condition_id===el("rcid").value).sort((a,b)=>starts.indexOf(a.initialization)-starts.indexOf(b.initialization)||a.seed-b.seed);
 set("rid",options(group.map(r=>[r.run_id,r.initialization+" · "+r.seed+(good(r)?"":" · partial")])));
 const found=group.find(r=>r.seed===seed&&r.initialization===start);if(found)el("rid").value=found.run_id;pickRun();
}
function openRun(a,c,s,i){show("runs");el("rat").value=a;runOptions(c,s,i)}
function pickRun(){
 current=data.runs.find(r=>r.attempt===el("rat").value&&r.run_id===el("rid").value);
 const r=current, done=good(r),hardware=r.hardware?.devices?.[0]?.device_kind||"unspecified GPU";
 const end=done?yes(r.final_valid):"Missing";
 set("runSummary",'<div class="box"><div class="kicker">'+esc(r.attempt)+' · '+esc(hardware)+' · '+r.updates_observed+'/'+r.planned_updates+' updates</div><h3>'+esc(r.run_id)+'</h3><p>'+esc(conditions.get(r.condition_id).question)+'</p><div class="metricrow">'+metric(fmt(r.best_valid_hard_score),"Best valid hard contact")+metric(fmt(r.best_new_valid_hard_score),"Best new valid")+metric(end,"Valid endpoint")+metric(done?(r.final_components??"—"):"—","Endpoint fragments")+metric(r.valid_state_count+"/"+r.states_observed,"Valid states observed")+metric(r.unique_valid_molecules,"Distinct valid identities")+'</div><p class="caption">Initial valid '+yes(r.initial_valid)+' · initial hard score '+fmt(r.initial_hard_score)+' · endpoint hard score '+(done?fmt(r.final_hard_score):"—")+' · valid in planned last 100 states '+(r.last100_states_observed?r.last100_valid_state_count+'/'+r.last100_states_observed:"—")+' · mean signed soft − hard gap '+fmt(r.all_state_soft_hard_gap_mean)+' · frozen checks '+(r.frozen===true?"passed":r.frozen===false?"failed":"unreported")+'</p>'+(done?"":'<p class="story"><b>Partial trajectory.</b> The last observation is update '+r.updates_observed+'. '+esc(r.error??r.exclusion_reason??"The planned endpoint is missing.")+'</p>')+(done&&r.final_smiles?'<details><summary>Final argmax SMILES</summary><p class="mono">'+esc(r.final_smiles)+'</p></details>':'')+'</div>');
 el("scrub").max=r.trace.length-1;const ix=r.trace.findIndex(row=>row[0]===r.best_valid_update);el("scrub").value=ix<0?r.trace.length-1:ix;
 const fields=["Rank","Hard contact","Best mapping at","Visits¹","Initial identity","Canonical SMILES"];
 set("candidateTable",r.molecules.length?gridTable(fields,r.molecules.map((m,i)=>[buttonLink(i,"candidate","View "+(i+1)),fmt(m[1],5),m[2],m[3],yes(m[8]),'<span class="mono">'+esc(m[0])+'</span>'])):'<p>No valid canonical molecule was recorded.</p>');
 el("molCSV").disabled=!r.molecules.length;renderTrace();pickMolecule(0);
}
const traces={
 score:[[1,"Hard contact","#20694d"],[2,"Relaxed contact","#be6535"]],
 components:[[4,"Connected components","#20694d"]],sulfur:[[5,"Sulfur atoms","#20694d"]],valence:[[9,"Valence excess","#20694d"]],
 schedule:[[6,"DeFoG weight","#20694d"],[7,"Teacher time","#be6535"],[8,"Temperature","#5174a4"]]
};
function renderTrace(){
 const r=current, rows=r.trace, selected=rows[+el("scrub").value], series=traces[el("traceMetric").value];
 const w=630,h=234,l=46,rr=18,t=12,b=31,x=n=>l+(w-l-rr)*n/r.planned_updates;
 const vals=rows.flatMap(v=>series.map(s=>v[s[0]]).filter(Number.isFinite));
 const lo=Math.min(0,...vals),hi=Math.max(...vals,lo+.01),y=v=>t+(h-b-t)*(hi-v)/(hi-lo);
 let g="";for(let j=0;j<=4;j++){const v=lo+(hi-lo)*j/4;g+='<path stroke="#d7dfd3" d="M'+l+' '+y(v)+'H'+(w-rr)+'"/><text x="'+(l-6)+'" y="'+(y(v)+4)+'" text-anchor="end">'+Number(v.toFixed(2))+'</text>'}
 for(let i=0;i<=r.planned_updates;i+=100)g+='<text x="'+x(i)+'" y="'+(h-9)+'" text-anchor="middle">'+i+'</text>';
 for(const [k,,color] of series){let d="",move=true;for(const a of rows){if(!Number.isFinite(a[k])){move=true;continue}d+=(move?"M":"L")+x(a[0]).toFixed(2)+" "+y(a[k]).toFixed(2)+" ";move=false}g+='<path fill="none" stroke="'+color+'" stroke-width="2" d="'+d+'"/>'}
 g+='<path stroke="#525c53" stroke-dasharray="4 4" d="M'+x(selected[0])+' '+t+'V'+(h-b)+'"/>';
 for(const [k,,color] of series)if(Number.isFinite(selected[k]))g+='<circle cx="'+x(selected[0])+'" cy="'+y(selected[k])+'" r="4" fill="'+color+'"/>';
 set("tracePlot",'<svg class="chart" role="img" aria-label="Observed trajectory by update" viewBox="0 0 '+w+' '+h+'">'+g+'</svg><div class="legend">'+series.map(s=>'<span><i class="sw" style="background:'+s[2]+'"></i>'+s[1]+'</span>').join("")+'</div>');
 set("scrubValue","<b>Update "+selected[0]+"</b> · valid and connected "+yes(selected[3])+" · fragments "+(selected[4]??"—")+" · hard "+fmt(selected[1],4)+" · relaxed "+fmt(selected[2],4));
 set("traceCaption",(rows.length===r.states_observed?"Every saved update is shown.":"The curve connects every 25th update plus the exact best and last observations.")+" All score rankings and state counts use the full observed trajectory.");
}
async function pickMolecule(index){
 const m=current.molecules[index], token=++molToken;if(figureURL){URL.revokeObjectURL(figureURL);figureURL=null}
 el("candidateTable").querySelectorAll("tbody tr").forEach((row,j)=>row.classList.toggle("sel",j===index));
 if(!m){set("molRank","No candidates");set("molDetail","<p>This attempt did not visit a valid connected argmax.</p>");return}
 set("molRank","Rank "+(index+1)+" / "+current.molecules.length);
 set("molDetail",'<img class="mol" id="rdmol" alt="2D molecular structure for rank '+(index+1)+'"><p class="mono">'+esc(m[0])+'</p><div class="metricrow">'+metric(fmt(m[1],4),"Hard contact")+metric(m[2],"Source update")+'</div><p class="caption">MW '+fmt(m[4],1)+' Da · QED '+fmt(m[5],3)+' · logP '+fmt(m[6],2)+' · TPSA '+fmt(m[7],1)+' Å² · initial identity '+yes(m[8])+'</p><p class="caption">¹ Visits are for the retained ordered hard graph. The ranked list keeps one representative per canonical SMILES.</p>');
 if(lib[m[0]]){const svg=await unpack(lib[m[0]]);if(token!==molToken)return;figureURL=URL.createObjectURL(new Blob([svg],{type:"image/svg+xml"}));el("rdmol").src=figureURL}else{el("rdmol").remove()}
}
