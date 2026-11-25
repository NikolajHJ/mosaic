"use strict";
const el=id=>document.getElementById(id), set=(id,html)=>{el(id).innerHTML=html};
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(x,n=3)=>typeof x==="number"&&Number.isFinite(x)?x.toFixed(n):"—";
const number=x=>x.toLocaleString("en-US"), yes=x=>x==null?"—":x?"Yes":"No";
const mean=xs=>{const v=xs.filter(Number.isFinite);return v.length?v.reduce((a,b)=>a+b,0)/v.length:null};
const sum=(a,k)=>a.filter(x=>x[k]===true).length;
const option=(v,t)=>'<option value="'+esc(v)+'">'+esc(t??v)+'</option>';
const label=(name,id,opts)=>'<label>'+esc(name)+'<select id="'+id+'">'+opts+'</select></label>';
const options=xs=>xs.map(v=>Array.isArray(v)?option(...v):option(v)).join("");
const btn=(t,id,kind="")=>'<button type="button" class="btn '+kind+'" id="'+id+'">'+t+'</button>';
const gridTable=(heads,rows)=>'<table><thead><tr>'+heads.map(h=>'<th scope="col">'+h+'</th>').join("")+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join("")+'</tr>').join("")+'</tbody></table>';
const metric=(v,t)=>'<div><b>'+v+'</b><span>'+t+'</span></div>';
const buttonLink=(val,act,text)=>'<button class="link" data-'+act+'="'+esc(val)+'">'+esc(text)+'</button>';
async function unpack(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text()}
function download(name,body,type){const a=document.createElement("a"),url=URL.createObjectURL(new Blob([body],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1500)}
function csv(name,rows){if(!rows.length)return;const keys=Object.keys(rows[0]);const field=v=>'"'+String(typeof v==="object"&&v!==null?JSON.stringify(v):v??"").replace(/"/g,'""')+'"';download(name,[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(field).join(",")).join("\r\n"),"text/csv;charset=utf-8")}
const pages={overview:"Overview",conditions:"Conditions",runs:"Runs & molecules",history:"Earlier studies",methods:"Methods & sources"};
function show(id){if(!(id in pages))id="overview";for(const k of Object.keys(pages)){el(k).classList.toggle("on",k===id);const n=el("tab-"+k);n.classList.toggle("on",k===id);n.setAttribute("aria-current",k===id?"page":"false")}if(location.hash!=="#"+id)history.replaceState(null,"","#"+id);window.scrollTo(0,0)}
let data,lib,conditions,original,lookup,current,molToken=0,condChoice="weight-0p5",conditionRows,shownConditions;
const key=(attempt,cid,seed,start)=>[attempt,cid,seed,start].join("|");
const good=r=>r?.status==="complete"&&r.complete_planned_trajectory&&r.scientific_usable;
const getRun=(a,c,s,i)=>lookup.get(key(a,c,s,i));
const readable=s=>s.replaceAll("_"," ");
const starts=["sparse","dense","fragmented"],seeds=[17,23,42];
function pairs(cid,attempt="Original",start="all",seed="all"){
 const out=[];for(const i of starts)for(const s of seeds){
 if((start!=="all"&&i!==start)||(seed!=="all"&&s!==+seed))continue;
 const a=attempt==="Recovered"&&i==="fragmented"&&s===17&&cid==="weight-0p5"?"Recovery":"Original";
 const t=getRun(a,cid,s,i),c=getRun(a,conditions.get(cid).paired_with,s,i);
 if(good(t)&&good(c))out.push({t,c,s,i,a})}
 return out;
}
function scored(ps){return ps.filter(p=>Number.isFinite(p.t.best_valid_hard_score)&&Number.isFinite(p.c.best_valid_hard_score))}
function delt(p){return p.t.best_valid_hard_score-p.c.best_valid_hard_score}
const weights=[["weight-0p015625","0.015625"],["weight-0p03125","0.03125"],["reference","0.0625"],["weight-0p125","0.125"],["weight-0p25","0.25"],["weight-0p5","0.5"]];
function pairChart(ps){
 const width=600,l=132,r=40,rowH=32,h=ps.length*rowH+49;
 const xmax=Math.max(.55,...ps.map(p=>Math.max(p.t.best_valid_hard_score,p.c.best_valid_hard_score)))+.01;
 const scale=n=>l+(width-l-r)*(n-.15)/(xmax-.15);
 let g="";
 for(let v=.2;v<=xmax;v+=.1){const x=scale(v);g+='<path d="M'+x+' 8V'+(h-29)+'" stroke="#d7dfd3"/><text x="'+x+'" y="'+(h-10)+'" text-anchor="middle">'+v.toFixed(1)+'</text>'}
 ps.forEach((p,j)=>{const x=scale(p.c.best_valid_hard_score),z=scale(p.t.best_valid_hard_score),y=20+j*rowH,col=z>x?"#177254":"#b15c2d";g+='<g><title>'+esc(p.i+' · seed '+p.s+' · '+p.a+' | control '+fmt(p.c.best_valid_hard_score)+' / guided '+fmt(p.t.best_valid_hard_score))+'</title><text x="2" y="'+(y+4)+'">'+p.i+' · '+p.s+(p.a==="Recovery"?" *":"")+'</text><path d="M'+x+' '+y+'H'+z+'" stroke="'+col+'" stroke-width="3"/><circle cx="'+x+'" cy="'+y+'" r="5" fill="white" stroke="#4b6055" stroke-width="2"/><circle cx="'+z+'" cy="'+y+'" r="5.5" fill="'+col+'"/></g>'});
 return '<svg class="chart" viewBox="0 0 '+width+' '+h+'" role="img" aria-label="Best valid hard contact by matched seed and initialization">'+g+'</svg><div class="axisnote">Best valid hard contact score →</div>';
}
function renderPair(){const cid=el("overviewWeight").value,pol=el("overviewPolicy");pol.disabled=cid!=="weight-0p5";if(pol.disabled)pol.value="original";const p=scored(pairs(cid,pol.value==="recovered"?"Recovered":"Original")),win=p.filter(x=>delt(x)>0).length;
 set("pairPlot",pairChart(p));set("pairHeadline",win+" of "+p.length+" pairs favored DeFoG");
 set("pairText",'<div class="metricrow">'+metric(fmt(mean(p.map(x=>x.t.best_valid_hard_score))),"Mean with DeFoG")+metric(fmt(mean(p.map(x=>x.c.best_valid_hard_score))),"Mean with projection alone")+'</div><p>Among these pairs, the final molecule was valid and connected in <b>'+p.filter(x=>x.t.final_valid).length+'/'+p.length+'</b> guided runs and <b>'+p.filter(x=>x.c.final_valid).length+'/'+p.length+'</b> controls.</p>');
 set("pairCaption",pol.value==="recovered"?"* Fragmented seed 17 uses both recovery attempts. The other eight pairs use original attempts. Three seeds contribute these nine pairs.":"The original fragmented seed-17 control stopped at update 226. Eight complete pairs across three seeds enter this comparison.");
}
const topics=[
 ["Guidance strength","weight-0p5","At fixed teacher time 0.8, weight 0.5 improved the best valid score in all eight complete original pairs. Four of its nine original endpoints were valid."],
 ["Teacher time & schedules","weight-reverse-0p125","A weight declining from 0.125 to zero gave a mean best valid score of 0.388 across nine runs. At the same mean weight, constant 0.0625 gave 0.318. Neither setting ended valid."],
 ["Valence & connectivity","projection-none-w-0p0625","Without projection, weight 0.0625 found a valid molecule in three of nine runs. With connected projection, the same weight found a new valid identity in all nine."],
 ["Cooling & fluidity","cool-fluidity-off-ramp-half-w-0p0625","With fluidity disabled and temperature falling to 0.5, three of eight completed guided endpoints were valid. The teacher-off arm also had three valid endpoints in nine runs."],
 ["Teacher information","mechanism-stale","Holding the initial teacher target fixed gave a mean best valid score of 0.347 across nine runs. A refreshed reference gave 0.318. This result leaves the benefit of refreshing open."],
 ["Simple chemistry loss","soft-valence-0p1-w-0p0625","With the 0.1 soft-valence loss, adding DeFoG raised the mean best valid score from 0.291 to 0.323 across nine starts. Both groups had zero valid endpoints."]
];
function chooseCondition(id){condChoice=id;el("family").value=conditions.get(id).family;el("csearch").value="";el("cstart").value="all";el("cseed").value="all";show("conditions");renderConditions();el("condDetails").scrollIntoView({block:"nearest"})}
function setupOverview(){
 const rr=original.filter(good);set("stats",[[number(rr.length), "original runs completed / 1,395"],[number(sum(rr,"any_valid")), "found a valid connected molecule"],[number(sum(rr,"any_new_valid")), "found a new valid identity"],[sum(rr,"final_valid"),"ended valid and connected"]].map(([v,l])=>'<div class="stat"><strong>'+v+'</strong><span>'+l+'</span></div>').join(""));
 set("overviewWeight",options(weights));el("overviewWeight").value="weight-0p5";el("overviewPolicy").value="recovered";el("overviewWeight").onchange=renderPair;el("overviewPolicy").onchange=renderPair;renderPair();
 el("seeRepeat").onclick=()=>openRun("Recovery","weight-0p5",17,"fragmented");
 set("findings",topics.map(([t,id,body])=>'<article class="box"><h3>'+t+'</h3><p>'+body+'</p>'+buttonLink(id,"cond","Compare conditions →")+'</article>').join(""));
}
function stats(c,start,seed){const eligible=r=>r.condition_id===c.id&&(start==="all"||r.initialization===start)&&(seed==="all"||r.seed===+seed);const all=original.filter(eligible),rs=all.filter(good),p=scored(pairs(c.id,"Original",start,seed));return {condition:c.id,family:c.family,planned:all.length,complete:rs.length,any:sum(rs,"any_valid"),new_valid:sum(rs,"any_new_valid"),final:sum(rs,"final_valid"),mean_best:mean(rs.map(r=>r.best_valid_hard_score)),scored:rs.filter(r=>Number.isFinite(r.best_valid_hard_score)).length,mean_distinct:mean(rs.map(r=>r.unique_valid_molecules)),signed_gap:mean(rs.map(r=>r.all_state_soft_hard_gap_mean)),paired_delta:mean(p.map(delt)),wins:p.filter(q=>delt(q)>0).length,scored_pairs:p.length}}
function setupConditions(){
 const fs=[...new Set([...conditions.values()].map(c=>c.family))].sort();
 set("condControls",label("Family","family",options([["all","All families"],...fs.map(f=>[f,readable(f)])]))+label("Start","cstart",options([["all","All starts"],...starts]))+label("Seed","cseed",options([["all","All seeds"],...seeds]))+label("Order by","csort",options([["delta","Matched gain"],["best","Best score"],["final","Valid endpoints"],["name","Condition name"]]))+'<label>Search conditions<input type="search" id="csearch" placeholder="e.g. reverse, bridge"></label>'+btn("Export filtered CSV","ccsv"));
 for(const id of ["family","cstart","cseed","csort"])el(id).onchange=renderConditions;el("csearch").oninput=renderConditions;el("ccsv").onclick=()=>csv("defog-conditions.csv",shownConditions);
 renderConditions();
}
function renderConditions(){const f=el("family").value,st=el("cstart").value,sd=el("cseed").value,search=el("csearch").value.trim().toLowerCase(),sort=el("csort").value;shownConditions=[...conditions.values()].filter(c=>(f==="all"||c.family===f)&&[c.id,c.family,c.question].join(" ").toLowerCase().includes(search)).map(c=>stats(c,st,sd));
 const k={delta:"paired_delta",best:"mean_best",final:"final"}[sort];
 shownConditions.sort((a,b)=>k?((b[k]??-Infinity)-(a[k]??-Infinity)||a.condition.localeCompare(b.condition)):a.condition.localeCompare(b.condition));
 set("condCount",shownConditions.length+' settings · original full-budget attempts • best-score mean excludes runs with no scored valid molecule • Δ = guided minus matched teacher-off score.');
 set("condTable",gridTable(["Setting","Complete / planned","Found valid / complete","Ended valid / complete","Mean best","Mean distinct","Mean signed gap","Mean paired Δ","Wins / scored pairs"],shownConditions.map(x=>[buttonLink(x.condition,"select",x.condition),x.complete+'/'+x.planned,x.any+'/'+x.complete,x.final+'/'+x.complete,fmt(x.mean_best),fmt(x.mean_distinct,1),fmt(x.signed_gap),fmt(x.paired_delta),x.scored_pairs?x.wins+'/'+x.scored_pairs:"—"])));
 if(!shownConditions.length){set("condDetails","No settings match these filters.");return}if(!shownConditions.some(x=>x.condition===condChoice))condChoice=shownConditions[0].condition;renderCondDetail();
}
function renderCondDetail(){const c=conditions.get(condChoice),st=el("cstart").value,sd=el("cseed").value;
 const rs=original.filter(r=>r.condition_id===c.id&&(st==="all"||r.initialization===st)&&(sd==="all"||r.seed===+sd)).sort((a,b)=>starts.indexOf(a.initialization)-starts.indexOf(b.initialization)||a.seed-b.seed);
 set("condDetails",'<div class="kicker">'+esc(readable(c.family))+'</div><h3>'+esc(c.id)+'</h3><p>'+esc(c.question)+'</p><p class="caption">Matched teacher-off condition: '+esc(c.paired_with??"This condition serves as a baseline")+'</p><div class="scroll">'+gridTable(["Run","Updates","Best valid","Control best¹","Δ¹","Best at","New identity","Final valid","Final fragments"],rs.map(r=>{const control=getRun("Original",c.paired_with,r.seed,r.initialization),ok=good(control)&&good(r)&&Number.isFinite(control.best_valid_hard_score)&&Number.isFinite(r.best_valid_hard_score);return [buttonLink(r.run_id,"run",r.initialization+" · "+r.seed),r.updates_observed+(good(r)?"":" (partial)"),fmt(r.best_valid_hard_score),good(control)?fmt(control.best_valid_hard_score):"—",ok?fmt(r.best_valid_hard_score-control.best_valid_hard_score):"—",r.best_valid_update??"—",yes(r.any_new_valid),yes(r.final_valid),r.final_components??"—"]}))+'</div><p class="caption">¹ Original full-budget controls. Each run link opens its trajectory and ranked molecules.</p><details><summary>Resolved teacher, projection, cooling and chemistry settings</summary><pre>'+esc(JSON.stringify(c,null,2))+'</pre></details>');
}
