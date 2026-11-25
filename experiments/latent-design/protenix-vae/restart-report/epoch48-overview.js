"use strict";
const data = JSON.parse(document.querySelector("#reportData").textContent);
const el = id => document.getElementById(id);
const finite = x => typeof x === "number" && Number.isFinite(x);
const fmt = (x, n = 3) => finite(x) ? x.toFixed(n) : "—";
const num = x => finite(x) ? x.toLocaleString("en-US") : "—";
const escapeHtml = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const opt = (value, label) => '<option value="' + escapeHtml(value) + '">' + escapeHtml(label) + '</option>';
const labels = {control:"Protenix", blend:"Protenix + Nesso-1", a:"Broad A", b:"Broad B"};
const colors = ["#087966","#9b4b40"];
let section = "m02", wave = 6, records = [], filtered = [], current = null;
const tree = job => 'trees.html#job=' + encodeURIComponent(job);
const short = arm => labels[arm.condition] || arm.label;
const armLabel = c => labels[c] || c;
const fullArm = c => data[section].arms.find(a => a.condition === c);
const th = x => "<th>" + escapeHtml(x) + "</th>";
const td = x => "<td>" + escapeHtml(x) + "</td>";
function selectSection(set, updateHash = true) {
    section = ["m02","broad","erk2"].includes(set) ? set : "m02";
    if (updateHash) history.replaceState(null,"","#" + section);
    document.querySelectorAll("[data-set]").forEach(b => {
        b.classList.toggle("active",b.dataset.set === section);
        b.setAttribute("aria-pressed",String(b.dataset.set === section));
    });
    el("brd4").hidden = section === "erk2";
    el("erkPanel").hidden = section !== "erk2";
    if (section === "erk2") {renderErk(); return;}
    wave = 6; current = null;
    const m02 = section === "m02", arms = data[section].arms;
    el("runFilter").innerHTML = opt("all","Both runs") + arms.map(a => opt(a.condition,short(a))).join("");
    el("filterGroup").hidden = !m02; el("hitFilter").value = "all"; el("molSearch").value = "";
    el("molSort").innerHTML = opt("contact","Lowest contact loss") + (m02 ? opt("affinity","Lowest affinity score") + opt("probability","Highest binder probability") : "") + opt("sa","Lowest SA score");
    el("lastCol").textContent = m02 ? "Affinity ↓" : "SA ↓";
    el("molIntro").textContent = m02 ? "Inspect every new molecule with a fresh contact score, or compare it with M02." : "Inspect the top-ranked molecules from each broad search. The trees include the full recorded trajectories.";
    records = m02 ? data.m02.molecules : arms.flatMap(a => a.top.map(r => ({...r,condition:a.condition})));
    el("legend").innerHTML = arms.map((a,i) => '<span class="series"><i class="dot ' + (i ? "alt" : "") + '"></i>' + escapeHtml(short(a)) + '</span>').join("");
    renderChart(); renderMetrics(); filterMolecules();
}

function renderChart() {
    const arms = data[section].arms, key = el("chartMode").value;
    const points = arms.flatMap(a => a.waves.map(w => w[key])).filter(finite);
    if (!points.length) {el("chart").innerHTML = ""; el("waveText").textContent = "No completed waves."; return;}
    const min = Math.min(...points), max = Math.max(...points);
    const gap = Math.max(.005,(max-min)*.12);
    const low = min - gap, high = max + gap;
    const X = i => 61+(i-1)*101, Y = v => 17+(high-v)/(high-low)*145;
    let s = "";
    for(let i=0;i<=4;i++){
        const y = 17+i*145/4;
        s += '<path d="M54 '+y+'H575" stroke="#dbe6e1"/><text x="46" y="'+(y+4)+'" text-anchor="end">'+fmt(high-i*(high-low)/4)+'</text>';
    }
    for(let i=1;i<=6;i++) s += '<text x="'+X(i)+'" y="186" text-anchor="middle">'+i+'</text>';
    s += '<text x="313" y="209" text-anchor="middle">Wave</text><path d="M'+X(wave)+' 13 V164" stroke="#a4b4ab" stroke-dasharray="3 4"/>';
    arms.forEach((a,i) => {
        const pts = a.waves.filter(w=>finite(w[key]));
        s += '<polyline fill="none" stroke="'+colors[i]+'" stroke-width="2.6" '+(i?'stroke-dasharray="6 3"':'')+' points="'+pts.map(w=>X(w.wave)+','+Y(w[key])).join(' ')+'"/>';
        pts.forEach(w => {
            const x=X(w.wave),y=Y(w[key]), desc=escapeHtml(short(a)+" · wave "+w.wave+" · "+fmt(w[key],5));
            const shape = i ? '<rect x="'+(x-4)+'" y="'+(y-4)+'" width="8" height="8" rx="1"/>' : '<circle cx="'+x+'" cy="'+y+'" r="4.7"/>';
            s += '<g class="point" tabindex="0" role="button" aria-label="'+desc+'" data-wave="'+w.wave+'" fill="'+colors[i]+'" stroke="white" stroke-width="1.3">'+shape+'<title>'+desc+'</title></g>';
        });
    });
    el("chart").innerHTML=s;
    function showWave(n){wave=Number(n);renderChart();}
    el("chart").querySelectorAll("[data-wave]").forEach(p=>{
        p.onclick=()=>showWave(p.dataset.wave);
        p.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();showWave(p.dataset.wave);}};
    });
    el("waveText").textContent="Wave "+wave+"  ·  "+arms.map(a=>short(a)+" "+fmt(a.waves.find(w=>w.wave===wave)?.[key],4)).join("   |   ");
}
function renderMetrics(){
    const m02=section==="m02", arms=data[section].arms;
    el("comparisonTitle").textContent=m02?"Which molecules beat M02?":"Broad search yield";
    el("comparisonSub").textContent=m02?"Same founder and seed · 9,000 latent updates per run":"Independent founder panels · 22,000 latent updates per run";
    const index = m02 ? [
        ["Distinct new molecules scored","scored"],["Better contact","contact_hits"],["Better affinity","affinity_hits"],
        ["Better contact and affinity","joint_hits"],["Also better binder probability","triple_hits"]
    ] : [["Distinct new molecules scored","scored"],["Discovery score attempts","attempts"],["Best contact loss","best_contact"],["Wave of best score","peak_wave"]];
    el("metrics").innerHTML="<thead><tr>"+th("Outcome")+arms.map(a=>th(short(a))).join("")+"</tr></thead><tbody>"+
        index.map(([label,k])=>'<tr class="'+(k==="joint_hits"||k==="best_contact"?"em":"")+'">'+td(label)+arms.map(a=>{
            if(k==="peak_wave") return td(a.waves.find(w=>Math.abs(w.best_contact-a.best_contact)<1e-8)?.wave);
            return td(k==="best_contact"?fmt(a[k],4):num(a[k]));
        }).join("")+"</tr>").join("")+"</tbody>";
    el("comparisonNote").textContent=m02?"Affinity scores were available for "+arms.map(a=>a.nesso_scored+" of "+a.scored+" "+short(a)+" molecules").join(" and ")+".":"Broad A’s valid founders lack discovery contact scores. In Broad B, 191 of 199 scored descendants beat their own valid founder.";
}
function filterMolecules(){
    const q=el("molSearch").value.trim().toLowerCase(), run=el("runFilter").value, hit=el("hitFilter").value, sort=el("molSort").value;
    filtered=records.filter(r=>(run==="all"||r.condition===run) && (hit==="all"||(hit==="founder"?r.founder===true:r[hit]===true)) &&
        (!q||String(r.smiles+" "+(r.branch??"")).toLowerCase().includes(q)));
    filtered.sort((a,b)=>{
        const x=a[sort],y=b[sort];if(!finite(x))return finite(y)?1:0;if(!finite(y))return -1;
        return (sort==="probability"?-1:1)*(x-y);
    });
    if(!filtered.includes(current)) current=filtered[0]||null;
    el("matches").textContent=num(filtered.length)+" matching records";
    el("molRows").innerHTML=filtered.map((r,i)=>'<tr data-n="'+i+'" class="'+(r===current?"selected":"")+'"><td><button type="button">'+escapeHtml(armLabel(r.condition))+'</button></td>'+
        td(r.founder===true?"Founder":(r.wave??"—")+" / "+(r.step??"—"))+td(fmt(r.contact,4))+td(fmt(section==="m02"?r.affinity:r.sa))+'</tr>').join("");
    el("molRows").querySelectorAll("tr").forEach(tr=>tr.onclick=()=>{current=filtered[Number(tr.dataset.n)];el("molRows").querySelectorAll("tr").forEach(x=>x.classList.toggle("selected",x===tr));detail();});
    detail();
}
function structure(s){
    return data.drawings[s]?'<img alt="2D molecular structure" src="data:image/webp;base64,'+data.drawings[s]+'">':'<p class="note">Structure drawing unavailable.</p>';
}
function detail(){
    const r=current, elview=el("molView");
    const job=fullArm(r?.condition||data[section].arms[0].condition).job;
    const branch=r?.branch?.match(/branch-\d+$/)?.[0];
    el("treeLink").href=r && branch ? "trees.html?"+new URLSearchParams({run:job,cycle:r.wave,branch,step:r.step}) : tree(job);
    el("treeLink").textContent=r && branch ? "Trace selected molecule ↗" : "Open search tree ↗";
    if(!r){elview.textContent="No molecule matches these filters.";return;}
    const title=r.founder===true?"Founder M02":armLabel(r.condition)+" · Wave "+(r.wave??"—")+", step "+(r.step??"—");
    const fields=[["Contact loss",r.contact],["Change in contact",r.contact_delta],["SA score",r.sa]];
    if(section==="m02") fields.push(["Affinity score",r.affinity],["Binder probability",r.probability],["Change in affinity",r.affinity_delta]);
    for(const [seed,score] of Object.entries(r.held_out||{})) fields.push(["Contact · geometry "+seed,score]);
    elview.innerHTML="<h3>"+escapeHtml(title)+"</h3><small>"+escapeHtml(r.branch||"")+"</small>"+structure(r.smiles)+'<div class="detailgrid">'+fields.map(([label,x])=>"<div><small>"+label+"</small><b>"+fmt(x)+"</b></div>").join("")+'</div><div class="smiles">'+escapeHtml(r.smiles)+"</div>";
}
const statusName = s => ({RUN:"Running",PEND:"Queued",DONE:"Completed",completed:"Completed",running:"Running",pending:"Queued",queued:"Queued"}[s] || s || "Unknown");
function directory(){
    let out="";
    for(const key of ["broad","m02","erk2"]){
        for(const a of data[key].arms){
            const st=key==="erk2"?statusName(a.status):"Completed";
            const title=key==="erk2"?"ERK2 · "+a.label:key==="broad"?"BRD4 · "+short(a):"M02 · "+short(a);
            out+="<tr>"+td(title)+td(a.job)+'<td class="status '+(st==="Queued"?"pending":"")+'">'+escapeHtml(st)+'</td>'+td(num(a.updates))+
                '<td>'+(key==="erk2"?'<a href="#erk2" data-erk>View run</a>':'<a href="'+tree(a.job)+'">Explore tree ↗</a>')+"</td></tr>";
        }
    }
    el("directory").innerHTML=out; el("directory").querySelectorAll("[data-erk]").forEach(a=>a.onclick=e=>{e.preventDefault();selectSection("erk2");el("erkPanel").scrollIntoView();});
    el("snapshot").textContent="Scheduler snapshot · "+(data.erk2.checked_at?new Date(data.erk2.checked_at).toLocaleString("en-GB",{dateStyle:"medium",timeStyle:"short",timeZone:"Europe/Paris"})+" Paris time":"25 November 2026");
}
function renderErk(){
    el("erkContent").innerHTML=data.erk2.arms.map(a=>{
        const founders=a.founders||[], rows=[...(a.observations||[]),...founders];
        const title='<h3>'+escapeHtml(a.label)+' · '+escapeHtml(statusName(a.status))+'</h3>';
        let html='<p class="note">Job '+escapeHtml(a.job)+' · '+escapeHtml(a.queue||a.gpu||"")+'</p>';
        if(rows.length){
            html+='<p>'+num(a.updates)+' observed updates · '+num(a.waves)+' of 6 waves complete · '+num(founders.length)+' scored founders.</p>';
            html+='<p class="note">Best founder loss '+fmt(a.founder_best,4)+' · best newly scored molecule '+fmt(a.best_new,4)+'. Step zero is the perturbed child before its first optimizer update.</p>';
            html+='<div class="toolbar"><label>Molecule<select data-erkpick="'+escapeHtml(a.condition)+'">'+rows.map((r,i)=>opt(i,(r.founder?"Founder · "+r.arm:"Wave "+r.wave+" · "+r.child+" · step "+r.step)+" · "+fmt(r.loss,4))).join("")+'</select></label></div><div class="mol-view" id="erkMol-'+escapeHtml(a.condition)+'"></div>';
        } else if(statusName(a.status)==="Queued")html+='<p class="note">Awaiting a GPU allocation.</p>';
        return title+html;
    }).join("");
    el("erkContent").querySelectorAll("[data-erkpick]").forEach(select=>{
        const arm=data.erk2.arms.find(a=>a.condition===select.dataset.erkpick), rows=[...arm.observations,...arm.founders];
        const display=()=>{const r=rows[Number(select.value)];el("erkMol-"+arm.condition).innerHTML=structure(r.smiles)+'<div class="detailgrid"><div><small>DBMol loss</small><b>'+fmt(r.loss,4)+'</b></div><div><small>SA score</small><b>'+fmt(r.sa,2)+'</b></div><div><small>Source</small><b>'+(r.founder?"Founder":"Child")+'</b></div></div><div class="smiles">'+escapeHtml(r.smiles)+'</div>';};
        select.onchange=display;display();
    });
}
document.querySelectorAll("[data-set]").forEach(b=>b.onclick=()=>selectSection(b.dataset.set));
["runFilter","hitFilter","molSort"].forEach(id=>el(id).onchange=filterMolecules);
el("molSearch").oninput=filterMolecules;el("chartMode").onchange=renderChart;
window.addEventListener("hashchange",()=>selectSection(location.hash.slice(1),false));
el("headline").textContent="Explore the BRD4 search trees, compare the Protenix and Nesso-1 predictions from founder M02, and follow the ERK2 runs.";
directory();selectSection(location.hash.slice(1),false);
