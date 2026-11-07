"use strict";
const snapshot = JSON.parse(document.getElementById("snapshot").textContent);
const jobs = snapshot.jobs;
const depictions = snapshot.depictions || {};
const structures = snapshot.structures || {};
const byId = id => document.getElementById(id);
const escape = value => String(value ?? "").replace(/[&<>"']/g, ch =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[ch]);
const num = value => Number.isFinite(value) ? value : null;
const fmt = (value, digits = 3) => num(value) === null ? "—" : value.toFixed(digits);
const count = value => num(value) === null ? "—" : value.toLocaleString("en-US");
const percent = (a, b) => num(a) === null || !(b > 0) ? "—" : `${(a / b * 100).toFixed(1)}%`;
const captureFormatter = new Intl.DateTimeFormat("en-GB", {
  timeZone:"Europe/Paris",day:"numeric",month:"short",year:"numeric",hour:"2-digit",minute:"2-digit",timeZoneName:"short",
});
const label = name => ({
  mutation:"Restart perturbation",cap:"Step cap",duration:"Branch length",lineage:"Elite versus independent",
  constant1:"Constant chemistry 1",constant10:"Constant chemistry 10","late-ramp":"Late chemistry ramp",
  untouched:"Untouched latent","all-row":"All-row 50/50",latent:"Latent kick",sparse:"Sparse perturbation",
  "cap-01":"Cap 0.1","cap-05":"Cap 0.5","four-short":"Four short branches","one-long":"One long branch",
  elite:"Elite restarts",independent:"Independent starts",mixed:"Mixed shortlist",relaxed:"Relaxed-rank shortlist",
})[name] || name;
const color = name => ({
  latent:"#13695e",sparse:"#5863a3",untouched:"#486f85","all-row":"#a25c26",
  "cap-01":"#5863a3","cap-05":"#13695e","four-short":"#13695e","one-long":"#a25c26",
  elite:"#13695e",independent:"#5863a3",constant1:"#13695e",constant10:"#5863a3","late-ramp":"#a25c26",
})[name] || "#486f85";
const state = {study:"mutation", panel:"", shortlist:"mixed", repeat:"matched", parent:"all",
  job:"", cycle:"", branch:"", metric:"relaxed", archive:"chemistry", archiveCycle:"", requestedStep:null};
let stopPlayback = () => {};
const statusNames = {DONE:"Completed",RUN:"Running",PEND:"Queued",EXIT:"Exited",UNKNOWN:"Unknown"};
const status = job => String(job.status || "UNKNOWN").toUpperCase();
const isDone = job => ["DONE","COMPLETED"].includes(status(job));
const title = job => job.campaign === "chemistry" ? label(job.task) :
  `${label(job.task.split("-")[0])} · panel ${job.task.split("-").at(-1).toUpperCase()}`;
const badge = job => `<span class="pill status-${escape(status(job).toLowerCase())}">${escape(statusNames[status(job)] || status(job))}</span>`;
const empty = message => `<div class="empty">${escape(message)}</div>`;
const metric = (value, name) => `<div class="metric"><strong>${escape(value)}</strong><span>${escape(name)}</span></div>`;
const option = (value, text, current) => `<option value="${escape(value)}"${String(value) === String(current) ? " selected" : ""}>${escape(text)}</option>`;
const select = (id, text, items, current) => `<label for="${id}">${escape(text)}<select id="${id}">${items.map(item => option(...item,current)).join("")}</select></label>`;
const pick = (current, entries, fallback) => entries.some(([key]) => String(key) === String(current)) ? String(current) : String(fallback ?? entries[0]?.[0] ?? "");
const table = (heads, rows, numeric = []) => `<div class="table-wrap"><table><thead><tr>${heads.map((head,i) => `<th scope="col"${numeric.includes(i) ? ' class="n"' : ""}>${escape(head)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((cell,i) => `<td${numeric.includes(i) ? ' class="n"' : ""}>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
const svg = (body, height, description) => `<div class="plot-wrap"><svg class="plot" viewBox="0 0 820 ${height}" role="img" aria-label="${escape(description)}">${body}</svg></div>`;
const text = (x,y,content,anchor = "start",cls = "") => `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${escape(content)}</text>`;
const line = (x1,y1,x2,y2,cls) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"/>`;
const titleMark = description => `<title>${escape(description)}</title>`;
const scale = (value,from,to,first,last) => first+(value-from)*(last-first)/(to-from || 1);

function switchTab(name) {
  stopPlayback();
  if (name !== "trees") {
    RestartTreeView.destroy();
    renderHeader(jobs,snapshot.captured_at);
  }
  for (const tab of document.querySelectorAll("nav [role=tab]")) {
    const active = tab.getAttribute("aria-controls") === name;
    tab.setAttribute("aria-selected", String(active));
    byId(tab.getAttribute("aria-controls")).hidden = !active;
  }
  if (name === "comparisons") renderComparisons();
  if (name === "trees") RestartTreeView.render();
  if (name === "trajectories") renderTrajectories();
  if (name === "archives") renderArchives();
}
for (const tab of document.querySelectorAll("nav [role=tab]")) {
  tab.addEventListener("click", () => switchTab(tab.getAttribute("aria-controls")));
  tab.addEventListener("keydown", event => {
    const tabs = [...document.querySelectorAll("nav [role=tab]")];
    if (!["ArrowRight","ArrowLeft"].includes(event.key)) return;
    event.preventDefault();
    const next = tabs[(tabs.indexOf(tab)+(event.key === "ArrowRight" ? 1 : -1)+tabs.length)%tabs.length];
    next.focus();
    next.click();
  });
}

function openJob(jobId, cycle = "", branch = "", step = null) {
  Object.assign(state,{job:String(jobId),cycle:String(cycle),branch:String(branch),requestedStep:step});
  switchTab("trajectories");
}

function renderHeader(currentJobs,capturedAt) {
  const date=new Date(capturedAt);
  const captureLabel=Number.isFinite(date.valueOf())?captureFormatter.format(date):capturedAt;
  const statuses = ["DONE","RUN","PEND","EXIT"].map(key =>
    [key,currentJobs.filter(job => key === "DONE" ? isDone(job) : status(job) === key).length]).filter(([,n]) => n);
  byId("header-info").innerHTML = statuses.map(([key,n]) =>
    `<span class="pill status-${key.toLowerCase()}">${n} ${escape(statusNames[key].toLowerCase())}</span>`).join("") +
    `<span class="quiet small">Snapshot ${escape(captureLabel)}</span><span class="quiet small">LR 150 · epoch-45 VAE · 25 occupied slots</span>`;
}

function renderOverview() {
  renderHeader(jobs,snapshot.captured_at);
  byId("overview-content").innerHTML = [["followup","Matched follow-ups"],["chemistry","Original chemistry schedules"]]
    .map(([campaign,heading]) => `<h3 class="group-name">${heading}</h3>` + table(
      ["Run","Job","Queue","Status","Saved updates","Scheduler note"],
      jobs.filter(job => job.campaign === campaign).map(job => {
        const total = num(job.budget_updates), done = num(job.completed_updates);
        const ratio = total && done !== null ? Math.min(100,100*done/total) : 0;
        const progress = `<div>${count(done)} / ${count(total)}</div><div class="meter" aria-hidden="true"><span style="width:${ratio}%"></span></div>`;
        return [`<button type="button" class="run-button" data-open-job="${escape(job.id)}">${escape(title(job))}</button>`,
          escape(job.id),escape(job.queue),badge(job),progress,escape(job.reason || "—")];
      }),[1,4])).join("");
  for (const btn of byId("overview-content").querySelectorAll("[data-open-job]")) {
    btn.addEventListener("click", () => openJob(btn.dataset.openJob));
  }
}

function gainPlot(groups) {
  const all = groups.flatMap(group => group.rows.map(row => row.gain)).filter(Number.isFinite);
  const lo = Math.min(0,...all), hi = Math.max(.05,...all);
  const width = hi-lo;
  const domain = [lo-width*.03,hi+width*.10];
  const x = value => scale(value,...domain,160,766);
  const h = 73 + groups.length*45;
  let marks = "";
  for (let i=0;i<=5;i++) {
    const value = domain[0]+(domain[1]-domain[0])*i/5;
    marks += line(x(value),17,x(value),h-38,"grid") + text(x(value),h-22,fmt(value,2),"middle");
  }
  marks += line(x(0),17,x(0),h-38,"axis") + text(462,h-4,"Fresh loss reduction (higher is better)","middle");
  groups.forEach(({arm,rows},index) => {
    const mean = rows.reduce((sum,row) => sum+row.gain,0)/rows.length;
    const y = 34+index*45;
    marks += text(147,y+4,label(arm),"end","value");
    marks += `<rect x="${Math.min(x(0),x(mean))}" y="${y-9}" width="${Math.max(1,Math.abs(x(mean)-x(0)))}" height="18" rx="4" fill="${color(arm)}" opacity=".42">${titleMark(`${label(arm)} mean ${fmt(mean)}`)}</rect>`;
    for (const row of rows) {
      const yy = y+((Number(row.parent_index)*13)%7-3)*3;
      marks += `<circle cx="${x(row.gain)}" cy="${yy}" r="4" fill="${color(arm)}" stroke="#fffefa" tabindex="0" data-parent="${escape(row.parent_index)}" aria-label="Parent ${Number(row.parent_index)+1}, ${label(arm)}, gain ${fmt(row.gain)}">${titleMark(`Parent ${Number(row.parent_index)+1} · ${fmt(row.gain)}`)}</circle>`;
    }
    marks += text(807,y+4,fmt(mean),"end","value");
  });
  return svg(marks,h,"Fresh loss reductions by policy. Dots are parents and bars are means.");
}

function renderComparisons() {
  const studies = ["mutation","cap","duration","lineage"].map(value => [value,label(value)]);
  const panels = jobs.filter(job => job.campaign === "followup" && job.task.startsWith(state.study+"-"));
  const panelOptions = panels.map(job => [job.task,`${job.task.at(-1).toUpperCase()} · ${job.id} · ${statusNames[status(job)] || status(job)}`]);
  state.panel = pick(state.panel,panelOptions,panels.find(job => job.result)?.task);
  const job = panels.find(row => row.task === state.panel);
  const paired = job?.result?.paired || [];
  const match = job?.result?.matched_first_branch;
  const source = state.repeat === "matched" && match?.length ? match : paired;
  const shortlists = [...new Set(source.map(row => row.shortlist))].map(value => [value,label(value)]);
  state.shortlist = pick(state.shortlist,shortlists,"mixed");
  const subset = source.filter(row => row.shortlist === state.shortlist && Number.isFinite(row.gain));
  const parents = [["all","All parents"],...[...new Set(subset.map(row => row.parent_index))].sort((a,b)=>a-b).map(i=>[i,`Parent ${i+1}`])];
  state.parent = pick(state.parent,parents);
  const rows = subset.filter(row => state.parent === "all" || String(row.parent_index) === state.parent);
  let controls = select("study","Comparison",studies,state.study) + select("panel","Panel",panelOptions,state.panel);
  if (shortlists.length > 1) controls += select("shortlist","Shortlist policy",shortlists,state.shortlist);
  if (match?.length) controls += select("repeat","Branch budget",[["matched","First branch matched"],["all","All branches"]],state.repeat);
  if (subset.length) controls += select("parent","Parent",parents,state.parent);
  byId("comparison-controls").innerHTML = controls;
  for (const key of ["study","panel","shortlist","repeat","parent"]) {
    byId(key)?.addEventListener("change", event => {
      state[key] = event.target.value;
      if (key === "study" || key === "panel") state.parent = "all";
      renderComparisons();
    });
  }
  const target = byId("comparison-content");
  if (!rows.length) {
    target.innerHTML = state.study === "lineage" && job?.archives?.length
      ? `<div class="empty">Saved lineage scores are available in Elite archives. <button type="button" id="view-lineage" class="run-button">Open archive</button></div>`
      : empty(job?.reason || "No completed paired scores in this snapshot.");
    byId("view-lineage")?.addEventListener("click", () => {
      state.archive = job.task;
      switchTab("archives");
    });
    return;
  }
  const groups = [...new Set(rows.map(row=>row.arm))].map(arm=>({arm,rows:rows.filter(row=>row.arm===arm)}));
  const summaryRows = groups.map(group => {
    const indices = new Set(group.rows.map(row=>row.parent_index));
    const traces = (job.branches || []).filter(branch=>branch.arm===group.arm && indices.has(branch.parent_index)
      && (!match?.length || state.repeat==="all" || branch.repeat===0));
    const rawValid = traces.reduce((sum,branch)=>sum+(branch.valid_states || 0),0);
    const total = traces.reduce((sum,branch)=>sum+(branch.states || 0),0);
    return [escape(label(group.arm)),fmt(group.rows.reduce((sum,row)=>sum+row.gain,0)/group.rows.length),
      `${group.rows.filter(row=>row.gain>0).length} / ${group.rows.length}`,percent(rawValid,total),
      count(group.rows.reduce((sum,row)=>sum+(row.requested_candidates || 0),0))];
  });
  target.innerHTML = gainPlot(groups) +
    `<p class="quiet small">Dots show the best reduction for each parent. Select a dot to isolate that parent.</p>` +
    table(["Policy","Mean gain","Parents improved","Raw-valid steps","Fresh candidate slots used"],summaryRows,[1,2,3,4]) +
    `<details><summary>Exact paired values</summary>` + table(["Parent","Policy","Parent loss","Best loss","Gain"],
      rows.map(row=>[escape(Number(row.parent_index)+1),escape(label(row.arm)),fmt(row.parent_fresh_loss),fmt(row.best),fmt(row.gain)]),[2,3,4]) + "</details>" +
    (state.repeat === "all" && match?.length ? `<p class="quiet small">The untouched policy has one deterministic branch per parent. The three stochastic policies have two.</p>` : "");
  for (const dot of target.querySelectorAll("[data-parent]")) {
    const activate = () => {state.parent=dot.dataset.parent;renderComparisons();byId("parent")?.focus();};
    dot.addEventListener("click",activate);
    dot.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();activate();}});
  }
}

function drawStructure(structure, description) {
  if (!structure?.atoms?.length) return "";
  const dense = structure.raw && structure.bonds.length > 2.2*structure.atoms.length;
  const atoms = structure.atoms.map(([x,y,label,charge],i)=>{
    const angle=2*Math.PI*i/structure.atoms.length-Math.PI/2;
    return dense ? [170+91*Math.cos(angle),120+91*Math.sin(angle),label,charge] : [62+216*x,12+216*y,label,charge];
  });
  const palette = {N:"#244eae",O:"#b52e36",S:"#946510",P:"#92572b",F:"#29703a",Cl:"#29703a",Br:"#813e28",I:"#6d427d",H:"#526858"};
  const bonds = structure.bonds.map(([first,second,kind])=>{
    if(!atoms[first]||!atoms[second])return "";
    const [x1,y1]=atoms[first], [x2,y2]=atoms[second];
    const dx=x2-x1, dy=y2-y1, length=Math.hypot(dx,dy)||1;
    const ox=dy/length, oy=-dx/length;
    const draw = (offset, dash = "") => `<line x1="${x1+ox*offset}" y1="${y1+oy*offset}" x2="${x2+ox*offset}" y2="${y2+oy*offset}" stroke-width="${dense ? .75 : dash ? 1.4 : 1.8}"${dense?' opacity=".21"':""}${dash?' stroke-dasharray="3 2"':""}/>`;
    switch (String(kind).toUpperCase()) {
      case "DOUBLE": return draw(-1.9)+draw(1.9);
      case "TRIPLE": return draw(-3.1)+draw(0)+draw(3.1);
      case "AROMATIC": return draw(-1.5)+draw(1.5,"dash");
      default: return draw(0);
    }
  }).join("");
  const symbols = atoms.map(([x,y,label,q])=>{
    const ink=palette[label]||"#263b2d";
    if(label==="C"&&!q)return `<circle cx="${x}" cy="${y}" r="1.9" fill="#45574c"/>`;
    const magnitude = Math.abs(q);
    const charge=q?`${magnitude>1?magnitude:""}${q>0?"+":"−"}`:"";
    return `<text x="${x}" y="${y+4}" font-size="13" text-anchor="middle" fill="${ink}">${escape(label)}</text>`+
      (charge?`<text class="atom-charge" x="${x+8}" y="${y-4}" fill="${ink}">${charge}</text>`:"");
  }).join("");
  return `<svg class="structure-svg" viewBox="0 0 340 240" role="img" aria-label="${escape(description)}">${bonds}${symbols}</svg>`;
}

function structureForPoint(point) {
  if (!point) return null;
  return structures[point[4] ? point[5] : point[7]] || structures[point[7]] || null;
}

function moleculeCard(row, tag = "", action = null) {
  const depiction = depictions[row.smiles];
  const image = drawStructure(structures[row.smiles],"2D depiction of the selected molecule") ||
    (depiction ? `<img src="${escape(depiction)}" alt="2D depiction of the selected molecule" loading="lazy">` : "");
  return `<article class="molecule"><h3>${escape(tag || "Candidate")}</h3><div class="small quiet">Fresh loss ${fmt(row.fresh_loss ?? row.loss,4)} · step ${count(row.step)}</div>${image}<code>${escape(row.smiles)}</code>${action||""}</article>`;
}

const series = {
  relaxed:[1,"Relaxed contact loss"],
  chemistry:[2,"Chemistry loss"],
  weight:[3,"Chemistry weight"],
  displacement:[6,"Outgoing latent displacement L2"],
};

function trajectoryPlot(points, field, maxStep) {
  const [column, description] = series[field];
  const numbers = points.map(row=>num(row[column])).filter(value=>value!==null);
  if (!numbers.length) return empty("This metric is unavailable for the selected branch.");
  let lo = Math.min(...numbers), hi = Math.max(...numbers);
  const margin = (hi-lo || Math.abs(lo) || 1)*.08;
  lo-=margin;hi+=margin;
  const x = step=>scale(step,0,maxStep,73,790);
  const y = value=>scale(value,lo,hi,183,24);
  const ticks = [];
  for(let i=0;i<=4;i++) {
    const xx = maxStep*i/4, yy=lo+(hi-lo)*i/4;
    ticks.push(line(x(xx),20,x(xx),191,"grid"),text(x(xx),214,Math.round(xx),"middle"),
      line(72,y(yy),791,y(yy),"grid"),text(66,y(yy)+4,fmt(yy,2),"end"));
  }
  const coords=points.filter(row=>num(row[column])!==null).map(row=>`${x(row[0])},${y(row[column])}`).join(" ");
  const valid=points.map(row=>`<circle cx="${x(row[0])}" cy="193" r="3" fill="${row[4]?"#138470":"#b27c51"}"/>`).join("");
  const body=ticks.join("")+line(73,191,791,191,"axis")+text(431,237,"Update","middle")+
    `<polyline points="${coords}" stroke="#13695e" stroke-width="2.5" fill="none"/>${valid}<line id="cursor-line" y1="20" y2="198" stroke="#9a561f" stroke-dasharray="4 3"/><circle id="cursor-dot" r="5" fill="#a25c26"/><rect class="hit-area" id="step-hit" x="73" y="20" width="717" height="179"/>`;
  return {html:svg(body,245,description+" by update with markers for sampled valid and invalid observations."),x,y,column};
}

function branchMap(branches, current, titleForBranch) {
  if (branches.length<2) return "";
  const trace = branches.flatMap(branch=>branch.points||[]).map(row=>row[1]).filter(Number.isFinite);
  if (!trace.length) return "";
  const min=Math.min(...trace),max=Math.max(...trace), limit=Math.max(1,...branches.map(branch=>branch.planned_updates||0));
  const rows=branches.map(branch=>{
    const coords=(branch.points||[]).filter(row=>Number.isFinite(row[1]))
      .map(row=>`${scale(row[0],0,limit,2,173)},${min===max?13:scale(row[1],min,max,23,3)}`).join(" ");
    const active=branch.name===current;
    return [`<button type="button" class="run-button" data-branch-jump="${escape(branch.name)}"${active?' aria-current="true"':""}>${escape(titleForBranch(branch))}</button>`,
      count(branch.completed_updates),percent(branch.valid_states,branch.states),
      `<svg width="175" height="26" viewBox="0 0 175 26" role="img" aria-label="Relaxed loss trace"><polyline points="${coords}" fill="none" stroke="${color(branch.arm)}" stroke-width="1.5"/></svg>`];
  });
  return `<details><summary>Compare all ${branches.length} branch traces in this phase</summary><p class="quiet small">The traces share update and relaxed-loss scales. Select a branch to open its molecule playback.</p>`+
    table(["Branch","Saved updates","Raw validity","Relaxed loss trace"],rows,[1,2])+"</details>";
}

function renderTrajectories() {
  stopPlayback();
  stopPlayback=()=>{};
  const choices=jobs.map(job=>[job.id,`${title(job)} · ${job.id}`]);
  const defaultJob=jobs.find(job=>job.campaign==="chemistry" && job.branches?.length) || jobs.find(job=>job.branches?.length);
  state.job=pick(state.job,choices,defaultJob?.id);
  const job=jobs.find(j=>String(j.id)===state.job);
  const branches=job?.branches || [];
  const cycleChoices=[...new Set(branches.map(branch=>branch.cycle ?? 0))].map(i=>[i,i ? `Cycle ${i}` : "Paired branches"]);
  state.cycle=pick(state.cycle,cycleChoices);
  const phase=branches.filter(branch=>String(branch.cycle ?? 0)===state.cycle);
  const branchTitle=branch=>branch.arm && branch.arm!==job.task && branch.parent_index!=null
    ? `${label(branch.arm)} · parent ${Number(branch.parent_index)+1} · branch ${Number(branch.repeat ?? 0)+1}` : branch.name;
  const branchOptions=phase.map(branch=>[branch.name,branchTitle(branch)]);
  state.branch=pick(state.branch,branchOptions);
  const branch=phase.find(row=>row.name===state.branch);
  byId("trajectory-controls").innerHTML=select("trajectory-job","Run",choices,state.job) +
    (cycleChoices.length>1?select("trajectory-cycle","Cycle",cycleChoices,state.cycle):"") +
    (branchOptions.length?select("trajectory-branch","Branch",branchOptions,state.branch):"") +
    (branch?.points?.length?select("trajectory-metric","Metric",Object.entries(series).map(([key,v])=>[key,v[1]]),state.metric):"");
  for (const key of ["job","cycle","branch","metric"]) {
    byId("trajectory-"+key)?.addEventListener("change",event=>{
      state[key]=event.target.value;
      if (key==="job") state.cycle="";
      if (key==="job" || key==="cycle") state.branch="";
      if (key!=="metric") state.requestedStep=null;
      renderTrajectories();
    });
  }
  const target=byId("trajectory-content");
  if(!branch?.points?.length) {
    target.innerHTML=empty(job?.reason || "No saved trajectory observations in this snapshot.");
    return;
  }
  const points=branch.points;
  const nearest = step => points.reduce((best,row,i)=>Math.abs(row[0]-step)<Math.abs(points[best][0]-step)?i:best,0);
  const branchStart=points[0];
  const parentIdentity=branch.parent_smiles || (branchStart[4] ? branchStart[5] : null);
  const reference=structures[parentIdentity] || structureForPoint(branchStart);
  const referenceSmiles=structures[parentIdentity] ? parentIdentity : branchStart[5];
  const referenceTitle=branch.parent_smiles && structures[parentIdentity] ? "Restart parent" : "Branch start";
  const referenceLabel=branch.parent_smiles && structures[parentIdentity]
    ? (Number.isFinite(branch.parent_fresh_loss) ? `Fresh loss ${fmt(branch.parent_fresh_loss,4)}` : "Saved parent graph")
    : `Update ${branchStart[0]} · ${branchStart[4] ? "Valid" : "Raw graph"}`;
  const maxStep=Math.max(branch.planned_updates||0,points.at(-1)[0],1);
  const plot=trajectoryPlot(points,state.metric,maxStep);
  const proposal=branch.proposal;
  const context=proposal?.parent && proposal?.proposal ? `<p class="quiet small">${escape(label(branch.arm))} · expected soft bonds ${fmt(proposal.parent.soft_bonds,1)} → ${fmt(proposal.proposal.soft_bonds,1)} at start · ${count(proposal.proposal.edit_count)} hard label edits</p>` : "";
  const candidates=[...new Map(Object.values(branch.candidates||{}).flat()
    .filter(row=>Number.isFinite(row.loss)).map(row=>[row.smiles,row])).values()].sort((a,b)=>a.loss-b.loss);
  const viewButton = row => `<button type="button" class="run-button" data-frame-step="${escape(row.step)}">${points[nearest(row.step)][0]===row.step?"View frame":"Nearest frame"}</button>`;
  const referenceCard = `<article class="structure-panel"><header><h3>${referenceTitle}</h3><p class="quiet small">${referenceLabel}</p></header>`+
    (drawStructure(reference,"2D structure of the restart reference") || empty("The start graph is unavailable in this snapshot."))+
    (referenceSmiles?`<code>${escape(referenceSmiles)}</code>`:"")+"</article>";
  target.innerHTML=`<div class="playback"><div class="playback-controls"><button type="button" id="previous" aria-label="Previous saved frame">Previous</button><button type="button" id="play" aria-pressed="false">Play</button><button type="button" id="next" aria-label="Next saved frame">Next</button><label class="speed-control" for="speed">Speed <select id="speed"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label><span id="frame-index" class="quiet small frame-index"></span></div><label class="slider-label" for="point">Saved trajectory frames<input id="point" type="range" min="0" max="${points.length-1}" value="0"></label></div><div class="structure-grid">${referenceCard}<article class="structure-panel current" id="selected-molecule"></article></div><div id="point-detail"></div>`+
    `<div class="metrics">${metric(`${count(branch.completed_updates)} / ${count(branch.planned_updates)}`,"Updates saved")}${metric(percent(branch.valid_states,branch.states),"Raw-valid observations")}${metric(count(branch.distinct_valid),"Distinct valid molecules")}${metric(percent(branch.bounded_steps,branch.completed_updates),"Steps at cap")}</div>`+context+
    (plot?.html||plot)+`<div class="legend quiet"><span><i class="swatch" style="background:#138470"></i>Valid sample</span><span><i class="swatch" style="background:#b27c51"></i>Invalid sample</span><span>${points.length} saved display points</span></div>`+
    (candidates.length?`<hr><h3>Fresh-scored candidates</h3><div class="cards">${candidates.slice(0,2).map((row,i)=>moleculeCard(row,`#${i+1}`,viewButton(row))).join("")}</div><details><summary>All scored molecules in the saved shortlist</summary>`+
      table(["Step","Fresh loss","SMILES","Trajectory"],candidates.map(row=>[count(row.step),fmt(row.loss,4),`<code>${escape(row.smiles)}</code>`,viewButton(row)]),[0,1])+"</details>":"")+
    branchMap(phase,branch.name,branchTitle);
  let timer=null;
  const toggle=byId("play");
  stopPlayback=()=>{
    clearInterval(timer);timer=null;
    toggle.textContent="Play";toggle.setAttribute("aria-pressed","false");
  };
  function setPoint(index) {
    const row=points[index], [step,relaxed,chem,weight,valid,smiles,displacement]=row;
    const slider=byId("point");
    slider.value=index;slider.setAttribute("aria-valuetext",`Update ${step}`);
    state.requestedStep=step;
    byId("frame-index").textContent=`Frame ${index+1} of ${points.length} · update ${step}`;
    byId("previous").disabled=index===0;byId("next").disabled=index===points.length-1;
    const structure=structureForPoint(row);
    const detail=valid && parentIdentity ? (smiles===parentIdentity ? "Same molecule as reference" : "Different valid molecule") :
      valid ? "Connected valid graph" : structure ? `Raw decoded topology · ${structure.atoms.length} atoms · ${structure.bonds.length} bonds` : "Raw decoded topology";
    const statusText=valid?"Valid":"Invalid";
    const statusStyle=valid?"status-done":"status-invalid";
    byId("selected-molecule").innerHTML=`<header><h3>Update ${step} <span class="pill ${statusStyle}">${statusText}</span></h3><p class="quiet small">${detail}</p></header>`+
      (drawStructure(structure,`${statusText} 2D graph at update ${step}`) || empty("This frame has a saved score and validity, but its structure is unavailable."))+
      (smiles?`<code>${escape(smiles)}</code>`:"");
    byId("point-detail").innerHTML=`<div class="metrics">${metric(fmt(relaxed,4),"Relaxed contact loss")}${metric(fmt(chem,4),"Chemistry loss")}${metric(fmt(weight,3),"Chemistry weight")}${metric(fmt(displacement,4),"Outgoing L2 step")}</div>`;
    if (plot?.x) {
      const cursor=byId("cursor-line"),dot=byId("cursor-dot");
      cursor.setAttribute("x1",plot.x(step));cursor.setAttribute("x2",plot.x(step));
      const score=num(row[plot.column]);
      dot.style.display=score===null?"none":"";
      if (score!==null){dot.setAttribute("cx",plot.x(step));dot.setAttribute("cy",plot.y(score));}
    }
  }
  function play() {
    toggle.textContent="Pause";toggle.setAttribute("aria-pressed","true");
    timer=setInterval(()=>{
      const current=Number(byId("point").value);
      if(current===points.length-1){stopPlayback();return;}
      setPoint(current+1);
      if(current+1===points.length-1)stopPlayback();
    },380/Number(byId("speed").value));
  }
  toggle.disabled=points.length<2;
  toggle.addEventListener("click",()=>{
    if(timer!==null){stopPlayback();return;}
    if(Number(byId("point").value)===points.length-1)setPoint(0);
    play();
  });
  byId("speed").addEventListener("change",()=>{if(timer!==null){stopPlayback();play();}});
  byId("previous").addEventListener("click",()=>{stopPlayback();setPoint(Number(byId("point").value)-1);});
  byId("next").addEventListener("click",()=>{stopPlayback();setPoint(Number(byId("point").value)+1);});
  byId("point").addEventListener("input",event=>{stopPlayback();setPoint(Number(event.target.value));});
  for (const button of target.querySelectorAll("[data-frame-step]")) button.addEventListener("click",()=>{
    stopPlayback();setPoint(nearest(Number(button.dataset.frameStep)));
    byId("selected-molecule").scrollIntoView({block:"nearest"});
  });
  for (const button of target.querySelectorAll("[data-branch-jump]")) button.addEventListener("click",()=>{
    state.branch=button.dataset.branchJump;
    state.requestedStep=null;
    renderTrajectories();
    byId("trajectory-branch")?.focus();
  });
  byId("step-hit")?.addEventListener("pointermove",event=>{
    if(timer!==null)return;
    const rect=event.currentTarget.getBoundingClientRect();
    const step=scale(event.clientX,rect.left,rect.right,0,maxStep);
    const index=nearest(step);
    if(index!==Number(byId("point").value))setPoint(index);
  });
  setPoint(state.requestedStep===null?0:nearest(state.requestedStep));
}

function archivePlot(rows) {
  const values=rows.flatMap(row=>row.points.map(p=>p.loss));
  const lo=Math.min(...values),hi=Math.max(...values),margin=(hi-lo||.06)*.14;
  const maximum=Math.max(5,...rows.flatMap(row=>row.points.map(p=>p.cycle)));
  const x=value=>scale(value,1,maximum,71,770),y=value=>scale(value,lo-margin,hi+margin,180,24);
  let body="";
  for(let i=1;i<=maximum;i++)body+=line(x(i),19,x(i),186,"grid")+text(x(i),207,i,"middle");
  for(let i=0;i<=4;i++){
    const v=lo-margin+(hi-lo+2*margin)*i/4;
    body+=line(70,y(v),770,y(v),"grid")+text(64,y(v)+4,fmt(v,2),"end");
  }
  body+=text(72,14,"Best fresh contact loss")+text(420,231,"Cycle","middle");
  for(const row of rows){
    body+=`<polyline points="${row.points.map(p=>`${x(p.cycle)},${y(p.loss)}`).join(" ")}" fill="none" stroke="${color(row.arm)}" stroke-width="2.5"/>`;
    for(const p of row.points){
      const desc=`${label(row.arm)} · cycle ${p.cycle} · fresh loss ${fmt(p.loss,4)}`;
      body+=`<circle cx="${x(p.cycle)}" cy="${y(p.loss)}" r="6" fill="${color(row.arm)}" stroke="#fffefa" tabindex="0" data-cycle="${p.cycle}" aria-label="${escape(desc)}">${titleMark(desc)}</circle>`;
    }
  }
  return svg(body,239,"Best saved fresh Protenix loss by cycle. Lower is better.");
}

function renderArchives(){
  const options=[["chemistry","Original chemistry schedules"],...jobs.filter(job=>job.task.startsWith("lineage-")).map(job=>[job.task,title(job)])];
  const relevant=jobs.filter(job=>state.archive==="chemistry"?job.campaign==="chemistry":job.task===state.archive);
  const rows=[];
  for(const job of relevant)for(const archive of job.archives||[])for(const [arm,elites] of Object.entries(archive.elites||{})){
    const available=elites.filter(row=>Number.isFinite(row.fresh_loss)).sort((a,b)=>a.fresh_loss-b.fresh_loss);
    if(!available.length)continue;
    let series=rows.find(row=>row.arm===arm && row.job.id===job.id);
    if(!series){series={arm,job,points:[]};rows.push(series);}
    series.points.push({cycle:Number(archive.cycle),loss:available[0].fresh_loss,elites:available});
  }
  const cycleOptions=[...new Set(rows.flatMap(row=>row.points.map(p=>p.cycle)))].sort((a,b)=>a-b).map(i=>[i,`Cycle ${i}`]);
  state.archiveCycle=pick(state.archiveCycle,cycleOptions,cycleOptions.at(-1)?.[0]);
  byId("archive-controls").innerHTML=select("archive-family","Campaign",options,state.archive)+
    (cycleOptions.length?select("archive-cycle","Saved cycle",cycleOptions,state.archiveCycle):"");
  byId("archive-family").addEventListener("change",event=>{state.archive=event.target.value;state.archiveCycle="";renderArchives();});
  byId("archive-cycle")?.addEventListener("change",event=>{state.archiveCycle=event.target.value;renderArchives();});
  const target=byId("archive-content");
  if(!rows.length){target.innerHTML=empty("No scored archive cycles saved for this campaign yet.");return;}
  target.innerHTML=archivePlot(rows)+`<div class="legend">${rows.map(row=>`<span><i class="swatch" style="background:${color(row.arm)}"></i>${escape(label(row.arm))}</span>`).join("")}</div>`+
    rows.map(series=>{
      const data=series.points.find(p=>String(p.cycle)===state.archiveCycle);
      if(!data)return "";
      return `<hr><h3>${escape(label(series.arm))} · cycle ${data.cycle}</h3><div class="cards">${data.elites.map((row,i)=>{
        const source=String(row.branch||"").match(/^cycle-(\d+)\/(.+)$/);
        const action=source && (series.job.branches||[]).some(branch=>Number(branch.cycle)===Number(source[1])&&branch.name===source[2])
          ? `<div class="molecule-controls"><button type="button" class="run-button" data-archive-job="${escape(series.job.id)}" data-archive-cycle="${Number(source[1])}" data-archive-branch="${escape(source[2])}" data-archive-step="${escape(row.step)}">Open trajectory frame</button></div>`
          : null;
        return moleculeCard(row,`Elite ${i+1}`,action);
      }).join("")}</div>`;
    }).join("");
  for(const dot of target.querySelectorAll("[data-cycle]")){
    const activate=()=>{state.archiveCycle=dot.dataset.cycle;renderArchives();byId("archive-cycle")?.focus();};
    dot.addEventListener("click",activate);
    dot.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();activate();}});
  }
  for(const button of target.querySelectorAll("[data-archive-job]")){
    button.addEventListener("click",()=>openJob(button.dataset.archiveJob,button.dataset.archiveCycle,
      button.dataset.archiveBranch,Number(button.dataset.archiveStep)));
  }
}

const coverage=snapshot.structure_coverage;
byId("footer").textContent="Offline snapshot from DTU scheduler and saved optimization artifacts. "+
  (coverage?`${count(coverage.drawn)} of ${count(coverage.frames)} sampled frames have a 2D graph layout. `:"")+
  "Fresh contact loss is a model score for this BRD4 pocket.";
renderOverview();
switchTab("trees");
