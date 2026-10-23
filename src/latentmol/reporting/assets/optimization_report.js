/* Standalone saved-artifact viewer. Rendering never invokes chemistry or model code. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  const svg = (tag, attrs = {}, text) => { const n = document.createElementNS("http://www.w3.org/2000/svg", tag); Object.entries(attrs).forEach(([k,v]) => n.setAttribute(k,String(v))); if (text !== undefined) n.textContent=text; return n; };
  const list = x => Array.isArray(x) ? x : [];
  const finite = x => typeof x === "number" && Number.isFinite(x);
  const fmt = (x, digits = 5) => finite(x) ? (Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-4 && x !== 0) ? x.toExponential(3) : x.toFixed(digits).replace(/\.?0+$/, "")) : "Not saved";
  const json = x => JSON.stringify(x === undefined ? null : x, null, 2);
  const palette = ["#087f83", "#d8753b", "#637bb3", "#a45c87", "#7a8853", "#ac7630"];
  const symbols = ["∅","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe"];
  const element = n => Number.isInteger(n) && symbols[n] ? symbols[n] : "Z" + String(n);
  let data;
  try { data = JSON.parse($("report-data").textContent); } catch (error) { $("global-warnings").append(el("p", "The embedded report data could not be read: " + error.message, "warning")); return; }
  const runs = list(data.runs);
  let runIndex = 0, trajectoryIndex = 0, frameIndex = 0, candidateIndex = 0, timer = null;
  const run = () => runs[runIndex] || {};
  const trajectory = () => list(run().trajectories)[trajectoryIndex] || {};
  const frames = () => list(trajectory().frames);
  const frame = () => frames()[frameIndex] || {};
  const color = i => palette[i % palette.length];
  const missing = (container, text = "Not saved in these artifacts.") => container.replaceChildren(el("p", text, "missing"));
  const option = (value, text) => { const n = el("option", text); n.value=String(value); return n; };
  const metric = (f,key) => f && f.metrics ? f.metrics[key] : null;
  const ratio = (a,b) => finite(a) && finite(b) ? a + " / " + b : "Not saved";
  const nameOf = (r,t) => (r.label || r.id || "Run") + (t.label || t.id ? " · " + (t.label || t.id) : "");
  const candidates = f => list(f.candidates);
  const argmax = f => candidates(f).find(c => String(c.kind).toLowerCase().includes("argmax")) || null;
  const validityText = c => !c ? "Readout not saved" : c.validity === "valid" ? "Chemical validity passed" : c.validity === "invalid" ? "Invalid raw graph" : "Validity not saved";
  const makeBadge = c => el("span", validityText(c), "badge " + (c && c.validity === "valid" ? "" : c && c.validity === "invalid" ? "invalid" : "unknown"));

  $("report-title").textContent = data.title || "Molecular optimization trajectories";
  document.title = data.title || "Molecular optimization report";
  $("created-at").textContent = data.created_at ? "Generated " + String(data.created_at).slice(0,10) : "";
  [runs.length + " saved run" + (runs.length === 1 ? "" : "s"), "One shared graph per optimization state", "Raw and repaired identities kept separate"].forEach(t => $("report-chips").append(el("span",t,"chip")));
  list(data.warnings).forEach(w => $("global-warnings").append(el("p",w,"warning")));
  $("source-provenance").textContent = json({report_version:data.report_version,created_at:data.created_at,comparison:data.comparison,sources:data.sources});

  function stop() { if (timer !== null) window.clearInterval(timer); timer=null; $("play").textContent="Play"; $("play").setAttribute("aria-pressed","false"); }
  function selectRun(i,ti = 0,fi = 0) {
    stop(); runIndex=Math.max(0,Math.min(runs.length-1,i)); trajectoryIndex=ti; frameIndex=fi; candidateIndex=0;
    $("run-select").value=String(runIndex); $("trajectory-select").replaceChildren();
    list(run().trajectories).forEach((t,j) => $("trajectory-select").append(option(j,t.label || t.id || "Trajectory " + (j+1))));
    if (trajectoryIndex >= list(run().trajectories).length) trajectoryIndex=0;
    $("trajectory-select").value=String(trajectoryIndex);
    const init = run().initialization;
    $("initialization-description").textContent = init && Object.keys(init).length ? "Saved initialization: " + JSON.stringify(init) : "Initialization metadata was not saved. The trajectory selector retains the recorded arm identity.";
    renderOverview(); renderEndpoints(); renderBestBindingStates(); renderEvidence(); chooseFrame(frameIndex);
  }
  function chooseFrame(i) {
    frameIndex=Math.max(0,Math.min(frames().length-1,i));
    const f=frame(), cs=candidates(f);
    if (candidateIndex >= cs.length) candidateIndex=0;
    $("candidate-select").replaceChildren();
    cs.forEach((c,j) => $("candidate-select").append(option(j,(c.kind || "candidate") + (c.draw !== null && c.draw !== undefined ? " · draw " + c.draw : "") + " · " + (c.validity || "unknown"))));
    if (!cs.length) $("candidate-select").append(option(0,"No discrete candidate saved"));
    $("candidate-select").value=String(candidateIndex); $("candidate-select").disabled=!cs.length;
    $("step-slider").max=String(Math.max(0,frames().length-1)); $("step-slider").value=String(Math.max(0,frameIndex)); $("step-slider").disabled=frames().length<2;
    $("first-step").textContent=frames().length ? String(frames()[0].step) : "—";
    $("last-step").textContent=frames().length ? String(frames()[frames().length-1].step) : "—";
    $("step-label").textContent=frames().length ? "Step " + f.step + " · saved state " + (frameIndex+1) + " / " + frames().length : "No saved trajectory states";
    $("previous").disabled=frameIndex<=0; $("next").disabled=frameIndex>=frames().length-1; $("play").disabled=frames().length<2;
    $("step-presets").replaceChildren();
    [...new Set(frames().length ? [0,Math.floor((frames().length-1)/4),Math.floor((frames().length-1)/2),Math.floor(3*(frames().length-1)/4),frames().length-1] : [])].forEach(i=> { const b=el("button","Step " + frames()[i].step); b.type="button"; b.onclick=()=>{stop();chooseFrame(i);}; $("step-presets").append(b); });
    $("trajectory-status").textContent=trajectory().status || run().status || "Status not saved";
    $("frame-events").replaceChildren(); eventsAt(frames(),frameIndex).forEach(e=>$("frame-events").append(el("span",e,"event-tag")));
    renderCandidate(); renderProbabilities(); renderPlots(); renderFrameMeasurements(); renderEvidence();
    $("raw-metrics").textContent=json(f.raw_metrics || {});
  }
  function renderOverview() {
    $("run-overview").replaceChildren();
    runs.forEach((r,ri)=>list(r.trajectories).forEach((t,ti)=> {
      const fs=list(t.frames), s=t.summary || {}, card=el("button",undefined,"overview-card" + (ri===runIndex&&ti===trajectoryIndex?" active":"")); card.type="button"; card.style.setProperty("--run-color",color(ri));
      card.append(el("div",nameOf(r,t),"card-label"),el("div",fmt(metric(fs[0],"loss"),5) + " → " + fmt(metric(fs[fs.length-1],"loss"),5),"card-loss"));
      if(s.step_control_version==="best_trial_v2") {
        card.append(el("p", "Applied updates: " + (s.applied_updates ?? "Not saved") + " · requested: " + (s.requested_steps ?? "Not saved")));
        card.append(el("p", "Strict improvements: " + (s.improving_updates ?? "Not saved") + " · forced fallbacks: " + (s.forced_fallback_updates ?? "Not saved") + " · no-ops: " + (s.no_op_updates ?? "Not saved")));
      } else card.append(el("p", "Accepted updates: " + (s.accepted_updates ?? "Not saved") + " · requested: " + (s.requested_steps ?? "Not saved")));
      card.append(el("p","Argmax valid: " + ratio(s.argmax_valid,s.argmax_total) + " · sampled valid: " + ratio(s.sampled_valid,s.sampled_total)));
      card.append(el("p","Audits passed: " + ratio(s.audits_passed,s.audits_total) + " · " + (r.status || "unknown status")));
      card.onclick=()=>selectRun(ri,ti,0); $("run-overview").append(card);
    }));
  }
  function eventsAt(fs,i) {
    const f=fs[i] || {}, p=fs[i-1], values=list(f.events).map(String);
    if (f.context_changed) values.push("Native context changed");
    if (p && f.repair_target_sha256 !== p.repair_target_sha256 && (f.repair_target_sha256 || p.repair_target_sha256)) values.push("Repair target changed");
    if (p && finite(metric(p,"repair_weight")) && finite(metric(f,"repair_weight")) && metric(p,"repair_weight")!==metric(f,"repair_weight")) values.push("Repair weight changed");
    return [...new Set(values)];
  }
  function comparable(r,t) {
    const keys=["target_sequence_sha256","scorer_state_sha256","graph_spec_sha256"];
    if (r!==run() && !keys.every(k=>r.identity && run().identity && r.identity[k] && r.identity[k]===run().identity[k])) return false;
    const a=new Set(list(t.frames).map(f=>f.context_sha256)), b=new Set(frames().map(f=>f.context_sha256));
    return a.size>0 && !a.has(null) && !a.has(undefined) && a.size===b.size && [...a].every(v=>b.has(v));
  }
  function traces() {
    const out=[];
    runs.forEach((r,ri)=>list(r.trajectories).forEach((t,ti)=> {
      if ((ri===runIndex&&ti===trajectoryIndex) || ($("overlay-runs").checked&&comparable(r,t))) out.push({r,t,ri,ti,fs:list(t.frames),color:color(ri),selected:ri===runIndex&&ti===trajectoryIndex});
    })); return out;
  }
  function renderPlots() {
    const ts=traces(); $("plot-legend").replaceChildren();
    ts.forEach(t=> { const item=el("span"), swatch=el("i",undefined,"swatch"); swatch.style.background=t.color; item.append(swatch,document.createTextNode(nameOf(t.r,t.t))); $("plot-legend").append(item); });
    const policy=data.comparison && data.comparison.policy;
    $("comparison-note").textContent=(policy==="matched_identity_checks"?"Backend identity checks passed for this declared comparison. ":"Descriptive comparison; causal matching has not been established. ") + "Overlays require saved matching target, scorer, graph schema and native-context identities. Changed guidance weights can change the total objective itself.";
    drawPlot($("total-plot"),ts,"loss"); drawPlot($("binding-plot"),ts,"binding_loss");
  }
  function drawPlot(container,ts,key) {
    const points=ts.flatMap(t=>t.fs.filter(f=>finite(f.step)&&finite(metric(f,key))));
    if (!points.length) { missing(container,"This objective was not saved."); return; }
    const W=container.clientWidth>0?container.clientWidth:560,H=225,m={l:57,r:15,t:18,b:35}, xs=points.map(f=>f.step), ys=points.map(f=>metric(f,key));
    let xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
    if (xmin===xmax) { xmin-=.5; xmax+=.5; } const pad=(ymax-ymin)*.09 || Math.max(Math.abs(ymin)*.05,.01); ymin-=pad; ymax+=pad;
    const X=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r), Y=y=>H-m.b-(y-ymin)/(ymax-ymin)*(H-m.t-m.b);
    const root=svg("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":key + " across saved optimization steps",tabindex:0}); root.append(svg("title",{},key + ". Click a state or use the step controls to inspect it."));
    for (let i=0;i<5;i++) { const v=ymin+(ymax-ymin)*i/4, y=Y(v); root.append(svg("line",{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:"#e5ece8"}),svg("text",{x:m.l-8,y:y+3,"text-anchor":"end"},fmt(v,3))); }
    for (let i=0;i<5;i++) { const v=xmin+(xmax-xmin)*i/4; root.append(svg("text",{x:X(v),y:H-15,"text-anchor":"middle"},fmt(v,1))); }
    root.append(svg("text",{x:(W+m.l-m.r)/2,y:H-1,"text-anchor":"middle"},"Optimization step"));
    ts.forEach(t=> {
      let path="", prev=null;
      t.fs.forEach(f=> { if (!finite(f.step)||!finite(metric(f,key))) {prev=null;return;} const objectiveJump=prev && key==="loss" && ((prev.repair_target_sha256??null)!==(f.repair_target_sha256??null) || (metric(prev,"repair_weight")??null)!==(metric(f,"repair_weight")??null)); const connected=prev && f.step===prev.step+1 && prev.context_sha256===f.context_sha256 && !objectiveJump; path+=(connected?"L":"M")+X(f.step)+","+Y(metric(f,key))+" "; prev=f; });
      root.append(svg("path",{d:path,fill:"none",stroke:t.color,"stroke-width":t.selected?2.6:1.7,opacity:t.selected?1:.7,"stroke-dasharray":t.selected?"none":"5 3"}));
      t.fs.forEach((f,i)=> {if (!finite(f.step)||!finite(metric(f,key))) return;
        const dot=svg("circle",{cx:X(f.step),cy:Y(metric(f,key)),r:3.2,fill:t.color,opacity:t.selected?1:.65}); dot.append(svg("title",{},nameOf(t.r,t.t)+" · step "+f.step+" · "+fmt(metric(f,key),7))); dot.style.cursor="pointer"; dot.onclick=e=>{e.stopPropagation();selectRun(t.ri,t.ti,i);}; root.append(dot);
        if (t.selected && eventsAt(t.fs,i).length) root.append(svg("line",{x1:X(f.step),y1:m.t,x2:X(f.step),y2:H-m.b,stroke:t.color,"stroke-width":.7,"stroke-dasharray":"2 4",opacity:.35}));
      });
    });
    if (finite(frame().step)) { root.append(svg("line",{x1:X(frame().step),x2:X(frame().step),y1:m.t,y2:H-m.b,stroke:"#19353d","stroke-dasharray":"4 3",opacity:.65})); if (finite(metric(frame(),key))) root.append(svg("circle",{cx:X(frame().step),cy:Y(metric(frame(),key)),r:5,fill:color(runIndex),class:"selected-point"})); }
    root.onclick=e=> {const rect=root.getBoundingClientRect(), target=xmin+((e.clientX-rect.left)*W/rect.width-m.l)/(W-m.l-m.r)*(xmax-xmin); let nearest=0,best=Infinity; frames().forEach((f,i)=>{if(finite(f.step)&&Math.abs(f.step-target)<best){best=Math.abs(f.step-target);nearest=i;}}); stop();chooseFrame(nearest);};
    root.onkeydown=e=> {if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();stop();chooseFrame(frameIndex+(e.key==="ArrowLeft"?-1:1));}}; container.replaceChildren(root);
  }
  function slotLayout() {
    const slots=new Set(); frames().forEach(f=>candidates(f).forEach(c=>list(c.graph && c.graph.atoms).forEach(a=>{if(Number.isInteger(a.slot))slots.add(a.slot);}))); return [...slots].sort((a,b)=>a-b);
  }
  function probabilityRow(p,slot) { if(!p)return -1; const rows=list(p.row_slots); return rows.length ? rows.indexOf(slot) : slot; }
  function graphView(c,f,compact=false) {
    const container=el("div",undefined,"molecule");
    if (!c || !c.graph || !list(c.graph.atoms).length) { container.append(el("p",c && c.failure ? c.failure : "Raw graph labels were not saved for this candidate.","molecule-empty")); return container; }
    const atoms=list(c.graph.atoms), layout=slotLayout(), allSlots=layout.length?layout:atoms.map(a=>a.slot), columns=Math.min(7,Math.max(3,Math.ceil(Math.sqrt(allSlots.length*1.55)))), rows=Math.ceil(allSlots.length/columns), W=Math.max(450,columns*78+45), H=Math.max(compact?210:280,rows*74+58), pos=new Map();
    allSlots.forEach((s,i)=>{const row=Math.floor(i/columns), col=i%columns;pos.set(s,{x:45+(row%2?columns-1-col:col)*78,y:45+row*74+(H-rows*74-58)/2});});
    const root=svg("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"Saved raw slot-labelled molecular graph; " + validityText(c)}); root.append(svg("title",{},validityText(c)+". Fixed slot layout, not physical coordinates."));
    const atomSet=new Set(atoms.map(a=>a.slot)), p=f.probabilities;
    list(c.graph.bonds).forEach(b=> {
      if(!atomSet.has(b.i)||!atomSet.has(b.j)||!pos.has(b.i)||!pos.has(b.j))return;
      const a=pos.get(b.i), d=pos.get(b.j), dx=d.x-a.x,dy=d.y-a.y,len=Math.hypot(dx,dy)||1,nx=-dy/len,ny=dx/len, type=String(b.type).toLowerCase();
      const n=type.includes("triple")||type==="3"?3:type.includes("double")||type==="2"?2:1, aromatic=type.includes("arom")||type==="4", pi=probabilityRow(p,b.i),pj=probabilityRow(p,b.j), bondRow=list(p&&p.bonds)[pi], bondProbs=list(bondRow)[pj], exists=Array.isArray(bondProbs)&&finite(bondProbs[0])?Math.max(0,Math.min(1,1-bondProbs[0])):null;
      for(let k=0;k<n;k++){const off=(k-(n-1)/2)*4; const line=svg("line",{x1:a.x+off*nx,y1:a.y+off*ny,x2:d.x+off*nx,y2:d.y+off*ny,stroke:"#719395","stroke-width":exists===null?2:.8+3.2*exists,"stroke-linecap":"round"}); if(aromatic)line.setAttribute("stroke-dasharray","5 4");line.append(svg("title",{},"Slots "+(b.i+1)+"–"+(b.j+1)+" · "+b.type+(exists===null?" · probability not saved; unweighted":" · saved shared-graph P(exists)="+fmt(exists,7))));root.append(line);}
    });
    const fills={C:"#e7eeee",N:"#dce7fc",O:"#fce1da",S:"#f9edbd",P:"#f7dfc5",F:"#dcefd5",Cl:"#dcefd5",Br:"#f0dbd0",I:"#eadff1"};
    atoms.forEach(a=>{const xy=pos.get(a.slot);if(!xy)return; const name=a.element || element(a.atomic_number), charge=finite(a.charge)?a.charge:0,row=probabilityRow(p,a.slot), probs=list(p && p.atoms)[row], ai=p&&p.graph_spec?list(p.graph_spec.atomic_numbers).indexOf(a.atomic_number):-1, probability=Array.isArray(probs)&&ai>=0&&finite(probs[ai])?Math.max(0,Math.min(1,probs[ai])):null; const group=svg("g");group.append(svg("circle",{cx:xy.x,cy:xy.y,r:probability===null?18:13+7*probability,fill:fills[name]||"#e7eeee",stroke:"#789394","stroke-width":1.1}));group.append(svg("text",{x:xy.x,y:xy.y+5,"text-anchor":"middle","font-size":15,"font-weight":750,fill:"#19353d"},name));group.append(svg("text",{x:xy.x,y:xy.y+32,"text-anchor":"middle","font-size":10,fill:"#60747a"},String(a.slot+1)));if(charge)group.append(svg("text",{x:xy.x+17,y:xy.y-12,"font-size":10,"font-weight":700,fill:"#19353d"},(Math.abs(charge)===1?"":Math.abs(charge))+(charge>0?"+":"−")));group.append(svg("title",{},"Slot "+(a.slot+1)+" · "+name+" · formal charge "+(finite(a.charge)?a.charge:"not saved")+(probability===null?" · probability not saved; unweighted":" · saved shared-graph P(element)="+fmt(probability,7))));root.append(group);});
    container.append(root);return container;
  }
  function renderCandidate() {
    const f=frame(), c=candidates(f)[candidateIndex]; $("molecule").replaceChildren(...graphView(c,f).childNodes);
    const b=makeBadge(c); $("candidate-validity").textContent=b.textContent; $("candidate-validity").className=b.className;
    const kind=c?String(c.kind || "candidate"):"candidate"; $("candidate-label").textContent=kind.includes("repair")?"Repair target · guidance only, separate identity":"Saved "+kind+" readout";
    $("candidate-smiles").textContent=c&&c.smiles?c.smiles:"SMILES not saved or unavailable for this raw graph.";
    $("candidate-failure").textContent=c&&c.failure?String(c.failure):"";
    const p=f.probabilities;$("graph-caption").textContent="Fixed slot schematic, not a physical conformation. "+(Array.isArray(p&&p.atoms)?"Atom radii use the saved shared-graph P(displayed element), on one absolute scale. ":"Atom probabilities were not saved; atom radii are unweighted. ")+(Array.isArray(p&&p.bonds)?"Bond widths use saved shared-graph P(exists), on one absolute scale.":"Bond probabilities were not saved; bond widths are unweighted.");
  }
  function renderFrameMeasurements() {
    const m=frame().metrics || {}, values=[["Total objective",m.loss],["Binding objective",m.binding_loss],["Affinity loss",m.affinity_loss],["Contact loss",m.contact_loss],["Confidence loss",m.confidence_loss],["Repair NLL",m.repair_loss],["Repair weight",m.repair_weight],["Gradient norm",m.gradient_norm],["Active / natural atom count",finite(m.active_count)&&finite(m.natural_count)?m.active_count+" / "+m.natural_count:null],["Latent displacement",m.latent_displacement]];
    const selection=frame().step_selection;
    if(selection) {
      values.push(["Following update selection",selection.selection_reason === "full_step_nonincrease" ? "Full step without objective increase" : selection.selection_reason || "Not saved"],
        ["Strictly improved objective",typeof selection.strictly_improved==="boolean"?(selection.strictly_improved?"Yes":"No"):"Not saved"],
        ["Applied objective improvement",selection.applied_loss_improvement],
        ["No representable movement",typeof selection.no_op==="boolean"?(selection.no_op?"Yes":"No"):"Not saved"]);
    }
    $("frame-measurements").replaceChildren(); values.forEach(([label,v])=>{const row=el("div");row.append(el("dt",label),el("dd",typeof v==="string"?v:fmt(v,6)));$("frame-measurements").append(row);});
  }
  function heatmap(container,rows,headers,rowLabels,{diagonal=false}={}) {
    if(!Array.isArray(rows)||!rows.length){missing(container);return;}
    const table=el("table",undefined,"heatmap"),head=el("thead"),hr=el("tr");hr.append(el("th","Slot"));headers.forEach(h=>hr.append(el("th",String(h))));head.append(hr);table.append(head);const body=el("tbody");
    rows.forEach((row,i)=>{const tr=el("tr"),rh=el("th",String(rowLabels[i]??i+1));rh.scope="row";tr.append(rh);headers.forEach((_,j)=>{const v=Array.isArray(row)?row[j]:null,td=el("td",diagonal&&i===j?"—":finite(v)?fmt(v,2):"—");if(finite(v)&&!(diagonal&&i===j)){const p=Math.max(0,Math.min(1,v));td.style.backgroundColor="rgba(8,127,131,"+(0.035+.82*p)+")";td.style.color=p>.62?"white":"#19353d";td.title="Slot "+(rowLabels[i]??i+1)+" · "+headers[j]+" · "+fmt(v,9);}tr.append(td);});body.append(tr);});table.append(body);container.replaceChildren(table);
  }
  function renderProbabilities() {
    const p=frame().probabilities, spec=p&&p.graph_spec || {}, rows=list(p&&p.row_slots), atoms=list(p&&p.atoms), charges=list(p&&p.charges);
    const labels=(rows.length?rows:atoms.map((_,i)=>i)).map(s=>s+1);
    heatmap($("atom-probabilities"),p&&p.atoms,list(spec.atomic_numbers).map(element),labels);
    heatmap($("charge-probabilities"),p&&p.charges,list(spec.formal_charges).map(q=>q>0?"+"+q:String(q)),rows.length?rows.map(s=>s+1):charges.map((_,i)=>i+1));
    const old=$("bond-category").value; $("bond-category").replaceChildren(option("exists","P(bond exists)"));list(spec.bond_names).forEach((b,i)=>$("bond-category").append(option(i,b)));if([...$("bond-category").options].some(o=>o.value===old))$("bond-category").value=old;
    $("bond-category").disabled=!Array.isArray(p&&p.bonds); renderBonds();
    $("probability-scope").textContent=p&&p.scope?String(p.scope):"Relaxed distributions were not saved for this state. Hard readouts are not converted into artificial one-hot probabilities.";
    const noatom=frame().raw_metrics&&frame().raw_metrics.natural_selection&&frame().raw_metrics.natural_selection.no_atom_probabilities;
    heatmap($("occupancy-probabilities"),Array.isArray(noatom)?noatom.map(v=>[v]):null,["P(NO_ATOM)"],Array.isArray(noatom)?noatom.map((_,i)=>i+1):[]);
  }
  function renderBonds() {
    const p=frame().probabilities, bonds=p&&p.bonds;
    if(!Array.isArray(bonds)||!bonds.length){missing($("bond-probabilities"),"Bond distributions were not saved. Connectivity in the drawing comes from saved hard labels.");return;}
    const category=$("bond-category").value, rows=list(p.row_slots), labels=(rows.length===bonds.length?rows:bonds.map((_,i)=>i)).map(s=>s+1);
    const values=bonds.map(row=>list(row).map(v=>Array.isArray(v)?category==="exists"?(finite(v[0])?1-v[0]:null):v[Number(category)]:null));heatmap($("bond-probabilities"),values,labels,labels,{diagonal:true});
  }
  function renderEndpoints() {
    $("endpoint-comparison").replaceChildren();const fs=frames();if(!fs.length){missing($("endpoint-comparison"),"No endpoint states were saved.");return;}
    [["Start",fs[0]],["Last saved state",fs[fs.length-1]]].forEach(([label,f])=>{const c=argmax(f),card=el("article",undefined,"endpoint-card"),heading=el("h3");heading.append(el("span",label+" · step "+f.step),makeBadge(c));card.append(heading,graphView(c,f,true),el("code",c&&c.smiles?c.smiles:c&&c.failure?c.failure:"Argmax readout not saved."),el("p","Total objective "+fmt(metric(f,"loss"),6)+" · binding "+fmt(metric(f,"binding_loss"),6),"caption"));$("endpoint-comparison").append(card);});
  }
  function renderBestBindingStates() {
    const container=$("best-binding-states"), states=(trajectory().summary || {}).best_binding_states;
    container.replaceChildren();
    if(!states || typeof states!=="object" || Array.isArray(states) || !Object.keys(states).length) {missing(container,"Best binding states were not saved. They are not reconstructed from the plotted losses.");return;}
    Object.entries(states).sort(([a],[b])=>a.localeCompare(b)).forEach(([context,record])=> {
      const saved=record && typeof record==="object"?record:{}, nativeContext=typeof saved.context_sha256==="string"?saved.context_sha256:context, card=el("article",undefined,"endpoint-card");
      card.append(el("h3","Saved best · context " + nativeContext.slice(0,12)),el("p","Step " + (saved.step ?? "Not saved") + " · binding objective " + fmt(saved.loss,7),"caption"),el("code",nativeContext),el("p","Saved active slots: " + (Array.isArray(saved.active_slots)?saved.active_slots.join(", "):"Not saved"),"caption"),el("p","Saved state: " + (saved.state_path || "Not saved"),"caption"));
      const index=frames().findIndex(f=>f.step===saved.step && f.context_sha256===nativeContext && (!Array.isArray(saved.active_slots) || json(saved.active_slots)===json(f.raw_metrics && f.raw_metrics.selection && f.raw_metrics.selection.active_slots))), button=el("button",index>=0?"Inspect saved state":"State is absent from saved frames");button.type="button";button.disabled=index<0;button.onclick=()=>{stop();chooseFrame(index);};card.append(button);container.append(card);
    });
  }
  function renderEvidence() {
    const f=frame(), audit=f.audit || {}, policy=audit.policy;let policyLabel="Not saved";if(typeof policy==="string")policyLabel=policy;else if(policy&&typeof policy==="object")policyLabel=[policy.version,policy.scope,typeof policy.full_network_fd_required==="boolean"?"Full-network finite differences required: "+(policy.full_network_fd_required?"yes":"no"):null].filter(Boolean).join(" · ")||JSON.stringify(policy);$("repair-audit").replaceChildren();const dl=el("dl");[["Audit status",audit.status||"Not saved"],["Derivative policy",policyLabel],["Current repair weight",fmt(metric(f,"repair_weight"),6)],["Frozen repair target SHA-256",f.repair_target_sha256||"No target identity saved for this state"],["Native context SHA-256",f.context_sha256||"Not saved"]].forEach(([k,v])=>dl.append(el("dt",k),el("dd",String(v))));$("repair-audit").append(dl);
    const follow=run().followups;$("native-followups").replaceChildren();if(follow&&Object.keys(follow).length)$("native-followups").append(el("pre",json(follow)));else missing($("native-followups"),"No fresh native follow-up was saved in this run. Relaxed optimization scores do not substitute for a fresh discrete-molecule prediction.");
    $("run-limitations").textContent=[run().claim_limit,run().error,...list(run().warnings)].filter(Boolean).join(" ");
    $("run-provenance").textContent=json({id:run().id,path:run().path,status:run().status,initialization:run().initialization,identity:run().identity,config:run().config,trajectory_summary:trajectory().summary});
  }
  runs.forEach((r,i)=>{const init=r.initialization||{}, kind=init.kind||init.type, seed=init.seed??(r.config&&r.config.seed);const description=[r.label||r.id||"Run "+(i+1),kind,seed!==null&&seed!==undefined?"seed "+seed:null].filter(Boolean).join(" · ");$("run-select").append(option(i,description));});
  $("run-select").onchange=()=>selectRun(Number($("run-select").value));
  $("trajectory-select").onchange=()=>selectRun(runIndex,Number($("trajectory-select").value));
  $("candidate-select").onchange=()=>{candidateIndex=Number($("candidate-select").value);renderCandidate();};
  $("bond-category").onchange=renderBonds;$("overlay-runs").onchange=renderPlots;
  $("previous").onclick=()=>{stop();chooseFrame(frameIndex-1);};$("next").onclick=()=>{stop();chooseFrame(frameIndex+1);};
  $("step-slider").oninput=()=>{stop();chooseFrame(Number($("step-slider").value));};
  $("play").onclick=()=>{if(timer!==null){stop();return;}if(frameIndex>=frames().length-1)chooseFrame(0);$("play").textContent="Pause";$("play").setAttribute("aria-pressed","true");timer=window.setInterval(()=>{chooseFrame(frameIndex+1);if(frameIndex>=frames().length-1)stop();},700);};
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
  let resizeTimer=null;window.addEventListener("resize",()=>{if(resizeTimer!==null)window.clearTimeout(resizeTimer);resizeTimer=window.setTimeout(()=>{resizeTimer=null;renderPlots();},120);});
  $("download-data").onclick=()=>{const url=URL.createObjectURL(new Blob([json(data)],{type:"application/json"})),a=el("a");a.href=url;a.download="molecular-optimization-report.json";document.body.append(a);a.click();a.remove();window.setTimeout(()=>URL.revokeObjectURL(url),1000);};
  if(runs.length)selectRun(0);else{missing($("run-overview"),"No runs were available in the supplied artifacts.");selectRun(0);}
})();
