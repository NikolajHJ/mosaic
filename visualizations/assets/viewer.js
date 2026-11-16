'use strict';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const state = {runs: [], run: null, id: null, frame: 0, selected: new Set(), timer: null,
  charts: [], atom: 0, pair: [0, 1], probability: 'atom', hiddenLoss: new Set(), loading: 0};
const finite = x => typeof x === 'number' && Number.isFinite(x);
const fmt = x => !finite(x) ? 'Unavailable' : x === 0 ? '0' : Math.abs(x) < 0.001 ? x.toExponential(2) : Number(x.toPrecision(4)).toString();
const colors = () => ['--blue','--orange','--green','--red','--purple','--teal'].map(k => getComputedStyle(document.documentElement).getPropertyValue(k).trim());
function node(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k,v]) => e.setAttribute(k, String(v)));
  if (text !== undefined) e.textContent = text;
  return e;
}
function error(message) { $('error').hidden = !message; $('error').textContent = message || ''; }
async function api(path, data) {
  const response = await fetch(`/api/${path}`, data === undefined ? {} : {method: 'POST',
    headers: {'Content-Type':'application/json', 'X-Mosaic-UI':'1'}, body: JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function handle(fn) { return async (...args) => { try { error(''); await fn(...args); } catch(e) { error(e.message); } }; }
function currentMeta() { return state.runs.find(r => r.id === state.id); }
function visibleRuns() {
  const filter = $('filter').value, query = $('search').value.toLowerCase();
  return state.runs.filter(r => !r.purged && Boolean(r.deleted) === (filter === 'trash') &&
    (filter !== 'starred' || r.starred) && `${r.name} ${r.description}`.toLowerCase().includes(query));
}
function selectionChanged() {
  const selected = state.runs.filter(r => state.selected.has(r.id));
  const trash = $('filter').value === 'trash';
  $('selection-count').textContent = selected.length ? `${selected.length} selected` : '';
  $('delete-selected').hidden = trash;
  $('restore-selected').hidden = !trash;
  $('purge-selected').hidden = !trash;
  $('delete-selected').disabled = !selected.length || selected.some(r => r.starred || r.deleted);
  $('restore-selected').disabled = !selected.length || selected.some(r => !r.deleted);
  $('purge-selected').disabled = !selected.length || selected.some(r => !r.deleted || r.starred);
  const visible = visibleRuns();
  $('select-all').checked = visible.length > 0 && visible.every(r => state.selected.has(r.id));
  $('select-all').indeterminate = ! $('select-all').checked && visible.some(r => state.selected.has(r.id));
}
function renderLibrary() {
  $('runs').replaceChildren();
  const visible = visibleRuns();
  if($('filter').value==='starred')visible.sort((a,b)=>(a.kind==='molecule'?0:1)-(b.kind==='molecule'?0:1)||a.name.localeCompare(b.name));
  $('library-count').textContent = `${state.runs.filter(r => !r.deleted && r.kind!=='molecule').length} runs · ${state.runs.filter(r => !r.deleted && r.kind==='molecule').length} molecules · ${state.runs.filter(r => r.starred && !r.deleted).length} starred`;
  if (!visible.length) { const p = document.createElement('p'); p.textContent = 'No runs in this view.'; $('runs').append(p); }
  for (const r of visible) {
    const row = document.createElement('div'); row.className = `run-row${r.id === state.id ? ' active' : ''}`;
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = state.selected.has(r.id);
    checkbox.setAttribute('aria-label', `Select ${r.name}`);
    checkbox.onchange = () => { checkbox.checked ? state.selected.add(r.id) : state.selected.delete(r.id); selectionChanged(); };
    const pick = document.createElement('button'); pick.className = 'run-pick'; pick.disabled = r.deleted;
    const name = document.createElement('span'); name.textContent = r.name;
    const detail = document.createElement('span'); detail.className = 'small muted'; detail.textContent = r.kind==='molecule'?`${r.backend} · molecule result`:`${r.backend} · ${r.frames} states${r.complete ? '' : ' · incomplete'}`;
    pick.append(name, detail); pick.onclick = handle(() => loadRun(r.id));
    const star = document.createElement('button'); star.className = 'star'; star.textContent = r.starred ? '★' : '☆';
    star.setAttribute('aria-label', `${r.starred ? 'Unstar' : 'Star'} ${r.name}`); star.setAttribute('aria-pressed', String(r.starred));
    star.disabled = r.deleted; star.onclick = handle(() => setStar(r.id, !r.starred));
    row.append(checkbox, pick, star); $('runs').append(row);
  }
  selectionChanged();
  const meta = currentMeta();
  if (meta) {
    $('star-current').textContent = meta.starred ? '★ Starred' : '☆ Star run';
    $('star-current').setAttribute('aria-pressed', String(meta.starred));
    $('delete-current').disabled = meta.starred;
  }
}
async function refresh() {
  const result = await api('runs'); state.runs = result.runs;
  for (const rid of state.selected) if (!state.runs.some(r => r.id === rid)) state.selected.delete(rid);
  renderLibrary();
  if (result.errors?.length) error(`${result.errors.length} runs could not be exported. See index.json for details.`);
}
async function setStar(id, starred) { state.runs = (await api('star', {id, starred})).runs; renderLibrary(); }
function askConfirmation(message, permanent=false) {
  const dialog=$('confirm-dialog');$('confirm-message').textContent=message;
  $('confirm-title').textContent=permanent?'Delete exported runs permanently?':'Move exported runs to trash?';
  $('confirm-yes').textContent=permanent?'Delete permanently':'Move to trash';
  return new Promise(resolve=>{
    const finish=value=>{dialog.close();$('confirm-yes').onclick=null;$('confirm-no').onclick=null;dialog.oncancel=null;resolve(value);};
    $('confirm-yes').onclick=()=>finish(true);$('confirm-no').onclick=()=>finish(false);
    dialog.oncancel=event=>{event.preventDefault();finish(false);};dialog.showModal();$('confirm-no').focus();
  });
}
async function removeRuns(ids) {
  if (!ids.length) return;
  if (!await askConfirmation(`Move ${ids.length} run${ids.length === 1 ? '' : 's'} to library trash? You can restore them. Original experiment files are kept.`)) return;
  state.runs = (await api('delete', {ids})).runs;
  state.selected.clear();
  if (ids.includes(state.id)) {
    stop(); state.loading++; state.run = null; state.id = null;
    $('comparison').hidden = true; $('experiment').hidden = true; $('empty').hidden = false;
  }
  renderLibrary();
}
async function loadRun(id) {
  const meta=state.runs.find(r=>r.id===id);
  if(meta?.kind==='molecule'){
    $('comparison-native').value='native_guided';const expected=state.loading+1;await openComparison();if(state.loading!==expected)return;
    state.id=id;state.run=null;$('pose-molecule').value=meta.molecule_id;updatePoseOptions();renderLibrary();
    const row=$(`comparison-${meta.molecule_id}`);row?.classList.add('score-focus');row?.scrollIntoView({block:'center'});return;
  }
  stop(); const request = ++state.loading;
  $('run-name').textContent = 'Loading saved steps…';
  const response = await fetch(`/runs/${id}/data.json`);
  if (!response.ok) throw new Error('Could not load this run. Refresh the library.');
  const run = await response.json();
  if (request !== state.loading) return;
  state.id = id; state.run = run; state.frame = 0; state.atom = 0; state.pair = [0,1]; state.probability = 'atom'; state.hiddenLoss.clear();
  for(const f of run.frames) for(const key of Object.keys(f.metrics)) if(key.startsWith('aux:')&&!key.includes('loss')) state.hiddenLoss.add(key);
  $('comparison').hidden = true; $('empty').hidden = true; $('experiment').hidden = false;
  $('run-name').textContent = run.name;
  $('run-mode').textContent = `${run.config.backend || 'Backend not recorded'} / ${run.config.stage2?.mode || 'Mode not recorded'}`;
  $('description').value = currentMeta().description || '';
  $('description-status').textContent = '';
  $('source').textContent = `Source: ${run.source}\nTrace SHA-256: ${run.trace_sha256}`;
  $('config').textContent = JSON.stringify(run.config, null, 2);
  $('step').max = run.frames.length - 1;
  renderLibrary(); nativeEvaluation(); setupCharts(); showFrame(0);
}
function nativeEvaluation() {
  const data=state.run.native_evaluation;
  $('native-evaluation').hidden=!data;
  $('native-bars').replaceChildren();$('native-table').replaceChildren();
  if(!data)return;
  const protocol=data.protocol;
  const settings=protocol?`Native settings: structure recycles ${protocol.structure_recycling_steps??'unavailable'}, affinity recycles ${protocol.affinity_recycling_steps??'unavailable'}, pipeline precision ${protocol.precision??'unavailable'}. `:'';
  $('native-scope').textContent=`${data.complete?'Completed evaluation.':'Evaluation not marked complete.'} ${settings}${data.scope} Exact scored pose means the pose passed to the affinity heads was captured and checked; basic geometry checks alone do not prove binding.`;
  const table=document.createElement('table'),head=document.createElement('tr');
  for(const label of ['Identity','Binder mean','Member 1','Member 2','Native affinity ↓','Pose checked','Bad bonds','Clashes <1.2 Å','Pocket ≤6 Å','Proposal']){
    const th=document.createElement('th');th.scope='col';th.textContent=label;head.append(th);
  }table.append(head);
  const width=720,left=190,plot=480,height=40*data.candidates.length+35;
  const svg=node('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':'Native binder-head predictions for selected molecules and positive control'});
  for(const tick of [0,.25,.5,.75,1]){const x=left+tick*plot;svg.append(node('line',{x1:x,y1:20,x2:x,y2:height-10,stroke:'var(--line)'}),node('text',{x,y:12,'text-anchor':'middle'},tick));}
  data.candidates.forEach((candidate,i)=>{
    const a=candidate.affinity,p=candidate.pose,tr=document.createElement('tr');
    const pose=candidate.exact_scored_pose_checked?'Exact scored pose':finite(p.maximum_bond_length_A)?'Pre-affinity only':'Unavailable';
    const cells=[candidate.name,a.affinity_probability_binary,a.affinity_probability_binary1,a.affinity_probability_binary2,a.affinity_pred_value,pose,p.bonds_outside_0_8_to_2_5A,p.protein_ligand_pairs_below_1_2A,p.ligand_fraction_within_6A_of_pocket];
    cells.forEach((value,j)=>{const td=document.createElement('td');td.textContent=typeof value==='string'?value:fmt(value);if(j===0)td.title=candidate.smiles||candidate.role;tr.append(td);});
    const td=document.createElement('td');
    if(Number.isInteger(candidate.proposal_frame)){const b=document.createElement('button');b.textContent=`Step ${candidate.proposal_step}`;b.onclick=()=>{stop();showFrame(candidate.proposal_frame);$('step').scrollIntoView({block:'center',behavior:'smooth'});};td.append(b);}else td.textContent='Not a trajectory state';tr.append(td);table.append(tr);
    const y=25+i*40;svg.append(node('text',{x:left-10,y:y+13,'text-anchor':'end'},candidate.name));
    [a.affinity_probability_binary,a.affinity_probability_binary1,a.affinity_probability_binary2].forEach((value,j)=>{
      if(!finite(value))return;
      const bar=node('rect',{x:left,y:y+j*7,width:plot*value,height:5,fill:colors()[j]});bar.append(node('title',{},`${candidate.name}: ${['mean','member 1','member 2'][j]} ${fmt(value)}`));svg.append(bar);
    });
  });
  $('native-bars').append(svg);$('native-table').append(table);
}
function stop() { clearInterval(state.timer); state.timer = null; $('play').textContent = 'Play'; }
function start() {
  stop(); if (!state.run) return;
  if (state.frame === state.run.frames.length - 1) showFrame(0);
  $('play').textContent = 'Pause';
  state.timer = setInterval(() => { showFrame(state.frame + 1); if (state.frame >= state.run.frames.length - 1) stop(); }, Number($('speed').value));
}
function atomIndices(frame) { return frame.argmax_atoms || frame.X.map(argmax); }
function bondIndices(frame) { return frame.argmax_bonds || frame.E.map(argmax); }
function graphKey(frame) { return JSON.stringify([atomIndices(frame), bondIndices(frame)]); }
function argmax(row) { let best=0; for(let i=1;i<row.length;i++) if(row[i]>row[best]) best=i; return best; }
function jumpChange(direction) {
  const key = graphKey(state.run.frames[state.frame]);
  for(let i=state.frame+direction;i>=0 && i<state.run.frames.length;i+=direction) if(graphKey(state.run.frames[i])!==key) { showFrame(i); return; }
}
function pairIndex(n, a, b) { const [i,j] = a<b ? [a,b] : [b,a]; return i*(2*n-i-1)/2+j-i-1; }
function showFrame(index) {
  if (!state.run) return;
  state.frame = Math.max(0, Math.min(state.run.frames.length-1, index));
  const f = state.run.frames[state.frame], metrics=f.metrics;
  $('step').value = state.frame; $('step-label').textContent = `Update ${f.step}`;
  $('frame-count').textContent = `${state.frame+1} / ${state.run.frames.length} saved states`;
  $('previous').disabled = state.frame===0; $('next').disabled = state.frame===state.run.frames.length-1;
  $('previous-change').disabled = state.frame===0; $('next-change').disabled = state.frame===state.run.frames.length-1;
  $('step-metrics').replaceChildren();
  for (const [key,label] of [['loss','Loss'],['atom_entropy','Atom entropy (nats)'],['bond_entropy','Bond entropy (nats)'],['charge_entropy','Charge entropy (nats)'],['components','Components'],['over_valence_atoms','Atoms above discrete cap']]) {
    const box=document.createElement('div');box.className='metric';const title=document.createElement('span');title.textContent=label;
    const value=document.createElement('strong');value.textContent=fmt(metrics[key]);box.append(title,value);$('step-metrics').append(box);
  }
  molecule(f); probabilityMatrices(f); valence(f);
  $('operator-receipt').textContent = f.operators ? JSON.stringify(f.operators,null,2) : 'No chemistry correction receipt was saved for this state.';
  $('operator-note').textContent = f.final ? 'Final evaluated state: no following update or chemistry corrections.' : `Receipts at update ${f.step} describe corrections applied after its gradient step. The molecule above is the state evaluated before that update.`;
  for (const chart of state.charts) if (chart.cursor) { const x=chart.x(f.step); chart.cursor.setAttribute('x1',x);chart.cursor.setAttribute('x2',x); }
}
function molecule(f) {
  const svg=$('molecule'), width=600,height=350,n=f.X.length, previous=state.run.frames[state.frame-1];
  const atoms=atomIndices(f), bonds=bondIndices(f), prevAtoms=previous?atomIndices(previous):undefined,prevBonds=previous?bondIndices(previous):undefined;
  const fixed=$('slot-layout').checked;
  const xy=fixed ? atoms.map((_,i)=>[Math.cos(2*Math.PI*i/n-Math.PI/2),Math.sin(2*Math.PI*i/n-Math.PI/2)]) : f.xy;
  const maxX=Math.max(...xy.map(p=>Math.abs(p[0])),.1),maxY=Math.max(...xy.map(p=>Math.abs(p[1])),.1);
  const scale=Math.min((width-65)/(2*maxX),(height-65)/(2*maxY));
  const points=xy.map(([x,y])=>[width/2+x*scale,height/2-y*scale]);
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.replaceChildren(node('title',{},`Raw argmax graph at update ${f.step}. ${f.layout}`));
  let k=0;
  for(let i=0;i<n;i++) for(let j=i+1;j<n;j++,k++) {
    const bond=bonds[k]; if(!bond) continue;
    const [x1,y1]=points[i],[x2,y2]=points[j],length=Math.hypot(x2-x1,y2-y1)||1;
    const dx=-(y2-y1)/length*2.6,dy=(x2-x1)/length*2.6;
    const span=state.run.frozen_distances_angstrom?.[i]?.[j];
    const radii=state.run.covalent_radii_angstrom,ratio=finite(span)&&radii?span/(radii[atoms[i]]+radii[atoms[j]]):null;
    const changed=prevBonds && prevBonds[k]!==bond;
    const stroke=finite(span)&&(span>2.5||span<.8) ? 'var(--red)' : finite(ratio)&&ratio>(state.run.covalent_stretch_threshold??1.35)?'var(--purple)':changed?'var(--orange)':'var(--muted)';
    const offsets=bond===2 ? [-.65,.65] : bond===3 ? [-1,0,1] : [0];
    for(const offset of offsets){const line=node('line',{x1:x1+dx*offset,y1:y1+dy*offset,x2:x2+dx*offset,y2:y2+dy*offset,stroke,'stroke-width':changed?2.5:1.5,...(bond===4?{'stroke-dasharray':'4 3'}:{})});line.append(node('title',{},`Atoms ${i+1}–${j+1}${finite(span)?`: ${span.toFixed(2)} Å frozen distance`:''}${finite(ratio)?`, ${ratio.toFixed(2)} × covalent-radius sum`:''}${changed?'; changed since previous state':''}`));svg.append(line);}
  }
  for(let i=0;i<n;i++) {
    const [x,y]=points[i],g=node('g');
    const changed=prevAtoms && atoms[i]!==prevAtoms[i];
    g.append(node('circle',{cx:x,cy:y,r:12,fill:'var(--surface)',stroke:changed?'var(--orange)':i===state.atom?'var(--blue)':'none','stroke-width':2}));
    const label=state.run.elements[atoms[i]]+($('atom-numbers').checked?i+1:'');
    const color={C:'var(--ink)',N:'var(--blue)',O:'var(--red)',S:'var(--orange)',F:'var(--green)',Cl:'var(--teal)'}[state.run.elements[atoms[i]]];
    g.append(node('text',{x,y:y+4,'text-anchor':'middle',fill:color,'font-size':12,'font-family':'system-ui'},label));
    g.append(node('title',{},`Atom ${i+1}: ${state.run.elements[atoms[i]]}${changed?'; changed since previous saved state':''}`));
    g.style.cursor='pointer';g.onclick=()=>selectAtom(i);svg.append(g);
  }
  const sanitized=f.metrics.sanitized;
  const geometryNote=state.run.frozen_distances_angstrom?` Red bonds fall outside 0.8–2.5 Å in the frozen coordinates.${state.run.covalent_radii_angstrom?' Purple bonds exceed 1.35 × their covalent-radius sum.':''} Warning colours take precedence over orange. This is a 2D graph, not that pose.`:'';
  $('molecule-status').textContent = `${sanitized===null?'Validity unknown':sanitized?'RDKit accepted this graph':'RDKit rejected this graph'} · ${f.metrics.components} component(s). ${fixed?'Fixed atom-slot layout':f.layout}. Orange or thicker bonds mark changes; aromatic bonds are dashed.${geometryNote}`;
  $('molecule-status').className=`small ${sanitized===1?'status-ok':sanitized===0?'status-bad':''}`;
  $('smiles').textContent=f.readout.canonical_smiles || f.readout.error || f.readout.conversion_error || 'No canonical SMILES saved.';
}
let heatPalette = [];
function heat(probability) {
  if(!heatPalette.length){
    const css=getComputedStyle(document.documentElement);
    const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');
    ctx.fillStyle=css.getPropertyValue('--heat-low').trim();ctx.fillRect(0,0,1,1);const a=ctx.getImageData(0,0,1,1).data;
    ctx.fillStyle=css.getPropertyValue('--heat-high').trim();ctx.fillRect(0,0,1,1);const b=ctx.getImageData(0,0,1,1).data;
    heatPalette=Array.from({length:256},(_,p)=>`rgb(${[0,1,2].map(i=>Math.round(a[i]+(b[i]-a[i])*p/255)).join(',')})`);
  }
  return heatPalette[Math.round(Math.max(0,Math.min(1,probability))*255)];
}
function probabilityMatrices(f) {
  const table=document.createElement('table'),head=document.createElement('tr');
  for(const text of ['Atom',...state.run.elements]){const th=document.createElement('th');th.textContent=text;th.scope='col';head.append(th);}table.append(head);
  f.X.forEach((row,i)=>{const tr=document.createElement('tr');tr.className=state.probability==='atom'&&state.atom===i?'atom-row-selected':'';
    const label=document.createElement('th');label.scope='row';label.textContent=i+1;tr.append(label);
    row.forEach((p,j)=>{const td=document.createElement('td'),button=document.createElement('button');button.className='probability-cell';
      button.textContent=`${(p*100).toFixed(1)}%`;button.style.background=heat(p);button.style.color=p>.55?'var(--surface)':'var(--ink)';
      button.setAttribute('aria-label',`Atom ${i+1}, ${state.run.elements[j]}: ${p}`);if(j===atomIndices(f)[i])button.style.fontWeight='700';
      button.onclick=()=>selectAtom(i);td.append(button);tr.append(td);});table.append(tr);});
  $('atom-matrix').replaceChildren(table);
  $('charge-note').textContent=f.Q?'Charge entropy is calculated from the recorded charge probabilities.':'Charge entropy is unavailable: this representation has no charge categories. Charge is fixed in the model context.';
  $('charge-matrix').replaceChildren();
  if(f.Q){const title=document.createElement('h4');title.textContent='Charge probabilities';$('charge-matrix').append(title);
    const text=document.createElement('pre');text.textContent=f.Q.map((r,i)=>`${i+1}: ${r.map(fmt).join('  ')}`).join('\n');$('charge-matrix').append(text);}
  const canvas=$('bond-matrix'),ctx=canvas.getContext('2d'),n=f.X.length,cell=16,offset=24,extent=n*cell+offset;
  canvas.width=extent;canvas.height=extent;ctx.clearRect(0,0,extent,extent);
  const channel=$('bond-channel').value;
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
    if(i===j) continue;const row=f.E[pairIndex(n,i,j)];const p=channel==='existence'?1-row[0]:row[Number(channel)];
    ctx.fillStyle=heat(p);ctx.fillRect(offset+j*cell,offset+i*cell,cell-1,cell-1);
  }
  ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();ctx.font='11px system-ui';ctx.textAlign='center';
  for(let i=0;i<n;i++){if(n<=30 || i%5===0){ctx.fillText(i+1,offset+i*cell+cell/2,16);ctx.fillText(i+1,12,offset+i*cell+12);}}
  const [a,b]=state.pair;ctx.strokeStyle=colors()[1];ctx.lineWidth=2;ctx.strokeRect(offset+b*cell,offset+a*cell,cell,cell);ctx.strokeRect(offset+a*cell,offset+b*cell,cell,cell);
  canvas.onclick=event=>{const box=canvas.getBoundingClientRect(),j=Math.floor(((event.clientX-box.left)*extent/box.width-offset)/cell),i=Math.floor(((event.clientY-box.top)*extent/box.height-offset)/cell);
    if(i>=0&&j>=0&&i<n&&j<n&&i!==j){state.pair=[Math.min(i,j),Math.max(i,j)];state.probability='bond';probabilityChart();showFrame(state.frame);}};
  const span=state.run.frozen_distances_angstrom?.[a]?.[b];
  const radii=state.run.covalent_radii_angstrom,types=atomIndices(f),reference=radii?radii[types[a]]+radii[types[b]]:null;
  $('bond-detail').textContent=`Selected pair: atoms ${a+1}–${b+1}${finite(span)?` · frozen coordinate distance ${span.toFixed(2)} Å`:''}${finite(span)&&finite(reference)?` · covalent-radius sum ${reference.toFixed(2)} Å · stretch ratio ${(span/reference).toFixed(2)}`:''}`;
  $('bond-values').textContent=state.run.bonds.map((name,i)=>`${name}: ${(f.E[pairIndex(n,a,b)][i]*100).toFixed(1)}%`).join(' · ');
}
function selectAtom(index) { state.atom=index;state.probability='atom';probabilityChart();showFrame(state.frame); }

function chart(id, series, unit, options={}) {
  state.charts=state.charts.filter(c=>c.id!==id);
  const root=$(id);root.replaceChildren();
  const frames=state.run.frames, available=series.filter(s=>frames.some(f=>finite(s.get(f))));
  if(!available.length){const p=document.createElement('p');p.className='chart-message';p.textContent=options.empty||'Not recorded for this run.';root.append(p);return;}
  const width=Math.max(280,root.getBoundingClientRect().width),height=205,margin={left:62,right:14,top:24,bottom:45};
  const first=frames[0].step,last=frames.at(-1).step;
  const numbers=available.flatMap(s=>frames.map(f=>s.get(f)).filter(finite));
  let min=Math.min(...numbers),max=Math.max(...numbers);if(options.zero)min=Math.min(0,min);
  if(min===max){const p=Math.max(Math.abs(min)*.05,.05);min-=p;max+=p;}else {const p=(max-min)*.06;min-=p;max+=p;}
  if(options.domain){[min,max]=options.domain;}
  const x=v=>margin.left+(v-first)/Math.max(1,last-first)*(width-margin.left-margin.right);
  const y=v=>height-margin.bottom-(v-min)/(max-min)*(height-margin.bottom-margin.top);
  const svg=node('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':`${unit} over saved updates`});
  svg.append(node('title',{},`${unit} over time. Click a curve to select a saved update.`));
  const ticks=width<400?3:4;
  for(let i=0;i<=ticks;i++){
    const value=min+(max-min)*i/ticks,py=y(value);
    svg.append(node('line',{x1:margin.left,y1:py,x2:width-margin.right,y2:py,class:'chart-grid'}));
    svg.append(node('text',{x:margin.left-8,y:py+4,'text-anchor':'end'},fmt(value)));
  }
  for(let i=0;i<=ticks;i++){
    const value=first+(last-first)*i/ticks;
    svg.append(node('text',{x:x(value),y:height-margin.bottom+18,'text-anchor':'middle'},Math.round(value)));
  }
  svg.append(node('rect',{x:margin.left,y:margin.top,width:width-margin.left-margin.right,height:height-margin.top-margin.bottom,class:'chart-frame'}));
  svg.append(node('text',{x:(width+margin.left-margin.right)/2,y:height-5,'text-anchor':'middle'},'Update'));
  svg.append(node('text',{x:margin.left,y:13},unit));
  const palette=colors();
  available.forEach((s,index)=>{
    let path='',active=false;
    for(const f of frames){const value=s.get(f);if(!finite(value)){active=false;continue;}path+=`${active?'L':'M'}${x(f.step).toFixed(2)},${y(value).toFixed(2)}`;active=true;}
    svg.append(node('path',{d:path,fill:'none',stroke:s.color||palette[index%palette.length],'stroke-width':1.8,...(s.dash?{'stroke-dasharray':s.dash}:{})}));
    if(frames.length===1){const f=frames[0];svg.append(node('circle',{cx:x(f.step),cy:y(s.get(f)),r:3,fill:s.color||palette[index%palette.length]}));}
  });
  const cursor=node('line',{x1:x(frames[state.frame].step),x2:x(frames[state.frame].step),y1:margin.top,y2:height-margin.bottom,class:'chart-cursor'});svg.append(cursor);
  const overlay=node('rect',{x:margin.left,y:margin.top,width:width-margin.left-margin.right,height:height-margin.top-margin.bottom,class:'chart-hit'});
  const indexAt=event=>{const rect=svg.getBoundingClientRect(),vx=(event.clientX-rect.left)*width/rect.width;const step=first+(vx-margin.left)/(width-margin.left-margin.right)*(last-first);
    let lo=0,hi=frames.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(frames[mid].step<step)lo=mid+1;else hi=mid;}return lo>0&&Math.abs(frames[lo-1].step-step)<Math.abs(frames[lo].step-step)?lo-1:lo;};
  overlay.onpointermove=event=>{const f=frames[indexAt(event)],tooltip=$('chart-tooltip');tooltip.replaceChildren();
    const title=document.createElement('strong');title.textContent=`Update ${f.step}`;tooltip.append(title);
    available.forEach(s=>{const row=document.createElement('div');row.textContent=`${s.label}: ${fmt(s.get(f))}`;tooltip.append(row);});
    tooltip.hidden=false;tooltip.style.left=`${Math.max(5,Math.min(event.clientX+12,innerWidth-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(5,Math.min(event.clientY+12,innerHeight-tooltip.offsetHeight-8))}px`;};
  overlay.onpointerleave=()=>{$('chart-tooltip').hidden=true;};overlay.onclick=event=>{stop();showFrame(indexAt(event));};svg.append(overlay);root.append(svg);
  const legend=document.createElement('div');legend.className='legend';
  available.forEach((s,index)=>{const item=document.createElement('span'),swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=s.color||palette[index%palette.length];
    item.append(swatch,document.createTextNode(s.label));legend.append(item);});root.append(legend);
  state.charts.push({id,x,cursor});
}
const metric=(key,label,color,dash)=>({label,get:f=>f.metrics[key],color,dash});
function setupCharts() {
  if(!state.run)return;
  const c=colors();state.charts=[];
  const lossKeys=['loss',...new Set(state.run.frames.flatMap(f=>Object.keys(f.metrics).filter(k=>k.startsWith('aux:')&&k!=='aux:forward_hardness')))];
  $('loss-legend').replaceChildren();
  lossKeys.forEach((key,i)=>{const button=document.createElement('button');button.textContent=key.replace('aux:','');button.setAttribute('aria-pressed',String(!state.hiddenLoss.has(key)));
    button.onclick=()=>{state.hiddenLoss.has(key)?state.hiddenLoss.delete(key):state.hiddenLoss.add(key);setupCharts();};$('loss-legend').append(button);});
  chart('loss-chart',lossKeys.filter(k=>!state.hiddenLoss.has(k)).map(k=>metric(k,k.replace('aux:',''),c[lossKeys.indexOf(k)%c.length])),'Recorded value');
  chart('entropy-chart',[metric('atom_entropy','Atom'),metric('bond_entropy','Bond'),metric('charge_entropy','Charge')],'Mean entropy (nats)');
  chart('validity-chart',[metric('sanitized','Sanitized'),metric('connected','Connected'),metric('valid_connected','Sanitized + connected',c[2],'5 3')],'Discrete indicator',{domain:[-.05,1.05]});
  chart('geometry-count-chart',[metric('frozen_long_bonds','Above 2.5 Å'),metric('frozen_short_bonds','Below 0.8 Å')],'Frozen bond count',{zero:true,empty:'No matching frozen-context geometry audit is available.'});
  chart('geometry-span-chart',[metric('maximum_frozen_bond_length_A','Longest frozen bond')],'Distance (Å)',{zero:true,empty:'No matching frozen-context geometry audit is available.'});
  chart('geometry-stretch-count-chart',[metric('covalent_stretched_bonds','Above 1.35 × radius sum')],'Stretched bond count',{zero:true,empty:'No matching element-aware geometry diagnostic is available.'});
  chart('geometry-stretch-chart',[metric('maximum_covalent_stretch_ratio','Largest radius-sum ratio'),{label:'Review threshold',get:f=>finite(f.metrics.maximum_covalent_stretch_ratio)?1.35:null,color:c[3],dash:'4 3'}],'Distance / covalent-radius sum',{zero:true,empty:'No matching element-aware geometry diagnostic is available.'});
  chart('projection-chart',[metric('projection_l2','Bond probabilities moved')],'L2 displacement');
  chart('projection-count-chart',[metric('projection_over_atoms','Above cap before'),metric('projection_changed_atoms','Valence changed')],'Atom count',{zero:true});
  chart('connectivity-chart',[metric('bridges_attempted','Baseline bridges proposed'),metric('bridges_surviving','Surviving'),metric('bridges_lost','Lost'),metric('experimental_bridges','Experimental reconnect'),metric('experimental_rewires','Saturated-fragment rewires')],'Bond/event count',{zero:true});
  chart('components-chart',[metric('components','Evaluated graph'),metric('components_after','After all corrections')],'Component count',{zero:true});
  chart('sparsity-chart',[metric('sparsity_removed','Bond mass removed')],'Expected bonds removed',{zero:true});
  chart('sparsity-count-chart',[metric('sparsity_pairs','Pairs changed')],'Pair count',{zero:true});
  chart('experimental-pruning-chart',[metric('readout_pruned_edges','Edges pruned for discrete valence')],'Experimental edge count',{zero:true,empty:'No experimental readout-safe valence correction in this run.'});
  chart('aromatic-merge-chart',[metric('aromatic_mass_merged','Aromatic probability moved to single')],'Experimental probability mass',{zero:true,empty:'No aromatic-channel merging in this run.'});
  chart('atom-support-chart',[metric('atom_support_atoms','Atom rows restricted'),metric('atom_support_changed_argmax','Argmax types changed by guard')],'Atom count',{zero:true,empty:'No atom-support valence alternative in this run.'});
  chart('atom-support-movement-chart',[metric('atom_support_l2','Atom probabilities moved by guard')],'L2 displacement',{zero:true,empty:'No atom-support valence alternative in this run.'});
  chart('hardening-chart',[metric('hardness','Recorded forward hardness')],'Hardness',{domain:[-.03,1.03],empty:'Hardening is off, or its schedule was not recorded.'});
  chart('gradient-chart',[metric('gradient_X','Atom gradient'),metric('gradient_E','Bond gradient')],'L2 gradient norm');
  chart('alignment-chart',[metric('descent_alignment_X','Atom movement'),metric('descent_alignment_E','Bond movement')],'Cosine with negative tangent gradient',{domain:[-1.05,1.05]});
  for(const [id,suffix] of [['atom-step-chart','X'],['bond-step-chart','E']]) chart(id,[metric(`gradient_step_${suffix}`,'Raw step'),metric(`simplex_step_${suffix}`,'After simplex'),metric(`observed_step_${suffix}`,'Observed next state')],'L2 step size',{zero:true});
  probabilityChart();
}
function probabilityChart() {
  if(!state.run)return;
  let series,label;
  if(state.probability==='atom'){series=state.run.elements.map((name,i)=>({label:name,get:f=>f.X[state.atom][i]}));label=`Atom ${state.atom+1} probability`;}
  else {const [a,b]=state.pair;series=state.run.bonds.map((name,i)=>({label:name,get:f=>f.E[pairIndex(f.X.length,a,b)][i]}));label=`Pair ${a+1}–${b+1} probability`;}
  chart('probability-chart',series,label,{domain:[-.03,1.03]});
}
function valence(f) {
  const root=$('valence-chart'),n=f.X.length,width=Math.max(300,root.getBoundingClientRect().width),height=270;
  const left=48,right=12,top=28,bottom=43,projection=f.projection;
  const caps=projection?.caps||f.valence.caps,before=projection?.before||f.valence.expected,after=projection?.after;
  const max=Math.max(1,...caps,...before,...(after||[]))*1.1;
  const x=i=>left+(i+.5)*(width-left-right)/n,y=v=>height-bottom-v/max*(height-top-bottom),slot=(width-left-right)/n;
  const svg=node('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':'Expected valence before and after correction, and valence ceiling per atom'});
  svg.append(node('title',{},'Each atom: valence before correction, after first valence projection, and its ceiling.'));
  for(let i=0;i<=4;i++){const v=max*i/4;svg.append(node('line',{x1:left,y1:y(v),x2:width-right,y2:y(v),class:'chart-grid'}));svg.append(node('text',{x:left-6,y:y(v)+4,'text-anchor':'end'},fmt(v)));}
  for(let i=0;i<n;i++){
    const beforeBar=node('rect',{x:x(i)-slot*.32,y:y(before[i]),width:slot*.29,height:y(0)-y(before[i]),fill:'var(--orange)'});
    beforeBar.append(node('title',{},`Atom ${i+1}: before ${fmt(before[i])}; cap ${fmt(caps[i])}`));svg.append(beforeBar);
    if(after){const bar=node('rect',{x:x(i)+slot*.03,y:y(after[i]),width:slot*.29,height:y(0)-y(after[i]),fill:'var(--blue)'});bar.append(node('title',{},`Atom ${i+1}: after ${fmt(after[i])}`));svg.append(bar);}
    svg.append(node('line',{x1:x(i)-slot*.4,x2:x(i)+slot*.4,y1:y(caps[i]),y2:y(caps[i]),stroke:'var(--ink)','stroke-width':2}));
    if(n<=15 || i%Math.ceil(n/(width<450?8:20))===0)svg.append(node('text',{x:x(i),y:height-bottom+17,'text-anchor':'middle'},i+1));
  }
  svg.append(node('text',{x:left,y:14},'Expected bond order per atom'));
  svg.append(node('text',{x:(left+width-right)/2,y:height-4,'text-anchor':'middle'},'Atom slot'));
  root.replaceChildren(svg);
  const legend=document.createElement('div');legend.className='legend';legend.textContent=`Orange: ${projection?'before projection':'evaluated state'} · Blue: after first projection · Black: vmax`;root.append(legend);
  $('valence-note').textContent=projection?`${projection.scope} Reconstructed movement matches the recorded receipt. Positive difference means above vmax; a negative value means unused capacity.`:`Evaluated state only. ${f.operators?.valence?'No verified reconstruction is available for the recorded correction; inspect its receipt.':'No following valence projection was recorded.'} Positive difference means above vmax; a negative value means unused capacity.`;
  const table=document.createElement('table'),head=document.createElement('tr');
  for(const title of ['Atom','vmax','Before','After','Before − vmax','Discrete valence','Discrete cap']){const th=document.createElement('th');th.textContent=title;head.append(th);}table.append(head);
  for(let i=0;i<n;i++){const tr=document.createElement('tr'),evaluated=atomIndices(f)[i],proposed=projection?.element_indices?.[i]??evaluated;
    const atomLabel=`${i+1} ${state.run.elements[evaluated]}${proposed!==evaluated?' → '+state.run.elements[proposed]:''}`;
    [atomLabel,fmt(caps[i]),fmt(before[i]),fmt(after?.[i]),fmt(before[i]-caps[i]),fmt(f.valence.discrete[i]),fmt(f.valence.discrete_caps[i])].forEach((text,j)=>{const td=document.createElement('td');td.textContent=text;if(j===4&&before[i]>caps[i]+2e-6)td.className='over';tr.append(td);});table.append(tr);}
  const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Per-atom values';details.append(summary,table);$('valence-table').replaceChildren(details);
}

$('search').oninput=()=>{state.selected.clear();renderLibrary();};
$('filter').onchange=()=>{state.selected.clear();renderLibrary();};
$('refresh').onclick=handle(refresh);
$('select-all').onchange=()=>{visibleRuns().forEach(r=>$('select-all').checked?state.selected.add(r.id):state.selected.delete(r.id));renderLibrary();};
$('delete-selected').onclick=handle(()=>removeRuns([...state.selected]));
$('restore-selected').onclick=handle(async()=>{state.runs=(await api('restore',{ids:[...state.selected]})).runs;state.selected.clear();renderLibrary();});
$('purge-selected').onclick=handle(async()=>{const ids=[...state.selected];if(!ids.length || !await askConfirmation(`Permanently delete ${ids.length} exported run${ids.length===1?'':'s'} from library trash and free their disk space? This cannot be undone. Original experiment files are kept.`,true))return;state.runs=(await api('purge',{ids,confirm_permanent:true})).runs;state.selected.clear();renderLibrary();});
$('star-current').onclick=handle(()=>setStar(state.id,!currentMeta().starred));
$('delete-current').onclick=handle(()=>removeRuns([state.id]));
$('save-description').onclick=handle(async()=>{if(!state.id)return;state.runs=(await api('description',{id:state.id,description:$('description').value})).runs;renderLibrary();$('description-status').textContent='Saved';});
$('description').oninput=()=>{$('description-status').textContent='Unsaved changes';};
$('step').oninput=event=>{stop();showFrame(Number(event.target.value));};
$('previous').onclick=()=>{stop();showFrame(state.frame-1);};$('next').onclick=()=>{stop();showFrame(state.frame+1);};
$('previous-change').onclick=()=>{stop();jumpChange(-1);};$('next-change').onclick=()=>{stop();jumpChange(1);};
$('play').onclick=()=>state.timer?stop():start();$('speed').onchange=()=>{if(state.timer)start();};
$('slot-layout').onchange=()=>showFrame(state.frame);$('atom-numbers').onchange=()=>showFrame(state.frame);$('bond-channel').onchange=()=>showFrame(state.frame);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
let resizeTimer;new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state.run){setupCharts();showFrame(state.frame);}},100);}).observe(document.querySelector('main'));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{heatPalette=[];if(state.run){setupCharts();showFrame(state.frame);}});
handle(refresh)();

const comparisonState={data:null,scene:null,yaw:-0.5,pitch:0.4,zoom:1,drag:null};
async function openComparison(){
  stop();const request=++state.loading;
  const response=await fetch('/evaluation.json');
  if(!response.ok)throw new Error('No scoring comparison has been published yet.');
  const raw=await response.json();if(request!==state.loading)return;
  const variant=$('comparison-native').value,guided=variant==='native_guided';
  const data={...raw,records:raw.records.map(r=>({...r,native:r[variant],delta_C25:{...r.delta_C25,native:r.delta_C25[variant]}}))};
  comparisonState.variant=variant;
  $('comparison-variant-note').textContent=guided?'Separate native-model variant: 200 steps in each diffusion, built-in physical potentials in both phases. No cavity/contact guidance or FK resampling. The standard protocol remains available. These scores have their own C25 baseline.':'Standard native pipeline with 25 steps in each diffusion and no physical guidance. This is shorter sampling than the default 200-step protocol. Both views use structure recycling 0, affinity recycling 5, one sample and seed 17.';
  comparisonState.data=data;$('experiment').hidden=true;$('empty').hidden=true;$('comparison').hidden=false;
  $('comparison-status').textContent=`${(guided?data.guided_complete:data.complete)?'Declared evaluations collected.':guided&&data.guided_status==='failed'?'Guided evaluation failed; partial outputs are retained.':'Evaluation in progress.'} ${data.records.filter(r=>r.native).length}/${data.records.length} native Boltz results (${data.records.filter(r=>r.native?.audit.stereo_match).length} preserve coordinate stereo); ${data.records.reduce((n,r)=>n+r.gnina.length,0)}/${data.records.length*data.protocol.sampling.length} GNINA first poses${data.missing_gnina_repeats?`; ${data.missing_gnina_repeats} preparation failure(s) retained`:''}.`;
  $('comparison-scope').textContent=data.scope+ (data.all_gnina_poses?` Placement check: ${data.first_gnina_poses.cavity_gate_pass}/${data.first_gnina_poses.total} first-written poses; ${data.all_gnina_poses.cavity_gate_pass}/${data.all_gnina_poses.total} of all saved GNINA poses. All saved poses are available in the picker.`:'');
  $('comparison-description').textContent=data.description||'No additional experiment description was recorded.';
  $('comparison-provenance').textContent=JSON.stringify({protocol:data.protocol,panel_sha256:data.panel_sha256,analysis_sha256:data.analysis_sha256,source_sha256:data.source_sha256,native_executions:data.native_executions,guided_declaration:data.guided_declaration},null,2);
  const table=document.createElement('table'),head=document.createElement('tr');
  const columns=['Molecule','Optimization source','Vina ↓; ΔC25','CNN pose ↑; ΔC25','CNN affinity ↑; ΔC25','Boltz affinity ↓; ΔC25','Boltz binder ↑; ΔC25','Native stereo / cavity / geometry','GNINA stereo / cavity / geometry'];
  for(const name of columns){const th=document.createElement('th');th.textContent=name;th.scope='col';head.append(th);}table.append(head);
  const delta=x=>finite(x)?`${x>0?'+':''}${fmt(x)}`:'Unavailable';
  for(const row of data.records){
    const tr=document.createElement('tr');tr.id=`comparison-${row.id}`;if(currentMeta()?.molecule_id===row.id)tr.classList.add('score-focus');if(row.id==='C25')tr.classList.add('comparison-baseline');
    const first=document.createElement('td'),pick=document.createElement('button');pick.textContent=row.id==='C25'?'C25 · START':row.id;pick.title=row.smiles;pick.onclick=()=>{$('pose-molecule').value=row.id;updatePoseOptions();$('pose-canvas').scrollIntoView({block:'center',behavior:'smooth'});};first.append(pick);tr.append(first);
    const sourceCell=document.createElement('td');
    for(const source of row.selection_sources||[]){const meta=state.runs.find(r=>r.source.endsWith('/'+source.run)||r.source.endsWith('/'+source.run+'/proposal'));if(!meta)continue;const b=document.createElement('button');b.textContent=`${meta.name} · update ${source.source_step??row.source_step}`;b.onclick=handle(async()=>{await loadRun(meta.id);const step=source.source_step??row.source_step;const i=state.run.frames.findIndex(f=>f.step===step);if(i>=0)showFrame(i);});sourceCell.append(b);}
    if(!sourceCell.childNodes.length)sourceCell.textContent=row.id==='C25'?'Starting graph':row.id==='positive-JQ1'?'Supplied reference':'Source not exported';tr.append(sourceCell);
    for(const field of ['minimizedAffinity','CNNscore','CNNaffinity']){const v=row.gnina_summary[field];const td=document.createElement('td');td.textContent=finite(v.mean)?`${fmt(v.mean)} [${fmt(v.min)}, ${fmt(v.max)}] (n=${v.n}/${data.protocol.sampling.length}) · Δ ${delta(row.delta_C25.gnina[field])}`:'Pending';td.title=`${v.matched_n}/${v.n} first poses preserve the explicit stereoisomer`;if(v.n&&v.matched_n!==v.n)td.className='comparison-warning';tr.append(td);}
    for(const field of ['affinity_pred_value','affinity_probability_binary']){const td=document.createElement('td');td.textContent=row.native?`${fmt(row.native.scores[field])} · Δ ${delta(row.delta_C25.native[field])}`:'Pending';if(row.native&&!row.native.audit.stereo_match)td.className='comparison-warning';tr.append(td);}
    const label=a=>`${a.stereo_match?'stereo OK':'STEREO MISMATCH'} / ${a.cavity_placement_pass?'in pocket':'outside pocket gate'} / ${a.basic_geometry_pass?'broad checks pass':`${a.bad_bonds} bad bonds, ${a.severe_clashes} severe overlaps`}`;
    const nt=document.createElement('td');nt.textContent=row.native?label(row.native.audit):'Pending';if(row.native&&!row.native.audit.stereo_match)nt.className='comparison-warning';tr.append(nt);
    const gt=document.createElement('td');gt.textContent=row.gnina.length?`${row.gnina.filter(p=>p.audit.stereo_match).length}/${row.gnina.length} stereo; ${row.gnina.filter(p=>p.audit.cavity_placement_pass).length}/${row.gnina.length} pocket; ${row.gnina.filter(p=>p.audit.basic_geometry_pass).length}/${row.gnina.length} geometry`:'Pending';tr.append(gt);table.append(tr);
  }
  $('comparison-table').replaceChildren(table);drawComparisonCharts(data);const previous=$('pose-molecule').value;$('pose-molecule').replaceChildren();
  for(const row of data.records){const o=document.createElement('option');o.value=row.id;o.textContent=row.id==='C25'?'C25 · starting chain':row.id;$('pose-molecule').append(o);}
  $('pose-molecule').value=data.records.some(r=>r.id===previous)?previous:'C25';updatePoseOptions();
}
function updatePoseOptions(){
  const row=comparisonState.data.records.find(r=>r.id===$('pose-molecule').value);$('pose-scorer').replaceChildren();
  const options=[];if(row.native)options.push([row.native.scene,comparisonState.variant==='native_guided'?'Native physical guidance · seed 17':'Standard native Boltz · seed 17']);for(const p of row.gnina_all_poses||row.gnina)options.push([p.scene,`GNINA · seed ${p.seed} · pose ${p.written_pose||1}${p.audit.cavity_placement_pass?'':' · placement check failed'}`]);
  for(const [key,label] of options){const o=document.createElement('option');o.value=key;o.textContent=label;$('pose-scorer').append(o);}selectPose();
}
function selectPose(){
  const key=$('pose-scorer').value,row=comparisonState.data.records.find(r=>r.id===$('pose-molecule').value),p=key.startsWith('native')?row.native:(row.gnina_all_poses||row.gnina).find(p=>p.scene===key);
  const bookmark=state.runs.find(r=>r.kind==='molecule'&&r.molecule_id===row.id);
  if(bookmark){state.id=bookmark.id;renderLibrary();}
  $('comparison-focus').textContent=bookmark?`${bookmark.name}: ${bookmark.description}`:'';
  comparisonState.scene=comparisonState.data.scenes[key]||null;
  $('pose-audit').textContent=p?`Coordinate stereo: ${p.audit.stereo_match?'matches':'MISMATCH'}. In cavity gate: ${p.audit.cavity_placement_pass?'yes':'no'}. Pocket fraction: ${fmt(p.audit.pocket_fraction)}. Anchor distances: ${Object.entries(p.audit.anchor_distances_A).map(([k,v])=>`${k}: ${fmt(v)} Å`).join(', ')}. Bad bonds: ${p.audit.bad_bonds}; severe protein overlaps: ${p.audit.severe_clashes}; nonlocal ligand pairs <1.5 Å: ${p.audit.ligand_nonlocal_pairs_below_1_5A}. ${finite(p.audit.JQ1_reference_RMSD_no_ligand_fit_A)?`Reference JQ1 RMSD (no ligand fit): ${fmt(p.audit.JQ1_reference_RMSD_no_ligand_fit_A)} Å.`:''} ${row.smiles}`:'No scored pose is available yet.';
  $('pose-details').textContent=JSON.stringify(p||{status:'pending'},null,2);
  drawPose();
}
function drawPose(){
  const canvas=$('pose-canvas'),ctx=canvas.getContext('2d'),s=comparisonState.scene;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#fafbfc';ctx.fillRect(0,0,canvas.width,canvas.height);if(!s)return;
  const center=[0,1,2].map(i=>s.reference.reduce((n,p)=>n+p[i],0)/s.reference.length),cy=Math.cos(comparisonState.yaw),sy=Math.sin(comparisonState.yaw),cp=Math.cos(comparisonState.pitch),sp=Math.sin(comparisonState.pitch),scale=20*comparisonState.zoom;
  const project=p=>{const [x,y,z]=p.map((v,i)=>v-center[i]),u=cy*x+sy*z,w=-sy*x+cy*z;return [canvas.width/2+u*scale,canvas.height/2-(cp*y-sp*w)*scale,sp*y+cp*w];};
  function line(a,b,color,width){const p=project(a),q=project(b);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
  for(let i=1;i<s.protein_ca.length;i++)line(s.protein_ca[i-1],s.protein_ca[i],'#b9c1ca',1);
  for(const atom of s.pocket_atoms){const p=project(atom);ctx.beginPath();ctx.arc(p[0],p[1],2,0,Math.PI*2);ctx.fillStyle='#c5cbd288';ctx.fill();}
  if($('pose-reference').checked)for(const [a,b] of s.reference_bonds)line(s.reference[a],s.reference[b],'#9872c6aa',2);
  for(const [a,b,order] of s.bonds)line(s.ligand[a],s.ligand[b],'#475569',order>=2?5:3);
  const palette={C:'#334155',N:'#2563eb',O:'#dc2626',S:'#d19900',Cl:'#16a34a',F:'#22a866'};
  s.ligand.map((v,i)=>({p:project(v),i})).sort((a,b)=>a.p[2]-b.p[2]).forEach(({p,i})=>{ctx.beginPath();ctx.arc(p[0],p[1],5,0,Math.PI*2);ctx.fillStyle=palette[s.elements[i]]||'#999';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1;ctx.stroke();if(s.elements[i]!=='C'){ctx.fillStyle='#111827';ctx.font='12px sans-serif';ctx.fillText(s.elements[i],p[0]+7,p[1]-5);}});
}
$('comparison-native').onchange=handle(openComparison);$('open-comparison').onclick=handle(openComparison);$('refresh-comparison').onclick=handle(openComparison);
$('pose-molecule').onchange=updatePoseOptions;$('pose-scorer').onchange=selectPose;$('pose-reference').onchange=drawPose;
$('pose-reset').onclick=()=>{Object.assign(comparisonState,{yaw:-0.5,pitch:0.4,zoom:1});drawPose();};
$('pose-canvas').onpointerdown=e=>{comparisonState.drag=[e.clientX,e.clientY];e.currentTarget.setPointerCapture(e.pointerId);};
$('pose-canvas').onpointermove=e=>{if(!comparisonState.drag)return;comparisonState.yaw+=(e.clientX-comparisonState.drag[0])*.01;comparisonState.pitch+=(e.clientY-comparisonState.drag[1])*.01;comparisonState.drag=[e.clientX,e.clientY];drawPose();};
$('pose-canvas').onpointerup=()=>{comparisonState.drag=null;};$('pose-canvas').onpointercancel=()=>{comparisonState.drag=null;};
$('pose-canvas').addEventListener('wheel',e=>{e.preventDefault();comparisonState.zoom=Math.max(.2,Math.min(5,comparisonState.zoom*Math.exp(-e.deltaY*.001)));drawPose();},{passive:false});

function drawComparisonCharts(data){
  $('comparison-charts').replaceChildren();
  for(const [kind,key,label] of [['gnina','minimizedAffinity','Vina energy ↓'],['gnina','CNNscore','CNN pose score ↑'],['gnina','CNNaffinity','CNN affinity ↑'],['native','affinity_pred_value','Native Boltz affinity ↓'],['native','affinity_probability_binary','Native Boltz binder output ↑']]){
    const rows=data.records.map(r=>{const g=r.gnina_summary[key],v=kind==='gnina'?g?.mean:r.native?.scores[key];return {r,v,lo:kind==='gnina'?g?.min:v,hi:kind==='gnina'?g?.max:v,qualified:kind==='gnina'?g?.n===data.protocol.sampling.length&&g?.matched_n===g?.n:r.native?.audit.stereo_match};});
    const numbers=rows.filter(r=>finite(r.v)).flatMap(r=>[r.lo,r.hi]);const section=document.createElement('div'),title=document.createElement('h4');title.textContent=label;section.append(title);$('comparison-charts').append(section);if(!numbers.length){const p=document.createElement('p');p.textContent='Pending';section.append(p);continue;}
    const width=370,left=104,right=20,top=28,height=430,bottom=30;let low=Math.min(...numbers),high=Math.max(...numbers);if(key.includes('probability')||key==='CNNscore'){low=0;high=1;}else{const pad=Math.max((high-low)*.1,.1);low-=pad;high+=pad;}const x=v=>left+(v-low)/(high-low)*(width-left-right),svg=node('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':`${label}. Solid points preserve selected stereo; hollow points have unqualified stereo. Dashed line is starting C25.`});
    for(let i=0;i<4;i++){const v=low+(high-low)*i/3;svg.append(node('line',{x1:x(v),y1:top,x2:x(v),y2:height-bottom,stroke:'var(--line)'}),node('text',{x:x(v),y:height-9,'text-anchor':'middle'},fmt(v)));}
    const baseline=rows.find(x=>x.r.id==='C25');if(finite(baseline?.v))svg.append(node('line',{x1:x(baseline.v),y1:top,x2:x(baseline.v),y2:height-bottom,stroke:'#64748b','stroke-dasharray':'4 4'}));
    rows.forEach((d,i)=>{const y=top+12+i*26,color=d.r.id==='positive-JQ1'?'#16846b':d.r.id==='C25'?'#64748b':d.qualified?'#217a9c':'#c97715';svg.append(node('text',{x:left-8,y:y+4,'text-anchor':'end','font-weight':d.r.id==='C25'?700:400},d.r.id));if(!finite(d.v))return;
      svg.append(node('line',{x1:x(d.lo),y1:y,x2:x(d.hi),y2:y,stroke:color,'stroke-width':2}));const dot=node('circle',{cx:x(d.v),cy:y,r:4,fill:d.qualified?color:'white',stroke:color,'stroke-width':2,tabindex:0,role:'button','aria-label':`Inspect ${d.r.id}, ${label}: ${fmt(d.v)}${d.qualified?'':', stereo unqualified'}`});dot.append(node('title',{},`${d.r.id}: ${fmt(d.v)} [${fmt(d.lo)}, ${fmt(d.hi)}]. ${d.qualified?'Stereo preserved.':'Coordinate stereo is mismatched or evaluation incomplete.'}`));const pick=()=>{$('pose-molecule').value=d.r.id;updatePoseOptions();const p=kind==='native'?d.r.native:d.r.gnina[0];if(p){$('pose-scorer').value=p.scene;selectPose();}$('pose-canvas').scrollIntoView({block:'center',behavior:'smooth'});};dot.onclick=pick;dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick();}};svg.append(dot);});section.append(svg);
  }
}
