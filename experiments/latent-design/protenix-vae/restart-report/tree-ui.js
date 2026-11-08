"use strict";

const RestartTreeView = (() => {
  let pendingData;
  let runId = "";
  let cleanup = () => {};
  let request = 0;
  const green = "#178553";
  const yellow = "#eac73b";
  const pathInk = "#294d92";

  function load() {
    pendingData ??= (async () => {
      const encoded = byId("tree-payload").textContent.trim();
      if (!encoded) return {jobs:[],structures:{},raw_graphs:{}};
      const bytes = typeof Uint8Array.fromBase64 === "function"
        ? Uint8Array.fromBase64(encoded) : Uint8Array.from(atob(encoded),char=>char.charCodeAt(0));
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      return new Response(stream).json();
    })();
    return pendingData;
  }

  const nodeLabel = node => [
    node.arm ? label(node.arm) : "",
    Number(node.cycle) ? `cycle ${node.cycle}` : "",
    node.branch || "initialization",
    `update ${node.step}`,
  ].filter(Boolean).join(" · ");
  const getNode = (model,node) => typeof node === "number" ? model.nodes[node] : node;
  const indexOf = node => node.index;

  function rawStructure(graph) {
    if (!graph?.atoms?.length) return null;
    const elements = {1:"H",5:"B",6:"C",7:"N",8:"O",9:"F",14:"Si",15:"P",16:"S",17:"Cl",34:"Se",35:"Br",53:"I"};
    return {
      raw:true,
      atoms:graph.atoms.map(([element,charge],index)=>{
        const angle=2*Math.PI*index/graph.atoms.length-Math.PI/2;
        return [.5+.42*Math.cos(angle),.5+.42*Math.sin(angle),elements[element]||String(element),charge];
      }),
      bonds:graph.bonds,
    };
  }

  function molecule(node,data) {
    const key = node.valid ? node.smiles : node.rawKey;
    return structures[key] || data.structures?.[key] || rawStructure(data.raw_graphs?.[node.rawKey]);
  }

  function destroy() {
    request++;
    cleanup();
    cleanup=()=>{};
  }

  async function render() {
    destroy();
    const current = request;
    const content = byId("tree-content");
    byId("tree-controls").textContent = "";
    content.innerHTML = empty("Loading the saved state trees…");
    try {
      const data = await load();
      if (request!==current || byId("trees").hidden) return;
      const jobs = data.jobs || [];
      if (!jobs.length) {
        content.innerHTML=empty("No saved tree data is available in this report.");
        return;
      }
      renderHeader(jobs,data.captured_at);
      const available = job => job.branches?.some(branch=>branch.states?.length);
      const hasFork=job=>job.branches?.some(branch=>branch.source&&branch.states?.length);
      const best = jobs.find(job=>isDone(job)&&hasFork(job)) || jobs.find(hasFork) ||
        jobs.find(job=>isDone(job)&&available(job)) || jobs.find(available) || jobs[0];
      if (!jobs.some(job=>String(job.id)===runId)) runId=String(best.id);
      const job=jobs.find(row=>String(row.id)===runId);
      const choices=[
        ["Completed runs",jobs.filter(row=>isDone(row))],
        ["Runs with partial or pending data",jobs.filter(row=>!isDone(row))],
      ];
      byId("tree-controls").innerHTML=`<label for="tree-run">Experiment<select id="tree-run">${choices.filter(([,group])=>group.length)
        .map(([name,group])=>`<optgroup label="${escape(name)}">${group.map(row=>option(row.id,`${title(row)} · ${row.id}`,runId)).join("")}</optgroup>`).join("")}</select></label>`+
        `<span class="quiet small">Tree snapshot ${escape(formatTreeDate(data.captured_at))}</span>`;
      byId("tree-run").addEventListener("change",event=>{runId=event.target.value;render();});
      if (!available(job)) {
        content.innerHTML = badge(job)+empty(job.reason || "This run has no saved state tree yet.");
        return;
      }
      const model=ProtenixTree.build(job);
      if(!model.nodes.length) {
        content.innerHTML=empty("This run has no distinct saved states to display.");
        return;
      }
      mount(job,model,data,content);
    } catch (error) {
      if(request===current) content.innerHTML=empty(`The tree could not be displayed. ${error.message}`);
    }
  }

  function formatTreeDate(value) {
    const date=new Date(value);
    return Number.isFinite(date.valueOf()) ? new Intl.DateTimeFormat("en-GB",{
      timeZone:"Europe/Paris",day:"numeric",month:"short",hour:"2-digit",minute:"2-digit",timeZoneName:"short",
    }).format(date) : (value||"");
  }

  function mount(job,model,data,content) {
    const roots=model.roots.map(node=>getNode(model,node));
    const edges=model.edges.map(edge=>({
      source:getNode(model,edge.source),target:getNode(model,edge.target),kind:edge.kind,
    }));
    const sources=new Set(edges.map(edge=>indexOf(edge.source)));
    const leaves=model.nodes.filter(node=>!sources.has(indexOf(node)))
      .sort((a,b)=>(Number(b.cycle)||0)-(Number(a.cycle)||0)||b.step-a.step||b.index-a.index);
    const forks=edges.filter(edge=>edge.kind==="restart").length;
    const missing=(job.branches||[]).reduce((total,branch)=>total+
      (Array.isArray(branch.missing_steps)?branch.missing_steps.length:Number(branch.missing_steps)||0),0);
    const summary=[
      `${count(model.nodes.length)} state nodes`,`${count(roots.length)} ${roots.length===1?"root":"roots"}`,
      `${count(leaves.length)} ${leaves.length===1?"leaf":"leaves"}`,`${count(forks)} recorded restarts`,
    ].join(" · ");
    const disclaimer = !isDone(job) ? " · partial tree" : "";
    content.innerHTML=`<div class="legend tree-legend">${badge(job)}<span>${escape(summary+disclaimer)}</span><span><i class="swatch tree-valid"></i>Valid</span><span><i class="swatch tree-invalid"></i>Invalid</span><span><i class="swatch tree-selected"></i>Selected path</span></div>
      <div class="tree-stack">
        <div class="tree-panel">
          <div class="tree-tools">
            <button type="button" id="tree-fit">Fit tree</button>
            <button type="button" id="tree-plus" aria-label="Zoom in">+</button>
            <button type="button" id="tree-minus" aria-label="Zoom out">−</button>
            <button type="button" id="tree-fit-path">Fit selected path</button>
            <button type="button" id="tree-latest">Latest leaf</button>
            <a href="#tree-inspection" class="tree-jump">Inspect selected path ↓</a>
          </div>
          <div class="tree-stage" id="tree-stage">
            <canvas id="tree-canvas" tabindex="0" aria-label="Molecular state tree. Drag to pan or scroll to zoom. Select a leaf from the leaf menu to browse the path with the keyboard."></canvas>
            <span class="tree-direction">Roots at bottom · later states ↑</span>
          </div>
          <p class="tree-note">Drag to pan and scroll to zoom. Click a node to see its molecule. A dashed edge marks a recorded restart.</p>
        </div>
        <section class="tree-inspector tree-chart-panel" id="tree-inspection" aria-labelledby="tree-loss-title">
          <div class="tree-section-header">
            <div><h3 id="tree-loss-title">Selected path loss</h3><p class="quiet small" id="tree-loss-coverage"></p></div>
            <div class="tree-chart-controls">
              ${select("tree-leaf","Leaf", [["","Select a leaf"], ...leaves.map(node=>[node.index,nodeLabel(node)])],"")}
              ${select("tree-loss-metric","Metric",[["relaxed","Relaxed contact loss"],["chemistry","Repair loss"],["valence","Valence excess"]],"relaxed")}
            </div>
          </div>
          <div id="tree-loss-chart" class="tree-loss-chart"></div>
          <div class="tree-chart-foot quiet small"><span>Click the chart to select an update · dashed lines mark restarts</span><span id="tree-loss-value"></span></div>
        </section>
        <section class="tree-inspector tree-molecule-panel" aria-label="Molecule on selected path">
          <div class="tree-path">
            <p class="quiet small" id="tree-path-info" aria-live="polite"></p>
            <label class="slider-label" for="tree-path">Browse updates on the selected path<input id="tree-path" type="range" min="0" max="0" value="0"></label>
            <div class="playback-controls">
              <button type="button" id="tree-root">Root</button>
              <button type="button" id="tree-prev">Previous</button>
              <button type="button" id="tree-next">Next</button>
              <button type="button" id="tree-end">Path end</button>
              <button type="button" id="tree-best">Lowest on chart</button>
              <button type="button" id="tree-export">Download path CSV</button>
            </div>
          </div>
          <div id="tree-node"></div>
        </section>
      </div>
      <p class="tree-note">An unchanged hard state occupies one node within a branch. An independent start has its own root.${missing ? ` Raw topology is unavailable for ${escape(count(missing))} saved observations.` : ""}</p>`;

    const canvas=byId("tree-canvas");
    const ctx=canvas.getContext("2d");
    const stage=byId("tree-stage");
    const slider=byId("tree-path");
    const leafSelect=byId("tree-leaf");
    const metricSelect=byId("tree-loss-metric");
    const chart=ProtenixPathChart.mount(byId("tree-loss-chart"),metricSelect,byId("tree-loss-value"),
      index=>showNode(index,true));
    let width=0,height=0,dpr=1;
    let view={sx:1,sy:1,ox:0,oy:0,zoom:1};
    let path=[],pathLinks=[],pathSet=new Set(),points=[],pathIndex=0;
    let pointer=null,scheduled=0;
    const rootIndices=new Set(roots.map(indexOf)), leafIndices=new Set(leaves.map(indexOf));
    const screenX=node=>view.ox+node.x*view.sx;
    const screenY=node=>view.oy+node.y*view.sy;

    function fit(nodes) {
      if(!nodes.length||!width||!height)return;
      let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
      for(const node of nodes){
        minX=Math.min(minX,node.x);maxX=Math.max(maxX,node.x);
        minY=Math.min(minY,node.y);maxY=Math.max(maxY,node.y);
      }
      const spanX=maxX-minX,spanY=maxY-minY;
      const sx=spanX?Math.max(1,width-84)/spanX:1;
      const sy=spanY?Math.max(1,height-74)/spanY:1;
      view={sx,sy,ox:width/2-(minX+maxX)*sx/2,oy:height-42-maxY*sy,zoom:1};
      queueDraw();
    }

    function resize() {
      const dims=stage.getBoundingClientRect();
      if(dims.width<2||dims.height<2)return;
      const nextDpr=Math.min(window.devicePixelRatio||1,2);
      if(width===dims.width&&height===dims.height&&dpr===nextDpr)return;
      width=dims.width;height=dims.height;dpr=nextDpr;
      canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
      fit(model.nodes);
    }

    function zoom(multiplier,x=width/2,y=height/2) {
      const target=Math.max(.25,Math.min(280,view.zoom*multiplier));
      const factor=target/view.zoom;
      view={sx:view.sx*factor,sy:view.sy*factor,
        ox:x+(view.ox-x)*factor,oy:y+(view.oy-y)*factor,zoom:target};
      queueDraw();
    }

    function queueDraw() {
      if(!scheduled) scheduled=requestAnimationFrame(()=>{scheduled=0;draw();});
    }

    function trace(list,kind) {
      ctx.beginPath();
      for(const edge of list) {
        if((edge.kind==="restart")!==(kind==="restart"))continue;
        const x1=screenX(edge.source),x2=screenX(edge.target),y1=screenY(edge.source),y2=screenY(edge.target);
        if((x1<-10&&x2<-10)||(x1>width+10&&x2>width+10)||(y1<-10&&y2<-10)||(y1>height+10&&y2>height+10))continue;
        ctx.moveTo(x1,y1);
        if(kind==="restart"&&Math.abs(x1-x2)>3){
          const mid=(y1+y2)/2;
          ctx.bezierCurveTo(x1,mid,x2,mid,x2,y2);
        } else ctx.lineTo(x2,y2);
      }
      ctx.stroke();
    }

    function visibleNode(node) {
      const x=screenX(node),y=screenY(node);
      return x>=-9&&x<=width+9&&y>=-9&&y<=height+9;
    }

    function dots(list,valid,radius) {
      ctx.beginPath();
      for(const node of list) {
        if(node.valid!==valid)continue;
        const x=screenX(node),y=screenY(node);
        ctx.moveTo(x+radius,y);ctx.arc(x,y,radius,0,Math.PI*2);
      }
      ctx.fillStyle=valid?green:yellow;ctx.fill();
    }

    function draw() {
      if(!width||!height)return;
      ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.clearRect(0,0,width,height);
      const active=points[pathIndex]?.node;
      const visible=model.visible((-10-view.ox)/view.sx,(-10-view.oy)/view.sy,
        (width+10-view.ox)/view.sx,(height+10-view.oy)/view.sy);
      ctx.globalAlpha=path.length>1?.35:.9;
      ctx.strokeStyle="#91a3a8";ctx.lineWidth=.95;ctx.setLineDash([]);
      trace(visible.edges,"step");
      ctx.strokeStyle="#718b80";ctx.setLineDash([4,3]);
      trace(visible.edges,"restart");
      const radius=Math.min(4,Math.max(1.35,view.sy*.72));
      ctx.globalAlpha=path.length>1?.30:1;
      dots(visible.nodes,true,radius);dots(visible.nodes,false,radius);
      ctx.setLineDash([]);
      ctx.strokeStyle="#314550";ctx.lineWidth=1;
      for(const node of visible.nodes) {
        if(!rootIndices.has(node.index)&&!leafIndices.has(node.index))continue;
        ctx.globalAlpha=path.length>1&&!pathSet.has(node.index)?.46:1;
        const x=screenX(node),y=screenY(node),r=Math.max(3.3,radius+1);
        ctx.beginPath();
        if(leafIndices.has(node.index)&&!rootIndices.has(node.index)){
          ctx.moveTo(x,y-r);ctx.lineTo(x+r,y);ctx.lineTo(x,y+r);ctx.lineTo(x-r,y);ctx.closePath();
        }else ctx.arc(x,y,r,0,Math.PI*2);
        ctx.fillStyle=node.valid?green:yellow;ctx.fill();ctx.stroke();
      }
      ctx.globalAlpha=1;
      if(path.length>1) {
        ctx.strokeStyle=pathInk;ctx.lineWidth=3;
        trace(pathLinks,"step");ctx.setLineDash([5,3]);trace(pathLinks,"restart");ctx.setLineDash([]);
        const activeNodes=visible.nodes.filter(node=>pathSet.has(node.index));
        dots(activeNodes,true,radius+.35);dots(activeNodes,false,radius+.35);
      }
      if(active&&visibleNode(active)) {
        const x=screenX(active),y=screenY(active);
        ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);
        ctx.fillStyle="#fbfffd";ctx.fill();ctx.lineWidth=2;ctx.strokeStyle=pathInk;ctx.stroke();
        ctx.beginPath();ctx.arc(x,y,4.2,0,Math.PI*2);
        ctx.fillStyle=active.valid?green:yellow;ctx.fill();
      }
    }

    function showNode(index,follow) {
      if(!points.length)return;
      pathIndex=Math.max(0,Math.min(index,points.length-1));
      const point=points[pathIndex],node=point.node;
      slider.value=String(pathIndex);
      slider.setAttribute("aria-valuetext",`Observation ${pathIndex+1} of ${points.length}, branch update ${point.step}`);
      byId("tree-prev").disabled=pathIndex===0;byId("tree-root").disabled=pathIndex===0;
      byId("tree-next").disabled=pathIndex===points.length-1;byId("tree-end").disabled=pathIndex===points.length-1;
      const restarts=pathLinks.filter(edge=>edge.kind==="restart").length;
      byId("tree-path-info").textContent=`Observation ${count(pathIndex+1)} of ${count(points.length)} · Path step ${count(point.depth)}${restarts?` · ${restarts} ${restarts===1?"restart":"restarts"}`:""}`;
      chart.show(pathIndex);
      const graph=molecule(node,data);
      const stageLabel=point.step===node.step&&rootIndices.has(node.index)?"Initialization":
        pathIndex===points.length-1&&leafIndices.has(node.index)?"Leaf":"State";
      const validLabel=node.valid?"Valid":"Invalid";
      const restart=point.restart&&node.incoming?.kind==="restart" ? `Restarted from ${node.incoming.source.branch}, update ${node.incoming.sourceStep}.` : "";
      byId("tree-node").innerHTML=`<h3>${stageLabel} <span class="pill ${node.valid?"status-done":"tree-invalid-label"}">${validLabel}</span></h3><p class="quiet small tree-node-meta">${escape(nodeLabel({...node,step:point.step}))}</p>`+
        (restart?`<p class="quiet small">${escape(restart)}</p>`:"")+
        (drawStructure(graph,`${validLabel} graph at update ${point.step}`)||empty("This state has no saved molecular graph."))+
        (node.smiles?`<code class="mono-block">${escape(node.smiles)}</code>`:"")+
        `<p class="quiet small">${Number.isFinite(point.relaxed)?`Relaxed contact loss ${fmt(point.relaxed,5)}`:""}${graph?.raw?` · Raw topology with ${graph.atoms.length} atoms and ${graph.bonds.length} bonds`:""}</p>`;
      if(follow&&!visibleNode(node)){
        view.ox=width/2-node.x*view.sx;
        view.oy=height/2-node.y*view.sy;
      }
      queueDraw();
    }

    function selectNode(node,follow=true) {
      const ancestry=model.ancestry(node);
      path=ancestry.nodes;
      pathLinks=ancestry.edges;
      pathSet=new Set(path.map(indexOf));
      const timeline=model.timeline(ancestry);
      points=timeline.points;
      chart.set(points);
      slider.max=String(Math.max(0,points.length-1));
      leafSelect.value=leafIndices.has(node.index)?String(node.index):"";
      byId("tree-loss-coverage").textContent=timeline.distinctOnly
        ? "Older snapshot coverage includes saved state nodes. Repeated updates may be absent."
        : `${count(points.length)} recorded updates, including repeated molecular states.`;
      showNode(points.length-1,follow);
    }

    const local=event=>{
      const rect=canvas.getBoundingClientRect();
      return {x:event.clientX-rect.left,y:event.clientY-rect.top};
    };
    canvas.addEventListener("wheel",event=>{
      event.preventDefault();
      const {x,y}=local(event);
      zoom(Math.exp(-Math.max(-180,Math.min(180,event.deltaY))*.0025),x,y);
    },{passive:false});
    canvas.addEventListener("pointerdown",event=>{
      if(event.button!==0)return;
      const {x,y}=local(event);
      pointer={id:event.pointerId,x,y,ox:view.ox,oy:view.oy,moved:false};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove",event=>{
      if(!pointer||pointer.id!==event.pointerId)return;
      const {x,y}=local(event);
      if(!pointer.moved&&Math.hypot(x-pointer.x,y-pointer.y)<4)return;
      pointer.moved=true;canvas.dataset.dragging="true";
      view.ox=pointer.ox+x-pointer.x;view.oy=pointer.oy+y-pointer.y;
      queueDraw();
    });
    canvas.addEventListener("pointerup",event=>{
      if(!pointer||pointer.id!==event.pointerId)return;
      const moved=pointer.moved;pointer=null;canvas.dataset.dragging="false";
      if(moved)return;
      const {x,y}=local(event);
      let best=null,distance=10,endpoint=null,endpointDistance=5;
      const nearby=model.visible((x-10-view.ox)/view.sx,(y-10-view.oy)/view.sy,
        (x+10-view.ox)/view.sx,(y+10-view.oy)/view.sy).nodes;
      for(const node of nearby){
        const d=Math.hypot(x-screenX(node),y-screenY(node));
        if(d<distance){best=node;distance=d;}
        if(d<endpointDistance&&(leafIndices.has(node.index)||rootIndices.has(node.index))){endpoint=node;endpointDistance=d;}
      }
      const picked=endpoint||best;
      if(picked){
        selectNode(picked,false);
      }
    });
    canvas.addEventListener("pointercancel",()=>{pointer=null;canvas.dataset.dragging="false";});
    byId("tree-fit").addEventListener("click",()=>fit(model.nodes));
    byId("tree-plus").addEventListener("click",()=>zoom(1.55));
    byId("tree-minus").addEventListener("click",()=>zoom(1/1.55));
    byId("tree-fit-path").addEventListener("click",()=>fit(path));
    byId("tree-latest").addEventListener("click",()=>{if(leaves.length){selectNode(leaves[0],false);fit(path);}});
    leafSelect.addEventListener("change",()=>{if(leafSelect.value!==""){selectNode(model.nodes[Number(leafSelect.value)],false);fit(path);}});
    byId("tree-prev").addEventListener("click",()=>showNode(pathIndex-1,true));
    byId("tree-next").addEventListener("click",()=>showNode(pathIndex+1,true));
    byId("tree-root").addEventListener("click",()=>showNode(0,true));
    byId("tree-end").addEventListener("click",()=>showNode(points.length-1,true));
    byId("tree-best").addEventListener("click",()=>{
      const metric=metricSelect.value;
      let best=-1;
      points.forEach((point,index)=>{
        if(Number.isFinite(point[metric])&&(best<0||point[metric]<points[best][metric]))best=index;
      });
      if(best>=0)showNode(best,true);
    });
    byId("tree-export").addEventListener("click",()=>{
      const quote=value=>'"'+String(value??"").replaceAll('"','""')+'"';
      const rows=points.map(({node,step,depth,relaxed,chemistry,weight,valence,restart})=>
        [depth,node.cycle,node.branch,step,relaxed,chemistry,weight,valence,node.valid,node.smiles,restart].map(quote).join(","));
      const csv=["path_step,cycle,branch,update,relaxed_loss,repair_loss,repair_weight,valence_excess,valid,smiles,restart",...rows].join("\n");
      const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));
      const link=document.createElement("a");
      link.href=url;link.download="protenix-"+job.id+"-path.csv";link.click();
      setTimeout(()=>URL.revokeObjectURL(url),0);
    });
    slider.addEventListener("input",()=>showNode(Number(slider.value),true));
    canvas.addEventListener("keydown",event=>{
      if(!["ArrowUp","ArrowDown","Home","End","+","-"].includes(event.key))return;
      event.preventDefault();
      if(event.key==="+")zoom(1.55);
      else if(event.key==="-")zoom(1/1.55);
      else showNode(event.key==="Home"?0:event.key==="End"?points.length-1:pathIndex+(event.key==="ArrowUp"?1:-1),true);
    });
    const observer=new ResizeObserver(resize);observer.observe(stage);
    cleanup=()=>{observer.disconnect();chart.destroy();if(scheduled)cancelAnimationFrame(scheduled);};
    resize();
    selectNode(leaves[0]||roots[0]||model.nodes[0],false);
  }

  return {render,destroy};
})();
