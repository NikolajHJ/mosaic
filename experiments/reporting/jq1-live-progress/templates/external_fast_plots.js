  // Presentation cache: retain every saved scalar; repaint only the cursor.
  const plotCache = new Map();
  function renderPlots() {
    const signature = [runIndex,trajectoryIndex,$("overlay-runs").checked,$("total-plot").clientWidth,$("binding-plot").clientWidth].join(':');
    if (plotCache.get('signature') !== signature) {
      const ts=traces(); $("plot-legend").replaceChildren();
      ts.forEach(t=>{const item=el('span'),swatch=el('i',undefined,'swatch');swatch.style.background=t.color;item.append(swatch,document.createTextNode(nameOf(t.r,t.t)));$("plot-legend").append(item);});
      $("comparison-note").textContent="All saved scalar values are plotted. Changed guidance weights or targets change the total objective; gaps are retained. Click the plot to inspect an exact saved step.";
      drawPlot($("total-plot"),ts,'loss');drawPlot($("binding-plot"),ts,'binding_loss');
      plotCache.set('signature',signature);
    }
    for(const key of ['loss','binding_loss']) {
      const p=plotCache.get(key);if(!p)continue;
      const x=p.X(frame().step),value=metric(frame(),key);
      p.cursor.setAttribute('x1',x);p.cursor.setAttribute('x2',x);
      p.dot.style.display=finite(value)?'':'none';
      if(finite(value)){p.dot.setAttribute('cx',x);p.dot.setAttribute('cy',p.Y(value));}
      p.root.setAttribute('aria-label',key+' · selected step '+frame().step+' · '+fmt(value,7)+' · '+p.count+' saved values');
    }
  }
  function drawPlot(container,ts,key) {
    let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity,count=0;
    for(const t of ts)for(const f of t.fs)if(finite(f.step)&&finite(metric(f,key))){xmin=Math.min(xmin,f.step);xmax=Math.max(xmax,f.step);ymin=Math.min(ymin,metric(f,key));ymax=Math.max(ymax,metric(f,key));count++;}
    if(!count){missing(container,'This objective was not saved.');plotCache.delete(key);return;}
    const W=container.clientWidth||560,H=225,m={l:57,r:15,t:18,b:35};
    if(xmin===xmax){xmin-=.5;xmax+=.5;}const pad=(ymax-ymin)*.09||Math.max(Math.abs(ymin)*.05,.01);ymin-=pad;ymax+=pad;
    const X=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r),Y=y=>H-m.b-(y-ymin)/(ymax-ymin)*(H-m.t-m.b);
    const wrap=el('div');wrap.style.cssText='position:relative;height:225px';
    const canvas=el('canvas'),dpr=Math.min(2,window.devicePixelRatio||1);canvas.width=W*dpr;canvas.height=H*dpr;canvas.style.cssText='position:absolute;inset:0;width:100%;height:225px';canvas.setAttribute('aria-hidden','true');canvas.dataset.points=String(count);
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    const root=svg('svg',{viewBox:`0 0 ${W} ${H}`,role:'img',tabindex:0});root.style.cssText='position:absolute;inset:0;width:100%;height:225px;cursor:crosshair';
    for(let i=0;i<5;i++){const v=ymin+(ymax-ymin)*i/4,y=Y(v);root.append(svg('line',{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:'#e5ece8'}),svg('text',{x:m.l-8,y:y+3,'text-anchor':'end'},fmt(v,3)));}
    for(let i=0;i<5;i++){const v=xmin+(xmax-xmin)*i/4;root.append(svg('text',{x:X(v),y:H-15,'text-anchor':'middle'},fmt(v,1)));}
    root.append(svg('text',{x:(W+m.l-m.r)/2,y:H-1,'text-anchor':'middle'},'Optimization step'));
    for(const t of ts){
      ctx.strokeStyle=ctx.fillStyle=t.color;ctx.globalAlpha=t.selected?1:.65;ctx.lineWidth=t.selected?1.6:1.1;ctx.setLineDash(t.selected?[]:[5,3]);ctx.beginPath();let prev=null;
      for(const f of t.fs){if(!finite(f.step)||!finite(metric(f,key))){prev=null;continue;}const jump=prev&&key==='loss'&&((prev.repair_target_sha256??null)!==(f.repair_target_sha256??null)||(metric(prev,'repair_weight')??null)!==(metric(f,'repair_weight')??null));const connected=prev&&f.step===prev.step+1&&prev.context_sha256===f.context_sha256&&!jump;if(connected)ctx.lineTo(X(f.step),Y(metric(f,key)));else ctx.moveTo(X(f.step),Y(metric(f,key)));prev=f;}
      ctx.stroke();ctx.setLineDash([]);ctx.beginPath();
      // Every finite saved point, including isolated points at objective breaks.
      for(const f of t.fs)if(finite(f.step)&&finite(metric(f,key))){const x=X(f.step),y=Y(metric(f,key));ctx.moveTo(x+1.25,y);ctx.arc(x,y,1.25,0,Math.PI*2);}ctx.fill();
      if(t.selected){ctx.globalAlpha=.2;ctx.lineWidth=.7;ctx.setLineDash([2,4]);ctx.beginPath();t.fs.forEach((f,i)=>{if(finite(f.step)&&eventsAt(t.fs,i).length){ctx.moveTo(X(f.step),m.t);ctx.lineTo(X(f.step),H-m.b);}});ctx.stroke();}
    }
    const cursor=svg('line',{y1:m.t,y2:H-m.b,stroke:'#19353d','stroke-dasharray':'4 3'}),dot=svg('circle',{r:5,fill:color(runIndex),class:'selected-point'});root.append(cursor,dot);
    const nearest=e=>{const rect=root.getBoundingClientRect(),px=(e.clientX-rect.left)*W/rect.width,py=(e.clientY-rect.top)*H/rect.height;let best=Infinity,result=null;for(const t of ts)t.fs.forEach((f,i)=>{if(!finite(metric(f,key))||!finite(f.step))return;const d=(X(f.step)-px)**2+(Y(metric(f,key))-py)**2;if(d<best){best=d;result={t,f,i};}});return result;};
    root.onclick=e=>{const hit=nearest(e);if(hit){stop();if(hit.t.ri===runIndex&&hit.t.ti===trajectoryIndex)chooseFrame(hit.i);else selectRun(hit.t.ri,hit.t.ti,hit.i);}};
    root.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();stop();chooseFrame(frameIndex+(e.key==='ArrowLeft'?-1:1));}};
    wrap.append(canvas,root);container.replaceChildren(wrap);plotCache.set(key,{root,X,Y,cursor,dot,count});
  }
