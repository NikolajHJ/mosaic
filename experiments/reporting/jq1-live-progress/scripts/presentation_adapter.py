"""Adapt only presentation code; preserve the embedded saved-data bytes exactly."""
import hashlib
import json
from pathlib import Path
import re

OWNER = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Presentation adapter input differs: ' + old[:90])
    return text.replace(old, new)


def adapt_external(raw, layouts):
    text = raw.decode('utf-8')
    marker = '<script id="report-data" type="application/json">'
    start = text.index(marker) + len(marker)
    end = text.index('</script>', start)
    saved = text[start:end]
    if hashlib.sha256(raw).hexdigest() != layouts['source_html_sha256'] or not layouts['identity_preserved']:
        raise ValueError('Display geometry belongs to a different saved report')
    original_js = re.search(r'<script>\s*(/\* Standalone saved-artifact viewer[\s\S]*?)</script>', text)
    if not original_js:
        raise ValueError('Unsupported external viewer')
    js = original_js.group(1)
    first, last = js.index('  function renderPlots()'), js.index('  function probabilityRow(')
    fast = (OWNER / 'templates/external_fast_plots.js').read_text(encoding='utf8')
    js = js[:first] + fast + '\n' + js[last:]
    js = replace_once(js, '  const runs = list(data.runs);', '  const displayLayouts = '+json.dumps(layouts['layouts'], separators=(',', ':'))+';\n  const runs = list(data.runs);')
    first, last = js.index('    const atoms=list(c.graph.atoms), layout='), js.index('    const atomSet=', js.index('  function graphView('))
    geometry = '''    const atoms=list(c.graph.atoms), key=JSON.stringify([atoms.map(a=>a.slot).sort((a,b)=>a-b),list(c.graph.bonds).map(b=>[Math.min(b.i,b.j),Math.max(b.i,b.j),b.type]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]||String(a[2]).localeCompare(String(b[2])))]), geometry=displayLayouts[key];
    if(!geometry){container.append(el('p','Display layout unavailable for this saved raw graph.','molecule-empty'));return container;}
    const W=720,H=440,pos=new Map(),layout=geometry.layout,scale=Math.min(640/layout.width,350/layout.height),left=(W-layout.width*scale)/2,top=(H-layout.height*scale)/2;
    geometry.nodes.forEach(n=>pos.set(n.id,{x:left+n.x*layout.width*scale,y:top+n.y*layout.height*scale}));
    const root=svg('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Saved raw molecular graph; '+validityText(c),'data-layout':'raw-bond-layout-v1'});root.append(svg('title',{},validityText(c)+'. Bond-aware display coordinates, not a physical pose.'));
'''
    js = js[:first] + geometry + js[last:]
    js = replace_once(js, 'Fixed slot schematic, not a physical conformation.', 'Bond-aware diagram using the Mosaic report layout; not a physical conformation.')
    js = replace_once(js, 'c && c.failure ? c.failure : "Raw graph labels were not saved for this candidate."', '"Raw graph labels were not saved at this update. Use the slider to browse retained molecules."')
    # Evidence includes frame-specific coefficients/targets, so retain its refresh.
    # Smaller stable atom markers keep dense raw graphs legible.
    js = replace_once(js, 'r:probability===null?18:13+7*probability', 'r:probability===null?12:10+4*probability')
    js = replace_once(js, 'y:xy.y+32', 'y:xy.y+25')
    # Browse retained molecules, while the plots/numeric field expose every loss.
    helpers = '''  const retainedCache=new WeakMap();
  function retained(){const fs=frames();if(!retainedCache.has(fs))retainedCache.set(fs,fs.map((f,i)=>candidates(f).some(c=>c.graph?.atoms?.length)?i:-1).filter(i=>i>=0));return retainedCache.get(fs);}
  function adjacentRetained(direction){const ids=retained();return direction>0?(ids.find(i=>i>frameIndex)??frames().length-1):([...ids].reverse().find(i=>i<frameIndex)??0);}
'''
    js = replace_once(js, '  function stop()', helpers+'  function stop()')
    js = replace_once(js, '$("step-slider").max=String(Math.max(0,frames().length-1)); $("step-slider").value=String(Math.max(0,frameIndex)); $("step-slider").disabled=frames().length<2;',
                      'const retainedIds=retained();$("step-slider").max=String(Math.max(0,retainedIds.length-1));$("step-slider").value=String(Math.max(0,retainedIds.findLastIndex(i=>i<=frameIndex)));$("step-slider").disabled=retainedIds.length<2;$("exact-step").value=String(f.step);')
    js = replace_once(js, 'chooseFrame(frameIndex-1);};$("next").onclick=()=>{stop();chooseFrame(frameIndex+1);', 'chooseFrame(adjacentRetained(-1));};$("next").onclick=()=>{stop();chooseFrame(adjacentRetained(1));')
    js = replace_once(js, 'chooseFrame(Number($("step-slider").value));', 'chooseFrame(retained()[Number($("step-slider").value)]);')
    js = replace_once(js, 'chooseFrame(frameIndex+1);if(frameIndex>=frames().length-1)stop();', 'chooseFrame(adjacentRetained(1));if(frameIndex>=frames().length-1)stop();')
    setup = '''  const exactLabel=el('label','Exact update '),exactInput=el('input');exactInput.id='exact-step';exactInput.type='number';exactInput.min='0';exactInput.max='50000';exactInput.step='1';exactInput.style.width='100px';exactLabel.append(exactInput);$('step-presets').before(exactLabel);
  exactInput.onchange=()=>{const wanted=Number(exactInput.value);if(!Number.isFinite(wanted))return;let closest=0;frames().forEach((f,i)=>{if(Math.abs(f.step-wanted)<Math.abs(frames()[closest].step-wanted))closest=i;});stop();chooseFrame(closest);};
  for(const input of [exactInput,$('step-slider')])for(const event of ['focus','pointerdown','keydown'])input.addEventListener(event,stop);
  const savedNote=el('p','Slider, arrows and playback browse retained molecule frames. Plots and Exact update retain every scalar loss; unsaved matrices are not reconstructed.','note');$('step-slider').closest('.scrubber').after(savedNote);
'''
    js = replace_once(js, '  if(runs.length)selectRun(0);', setup+'  if(runs.length)selectRun(0);')
    adapted = text[:original_js.start(1)] + js + text[original_js.end(1):]
    adapted = replace_once(adapted, 'Slot positions stay fixed within each trajectory.', 'Bond-aware diagrams preserve the saved atom identities and connectivity.')
    adapted = adapted.replace('aria-label="Previous saved state"', 'aria-label="Previous retained molecule"').replace('aria-label="Next saved state"', 'aria-label="Next retained molecule"')
    assert adapted[adapted.index(marker)+len(marker):adapted.index('</script>', adapted.index(marker))] == saved
    return adapted.encode('utf8'), {'saved_data_sha256': hashlib.sha256(saved.encode()).hexdigest(),
                                   'saved_data_bytes_unchanged': True, 'layout_job_id': layouts['job_id']}


def adapt_mosaic(raw, edition):
    text = raw.decode('utf8')
    # The same canonical template functions are also used by future reports.
    template = (OWNER / 'templates/report.html').read_text(encoding='utf8')
    begin, end = ' function drawSeries(', ' function clearFrame('
    text = text[:text.index(begin)] + template[template.index(begin):template.index(end)] + text[text.index(end):]
    begin, end = ' async function chooseArm(', ' function navigation('
    text = text[:text.index(begin)] + template[template.index(begin):template.index(end)] + text[text.index(end):]
    text = replace_once(text, 'assetURL(config.manifest,location.href)', 'assetURL(config.manifest,document.baseURI)')
    text = replace_once(text, '<head>', '<head><base href="../../editions/'+edition+'/">')
    return text.encode('utf8')
