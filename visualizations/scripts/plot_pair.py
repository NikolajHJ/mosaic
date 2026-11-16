"""Export a standalone figure for an explicitly matched pair of saved runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

cache=Path(tempfile.gettempdir())/'mosaic-figure-cache'
cache.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(cache))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(directory):
    trajectory=directory/'trajectory'
    trace=[json.loads(line) for line in (trajectory/'trace.jsonl').read_text().splitlines()]
    by_step={row['evaluated_step']:row for row in trace}
    rows=[];first_smiles=None;bridges=0
    for row in trace:
        ro=read(trajectory/row['readout']) if row.get('readout') else {}
        if row is trace[0]:first_smiles=ro.get('canonical_smiles')
        operators=row.get('post_step_operators') or {}
        extra=operators.get('experimental') or {}
        if row.get('post_step_operators'):
            bridges+=len((extra.get('reconnect') or {}).get('bridges',[]))+len((operators.get('connectivity') or {}).get('bridges',[]))
        rows.append(dict(step=row['evaluated_step'],loss=row['loss'],
                         valid_connected=bool(ro.get('sanitized') and ro.get('fragment_count')==1) if ro else None,
                         cumulative_bridges=bridges))
    hard=[]
    for point in read(directory/'soft_hard_gap.json'):
        row=by_step[point['step']]
        ro=read(trajectory/row['readout']) if row.get('readout') else {}
        if ro.get('sanitized') and ro.get('fragment_count')==1 and ro.get('canonical_smiles')!=first_smiles:
            hard.append(dict(step=point['step'],value=point['hard_loss']))
    return rows,hard


def plot(control, treatment, output, title, labels):
    directories=[control,treatment]
    configs=[read(p/'stage2_config.json') for p in directories]
    if configs[0].get('backend') != 'boltz2' or not configs[0].get('context_sha256'):
        raise ValueError('This figure requires an identified fixed Boltz context')
    for key in ('backend','context_sha256','eta_X','eta_E','score_metric','optimization_objective'):
        if configs[0].get(key)!=configs[1].get(key):raise ValueError(f'Pair has different {key}; quantitative losses must be comparable')
    if configs[0].get('optimization_objective','affinity_pred_value')!='affinity_pred_value':raise ValueError('This plot is scoped to quantitative-affinity trajectories')
    checkpoint=[c.get('worker_identity',{}).get('checkpoint_sha256') for c in configs]
    if not checkpoint[0] or checkpoint[0]!=checkpoint[1]:raise ValueError('Pair checkpoint identities differ or are missing')
    first_arrays=[]
    for directory in directories:
        entry=json.loads((directory/'trajectory/trace.jsonl').read_text().splitlines()[0])
        with np.load(directory/'trajectory'/entry['evaluated_arrays'],allow_pickle=False) as z:first_arrays.append({k:z[k].copy() for k in ('X','E_upper')})
    if any(not np.array_equal(first_arrays[0][k],first_arrays[1][k]) for k in ('X','E_upper')):raise ValueError('Pair initial probability arrays differ')
    colors=['#ae4b40','#1b7c70']
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),sharex=True)
    identities=[]
    for directory,label,color in zip(directories,labels,colors):
        config=read(directory/'stage2_config.json');rows,hard=load(directory)
        steps=[r['step'] for r in rows]
        axes[0,0].step(steps,[float(r['valid_connected']) if r['valid_connected'] is not None else float('nan') for r in rows],where='post',label=label,color=color,lw=1.8)
        axes[0,1].plot(steps,[r['loss'] for r in rows],label=label,color=color,lw=1.35)
        axes[1,0].scatter([p['step'] for p in hard],[p['value'] for p in hard],label=label,color=color,s=22,alpha=.85)
        axes[1,1].plot(steps,[r['cumulative_bridges'] for r in rows],label=label,color=color,lw=1.8)
        identities.append(dict(source=str(directory.resolve()),trace_sha256=digest(directory/'trajectory/trace.jsonl'),
                               config_sha256=digest(directory/'stage2_config.json'),hard_scores_sha256=digest(directory/'soft_hard_gap.json')))
    axes[0,0].set(title='Raw graph sanitizes and is connected',yticks=[0,1],yticklabels=['No','Yes'],ylim=(-.08,1.08))
    axes[0,1].set(title='Recorded proposal loss',ylabel='Fixed-pose quantitative loss')
    axes[1,0].set(title='Scored changed, connected valid readouts',ylabel='Hard fixed-pose quantitative loss')
    axes[1,1].set(title='Connectivity (5.3) intervention events',ylabel='Recorded bridge events, cumulative')
    for ax in axes.flat:
        ax.grid(alpha=.17);ax.spines[['right','top']].set_visible(False)
        ax.set_xlim(left=0);ax.tick_params(labelsize=10)
    for ax in axes[1]:ax.set_xlabel('Update')
    axes[0,0].legend(loc='center right',frameon=False,fontsize=10)
    fig.suptitle(title,fontsize=17,x=.07,ha='left')
    fig.text(.07,.92,f"Same initial probabilities and frozen model context; {configs[0]['steps']} / {configs[1]['steps']} updates; atom/bond rates {configs[0]['eta_X']} / {configs[0]['eta_E']}.",fontsize=11,color='#4b5563')
    fig.text(.07,.035,'Hard scores show only explicitly evaluated checkpoints. Bridge events can repeat the same edge.\nStates are correlated, not independent samples. These fixed-pose scores and broad geometry checks do not establish binding.',fontsize=9,color='#4b5563')
    fig.subplots_adjust(left=.075,right=.97,top=.85,bottom=.15,hspace=.36,wspace=.27)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor='white')
    fig.savefig(output.with_suffix('.svg'),facecolor='white')
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(sources=identities,script_sha256=digest(__file__),matplotlib_version=matplotlib.__version__,title=title),indent=2)+'\n')
    print(output.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('control',type=Path);p.add_argument('treatment',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--title',default='Coupled valence projection (5.2) and connectivity (5.3)')
    p.add_argument('--control-label');p.add_argument('--treatment-label')
    a=p.parse_args();plot(a.control,a.treatment,a.out,a.title,[a.control_label or a.control.name,a.treatment_label or a.treatment.name])
