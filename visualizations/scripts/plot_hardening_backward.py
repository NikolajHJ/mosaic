"""Plot same-state backward-rule diagnostics without new model evaluations."""
import argparse
import json
from pathlib import Path

from plot_pair import plt, np, read, digest


def plot(audit_path, output):
    audit = read(audit_path)
    rows = audit['rows']
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    colors = {'X':'#246f8b','E_upper':'#b75b35'}
    labels = {'X':'Atoms (6 categories)','E_upper':'Bonds (5 categories)'}
    for field,color in colors.items():
        steps = [row['step'] for row in rows]
        value = lambda key, series=rows:[row['fields'][field].get(key) for row in series]
        axes[0,0].plot(steps,value('inherited_to_passthrough_norm_ratio'),color=color,lw=1.2,label=labels[field])
        axes[0,0].plot(steps,value('scalar_attenuation_estimate'),color=color,ls='--',lw=1,alpha=.6)
        axes[0,1].plot(steps,value('rescaled_to_passthrough_norm_ratio'),color=color,lw=1.2)
        axes[1,0].plot(steps,value('inherited_passthrough_direction_cosine'),color=color,lw=1.2)
        updates = [row for row in rows if not row['final']]
        axes[1,1].plot([row['step'] for row in updates],value('inherited_to_passthrough_simplex_ratio',updates),color=color,lw=1.2)
        axes[1,1].plot([row['step'] for row in updates],value('rescaled_to_passthrough_simplex_ratio',updates),color=color,lw=1.1,ls='--')
    axes[0,0].set(title='Inherited tangent-gradient attenuation',ylabel='Norm / identity-rule norm')
    axes[0,0].text(.03,.06,'Dashes: scalar estimate (1−h)+h/K',transform=axes[0,0].transAxes,fontsize=9,color='#475569')
    axes[0,1].set(title='Scalar compensation at the same state',ylabel='Rescaled norm / identity-rule norm')
    axes[0,1].axhline(1,color='#64748b',lw=.8,alpha=.5)
    axes[1,0].set(title='Inherited / identity-rule gradient direction',ylabel='Cosine',ylim=(-.02,1.05))
    axes[1,1].set(title='Local simplex-step magnitude',ylabel='Step norm / identity-rule step norm')
    axes[1,1].text(.03,.06,'Solid: inherited; dashed: rescaled',transform=axes[1,1].transAxes,fontsize=9,color='#475569')
    axes[1,1].axhline(1,color='#64748b',lw=.8,alpha=.5)
    for axis in axes.flat:
        axis.set_xlabel('Evaluated update')
        axis.grid(alpha=.17)
        axis.spines[['top','right']].set_visible(False)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle('Hardening (5.5): backward rules at the same saved states',x=.07,ha='left',fontsize=16,y=.985)
    source=Path(audit['run']);metrics=audit['source_metrics']
    text=lambda key:'unknown' if metrics.get(key) is None else f"{metrics[key]:.3f}"
    fig.text(.07,.93,f"Source: {source.parent.name}/{source.name}. Graph-valid tail {text('tail_valid_connected_fraction')}; broad-supported tail {text('tail_frozen_supported_valid_fraction')}.",fontsize=9.5,color='#475569')
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=9)
    fig.text(.07,.03,'No new model calls. Inferred tangent gradients use J=(1−h)I+hJsoftmax(p), checked separately against the retained JAX backward code.\n'
             'Local alternatives use the same forward input per state. Chemistry corrections and alternative complete trajectories are not simulated.\n'
             'Only tangent components are identified at full hardness. Final-state virtual steps are omitted from the step plot.\n'
             'Model derivatives, molecular validity, realistic poses and binding remain separate checks.',fontsize=8.5,color='#475569')
    fig.subplots_adjust(left=.09,right=.97,top=.8,bottom=.2,hspace=.43,wspace=.3)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor='white');fig.savefig(output.with_suffix('.svg'),facecolor='white');plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'audit_source':str(audit_path.resolve()),'audit_sha256':digest(audit_path),
        'plot_source_sha256':digest(__file__),'plot_helper_sha256':digest(Path(__file__).with_name('plot_pair.py'))},indent=2)+'\n')
    print(output.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audit',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plot(args.audit,args.out)
