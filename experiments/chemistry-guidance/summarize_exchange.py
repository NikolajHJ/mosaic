"""Aggregate saved matched exchange pilots, preserving seed and failure denominators."""
import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import rdBase

from benchmark_exchange import diversity


def summarize_native(folders):
    """First-50 and full-500 native readouts; never treat missing runs as passes."""
    incomplete=[str(folder) for folder in folders if not (folder/'comparison.json').exists()]
    if incomplete:return dict(status='incomplete',missing_comparisons=incomplete)
    runs=[json.loads((folder/'comparison.json').read_text()) for folder in folders]
    config=runs[0]['config']
    if any(r['status']!='completed' or r['config']!=config for r in runs):
        raise ValueError('Unmatched/incomplete native cohorts')
    if len({r['seed'] for r in runs})!=len(runs):raise ValueError('Duplicate native cohort')
    completed_seeds=[r['seed'] for r in runs]
    expected_seeds=config.get('seeds',completed_seeds)
    if not set(completed_seeds).issubset(expected_seeds):raise ValueError('Unexpected native cohort seed')
    missing_seeds=[seed for seed in expected_seeds if seed not in completed_seeds]
    arm_data={arm:[] for arm in config['arms']}
    identities=[]
    for folder,run in zip(folders,runs):
        for arm in config['arms']:
            complete=json.loads((folder/arm/'result.json').read_text())
            if complete['status']!='completed' or not all(complete['technical_checks'][k] for k in
                    ('finite_required_binding_path','finite_chemistry_path','frozen_decoder_unchanged','frozen_native_close_completed')):
                raise ValueError('Native/frozen checks did not complete')
            metadata=json.loads((folder/arm/'run.json').read_text())
            identities.append(metadata['native_identity'])
            records=folder/arm/'arms'/'latent'
            read=lambda path:[json.loads(line) for line in path.read_text().splitlines()]
            data=dict(seed=run['seed'],candidates=read(records/'candidates.jsonl'),metrics=read(records/'metrics.jsonl'),
                updates=read(records/'updates.jsonl'),observations=json.loads((folder/arm/'chemistry-observations.json').read_text()),
                hard=json.loads((folder/arm/'hard-readouts.json').read_text()))
            for key in ('candidates','metrics','observations'):
                if [r['step'] for r in data[key]]!=list(range(config['design']['updates']+1)):
                    raise ValueError('Missing/duplicate native trajectory states')
            arm_data[arm].append(data)
    if any(identity!=identities[0] for identity in identities):raise ValueError('Native objective/context identities differ')
    stages={}
    for steps in (50,config['design']['updates']):
        stage={}
        for arm,cohorts in arm_data.items():
            endpoints=[d['candidates'][steps] for d in cohorts]
            pool={row['smiles']:row for d in cohorts for row in d['candidates'][:steps+1] if row['valid']}
            # Limit fingerprint pair work, retaining the exact distinct count.
            unique=list(pool.values());sample=unique
            if len(sample)>1000:
                ids=np.random.default_rng(111211).choice(len(sample),1000,replace=False)
                sample=[sample[i] for i in ids]
            per_seed=[]
            for d in cohorts:
                start=d['observations'][0];end=d['observations'][steps]
                per_seed.append(dict(seed=d['seed'],final_valid=d['candidates'][steps]['valid'],
                    initial_relaxed_loss=d['metrics'][0]['score']['binding_loss'],
                    final_relaxed_loss=d['metrics'][steps]['score']['binding_loss'],
                    relaxed_loss_delta=d['metrics'][steps]['score']['binding_loss']-d['metrics'][0]['score']['binding_loss'],
                    hard_sulfur_delta=end['hard_sulfur']-start['hard_sulfur'],
                    expected_sulfur_delta=end['expected_sulfur']-start['expected_sulfur'],
                    max_hard_sulfur_increase=max(o['hard_sulfur']-start['hard_sulfur'] for o in d['observations'][:steps+1]),
                    final_atom_exchanges=end['atom_exchanges'],final_substructure_changes=end['substructure_changes'],
                    bound_hits=sum(u['bound_factor']<1 for u in d['updates'][:steps]),
                    unique_valid=len({r['smiles'] for r in d['candidates'][:steps+1] if r['valid']})))
            stage[arm]=dict(endpoint_diversity=diversity(endpoints),
                valid_source_fraction=sum(r['valid'] for d in cohorts for r in d['candidates'][:steps])/(steps*len(cohorts)),
                unique_valid_trajectory=len(pool),trajectory_diversity=diversity(sample) if sample else None,
                trajectory_diversity_sampled=len(sample)<len(unique),per_seed=per_seed)
        stages[str(steps)]=stage
    hard={}
    for arm,cohorts in arm_data.items():
        hard[arm]=[]
        for d in cohorts:
            first=next(r for r in d['hard'] if r['role']=='initial')
            for r in d['hard']:
                hard[arm].append(dict(r,seed=d['seed'],loss_delta_from_initial=
                    r['loss']-first['loss'] if r['loss'] is not None and first['loss'] is not None else None))
    return dict(status='partial' if missing_seeds else 'completed',config=config,seeds=completed_seeds,
        expected_seeds=expected_seeds,missing_seeds=missing_seeds,
        rdkit_summary_version=rdBase.rdkitVersion,
        starts=[r['start'] for r in runs],unique_start_identities=len({r['start']['smiles'] for r in runs}),
        native_identity=identities[0],stages=stages,hard_readouts=hard,
        limitations=['Native Protenix contact surrogate, not affinity.',
                    'Hard readouts retain fixed geometry-off context; not fresh pose predictions.',
                    f'{len(runs)} completed independent start(s) of {len(expected_seeds)} planned; within-trajectory observations are correlated.',
                    'Fingerprint/scaffold diversity uses parseable saved valid SMILES; parse failures are reported without changing recorded validity denominators. Trajectory fingerprints use unique identities.'])


def summarize(folders):
    runs=[json.loads((folder/'result.json').read_text()) for folder in folders]
    trajectories=[[json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()] for folder in folders]
    seeds=[r['config']['seed'] for r in runs]
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate seed')
    config={k:v for k,v in runs[0]['config'].items() if k!='seed'}
    for r in runs:
        if {k:v for k,v in r['config'].items() if k!='seed'}!=config:raise ValueError('Unmatched configs')
        if r['checkpoint_sha256']!=runs[0]['checkpoint_sha256'] or not r['frozen_decoder_unchanged']:
            raise ValueError('Checkpoint/frozen-state mismatch')
    count=config['count'];updates=config['updates'];total=count*len(runs)
    initial=[];reference_smiles=[]
    for rows in trajectories:
        baseline=sorted([x for x in rows if x['arm']=='bond_only' and x['step']==0],key=lambda x:x['index'])
        if len(baseline)!=count:raise ValueError('Incomplete initial cohort')
        initial.extend(baseline)
    # Bootstrap matched starts within each seed. These are exploratory intervals,
    # conditional on the selected valid-reference distribution, not native claims.
    rng=np.random.default_rng(111130)
    draws=np.concatenate([rng.integers(count,size=(2000,count))+i*count for i in range(len(runs))],axis=1)
    def interval(values):
        values=np.asarray(values,dtype=float)
        return dict(mean=float(values.mean()),paired_start_bootstrap_95=np.quantile(values[draws].mean(1),[.025,.975]).tolist())
    endpoints={arm:[x for r in runs for x in sorted(r['arms'][arm]['endpoints'],key=lambda x:x['index'])] for arm in config['arms']}
    base=endpoints['bond_only'];summary={}
    for arm,ends in endpoints.items():
        if len(ends)!=total:raise ValueError('Incomplete endpoints')
        allrows=[x for rows in trajectories for x in rows if x['arm']==arm]
        if len(allrows)!=total*(updates+1):raise ValueError('Incomplete trajectory')
        common=[i for i,(a,b) in enumerate(zip(ends,base)) if a['valid'] and b['valid']]
        sulfur_deltas=[e['hard_sulfur']-s['hard_sulfur'] for e,s in zip(ends,initial)]
        summary[arm]=dict(diversity=diversity(ends),
            per_seed_valid={str(seed):r['arms'][arm]['valid'] for seed,r in zip(seeds,runs)},
            final_validity_difference_vs_bond=interval([int(e['valid'])-int(b['valid']) for e,b in zip(ends,base)]),
            substructure_difference_vs_bond=interval([e['substructure_changes']-b['substructure_changes'] for e,b in zip(ends,base)]),
            mean_atom_exchanges=float(np.mean([e['atom_exchanges'] for e in ends])),
            mean_substructure_changes=float(np.mean([e['substructure_changes'] for e in ends])),
            expected_sulfur_drift=interval([e['expected_sulfur']-s['expected_sulfur'] for e,s in zip(ends,initial)]),
            hard_sulfur_drift=interval(sulfur_deltas),sulfur_increase_starts=sum(x>0 for x in sulfur_deltas),
            max_individual_sulfur_increase=max(sulfur_deltas),min_individual_sulfur_change=min(sulfur_deltas),
            mean_sulfur_error_vs_reference=float(np.mean([abs(e['hard_sulfur']-e['reference_hard_sulfur']) for e in ends])),
            initially_invalid=sum(not s['valid'] for s in initial),
            recovered=sum(e['valid'] and not s['valid'] for e,s in zip(ends,initial)),
            broke_initially_valid=sum(not e['valid'] and s['valid'] for e,s in zip(ends,initial)),
            trajectory_validity=float(np.mean([x['valid'] for x in allrows if x['step']<updates])),
            common_valid_count=len(common),
            common_valid_diversity=diversity([ends[i] for i in common]) if common else None,
            common_valid_baseline_diversity=diversity([base[i] for i in common]) if common else None,
            checks=sum(r['arms'][arm]['checks'] for r in runs),censored=sum(r['arms'][arm]['censored'] for r in runs),
            invalid_search_attempts=sum(r['arms'][arm]['invalid_attempts'] for r in runs),
            target_edits=None if arm=='bond_only' else {k:sum(r['arms'][arm]['target_edits'][k] for r in runs) for k in ('atom','charge','bond')},
            perturbation_strata={str(radius):diversity([x for x in ends if abs(x['perturbation_norm']-radius)<1e-4]) for radius in config.get('perturbation_norms',[config['perturbation_norm']])})
    return dict(seeds=seeds,starts=total,updates_per_arm=total*updates,config=config,
        checkpoint_sha256=runs[0]['checkpoint_sha256'],rdkit_summary_version=rdBase.rdkitVersion,
        baseline=diversity(initial),arms=summary,
        summed_run_seconds=sum(r['elapsed_seconds'] for r in runs),
        warnings=['Chemistry-only recovery, no native binding objective or training.',
                  'Paired intervals resample starts within each seed; tiny sample and multiple exploratory comparisons.',
                  'Pooled fingerprint/scaffold diversity excludes invalid endpoints; common-valid comparisons supplied.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folders',type=Path,nargs='+')
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize_native(args.folders) if args.native else summarize(args.folders)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
