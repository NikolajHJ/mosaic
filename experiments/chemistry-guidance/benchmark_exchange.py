"""Small matched frozen-VAE chemistry-only trajectories; no native scorer/training."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import LatentDesignConfig, structural_target, _model_identity
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.exchange import ExchangeConfig, StructureReference, exchange_guidance, structure_distance
from latentmol.repair_guidance.loss import fixed_target_nll


def labels(graph):
    return tuple(getattr(graph,k)[0].detach().argmax(-1).cpu().numpy() for k in ('atoms','charges','bonds'))


def observe(graph, spec, reference):
    raw = labels(graph)
    result = inspect_graph(*raw,spec)
    active = reference.target.active.numpy()
    ref = reference.labels()
    sulfur = spec.atomic_numbers.index(16)
    return dict(valid=result.smiles is not None, smiles=result.smiles, failure=result.failure_reason,
        hard_sulfur=int(np.count_nonzero(raw[0]==sulfur)), expected_sulfur=float(graph.atoms[0,:,sulfur].detach().sum()),
        atom_exchanges=int(np.count_nonzero(raw[0][active]!=ref[0][active])),
        charge_exchanges=int(np.count_nonzero(raw[1][active]!=ref[1][active])),
        substructure_changes=structure_distance(raw,reference),
        reference_hard_sulfur=int(np.count_nonzero(ref[0]==sulfur)),
        protected_atoms=int(reference.atom_mask.sum()),protected_pairs=int(np.triu(reference.pair_mask,1).sum()),
        element_counts={str(z):int(np.count_nonzero(raw[0]==i)) for i,z in enumerate(spec.atomic_numbers) if i})


def diversity(rows):
    smiles = [r['smiles'] for r in rows if r['valid']]
    parsed = [(s,Chem.MolFromSmiles(s)) for s in smiles]
    molecules = [m for _,m in parsed if m is not None]
    unparseable = [s for s,m in parsed if m is None]
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fingerprints = [fp.GetFingerprint(m) for m in molecules]
    distances = [1-DataStructs.TanimotoSimilarity(fingerprints[i],fingerprints[j])
                 for i in range(len(molecules)) for j in range(i)]
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in molecules]
    return dict(valid=len(smiles), attempts=len(rows), validity=len(smiles)/len(rows), unique_valid=len(set(smiles)),
        unique_fraction=len(set(smiles))/len(smiles) if smiles else None,
        fingerprint_evaluable=len(molecules),unparseable_valid=len(unparseable),
        unparseable_valid_smiles=sorted(set(unparseable)),
        mean_pairwise_morgan_distance=float(np.mean(distances)) if distances else None,
        unique_nonempty_scaffolds=len(set(scaffolds)-{''}), acyclic_valid=scaffolds.count(''))


def run(config, checkpoint, output):
    radii = config.get('perturbation_norms',[config['perturbation_norm']])
    if not radii or any(not np.isfinite(x) or x<0 for x in radii):
        raise ValueError('Require finite nonnegative perturbation radii')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    model,_=load_model(checkpoint,'cpu');model.eval().requires_grad_(False)
    initial_hash=_model_identity(model)
    generator=torch.Generator().manual_seed(config['seed'])
    references=[]; seen=set(); rejected=0
    for draw in range(config['max_reference_draws']):
        if time.monotonic()-started>config['seconds']:raise TimeoutError('Pilot wall budget')
        with torch.no_grad():
            z=torch.randn((1,model.config.latent),generator=generator)
            decoded=model.decode(z)
            active=decoded.atoms[0].argmax(-1)!=0
            if int(active.sum())<3:continue
            graph=graph_from_decoder(decoded,model.spec,temperature=config['temperature'],fixed_active=active)
            raw=labels(graph); checked=inspect_graph(*raw,model.spec)
        if checked.smiles is None or checked.smiles in seen:continue
        try:ref=StructureReference.create(raw,model.spec)
        except ValueError:
            rejected+=1;continue
        seen.add(checked.smiles)
        radius = radii[len(references)%len(radii)]
        noise=torch.randn(z.shape,generator=generator); noise*=radius/noise.norm()
        references.append((z,z+noise,active,ref))
        if len(references)==config['count']:break
    if len(references)!=config['count']:raise ValueError('Insufficient distinct supported valid references')
    torch.save([dict(reference_z=z,initial_z=p,active=a,labels=[torch.tensor(x) for x in ref.labels()],
        motifs=ref.motifs) for z,p,a,ref in references],output/'starts.pt')
    metadata=dict(config=config,checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
        job_id=os.environ.get('LSB_JOBID'),scope='chemistry-only latent recovery from perturbed valid prior samples; no binding or training',
        reference_draws=draw+1,reference_label_rejections=rejected,graph_spec=model.spec.to_dict())
    (output/'run.json').write_text(json.dumps(metadata,indent=2))
    baseline=[];reference_rows=[];results={}
    with torch.no_grad():
        for original,initial,active,ref in references:
            reference_rows.append(observe(graph_from_decoder(model.decode(original),model.spec,
                temperature=config['temperature'],fixed_active=active),model.spec,ref))
            baseline.append(observe(graph_from_decoder(model.decode(initial),model.spec,
                temperature=config['temperature'],fixed_active=active),model.spec,ref))
    legacy=LatentDesignConfig(repair_search=config['search'])
    with (output/'trajectory.jsonl').open('x') as log:
        for arm,settings in config['arms'].items():
            policy=None if settings is None else ExchangeConfig(**config['search'],**settings)
            endrows=[]; allrows=[]; checks=success=censored=invalid=0; norms=[]
            target_edits=dict(atom=0,charge=0,bond=0)
            for index,(original,initial,active,ref) in enumerate(references):
                point=initial.clone()
                for step in range(config['updates']+1):
                    if time.monotonic()-started>config['seconds']:raise TimeoutError('Pilot wall budget')
                    point=point.detach().requires_grad_(True)
                    decoded=model.decode(point)
                    if any(not bool(getattr(decoded,k).isfinite().all()) for k in ('atoms','charges','bonds')):
                        raise ValueError('Nonfinite decoder')
                    graph=graph_from_decoder(decoded,model.spec,temperature=config['temperature'],fixed_active=active)
                    row=observe(graph,model.spec,ref);row.update(arm=arm,index=index,step=step,
                        perturbation_norm=float((initial-original).norm()))
                    if step==config['updates']:
                        endrows.append(row);log.write(json.dumps(row)+'\n');log.flush();break
                    if policy is None:
                        target,report=structural_target(graph,model.spec,active,legacy,step)
                        loss=sum(getattr(decoded,k).sum()*0 for k in ('atoms','charges','bonds'))
                        detail={}
                        if target is not None:
                            loss,detail=fixed_target_nll(decoded,graph,target,model.spec,temperature=config['temperature'],normalization='mean_per_decision')
                    else:
                        loss,detail,target,report=exchange_guidance(decoded,graph,model.spec,ref,policy,temperature=config['temperature'],step=step)
                    gradient,=torch.autograd.grad(loss,point,allow_unused=True)
                    if gradient is None or not bool(gradient.isfinite().all()) or not bool(loss.isfinite()):
                        raise ValueError('Missing/nonfinite latent path')
                    norm=float(gradient.norm());norms.append(norm)
                    delta=-config['learning_rate']*gradient
                    delta*=min(1.,config['max_step_norm']/max(float(delta.norm()),1e-30))
                    point=(point+delta).detach()
                    checks+=report['checks'];success+=int(report['status']=='witness_found')
                    censored+=int(report['status']=='censored');invalid+=int(not row['valid'])
                    edits=report.get('weighted_witness',{}).get('edit_counts',{}) if report.get('weighted_witness') else {}
                    for key,value in edits.items():target_edits[key]+=value
                    row.update(loss=float(loss.detach()),gradient_norm=norm,repair=report['status'],detail=detail,
                        target_edits=edits,checked_operations=report.get('checked_operations'))
                    allrows.append(row);log.write(json.dumps(row)+'\n')
                print(json.dumps(dict(arm=arm,index=index,endpoint=endrows[-1],elapsed=time.monotonic()-started)),flush=True)
            summary=diversity(endrows)
            summary.update(mean_hard_sulfur=float(np.mean([r['hard_sulfur'] for r in endrows])),
                mean_expected_sulfur=float(np.mean([r['expected_sulfur'] for r in endrows])),
                hard_sulfur_drift_from_start=float(np.mean([r['hard_sulfur']-s['hard_sulfur'] for r,s in zip(endrows,baseline)])),
                expected_sulfur_drift_from_start=float(np.mean([r['expected_sulfur']-s['expected_sulfur'] for r,s in zip(endrows,baseline)])),
                mean_atom_exchanges_from_reference=float(np.mean([r['atom_exchanges'] for r in endrows])),
                mean_charge_exchanges_from_reference=float(np.mean([r['charge_exchanges'] for r in endrows])),
                mean_substructure_changes=float(np.mean([r['substructure_changes'] for r in endrows])),
                recovered_initially_invalid=sum(r['valid'] and not s['valid'] for r,s in zip(endrows,baseline)),
                changed_valid_reference_identity=sum(r['valid'] and r['smiles']!=s['smiles'] for r,s in zip(endrows,reference_rows)),
                mean_element_counts={k:float(np.mean([r['element_counts'][k] for r in endrows])) for k in endrows[0]['element_counts']},
                trajectory_validity=float(np.mean([r['valid'] for r in allrows])),repair_successes=success,
                censored=censored,invalid_attempts=invalid,checks=checks,
                target_edits=target_edits if policy is not None else None,
                nonzero_gradient_steps=sum(n>0 for n in norms),endpoints=endrows)
            results[arm]=summary
            (output/'partial.json').write_text(json.dumps(results,indent=2))
    if _model_identity(model)!=initial_hash or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen decoder modified')
    result=dict(metadata,baseline=diversity(baseline),reference=diversity(reference_rows),arms=results,
        elapsed_seconds=time.monotonic()-started,frozen_decoder_unchanged=True,
        warnings=['Short chemistry-only trajectories cannot establish binding quality or rule out long-run sulfur collapse.',
                  'Element drift uses ALL outputs; fingerprint/scaffold diversity uses valid endpoints only.',
                  'Bond-only uses existing structural proposals; mixed arms use bounded beam. Their search spaces and actual checks differ.'])
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();torch.set_num_threads(4)
    config=json.loads(args.config.read_text())
    if args.seed is not None:config['seed']=args.seed
    run(config,args.checkpoint,args.output)
