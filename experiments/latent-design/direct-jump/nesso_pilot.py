"""Paired direct-graph Nesso-1 pilot with fresh, native BRD4 affinity readouts."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from repair_search import finite_contact, project, update
from repair_signal import canonical_valid_smiles, evaluate_repair
from study import hard_smiles, write


def timing(samples):
    return dict(count=len(samples), mean_s=float(np.mean(samples)), median_s=float(np.median(samples)),
                p90_s=float(np.percentile(samples, 90))) if samples else None


def optimize(backend, initial, founder, cfg, direction, folder):
    folder.mkdir()
    graph = initial
    observed, statuses, grad_times, update_times = {}, Counter(), [], []
    with (folder / 'trajectory.jsonl').open('x', buffering=1) as stream:
        for step in range(cfg['updates'] + 1):
            began = perf_counter()
            value, grad = finite_contact(backend, graph)
            grad_seconds = perf_counter() - began
            repair, da, db, info = evaluate_repair(*graph)
            if step < cfg['updates']:
                statuses[info['status']] += 1
            smi = info['smiles']
            if smi and smi != founder and (smi not in observed or value < observed[smi]['relaxed_affinity']):
                observed[smi] = dict(smiles=smi, step=step, relaxed_affinity=value)
            row = dict(step=step, smiles=smi, relaxed_affinity=value, repair_loss=repair,
                       repair_status=info['status'], model_value_and_gradient_s=grad_seconds,
                       atom_gradient_norm=float(np.linalg.norm(grad[0])), bond_gradient_norm=float(np.linalg.norm(grad[1])))
            if step < cfg['updates']:
                combined = tuple(direction * g + cfg['repair_weight'] * r for g, r in zip(grad, (da, db)))
                graph, row['update'] = update(graph, combined, cfg)
                row['update_wall_s'] = perf_counter() - began
                if step:
                    grad_times.append(grad_seconds)
                    update_times.append(row['update_wall_s'])
                else:
                    start_value = value
                    first_step_seconds = row['update_wall_s']
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    # Use the same molecular selection rule and allowance in the two arms.
    ranked = sorted(observed.values(), key=lambda r: (r['relaxed_affinity'], r['smiles']))
    result = dict(start_relaxed=start_value, endpoint_relaxed=value, endpoint_smiles=smi,
                  distinct_new_valid=len(ranked), valid_update_fraction=statuses['already_valid']/cfg['updates'],
                  repair_status_counts=dict(statuses), first_update_s=first_step_seconds,
                  warm_model_value_and_gradient=timing(grad_times), warm_update=timing(update_times),
                  fresh_shortlist=ranked[:cfg['fresh_new_per_arm']])
    write(folder / 'optimization.json', result)
    return result


def native_readout(model, features):
    import torch
    torch.cuda.synchronize()
    began = perf_counter()
    with torch.inference_mode():
        prediction = model.predict_step(features, batch_idx=0)
    if prediction.get('exception'):
        raise RuntimeError('Native Nesso-1 inference failed; see model log')
    row = {k: float(prediction[k].item()) for k in
           ('affinity_pred_value', 'affinity_probability_binary', 'entropy_crop_pl')}
    if not all(math.isfinite(x) for x in row.values()):
        raise FloatingPointError('Nonfinite native Nesso-1 readout')
    row['native_forward_s'] = perf_counter() - began
    row['has_interface'] = row['entropy_crop_pl'] > 0
    return row


def main():
    from nesso.model.models.nesso1 import Nesso1
    from nesso_inputs import ConformerUnavailable, NessoInputs, prepare_esm
    from nesso_native import NessoBackend, NoProteinInCrop
    from latentmol.multimodel.external import tensor_tree_sha256
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(Path(__file__).with_name('nesso-pilot.json').read_bytes())
    founder = cfg['founders'][str(args.seed)]
    if not os.environ.get('LSB_JOBID', '').isdigit() or torch.cuda.device_count() != 1:
        raise RuntimeError('This pilot requires one allocated visible GPU')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    began = perf_counter()
    prepare_esm(args.output/'esm', model_name=str(args.assets/'esm'), device='cuda')
    inputs = NessoInputs(args.output/'features', args.assets/'ccd.pkl', args.output/'esm', device='cuda')
    model = Nesso1.from_pretrained(str(args.assets/'checkpoint')).eval().requires_grad_(False).to('cuda')
    model.use_kernels = False
    model.predict_args = dict(model.predict_args or {}, recycling_steps=cfg['hard_recycles'],
                              refine_protein_inference=True, refine_protein_tokens_budget=256,
                              refine_protein_cutoff=22., affinity_protein_cutoff=15., pose_protein_cutoff=15.,
                              save_metadata=False)
    before = tensor_tree_sha256(model.state_dict())
    features, slots, atom, bond = inputs.featurize(founder)
    if canonical_valid_smiles(hard_smiles((atom, bond))) != founder:
        raise ValueError('Native founder atom/bond rows changed the original molecule')
    founder_readout = native_readout(model, features)
    repeat = [native_readout(model, features)['native_forward_s'] for _ in range(3)]
    init = dict(seed=args.seed, gpu=torch.cuda.get_device_name(), fp32=True, settings=cfg,
                model_digest=before, founder=founder, founder_readout=founder_readout,
                warm_native_five_recycle_forward=timing(repeat), preparation_and_reference_s=perf_counter()-began)
    write(args.output/'founder.json', init)
    if marker := os.environ.get('DBMOL_LAUNCH_READY'):
        write(Path(marker), dict(status='models_and_inputs_ready', job_id=os.environ['LSB_JOBID'],
                                 nonce=os.environ['DBMOL_LAUNCH_NONCE']))
    empty_crops = []
    for recycles in cfg['gradient_recycles_to_try']:
        try:
            backend = NessoBackend(model, features, slots, atom, bond, recycling_steps=recycles)
            break
        except NoProteinInCrop as exc:
            empty_crops.append(dict(recycles=recycles, ligand_tokens=exc.ligand_tokens, affinity=exc.native_score))
    else:
        write(args.output/'result.json', dict(init, status='no_protein_in_founder_gradient_crop', empty_crops=empty_crops))
        return
    bridge_difference = abs(backend.founder_bridge_score - backend.founder_native_score)
    if bridge_difference > 0.001:
        raise ValueError(f'Fixed-context founder and equivalent native affinity differ by {bridge_difference:g}')
    initial = tuple(project((1-m)*x + m/x.shape[1], cfg['probability_floor']) for x,m in zip((atom,bond),(.02,.002)))
    if hard_smiles(initial) != founder:
        raise ValueError('Softened initialization changed founder')
    results, cache = {}, {}
    for name, direction in cfg['arms'].items():
        folder = args.output/name
        arm = optimize(backend, initial, founder, cfg, direction, folder)
        fresh = []
        for hit in arm['fresh_shortlist']:
            smi = hit['smiles']
            if smi not in cache:
                mark = perf_counter()
                try:
                    candidate, _, _, _ = inputs.featurize(smi)
                except ConformerUnavailable as exc:
                    cache[smi] = dict(status='feature_failed', reason=str(exc))
                else:
                    cache[smi] = dict(status='scored', **native_readout(model, candidate))
                cache[smi]['hard_total_s'] = perf_counter()-mark
            row = dict(hit, **cache[smi])
            if row['status'] == 'scored':
                row['delta_vs_founder'] = row['affinity_pred_value'] - founder_readout['affinity_pred_value']
            fresh.append(row)
        improved = [r for r in fresh if r.get('delta_vs_founder', 0) < 0 and r['has_interface'] and founder_readout['has_interface']]
        arm.update(fresh=fresh, improved_with_interface=len(improved), best_improved=min(improved, default=None,
                   key=lambda r: r['affinity_pred_value']))
        write(folder/'result.json', arm)
        results[name] = arm
    audit = backend.close()
    if before != tensor_tree_sha256(model.state_dict()):
        raise RuntimeError('Frozen Nesso-1 weights or buffers changed')
    write(args.output/'result.json', dict(init, status='completed', gradient_recycles=recycles,
          founder_gradient_protocol_native=backend.founder_native_score, founder_bridge=backend.founder_bridge_score,
          gradient_crop_protein_tokens=backend.crop_protein_tokens, empty_crops=empty_crops, arms=results, audit=audit,
          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), full_job_seconds=perf_counter()-began))


if __name__ == '__main__':
    main()
