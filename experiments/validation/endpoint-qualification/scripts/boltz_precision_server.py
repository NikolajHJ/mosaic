"""Same-job native producer/reference server; no optimizer or saved-state writer."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

def require(value,message):
    if not value:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pin',required=True);parser.add_argument('--arm',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--startup-only',action='store_true')
    args=parser.parse_args()
    import boltz_precision_ops as ops
    base,plan=ops.context(args.pin);require(args.arm in plan['arms'],'Undeclared arm')
    expected=ops.ROOT/('cpu/payload/startup-'+args.arm if args.startup_only else 'gpu-'+args.arm+'/payload/native')
    require(args.output==expected and not args.output.exists(),'Exact fresh native output required');args.output.mkdir()
    if args.startup_only:
        require(os.environ.get('LSB_QUEUE')=='hpc','Actual CPU startup required')
        ops.write(args.output/'server-startup.json',{'status':'passed_actual_server_parser_and_provenance',
            'job_id':os.environ['LSB_JOBID'],'arm':args.arm,'model_constructed':False,'updates':0});return
    require(os.environ.get('LSB_QUEUE')=='gpul40s','Actual GPU allocation required')
    science=Path(plan['science']['path']);sys.path.insert(0,str(science/'src'))
    import numpy as np
    from boltz_objective_reference import Evaluator
    precision=ops.module(Path(plan['capacity_science']['path'])/'scripts/dbmol_reference.py','_qualified_original_Torch_FP64_helpers')
    case=plan['arms'][args.arm];original=np.load(case['point']['path'],allow_pickle=False)
    baseline={f:original[f].copy() for f in ('X','E_upper')};original.close()
    with contextlib.redirect_stdout(sys.stderr):evaluation=Evaluator(plan,args.arm=='dbmol',precision,baseline)
    def persist(name,row,values):
        gradients=row.get('gradients',{});path=args.output/(name+'.npz')
        np.savez_compressed(path,**values,value=np.asarray(row['value']),
            **{'gradient_'+f:g for f,g in gradients.items()})
        row={k:v for k,v in row.items() if k!='gradients'}
        if gradients:row['gradients']={f:g.tolist() for f,g in gradients.items()}
        row['raw_point']=ops.binding(path);return row
    before=persist('production-before-reference',evaluation.before,baseline)
    during=persist('production-during-reference',evaluation.during,baseline)
    print(json.dumps({'status':'ready_complete_objective_reference','arm':args.arm,'derivation':evaluation.derivation,
        'production_before':before,'production_during':during,'promotions':evaluation.promotions,
        'frozen':evaluation.verify(),'optimizer_updates':0}),flush=True)
    sequence=0
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('action')=='close':
            frozen,after=evaluation.close(baseline);after=persist('production-after-reference',after,baseline)
            ops.write(args.output/'frozen-and-residency.json',{'status':'passed_exact_production_before_during_after_reference',
                'production_before':before,'production_during':during,'production_after':after,'frozen':frozen,
                'optimizer_updates':0,'original_results_changed':False})
            print(json.dumps({'status':'closed','requests':sequence,'frozen':frozen,'production_after':after}),flush=True);return
        require(request.get('id')==sequence+1 and type(request.get('reference')) is bool and type(request.get('gradient')) is bool,
                'Exact sequenced typed request required')
        sequence+=1;dtype=np.float64 if request['reference'] else np.float32
        values={f:np.asarray(request[f],dtype=dtype) for f in ('X','E_upper')}
        require([v.shape for v in values.values()]==[(25,6),(300,5)] and all(np.isfinite(v).all() for v in values.values()),'Malformed graph')
        require(hashlib.sha256(b''.join(values[f].tobytes() for f in ('X','E_upper'))).hexdigest()==request['graph_sha256'],
                'RPC point bytes changed')
        objective=request['objective'];require(objective in (('dbmol_affinity_loss','dbmol_contact_loss','dbmol_confidence_loss','dbmol_total_loss')
            if args.arm=='dbmol' else ('dbmol_affinity_loss',)),'Undeclared objective')
        row=evaluation.measure(values,objective,request['gradient'],request['reference'])
        row=persist('point-'+str(sequence).zfill(4),row,values);row.update(id=sequence,graph_sha256=request['graph_sha256'])
        ops.budget(plan)
        print(json.dumps(row,allow_nan=False),flush=True)
    raise ValueError('Native reference exited without frozen close handshake')

if __name__=='__main__':main()
