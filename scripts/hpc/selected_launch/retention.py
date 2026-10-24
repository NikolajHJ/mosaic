"""Stride20 persistence only; original evaluations and update callables run intact."""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
from pathlib import Path
import re
import runpy
import shutil
import sys
import textwrap

STRIDE = 20
VERSION = 'mosaic-loss-every-step-matrices-every20-v1'


def retained(step, total):
    if type(step) is not int or type(total) is not int or not 0 <= step <= total:
        raise ValueError('Invalid retention step/budget')
    return step == 0 or step == total or step % STRIDE == 0


def policy():
    return {'version': VERSION, 'matrix_stride': STRIDE, 'loss_stride': 1,
            'initial_terminal_and_segment_boundaries': True,
            'rolling_latest_and_best_checkpoints': True,
            'interpolation_or_reconstruction': False, 'optimization_arithmetic_changed': False}


def full_projection(unit, total, segments):
    if any(type(x) is not int or x < 1 for x in (unit,total,segments)) or total % segments:
        raise ValueError('Invalid sparse projection')
    length=total//segments
    # Each segment keeps initial, rolling latest and best in addition to every20.
    # Trials/references fit inside the same empirical triple-file bound.
    bundles=segments*((length+STRIDE-1)//STRIDE+3)
    return 2*unit*bundles+segments*1024**3


def thin_readout(row):
    row['readout']={k:v for k,v in row.get('readout',{}).items()
                    if k not in ('element_indices','bond_upper_indices','pairs','elements')}
    row['readout_retained']=False


def rolling_snapshot(original):
    """Derive only a filename change from the authenticated original writer."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0];function.name='_stride20_rolling_snapshot'
    function.decorator_list=[]
    changed=0
    for n in function.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='path':
            n.value=ast.parse("directory / 'latest-state.npz'",mode='eval').body;changed+=1
    if changed!=1:raise ValueError('Original snapshot writer shape changed')
    ast.fix_missing_locations(tree)
    namespace={}
    exec(compile(tree,original.__code__.co_filename+'[stride20-filename-only]','exec'),original.__globals__,namespace)
    return namespace[function.name]


@contextmanager
def trajectory_writes(module, directory, total):
    """Change file destinations/metadata only, never proposal/evaluation arithmetic."""
    directory=Path(directory)
    old_snapshot,old_trials,old_write=module._snapshot,module._save_trials,module.write_json
    rolling=rolling_snapshot(old_snapshot)
    best=None
    def snapshot(root,step,evaluation,**kwargs):
        nonlocal best
        if Path(root)!=directory:raise ValueError('Sparse writer escaped declared trajectory')
        keep=retained(step,total)
        row=(old_snapshot if keep else rolling)(root,step,evaluation,**kwargs)
        source=directory/row['snapshot']
        loss=row[evaluation.native_name+'_loss']
        if best is None or loss<best['loss']:
            target=directory/'best-observed.npz';tmp=directory/'best-observed.partial'
            shutil.copyfile(source,tmp);tmp.replace(target)
            best={'step':step,'loss':loss,'snapshot':target.name,'snapshot_sha256':module.file_sha256(target)}
            old_write(directory/'best-observed.json',best)
        if not keep:
            old_write(directory/'latest-state.json',{**row,'retention_policy':policy()})
            row['snapshot']=row['snapshot_sha256']=None
            thin_readout(row)
        row['snapshot_retained']=keep
        return row
    def trials(path,evaluations):
        path=Path(path);match=re.fullmatch(r'step-(\d+)\.npz',path.name)
        if not match or path.parent!=directory/'trials':raise ValueError('Unexpected trial persistence path')
        if not retained(int(match[1])+1,total):path=path.with_name('latest-trial.npz')
        return old_trials(path,evaluations)
    def write(path,value):
        path=Path(path)
        if path.parent==directory/'references':
            match=re.fullmatch(r'step-(\d+)\.json',path.name)
            if not match:raise ValueError('Unexpected reference persistence path')
            if not retained(int(match[1]),total):path=path.with_name('latest-reference.json')
        if path==directory/'report.json':
            value['artifact_retention']=policy()
            if best is not None:
                key='best_observed_'+next((k.removeprefix('best_observed_') for k in value if k.startswith('best_observed_')), '')
                if key in value:value[key].update(best)
        return old_write(path,value)
    module._snapshot,module._save_trials,module.write_json=snapshot,trials,write
    try:yield
    finally:module._snapshot,module._save_trials,module.write_json=old_snapshot,old_trials,old_write


def verify_case(case, directory, *, require_complete=True):
    directory=Path(directory);total=case['requested_updates']
    if case.get('artifact_retention')!=policy():raise ValueError('Missing exact stride20 policy')
    states=case['states'];updates=case['updates']
    if require_complete and (len(states)!=total+1 or len(updates)!=total):raise ValueError('Every scalar state/update is mandatory')
    selected=[]
    for index,row in enumerate(states):
        if row['step']!=index or row.get('snapshot_retained')!=retained(index,total):raise ValueError('Wrong matrix stride/boundary')
        if retained(index,total):
            path=directory/row['snapshot']
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):raise ValueError('Missing/escaped retained frame')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['snapshot_sha256']:raise ValueError('Retained frame bytes changed')
            selected.append({'step':index,'global_step':row.get('global_step',index),'snapshot':row['snapshot'],'snapshot_sha256':row['snapshot_sha256']})
        elif row['snapshot'] is not None or row['snapshot_sha256'] is not None:raise ValueError('Unsaved frame must have no invented reference')
    return {'policy':policy(),'scalar_states':len(states),'scalar_updates':len(updates),'retained_frames':selected}


def install_trajectory(module, output):
    original=module.run_chemical_updates
    def run(*args,**kwargs):
        directory=Path(kwargs['directory'])
        if directory!=Path(output)/'case-0':return original(*args,**kwargs)
        with trajectory_writes(module,directory,kwargs['steps']):result=original(*args,**kwargs)
        index=verify_case(result,directory)
        module.write_json(directory/'retained-frames.json',index)
        return result
    module.run_chemical_updates=run
    return original


def adapt_certifier(op, profile):
    """Keep the entire original acceptance function; skip only nonexistent frames."""
    original=op.certify_segment
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));fn=tree.body[0]
    found=0
    for node in ast.walk(fn):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='state' and ast.dump(node.iter)==ast.dump(ast.parse("case['states']",mode='eval').body):
            node.body.insert(0,ast.parse("if not state['snapshot_retained']:\n    continue").body[0]);found+=1
    if found!=1:raise ValueError('Original acceptance snapshot loop differs')
    ast.fix_missing_locations(tree);namespace={}
    exec(compile(tree,original.__code__.co_filename+'[stride20-persistence-only]','exec'),original.__globals__,namespace)
    compiled=namespace[fn.name]
    def certify(*args,**kwargs):
        # Producer is selected by original arm/index decoding, never user path input.
        if profile == 'dbmol10k':
            parent,index,manifest=args;producer=op.ROOT/op.label('segment',index)
        else:
            arm,index,manifest=args[:3];producer=op.ROOT/op.name(arm,'segment',index)
        case_dir=producer/'payload/study/case-0'
        case=op.read(case_dir/'report.json');index_record=verify_case(case,case_dir)
        existing=op.read(case_dir/'retained-frames.json')
        if existing!=index_record:raise ValueError('Retained-frame index differs from full loss stream')
        return compiled(*args,**kwargs)
    op.certify_segment=certify


def noise_append_hook(original, output, total=1000):
    output=Path(output);trajectory=output/'trajectory';best=None
    def append(path,row):
        nonlocal best
        path=Path(path)
        if path==trajectory/'states.jsonl':
            step=row['step'];keep=retained(step,total);source=output/row['snapshot']
            if best is None or row['contact_loss']<best:
                target=trajectory/'best-observed.npz';tmp=target.with_suffix('.partial')
                shutil.copyfile(source,tmp);tmp.replace(target);best=row['contact_loss']
            if not keep:
                target=trajectory/'latest-state.npz';source.replace(target)
                (trajectory/'latest-state.json').write_text(json.dumps({**row,'rolling_snapshot':str(target.relative_to(output))},allow_nan=False)+'\n')
                row['snapshot']=row['snapshot_sha256']=None;thin_readout(row)
            row['snapshot_retained']=keep
        elif path==trajectory/'updates.jsonl':
            step=row['update'];keep=retained(step,total)
            source=trajectory/'noise'/f'update-{step:04d}.npz'
            row['noise_snapshot_retained']=keep
            row['noise_snapshot']=str(source.relative_to(output)) if keep else None
            if not keep:
                source.replace(source.with_name('latest-noise.npz'))
                row['noise_sha256']=None
        return original(path,row)
    return append


def execute_noise(original, output):
    namespace=runpy.run_path(str(original),run_name='stride20_original_noisy_driver')
    main=namespace['main'];g=main.__globals__
    g['append']=noise_append_hook(g['append'],output)
    main()
    directory=Path(output)/'trajectory'
    states=[json.loads(x) for x in (directory/'states.jsonl').read_text().splitlines()]
    updates=[json.loads(x) for x in (directory/'updates.jsonl').read_text().splitlines()]
    # Noise-driver snapshot paths are relative to the study output, not case dir.
    index=verify_case({'requested_updates':1000,'artifact_retention':policy(),'states':states,'updates':updates},output)
    (directory/'retained-frames.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')


class ProcessProxy:
    def __init__(self, original, profile, manifest):
        self.original,self.profile,self.manifest=original,profile,manifest
    def __getattr__(self,name):return getattr(self.original,name)
    def run(self,args,*pos,**kwargs):
        from selected_contract import LAUNCHER,PYTHON
        if isinstance(args,list) and '--mode' in args and args[args.index('--mode')+1]=='run' and '--output' in args:
            output=Path(args[args.index('--output')+1]);parent=output.parent.parent
            wrapped=[PYTHON,str(LAUNCHER/'retention.py'),'--profile',self.profile,'--manifest',self.manifest,'--parent',str(parent),'--',*args]
            from selected_contract import exclusive
            exclusive(output.parent/'retention-invocation.json',{'args':wrapped,'original_driver_args':args,'policy':policy()})
            return self.original.run(wrapped,*pos,**kwargs)
        return self.original.run(args,*pos,**kwargs)


def validate_entry(argv):
    p=argparse.ArgumentParser()
    p.add_argument('--startup-only',action='store_true')
    p.add_argument('--profile',required=True);p.add_argument('--manifest',required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('original',nargs=argparse.REMAINDER)
    args=p.parse_args(argv);original=args.original
    if original and original[0]=='--':original=original[1:]
    from selected_contract import profile,PYTHON
    source=profile(args.profile)['source'];driver=source/('scripts/dbmol_driver.py' if args.profile=='dbmol10k' else 'examples/brd4_25_atoms.py')
    if original[:2]!=[PYTHON,str(driver)] or original.count('--output')!=1 or original.count('--mode')!=1 or original[original.index('--mode')+1]!='run':raise ValueError('Exact original production entry required')
    output=Path(original[original.index('--output')+1])
    if output!=args.parent/'payload/study':raise ValueError('Original output owner differs')
    return args,original,output


def main():
    args,original,output=validate_entry(sys.argv[1:])
    from selected_contract import verify_own,verify_source,read
    verify_own(args.manifest);verify_source(args.profile)
    if args.startup_only:
        if any(k=='jax' or k.startswith('jax.') or k=='torch' or k.startswith('torch.') for k in sys.modules):raise ValueError('Startup metadata may not import numerical backends')
        print(json.dumps({'status':'actual_retention_cli_accepted_without_model','optimizer_updates':0,'scientific_readiness_passed':False,'policy':policy()}))
        return
    from selected_ops import metadata_certificate
    metadata_certificate(args.manifest)
    if read(output.parent/'driver-invocation.json')['args']!=original:raise ValueError('Declared driver command differs')
    import mosaic.small_molecule.chemistry.trajectory as module
    install_trajectory(module,output)
    sys.argv=[original[1],*original[2:]]
    runpy.run_path(original[1],run_name='__main__')


if __name__=='__main__':main()
