"""Fresh-process exact chemistry execution and unchanged original acceptance.

Only --execute imports numerical implementations. Metadata startup runs the
actual original argparse AST; GPU startup separately executes the real original
entrypoint through its authenticated intentional pre-model boundary. Terminal
acceptance retains the complete original certifier with only stride20 file-loop
adaptation and new, independently verified metadata routing.
"""
from contextlib import ExitStack
import argparse
import ast
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

import production_contract as c
import production_ops as op


def parse_original(driver, argv):
    """Execute the exact original main's argparse prefix, without model imports."""
    tree = ast.parse(Path(driver).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    op.require(len(functions) == 1, 'Original real main entrypoint differs')
    prefix = []
    for statement in functions[0].body:
        prefix.append(statement)
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == 'args'):
            op.require(isinstance(statement.value, ast.Call)
                       and ast.dump(statement.value.func) == ast.dump(ast.parse('parser.parse_args', mode='eval').body),
                       'Original argparse endpoint shape differs')
            break
    else: raise ValueError('Actual original argparse result is missing')
    allowed = {'parser', 'args'}
    for statement in prefix:
        op.require(isinstance(statement, (ast.Assign, ast.Expr)), 'Original parser prefix contains unexpected execution')
        if isinstance(statement, ast.Assign):
            op.require(all(isinstance(target, ast.Name) and target.id in allowed for target in statement.targets),
                       'Unexpected original parser assignment')
    prefix.append(ast.Return(ast.Name('args', ast.Load())))
    fn = ast.FunctionDef(name='_actual_parser', args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
                         body=prefix, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace = {'argparse': argparse, 'Path': Path, '__doc__': ast.get_docstring(tree)}
    old = sys.argv
    try:
        sys.argv = [str(driver), *argv[2:]]
        exec(compile(module, str(driver)+'[exact-argparse-only]', 'exec'), namespace)
        return namespace['_actual_parser']()
    finally: sys.argv = old


def import_original(arm):
    spec = c.profile(arm); source = Path(spec['source'])
    op.verify(source, spec['manifest'])
    sys.path[:0] = [str(source/'scripts'),str(source/'src'),str(source/'tests')]
    filename = 'extended_ops.py' if arm=='protenix' else 'boltz_serial_ops.py'
    original = op.module(source/'scripts'/filename,'_chemistry_original_'+arm)
    original.ROOT = op.ROOT/arm
    return original


def require_startup(arm, phase, index, pin):
    parent = op.phase_parent(phase,arm,index)
    proof = op.read(parent/'startup-gpu.json')
    command = op.read(parent/'startup-command.json')
    plan = c.validate_plan(op.read(op.SOURCE/'plan.json'))
    args = op.driver_args(arm,phase,index,parent/'payload/study',plan)
    op.require(proof['status']=='startup_only_passed' and proof['production_manifest_sha256']==pin
               and proof['arm']==arm and proof['phase']==phase and proof['index']==index
               and proof['job_id']==op.read(parent/'submission.json')['job_id']
               and proof['source_manifest_sha256']==c.profile(arm)['manifest']
               and proof['gradient_passed'] is False and proof['model_loaded'] is False
               and proof['optimizer_updates']==0 and command['driver_args']==args
               and command['production_manifest_sha256']==pin,
               'Actual current generated startup proof differs')
    summary=proof['result']['summary']
    expected_error=('ProtenixPremodelBoundaryReached: Intentional Protenix GPU startup-only boundary; no native model loaded'
                    if arm=='protenix' else 'PremodelBoundaryReached: Intentional startup-only boundary; no native model loaded')
    op.require(summary['status']=='failed' and summary['error']==expected_error
               and summary['source_manifest_sha256']==c.profile(arm)['manifest']
               and summary['mode']==('validate' if phase=='native' else 'run')
               and summary['config_sha256']==op.sha(args[args.index('--config')+1])
               and not summary.get('cases') and not summary.get('native_gradient_checks'),
               'Preserve exact actual intentional pre-model sentinel, never a scientific pass')
    op.require(op.read(parent/'payload/driver-invocation.json')['args']==args,
               'Scientific command differs from accepted startup command')
    return proof


def startup(arm,phase,index,pin,plan,parent):
    op.require(os.environ.get('LSB_QUEUE')=='gpul40s', 'Genuine allocated GPU startup required')
    source=Path(c.profile(arm)['source']); original=import_original(arm)
    original_args=op.driver_args(arm,phase,index,parent/'payload/study',plan)
    parse_original(original_args[1],original_args)
    tmp = Path(os.environ.get('TMPDIR',''))
    op.require(tmp.is_absolute() and tmp.is_dir() and not tmp.is_symlink()
               and tmp.stat().st_uid==os.getuid(), 'Exact owned node TMPDIR required')
    directory=Path(tempfile.mkdtemp(prefix='chem-startup-',dir=tmp))
    args=list(original_args); args[args.index('--output')+1]=str(directory/'study')
    if arm=='protenix':
        helper=op.module(source/'scripts/extended_startup.py','_chemistry_original_gpu_startup')
        result=helper.run_startup_child(args,source,c.profile(arm)['manifest'],directory)
    else:
        helper=op.module(source/'scripts/premodel_startup.py','_chemistry_original_boltz_startup')
        result=helper.execute_generated_command(args,source,c.profile(arm)['manifest'],arm='boltz2_chemistry')
    # Raw startup summaries intentionally remain failed sentinels; this is only
    # successful pre-model command execution, never a native/derivative pass.
    op.write(parent/'startup-command.json', {'driver_args':original_args,'startup_args':args,
                                            'production_manifest_sha256':pin})
    op.write(parent/'startup-gpu.json', {'status':'startup_only_passed','arm':arm,'phase':phase,'index':index,
        'job_id':os.environ['LSB_JOBID'],'production_manifest_sha256':pin,
        'source_manifest_sha256':c.profile(arm)['manifest'],'result':result,
        'model_loaded':False,'gradient_passed':False,'optimizer_updates':0})


def verify_native(arm,index,pin):
    original=import_original(arm); spec=c.profile(arm)
    # Exact original readiness, boundary replay, nonzero/coupled/raw/native/
    # finite/frozen/runtime/projection checks execute unmodified in this fresh
    # namespace. Only ROOT selects the independent new output owner.
    ready,value=original.native_assessment(spec['arm'],index,spec['manifest'])
    op.require(ready==op.phase_parent('native',arm,index)/'payload/study/assessment.json',
               'Original readiness escaped fresh chemistry producer')
    require_startup(arm,'native',index,pin)
    return value


def install_terminal_metadata_routes(original,arm,index,pin,parent,producer,lsf):
    """Route only metadata calls with signatures used by the pinned certifiers.

    Allocated contracts execute the exact original AST call sites against these
    functions before a production run. No scientific assertion/function is
    replaced by this adapter.
    """
    spec=c.profile(arm)
    op.require(producer==op.phase_parent('segment',arm,index)
               and parent==op.phase_parent('after-segment',arm,index), 'Wrong terminal routing owner')
    guard_path=parent/'payload/production-storage-after-segment-entry.json'
    c.require_storage(op.read(guard_path))
    # No original scientific acceptance condition is replaced. These narrow
    # callables only route current authenticated scheduler/startup/storage proof
    # and prevent the old controller submitting into its historical campaign.
    def original_final(owner,queue):
        op.require(owner==producer and queue=='gpul40s','Original terminal producer/queue differs')
        terminal_proofs=op.read(parent/'terminal-proofs.json')['proofs']
        for row in terminal_proofs:op.check_binding(row)
        return [{'path':row['path'],'sha256':row['sha256']} for row in
            (*terminal_proofs,op.binding(producer/'submission.json'),lsf,op.binding(producer/'startup-gpu.json'),
             op.binding(op.SOURCE/'source_manifest.json'),op.binding(op.ROOT/'cpu-certificate.json'),
             op.binding(op.ROOT/arm/'adoption.json'),op.binding(guard_path))]
    original.final_job=original_final
    def verify_gpu_startup(old_arm,phase,old_index,manifest,owner):
        op.require(old_arm==spec['arm'] and phase=='segment' and old_index==index
                   and manifest==spec['manifest'] and owner==producer,
                   'Original five-argument GPU startup call differs')
        return require_startup(arm,phase,old_index,pin)
    original.verify_gpu_startup=verify_gpu_startup
    def verify_startup(old_arm,phase,old_index,manifest):
        op.require(old_arm==spec['arm'] and phase=='segment' and old_index==index
                   and manifest==spec['manifest'], 'Original four-argument startup call differs')
        return require_startup(arm,phase,old_index,pin)
    original.verify_startup=verify_startup
    def command_path(old_arm,phase,old_index):
        op.require(old_arm==spec['arm'] and phase=='segment' and old_index==index,
                   'Original three-argument startup-command call differs')
        return producer/'startup-command.json'
    original.command_path=command_path
    def all_budgets():
        value=op.read(guard_path);c.require_storage(value)
        full=value['chemistry'][arm]['full']
        op.require(type(full) is int and full>0, 'Full current chemistry projection missing')
        return {spec['arm']:full}
    def same_guard(owner,phase,full,*,existing_full=False):
        op.require(owner==parent and phase=='next-native' and existing_full is True
                   and full==all_budgets(), 'Original next-native storage call differs')
        value=op.read(guard_path);c.require_storage(value);return value
    original.storage_gate=same_guard
    original.all_budgets=all_budgets
    transitions=[]
    def record_next(old_arm,next_index,manifest,owner):
        op.require(old_arm==spec['arm'] and next_index==index+1 and next_index<spec['segments']
                   and manifest==spec['manifest'] and owner==parent,
                   'Original next-native intent differs from exact serial boundary')
        transitions.append({'args':[old_arm,next_index,manifest,str(owner)],'kwargs':{},
                            'submission_performed':False})
    original.queue_native=record_next
    return transitions


def terminal(arm,index,pin,parent):
    original=import_original(arm); spec=c.profile(arm)
    producer=op.phase_parent('segment',arm,index)
    sub=op.read(producer/'submission.json')
    op.require(sub['status']=='submitted' and sub['argv']==op.argv('segment',pin,arm,index),
               'Exact original producer allocation/launcher differs')
    lsf=op.lsf_success(producer,sub); require_startup(arm,'segment',index,pin)
    transitions=install_terminal_metadata_routes(original,arm,index,pin,parent,producer,lsf)
    op.verify(op.RETENTION_SOURCE,op.RETENTION_PIN)
    retention=op.module(op.RETENTION_SOURCE/'retention.py','_chemistry_exact_retention')
    retention.adapt_certifier(original,'chemistry')
    if arm=='protenix':original.certify_segment(spec['arm'],index,spec['manifest'])
    else:original.certify_segment(spec['arm'],index,spec['manifest'],parent)
    certificate=op.ROOT/arm/(c.stage_label(arm,'segment',index)+'-certificate.json')
    passed=op.read(certificate)
    op.require(passed['status']=='passed' and passed['start']==index*10000
               and passed['end']==(index+1)*10000,'Original complete terminal certifier did not accept')
    op.write(parent/'original-certifier-result.json',{'status':'passed_original_terminal_acceptance',
        'certificate':op.binding(certificate),'original_scientific_source_manifest_sha256':spec['manifest'],
        'production_manifest_sha256':pin,'next_phase_intents':transitions,
        'scientific_checks_removed':False,'saved_results_changed':False})


def execute_driver(arm,phase,index,pin,plan,parent):
    native=op.arm_plan(arm,plan); source=Path(native['source']['path'])
    args=op.driver_args(arm,phase,index,parent/'payload/study',plan)
    parse_original(args[1],args)
    proof=op.read(parent/'startup-gpu.json')
    op.require(proof['status']=='startup_only_passed' and proof['production_manifest_sha256']==pin,
               'Actual startup must pass before native model work')
    op.write(parent/'payload/driver-invocation.json', {'args':args,'production_manifest_sha256':pin,
        'original_source_manifest_sha256':native['source']['manifest_sha256'],
        'original_saved_runs_read_only':True})
    # Authenticate exact adapter source before its first import, and preserve
    # the original source before scripts insert their own paths.
    pilot=plan['pilots'][arm];pilot_source=Path(pilot['source'])
    op.adapter_parity(plan)
    op.verify(pilot_source,pilot['manifest_sha256'])
    sys.path[:0]=[str(pilot_source/'scripts'),str(source/'scripts'),str(source/'src'),str(source/'tests')]
    adapters=op.module(pilot_source/'scripts/full_model_pilot.py','_production_numerical_adapters_'+arm)
    op.require(Path(adapters.__file__).resolve()==(pilot_source/'scripts/full_model_pilot.py').resolve(),
               'Combined adapter imported from an unpinned per-arm source')
    old_argv,old_cwd=sys.argv,Path.cwd()
    try:
        with ExitStack() as stack:
            stack.enter_context(adapters.combined_adapters(native['profile'],parent/'payload/acceleration'))
            if phase=='segment':
                op.verify(op.RETENTION_SOURCE,op.RETENTION_PIN)
                retention=op.module(op.RETENTION_SOURCE/'retention.py','_chemistry_production_retention')
                import mosaic.small_molecule.chemistry.trajectory as trajectory
                retention.install_trajectory(trajectory,parent/'payload/study')
            sys.argv=args[1:];os.chdir(source)
            runpy.run_path(args[1],run_name='__main__')
    finally:sys.argv=old_argv;os.chdir(old_cwd)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--arm',choices=tuple(c.PROFILES),required=True)
    parser.add_argument('--phase',choices=('native','segment'),required=True)
    parser.add_argument('--index',type=int,required=True)
    modes=parser.add_mutually_exclusive_group(required=True)
    for option in ('metadata-startup','startup-only','certify-terminal','execute'):modes.add_argument('--'+option,action='store_true')
    args=parser.parse_args(argv)
    op.require(os.environ.get('LSB_JOBID'),'Allocated HPC only')
    manifest,plan=op.own_plan(args.manifest)
    if args.metadata_startup:
        op.require(os.environ.get('LSB_QUEUE')=='hpc' and args.index==0 and args.phase=='segment',
                   'Only the exact initial CPU metadata startup route is declared')
        evidence=op.pilot_pass(args.arm,plan)
        command=op.driver_args(args.arm,args.phase,args.index,op.phase_parent('segment',args.arm,0)/'payload/study',
                               plan,initial_ready=evidence['readiness'])
        parsed=parse_original(command[1],command)
        op.require(parsed.mode=='run' and parsed.offset==0 and parsed.segment_steps==10000,
                   'Actual original parser changed production scope')
        print(json.dumps({'status':'actual_original_parser_metadata_accepted','args':command,
                          'native_or_gradient_passed':False,'optimizer_updates':0}))
        return
    callback_phase='after-'+args.phase if args.certify_terminal else args.phase
    parent,submission=op.own(callback_phase,args.manifest,args.arm,args.index)
    op.cpu_certificate(args.manifest,manifest);op.adoption(args.arm)
    if args.startup_only:startup(args.arm,args.phase,args.index,args.manifest,plan,parent)
    elif args.certify_terminal:
        if args.phase=='native':
            value=verify_native(args.arm,args.index,args.manifest)
            op.write(parent/'original-readiness-result.json',{'status':'passed_original_readiness',
                'readiness_sha256':op.sha(op.phase_parent('native',args.arm,args.index)/'payload/study/assessment.json'),
                'source_manifest_sha256':value['source_manifest_sha256']})
        else:terminal(args.arm,args.index,args.manifest,parent)
    else:execute_driver(args.arm,args.phase,args.index,args.manifest,plan,parent)


if __name__=='__main__':main()
