"""Exact original chemistry entrypoints with separately owned prospective gates.

All model/array imports occur only after the real allocated parser/startup path.
The original optimizer, fixed steps, ramp, teacher and initialization execute
unchanged. Private AST changes affect job horizon counts and an additional
actual-global-coefficient check only; old source bytes stay read-only.
"""
import argparse
import ast
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import sys

import chemistry_continuation_contract as c
import chemistry_continuation_hooks as hooks


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def modules():
    sys.path.insert(0,str(c.CURRENT/'scripts'))
    base=module(c.CURRENT/'scripts/continuation_driver.py','_chemistry_inherited_driver')
    precision=module(c.CURRENT/'scripts/precision_continuation_hooks.py','_chemistry_precision_hooks')
    return base,precision


def authenticated_plan(path,pin):
    """Carry the verified owning hash into the existing reference expansion."""
    c.require(c.sha(path)==pin,'Owning plan digest changed')
    plan=c.validate_plan(c.read(path));plan['_sha256']=pin
    return plan


def argv(plan,arm,phase,output,ready=None):
    row=c.configured_arm(plan,arm);old=row['original_argv'];result=old[:2]
    replaced={'--mode','--output','--offset','--segment-steps','--resume-snapshot','--resume-sha256','--ready','--ready-sha256'}
    c.require(len(old)%2==0 and all(old.count(k)<=1 for k in replaced),'Original argument vector malformed')
    for i in range(2,len(old),2):
        if old[i] not in replaced:result+=old[i:i+2]
    c.require(phase in ('validate','segment') and (ready is not None)==(phase=='segment'),'New exact readiness required')
    result+=['--mode','validate' if phase=='validate' else 'run','--output',str(output),
             '--offset',str(row['start']),'--segment-steps',str(10000 if phase=='validate' else row['remaining']),
             '--resume-snapshot',row['boundary']['path'],'--resume-sha256',row['boundary']['sha256']]
    if ready:result+=['--ready',ready['path'],'--ready-sha256',ready['sha256']]
    return result


def actual_coefficient_tree(tree):
    """Add a fixed actual-global coefficient cohort before either original pilot.

    Original low/high cohorts and both ten-update pilots remain in place. The
    added check uses the exact saved boundary/teacher and global schedule value.
    """
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)
            and t.id=='means' for t in n.targets)]
    # Original code assigns means,sizes as a tuple, so find the exact statement.
    needle=ast.dump(ast.parse('means, sizes = [], []').body[0]);count=0
    addition=ast.parse('''
if chemistry:
    actual_fixed = prepare(initial, args.offset)
    actual_options = {**options, 'repair_weight': schedule.for_state(args.offset)}
    actual_evaluation = evaluate_combined(loss, initial, actual_fixed, spec, reference, key=key, config=actual_options)
    check_combined_directions(loss, actual_evaluation, actual_fixed, spec, reference, key=key, config=actual_options, directory=args.output / 'actual_coefficient_coupled_checks')
''').body
    class Add(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal count
            if ast.dump(node)==needle:count+=1;return [*addition,node]
            return node
    tree=Add().visit(tree);c.require(count==1,'Original complete pilot boundary differs')
    return ast.fix_missing_locations(tree)


def execute(plan,arm,phase,output,ready=None,startup=False,defer_ready=False):
    c.validate_plan(plan);row=plan['arms'][arm]
    c.require(os.environ.get('LSB_JOBID','').isdigit(),'Real allocated startup/execution required')
    base,precision=modules();source=Path(row['science_source']['path'])
    base.verify_source(source,row['science_source']['manifest_sha256'],row['science_source']['files'])
    command=argv(plan,arm,phase,output,ready);driver=Path(command[1]);raw=driver.read_bytes()
    for b in (row['boundary'],row['config'],*row['argv_inputs']):c.bound(b)
    with base.execution_context(source,command):
        protocol=__import__('extended_protocol' if arm=='protenix' else 'boltz_serial_protocol')
        changed=[]
        try:
            if phase=='segment':
                for name in ('segment_scope','require_boundary'):
                    old=getattr(protocol,name);changed.append((name,old))
                    setattr(protocol,name,hooks.protocol_function(old,arm,name,row['remaining']))
            if startup:
                tree=base.startup_tree(raw,str(driver),defer_ready=defer_ready)
                namespace={'__name__':'_chemistry_actual_startup','__file__':str(driver),
                           'PremodelBoundaryReached':base.PremodelBoundaryReached}
                exec(compile(tree,str(driver),'exec'),namespace)
                try:namespace['main']()
                except base.PremodelBoundaryReached:
                    result={'status':'passed_metadata_only','job_id':os.environ['LSB_JOBID'],
                            'original_argv':command,'numerical_setup_entered':False,'updates':0,
                            'readiness_deferred':defer_ready}
                    c.write(Path(output).parent/'endpoint-startup.json',result);return result
                raise ValueError('Actual original parser did not reach its pre-model sentinel')
            c.require(os.environ.get('LSB_QUEUE')=='gpul40s','Scientific execution requires own GPU allocation')
            adapter=row['adapter_source'];base.verify_source(Path(adapter['path']),adapter['manifest_sha256'],adapter['files'])
            sys.path.insert(0,str(Path(adapter['path'])/'scripts'))
            pilot=module(Path(adapter['path'])/'scripts/full_model_pilot.py','_chemistry_exact_acceleration_'+arm)
            directory=Path(output).parent/'acceleration';directory.mkdir(exist_ok=False)
            tree=actual_coefficient_tree(ast.parse(raw,filename=str(driver)))
            if phase=='segment':tree=hooks.driver_tree(ast.unparse(tree),str(driver),arm)
            tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))]
            with ExitStack() as stack:
                stack.enter_context(pilot.combined_adapters(row['profile'],directory))
                if phase=='segment':
                    retention=plan['retention_source'];base.verify_source(Path(retention['path']),retention['manifest_sha256'])
                    tool=module(Path(retention['path'])/'retention.py','_chemistry_unchanged_retention')
                    import mosaic.small_molecule.chemistry.trajectory as trajectory
                    tool.install_trajectory(trajectory,Path(output))
                state=stack.enter_context(hooks.installed(base,precision,plan,arm,output))
                namespace={'__name__':'_chemistry_actual_original','__file__':str(driver)}
                try:exec(compile(ast.fix_missing_locations(tree),str(driver),'exec'),namespace);namespace['main']()
                finally:
                    c.write(Path(output).parent/'prospective-evidence.json',{'native':state.records,
                            'combined':state.logit_records,'original_results_changed':False,
                            'original_failures_preserved':True,'optimizer_settings_changed':False})
            return command
        finally:
            for name,old in changed:setattr(protocol,name,old)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--arm',choices=tuple(c.ARMS),required=True);p.add_argument('--phase',choices=('validate','segment'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ready');p.add_argument('--ready-sha256')
    p.add_argument('--startup-only',action='store_true');p.add_argument('--startup-defer-ready',action='store_true')
    a=p.parse_args();plan=authenticated_plan(a.plan,a.plan_sha256);ready=None
    if a.ready:
        ready={'path':a.ready,'sha256':a.ready_sha256,'bytes':Path(a.ready).stat().st_size};c.bound(ready)
    execute(plan,a.arm,a.phase,a.output,ready,a.startup_only,a.startup_defer_ready)


if __name__=='__main__':main()
