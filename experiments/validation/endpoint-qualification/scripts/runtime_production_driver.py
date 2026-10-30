"""Single original Protenix SEGMENT, with an authenticated new runtime owner."""
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import inspect
import os
from pathlib import Path


def require(value, message):
    if not value: raise ValueError(message)


def check_segment_startup(value, ready, job):
    """Authenticate the ORIGINAL raw CLI receipt, which has no readiness object."""
    require(value['status']=='passed_metadata_only' and value['job_id']==job
        and value['numerical_setup_entered'] is False and value['updates']==0
        and value['readiness_deferred'] is False,'Actual ready-bound segment startup failed')
    argv=value['original_argv'];require(type(argv) is list,'Original parser argv required')
    for flag,expected in {'--ready':ready['path'],'--ready-sha256':ready['sha256'],
        '--offset':'20000','--segment-steps':'30000','--resume-sha256':'6be2b3868d58162d4aa3cc9d960cfb85a84d4078748f0b24d50b138ae3e47dbd',
        '--mode':'run'}.items():
        require(argv.count(flag)==1 and argv.index(flag)+1<len(argv) and argv[argv.index(flag)+1]==expected,
                'Actual original ready/initialization/whole-remainder argv differs: '+flag)
    return value


def segment_tree(source):
    """Change only phase/readiness and remove the diagnostic stop, never model math."""
    tree=ast.parse(source);require(len(tree.body)==1 and tree.body[0].name=='plain','Original plain entry required')
    fn=tree.body[0]
    vectors=[n for n in ast.walk(fn) if isinstance(n,ast.List) and any(
        isinstance(v,ast.Constant) and v.value=='--phase' for v in n.elts)]
    require(len(vectors)==1,'One original CLI vector required');values=vectors[0].elts
    at=next(i for i,v in enumerate(values) if isinstance(v,ast.Constant) and v.value=='--phase')
    require(values[at+1].value=='validate','Original validation phase differs');values[at+1]=ast.Constant('segment')
    values.extend(ast.parse("['--ready',plan['qualified']['ready']['path'],'--ready-sha256',plan['qualified']['ready']['sha256']]",mode='eval').body.elts)
    class Full(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='stopped_tree':
                require(len(node.args)==1 and not node.keywords,'Stop shape differs');self.count+=1;return node.args[0]
            return node
        def visit_ExceptHandler(self,node):
            require(isinstance(node.type,ast.Name) and node.type.id=='BaseException','Exception shape differs')
            node.body=[ast.Raise()];return node
    change=Full();tree=change.visit(tree)
    require(change.count==1 and isinstance(fn.body[-1],ast.Raise),'Exactly one diagnostic stop required')
    fn.body[-1]=ast.Return(ast.Name('records',ast.Load()))
    return ast.fix_missing_locations(tree)


def validation_lookup(adapter, summary):
    """Original segment verifier uses the actual newly qualified summary as its clock basis."""
    tree=ast.parse(inspect.getsource(adapter.verify_precision_result));found=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='expected_validation':
            require(isinstance(node.value,ast.IfExp) and ast.unparse(node.value.test)=="phase == 'validate'",
                    'Original conditional validation lookup differs')
            node.value.orelse=ast.Call(ast.Name('Path',ast.Load()),[ast.Constant(str(summary))],[]);found+=1
    require(found==1,'One original validation metadata lookup required')
    exec(compile(ast.fix_missing_locations(tree),adapter.__file__,'exec'),adapter.__dict__)


@contextmanager
def reference_budget(helper, precision, study, output):
    """Explicit128MiB per native/terminal metadata tree; complete inventory unchanged."""
    require(precision.SESSION_CAP==64*1024**2,'Original metadata identity differs')
    original=precision._budget;namespace=dict(original.__globals__)
    exec(compile(helper.diagnostic_budget_tree(inspect.getsource(original)),original.__code__.co_filename,'exec'),namespace)
    inventory=namespace['_budget'];receipts=output/'reference-budget-checks';receipts.mkdir();count=0
    def budget(directory,config):
        nonlocal count
        require(Path(directory) in (study/'native_gradient_checks-prospective',study/'terminal_gradient_checks-prospective'),
                'Only exact new native/terminal reference metadata scopes allowed')
        total=inventory(directory,config)
        helper.write(receipts/('check-'+str(count).zfill(4)+'.json'),{'status':'passed_complete_inventory_explicit_metadata_budget',
            'directory':str(directory),'bytes':total,'inherited_configuration_cap_bytes':config['max_session_bytes'],
            'effective_metadata_cap_bytes':128*1024**2,'all_output_retained':True,'scientific_comparisons_changed':False})
        count+=1;return total
    precision._budget=budget
    try:yield
    finally:precision._budget=original


def run(helper, plan, output, startup=False):
    require(plan['production_policy']=={'arm':'protenix','phase':'segment','offset':20000,'updates':30000,
        'global_target':50000,'learning_rate':0.3,'matrix_stride':20,'replicas_per_native_checkpoint':16},
        'Closed unchanged full-remainder production policy required')
    output.mkdir();study=output/'payload/study';namespace=dict(vars(helper));old_module=helper.module
    def module(path,name):
        value=old_module(path,name)
        if Path(path).name=='precision_continuation_driver.py':validation_lookup(value,plan['qualified']['summary']['path'])
        if Path(path).name=='precision_continuation_hooks.py' and not startup:
            stack.enter_context(reference_budget(helper,value,study,output))
        return value
    namespace['module']=module
    # The guard permits only this exact NEW scientific/startup directory.
    namespace['study_path']=lambda current,path: study if Path(path)==output else (_ for _ in ()).throw(ValueError('Foreign study scope'))
    exec(compile(segment_tree(inspect.getsource(helper.plain)),helper.__file__,'exec'),namespace)
    from contextlib import ExitStack
    with ExitStack() as stack:
        if not startup:helper.guard_candidate_environment(os.environ)
        result=namespace['plain'](plan,output,startup=startup)
    if startup:
        require(result['status']=='passed_metadata_only' and result['updates']==0 and result['numerical_setup_entered'] is False,
                'Actual original segment startup failed')
        return result
    summary=helper.read(study/'summary.json')
    require(summary['status']=='passed' and summary['job_id']==os.environ['LSB_JOBID'],'Original full segment failed')
    helper.write(output/'runtime-production-assessment.json',{'status':'passed_original_segment_pending_independent_capture',
        'execution_profile':plan['execution_profile'],'original_source':plan['capacity_source'],
        'original_results_changed':False,'original_failures_preserved':True,'global_start':20000,'global_end':50000,
        'original_summary':{'path':str(study/'summary.json'),'bytes':(study/'summary.json').stat().st_size,'sha256':helper.sha(study/'summary.json')},
        'measurements':result})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pin',required=True)
    parser.add_argument('--mode',choices=('startup','segment'),required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import runtime_production_ops as ops
    original,plan=ops.context(args.pin)
    expected=ops.ROOT/('cpu/payload/startup/plain' if args.mode=='startup' else 'gpu-plain/payload/replay/plain')
    require(args.output==expected,'Exact new owned output required')
    helper=ops.module(Path(plan['helper_source']['path'])/'scripts/continuation_failure_diagnostics.py','_runtime_production_helpers')
    value=run(helper,plan,args.output,startup=args.mode=='startup')
    if args.mode=='startup':ops.write(args.output/'actual-startup.json',value)


if __name__=='__main__':main()
