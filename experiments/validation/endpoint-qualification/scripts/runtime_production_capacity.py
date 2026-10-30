"""Namespace-only adoption of the original complete shared capacity algorithm."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shlex
from types import SimpleNamespace


def require(value, message):
    if not value: raise ValueError(message)


def replace_expression(tree, before, after, count=1):
    old=ast.dump(ast.parse(before,mode='eval').body);new=ast.parse(after,mode='eval').body
    class Rewrite(ast.NodeTransformer):
        seen=0
        def visit(self,node):
            if isinstance(node,ast.expr) and ast.dump(node)==old:
                self.seen+=1;return ast.copy_location(deepcopy(new),node)
            return super().visit(node)
    rewrite=Rewrite();tree=rewrite.visit(tree)
    require(rewrite.seen==count,'Exact inherited namespace expression differs: '+before)
    return ast.fix_missing_locations(tree)


def proxy(op, plan, source, run, ready, result):
    """One replacement future; original failed files remain charged and untouched."""
    namespace=dict(vars(op));root=Path(plan['root']);source=Path(source);run=Path(run)
    def parent(phase,arm='all',offset=0):
        if arm=='protenix' and offset==20000:
            if phase=='segment':return run/'gpu-plain/payload/replay/plain'
            if phase=='validate':return Path(plan['qualified']['study']).parent.parent
        return op.parent_path(phase,arm,offset)
    def read(path):
        if Path(path)==root/'projections.json':
            return {'protenix':{'remaining_offsets':[20000],
                'unit_bytes':result['projections']['maximum_pilot_update_bytes']}}
        return op.read(path)
    def verify_ready(current,pin,arm,offset):
        require(arm=='protenix' and offset==20000 and ready['source_manifest_sha256']==pin,
                'Only newly authenticated Protenix readiness can own the replacement future')
        for row in ready['proofs']:op.check_binding(row)
        return ready
    def verify_result(current,pin,phase,arm,offset):
        verify_ready(current,pin,arm,offset);require(phase=='validate','Only admitted validation supplies the forecast')
        return result
    namespace.update(ROOT=root,SOURCE=source,parent_path=parent,DISPATCH_ARMS=('protenix',),read=read)
    # Preserve the complete byte walk and exclude the actual NEW scientific root.
    exec(compile(ast.parse(inspect.getsource(op.metadata_bytes)),op.__file__,'exec'),namespace)
    namespace.update(verify_ready=verify_ready,verify_result=verify_result,
        runtime_root=run,runtime_ready=run/'qualified-ready.json')
    return SimpleNamespace(**namespace)


def configure(module, owner, plan):
    """Alter only claim/readiness paths and exact registered command ownership."""
    run=Path(plan['run_root']);scope=Path(plan['root'])
    old_candidate=module.candidate_from_parent
    def candidate(op,parent,phase):
        if phase=='dispatch' and Path(parent)==run/'after-cpu':return 'protenix',20000
        return old_candidate(op,parent,phase)
    module.candidate_from_parent=candidate
    def namespace_owned(row,own_jobs,source):
        if row['JOBID'] not in own_jobs:return False
        registry=owner.read(scope/'jobs.json')
        names=[name for name,jid in registry.items() if jid==row['JOBID']]
        require(len(names)==1,'Registered ownership is ambiguous')
        record=owner.read(scope/names[0]/'submission.json');argv=record['argv']
        require(record['status']=='submitted' and record['job_id']==row['JOBID'],'Registered allocation differs')
        starts=[i for i in range(len(argv)-1) if argv[i:i+2]==['bash','-l']]
        require(len(starts)==1,'One exact recorded stage command required')
        actual=shlex.join(argv[starts[0]:])
        if actual!=row['COMMAND']:return False
        sources={p['path']:p['manifest_sha256'] for p in owner.read(Path(source)/'source_manifest.json')['parent_sources']}
        sources[str(source)]=owner.sha(Path(source)/'source_manifest.json')
        selected=[path for path,pin in sources.items() if str(Path(path)/'scripts')+'/' in actual
                  and record['source_manifest_sha256']==pin]
        return len(selected)==1
    module.__dict__['_runtime_namespace_owned']=namespace_owned
    expressions={
      'build_scopes':[("op.ROOT / 'storage-claims' / (sid + '.json')", "Path(op.runtime_root) / 'storage-claims' / (sid + '.json')"),
        ("op.ROOT / (op.label('validate', arm, offset) + '-certificate.json')",'Path(op.runtime_ready)'),
        ("read(op.ROOT / 'projections.json')", "op.read(op.ROOT / 'projections.json')")],
      '_storage':[("op.ROOT / 'storage-claims'",'Path(op.runtime_root) / "storage-claims"')],
      'segment_certificate':[("op.ROOT / (op.label('segment', arm, offset) + '-certificate.json')",
                             'Path(op.runtime_root) / "completion-certificate.json"')],
      'classify_jobs':[("str(source) in command and jid in own_jobs",
                        "(str(source) in command and jid in own_jobs) or _runtime_namespace_owned(row, own_jobs, source)")],
    }
    for name,changes in expressions.items():
        tree=ast.parse(inspect.getsource(getattr(module,name)))
        for before,after in changes:tree=replace_expression(tree,before,after)
        exec(compile(tree,module.__file__,'exec'),module.__dict__)
    return module
