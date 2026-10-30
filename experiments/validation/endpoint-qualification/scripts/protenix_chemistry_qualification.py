"""Fresh complete chemistry qualification with an independent combined oracle."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(s);sys.modules[name]=v;s.loader.exec_module(v);return v


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--pin',required=True)
    p.add_argument('--mode',choices=('startup','chemistry'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import failure_diagnostic_ops as ops
    plan=ops.read(a.plan)
    assert ops.sha(a.plan)==a.pin and os.environ.get('LSB_JOBID','').isdigit()
    assert plan['combined_reference_policy']=='protenix-fixed-chemistry-full-reference-v1'
    q=load(Path(plan['qualified_helper']['path'])/'scripts/continuation_failure_diagnostics.py','_full_chemistry_validation')
    q.validate_policy(plan)
    helper=ops.helper(plan);helper.CAP=q.CAP
    if a.mode=='startup':
        q.startup(helper,plan,a.output)
        return
    helper.guard_candidate_environment(os.environ)
    source=Path(plan['chemistry_source']['path']);sys.path.insert(0,str(source/'scripts'))
    import chemistry_continuation_hooks as hooks
    original=ops.read(source/'plan.json');original['_sha256']=ops.sha(source/'plan.json')
    precision=load(Path(plan['capacity_source']['path'])/'scripts/precision_continuation_hooks.py','_combined_reference_only_RPC')
    from protenix_combined_reference import install
    with install(hooks,original,precision):
        q.full_child(helper,plan,a.output,'chemistry')


if __name__=='__main__':main()
