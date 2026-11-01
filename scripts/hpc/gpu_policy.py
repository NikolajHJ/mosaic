"""Explicit GPU profiles for queue-distributed exploratory plans.

New batch planners assign compatible profiles under docs/GPU_POLICY.md. This
module does not discover queues, balance jobs or alter existing allocations.
"""
import os
import re

# Compatibility fallback for older callers; new batch plans declare placements.
DEFAULT_PROFILE='v100-32gb-v1'
PROFILES={
    'v100-32gb-v1':{'queue':'gpuv100','family':'V100','minimum_memory_MiB':30000,
        'resource':'num=1:mode=shared:j_exclusive=yes:mps=no:gmem=30G'},
    'l40s-48gb-v1':{'queue':'gpul40s','family':'L40S','minimum_memory_MiB':45000,
        'resource':'num=1:mode=shared:j_exclusive=yes:mps=no'},
    'a10-24gb-v1':{'queue':'gpua10','family':'A10','minimum_memory_MiB':22000,
        'resource':'num=1:mode=shared:j_exclusive=yes:mps=no:gmem=22G'},
    'a100-40gb-v1':{'queue':'gpua100','family':'A100','minimum_memory_MiB':39000,
        'resource':'num=1:mode=shared:j_exclusive=yes:mps=no:gmem=38G'},
    'a40-48gb-v1':{'queue':'gpua40','family':'A40','minimum_memory_MiB':45000,
        'resource':'num=1:mode=shared:j_exclusive=yes:mps=no:gmem=44G'},
}


def profile(name=None):
    name=DEFAULT_PROFILE if name is None else name
    if name not in PROFILES:raise ValueError('Unknown explicit GPU profile: '+str(name))
    return {'id':name,**PROFILES[name]}


def selected(row):
    name=row.get('gpu_profile')
    if name is None and row.get('queue'):
        matches=[k for k,v in PROFILES.items() if v['queue']==row['queue']]
        if len(matches)!=1:raise ValueError('Unknown GPU queue')
        name=matches[0]
    result=profile(name)
    if row.get('queue',result['queue'])!=result['queue']:raise ValueError('GPU queue/profile conflict')
    return result


def resource(row):
    expected=selected(row);value=row.get('gpu_request',expected['resource'])
    if expected['family']=='V100':
        fields=[x.split('=',1)[1] for x in value.split(':') if x.startswith('gmem=')]
        if not fields:value+=':gmem=30G'
        elif fields!=['30G']:raise ValueError('The 32 GB V100 profile requires gmem=30G')
    return value


def legacy_adapter_profile():
    # Standalone historical adapter calls remain explicitly L40S unless a new
    # launch supplies its assigned per-arm profile through MOSAIC_GPU_PROFILE.
    return profile(os.environ.get('MOSAIC_GPU_PROFILE','l40s-48gb-v1'))


def validate_hardware(name,total_mib,selection=None):
    expected=profile() if selection is None else selection
    matches = expected['family'] in name
    if expected['family'] == 'A10':
        matches = re.search(r'(?<![A-Za-z0-9])A10(?![A-Za-z0-9])', name) is not None
    if not matches or int(total_mib)<expected['minimum_memory_MiB']:
        raise ValueError('GPU kind/memory differs from declared profile')
    return {'profile':expected['id'],'name':name,'memory_total_MiB':int(total_mib)}
