"""Select an actual checkpoint branch without changing saved source records."""
from copy import deepcopy
from pathlib import Path
import json
import quick_inputs as inputs


def capture(segment, spec):
    if 'through_global_update' not in spec:
        return segment
    end = spec['through_global_update'];offset = segment['global_offset']
    count = segment['expected_counts'];keep = end-offset
    inputs.require(type(end) is int and 0 < keep <= count['committed_updates'], 'Invalid saved prefix boundary')
    retained = 0;boundary = None
    with Path(segment['states']['path']).open() as stream:
        for index,line in enumerate(stream):
            if index > keep:break
            inputs.require(line.endswith('\n'), 'Uncommitted prefix row')
            row=json.loads(line)
            inputs.require(row['step']==index and row.get('global_step',offset+index)==offset+index,
                           'Noncontiguous selected prefix')
            inputs.require(type(row.get('snapshot_retained')) is bool, 'Prefix requires original retained markers')
            retained+=int(row['snapshot_retained']);boundary=row
    inputs.require(boundary is not None and boundary['step']==keep and boundary['snapshot_retained'],
                   'Prefix must end at an actual retained checkpoint')
    result=deepcopy(segment)
    result.update(report_stop_global=end,source_expected_counts=count,
                  expected_counts={'committed_updates':keep,'scalar_states':keep+1,'retained_frames':retained})
    return result


def collect(result, segment):
    """Called only by allocated-CPU collectors after validating original data."""
    if 'report_stop_global' not in segment:return result
    import numpy as np
    import report_data as data
    end=segment['report_stop_global'];keep=end-segment['global_offset']
    data.require(0<keep<=len(result['updates']), 'Collected prefix exceeds source')
    retained=[r for r in result['retained'] if r[0]['global_step']<=end]
    data.require(retained and retained[-1][0]['global_step']==end, 'Prefix checkpoint is absent')
    frame=retained[-1][0]
    with np.load(data.bound_file(frame['source_snapshot']),allow_pickle=False) as arrays:
        point={name:arrays[name].copy() for name in ('X','E_upper','atom_logits','bond_logits','key_data')}
    proof=deepcopy(result['proof'])
    proof.update(committed_updates=keep,retained_frames=len(retained),
                 report_prefix={'through_global_update':end,'source_updates':len(result['updates']),
                                'excluded_tail_updates':len(result['updates'])-keep,
                                'boundary_snapshot':frame['source_snapshot'],
                                'scope':'Actual saved prefix; later source tail remains untouched'})
    return {**result,'states':result['states'][:keep+1],'updates':result['updates'][:keep],
            'retained':retained,'last_arrays':point,'proof':proof}
