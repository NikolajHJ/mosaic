"""Allocated CPU display geometry for already-saved external raw graphs.

Uses the existing Mosaic report layout, with no chemistry or model evaluation.
"""
import hashlib
import json
import os
from pathlib import Path

from report_data import layout_raw_graph
from storage_accounting import assess_output


def signature(graph):
    return json.dumps([sorted(a['slot'] for a in graph['atoms']),
                      sorted([min(b['i'], b['j']), max(b['i'], b['j']), b['type']]
                             for b in graph['bonds'])], separators=(',', ':'))


def main():
    root = Path(__file__).resolve().parent
    assert os.environ.get('LSB_JOBID'), 'An allocated job is required'
    (root / 'capacity.json').write_text(json.dumps(assess_output(root, 16 * 1024**2, 128)))
    source = (root / 'graphs.json').read_bytes()
    packet = json.loads(source)
    layouts = {}
    for digest, graph in packet['graphs'].items():
        assert hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == digest
        nodes = [{'id': a['slot']} for a in graph['atoms']]
        edges = [{'source': b['i'], 'target': b['j'], 'type': b['type']} for b in graph['bonds']]
        result = layout_raw_graph(nodes, edges)
        assert result['edges'] == edges
        assert [n['id'] for n in result['nodes']] == [n['id'] for n in nodes]
        key = signature(graph)
        if key in layouts:
            assert layouts[key] == result
        layouts[key] = result
    output = {'version': 'mosaic-external-display-layout-v1', 'job_id': os.environ['LSB_JOBID'],
              'source_html_sha256': packet['source_html_sha256'],
              'input_sha256': hashlib.sha256(source).hexdigest(),
              'layout_source_sha256': hashlib.sha256((root / 'report_data.py').read_bytes()).hexdigest(),
              'graphs': len(packet['graphs']), 'layouts': layouts,
              'identity_preserved': True, 'scope': 'Display geometry only; no repair, scoring or molecular pose'}
    with (root / 'layouts.json').open('x') as stream:
        json.dump(output, stream, sort_keys=True)
    print(json.dumps({'status': 'passed', 'job_id': os.environ['LSB_JOBID'], 'graphs': output['graphs']}))


if __name__ == '__main__':
    main()
