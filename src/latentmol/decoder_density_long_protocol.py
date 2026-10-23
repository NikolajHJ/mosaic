"""Authenticated metadata-only100epoch extension of the source077 study."""
import json
from pathlib import Path
import re

from latentmol.decoder_density_protocol import read_protocol, cases
from latentmol.experiments import file_sha256, source_manifest

FIELDS = {'version', 'base_release', 'base_archive_sha256', 'base_manifest_sha256',
    'base_config_sha256', 'base_campaign_root', 'base_receipts_root', 'endpoint_epochs',
    'milestones', 'segment_budget_seconds', 'evaluation_reserve_seconds', 'gate_budget_seconds',
    'prefix_policy', 'checkpoint_keep_recent', 'archive_evaluation_rows'}


def read_long_protocol(path, expected_sha):
    path = Path(path)
    if file_sha256(path) != expected_sha:
        raise ValueError('Long density protocol hash mismatch')
    return validate_long_protocol(json.loads(path.read_text()))


def validate_long_protocol(value):
    if set(value) != FIELDS or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('Unknown long density fields/version')
    for key in ('base_archive_sha256', 'base_manifest_sha256', 'base_config_sha256'):
        if not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError('Long study needs authenticated source/config lineage')
    for key in ('base_release', 'base_campaign_root', 'base_receipts_root'):
        if not isinstance(value[key], str) or not value[key].startswith('/work3/'):
            raise ValueError('Long study paths must be declared DTU work3 paths')
    if (type(value['endpoint_epochs']) is not int or value['endpoint_epochs'] != 100
            or value['milestones'] != [25, 50, 75, 100]
            or any(type(epoch) is not int for epoch in value['milestones'])):
        raise ValueError('Prescribed cumulative milestones are25/50/75/100')
    for key in ('segment_budget_seconds', 'evaluation_reserve_seconds', 'gate_budget_seconds'):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError('Invalid long-study budget')
    if value['segment_budget_seconds'] + value['evaluation_reserve_seconds'] > 21 * 3600:
        raise ValueError('Reserve two-hour allocation headroom')
    if value['gate_budget_seconds'] > 3600:
        raise ValueError('Native gate must fit its two-hour allocation')
    if (value['prefix_policy'] != 'completed10_or_original'
            or type(value['checkpoint_keep_recent']) is not int or value['checkpoint_keep_recent'] != 2
            or value['archive_evaluation_rows'] is not True):
        raise ValueError('Require declared prefix fallback and lossless bounded checkpoint storage')
    return value


def authenticate_base(protocol):
    """Authenticate original immutable source, protocol and successful gate."""
    release = Path(protocol['base_release'])
    if file_sha256(Path(str(release) + '.tar.gz')) != protocol['base_archive_sha256']:
        raise ValueError('Base source archive changed')
    if source_manifest(release)['sha256'] != protocol['base_manifest_sha256']:
        raise ValueError('Base executable manifest changed')
    base = read_protocol(release / 'configs/decoder_density_study.json', protocol['base_config_sha256'])
    gate = json.loads((Path(protocol['base_campaign_root']) / 'gate/passed.json').read_text())
    if (gate['status'] != 'passed' or gate['protocol'] != base
            or gate['source_archive_sha256'] != protocol['base_archive_sha256']):
        raise ValueError('Base native gate identity changed')
    return base


def segments(base, protocol):
    """Four trajectories × four segments, serialized in declared case order."""
    return [{**case, 'stop_epoch': epoch, 'start_epoch': 10 if number == 0 else protocol['milestones'][number - 1],
             'segment_index': number, 'label': case['label'] + f'-e{epoch:03d}'}
            for case in cases(base) for number, epoch in enumerate(protocol['milestones'])]
