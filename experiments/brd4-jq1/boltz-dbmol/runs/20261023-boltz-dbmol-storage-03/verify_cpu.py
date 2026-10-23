"""Independent stdlib audit of completion-aware launcher03 CPU delivery.

Old failed CPU jobs stay historical. Current IDs come from the authenticated
storage-adoption-03/submission-complete.json, never from a guessed job number.
Admission-only and failure scopes cannot establish full CPU or native success.
"""
import argparse
import ast
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT = BASE + '/boltz-dbmol-pilot-01'
SCIENCE = BASE + '/boltz-dbmol-pilot-release-01/source'
LAUNCHER = BASE + '/boltz-dbmol-storage-release-03/source'
PREVIOUS_LAUNCHER = BASE + '/boltz-dbmol-storage-release-02/source'
PREVIOUS_MANIFEST = '5f479f2e5342469efd8606c9457c946a51dbcf0dd90d837ee472babfdd820a4b'
PREVIOUS_ARCHIVE = 'a26dafcac4fbc342a275dedb1c93ed8880062f9e3dafe48d188937c3137dd750'
MANIFEST = '24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
ARCHIVE = 'ec333fd3e0b33d9e5b82e0cd5e4e12c14ffb4f91782c18cd820d4bfab53de1f8'
CONFIG = 'b3b56f409a59b3b1d00228bb7dcd3f74fc08f17fdf06facfc1469922778d5497'
ORIGINAL_MANIFEST = 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
SUITE = 'mosaic-dbmol-pilot-native-components-v1'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
REFERENCE = '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
GIB = 1024 ** 3
OLD_COUNTS = {'hpc-contract': 500, 'legacy': 239, 'native-mapping': 34, 'chemistry': 59}
NEW_COUNTS = {'dbmol-native': 83, 'dbmol-transport': 148, 'launcher-adoption': 72}
NEW_MODULES = {'dbmol-native': ('test_dbmol_objective', 'test_dbmol_native_paths'),
               'dbmol-transport': ('test_dbmol_gradient_contract', 'test_dbmol_bridge', 'test_dbmol_pilot_contract'),
               'launcher-adoption': ('fixture_storage_adoption',)}
ASSETS = {
    'affinity': ('/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt', 'dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e'),
    'structural': ('/work3/s234818/latentmol/cache/boltz2/boltz2_conf.ckpt', '090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1'),
    'prepared_context': ('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared/context.pt', 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093')}
OLD_FAILED = {
    'cpu/submission.json': 'bbc02e915d3cda0d58cfcfe0806b7c5544df9328214bfabb319669ea27c7103d',
    'after-cpu/submission.json': 'f22d46432171ff8cf5b0fc3ceae8d05b65309dd5185777fa103338f8e7f558e9',
    'cpu/stdout.29798549.log': 'c1b559a85182bf1fd485e987480c43323a2a7538ea22fd58e51f5df21dc3a33e',
    'after-cpu/stdout.29798550.log': '373813d3d4faabc87af19299827c1983ef62de6b2e43fab08dc3542a7cbcd2cf',
    'cpu/storage-cpu-entry/gate.json': '0cb5b96b49ebefd8f483e882d3b796d040554e9c60b7e1f2ee194fceda2e8afd'}
PREVIOUS_FAILED = {
    'cpu-02/submission.json': 'cb0d36f854b15cda59f0261366c209ec89f52f9ee57f6045cdc0758904c0bc0e',
    'after-cpu-02/submission.json': '5b79746f9331ebb3a8eb5da7d16fe9798ae3aa32a3de256ed344efdd63071dd1',
    'cpu-02/stdout.29798699.log': '981fe2fa535f613a131d31aaafc8bfea733829282065c3307e17bcaa7ecf019d',
    'after-cpu-02/stdout.29798700.log': '66d1335c49c5672c82f33b200093cd30a303901a65645631a714b9d20fdf3fa8',
    'cpu-02/storage-cpu-entry/gate.json': '7e5a75a5a0b132df3ee2f8eb164b298be95cba30950f9de4d8c7945a96498b49',
    'cpu-02/payload/hpc-contract.xml': '00251c0632f2cec8b97e17140899192bb164ba34f97887e96b43c2d6e5fded71',
    'cpu-02/payload/hpc-contract.log': 'b074fd8adb9b31db1f9ad34acde6034f96b2e2b0fb47f9e27971cbe7244c07fc',
    'cpu-02/payload/legacy.xml': '7e04fb5fac25026b5550273a402f2b2c4c3a1b8734f5e942f5bd345e42622853',
    'cpu-02/payload/legacy.log': 'a92649d51ff960fba2d3812290eb16d448c4ed9a9bf3cb0a612b7edbbacd4e45',
    'cpu-02/payload/native-mapping.xml': 'e79c101288fb95d8544fbac5a3b379305e7856de92d70772203b88655c42b5fa',
    'cpu-02/payload/native-mapping.log': '35d2cf6a71308872d5b4b2af41d250375ce1559b123380fb9623f10f4dc7da01',
    'cpu-02/payload/chemistry.xml': '97922866e2a72d497ed385c57aae7a9344176a9ca606c9b484907952293fb6b2',
    'cpu-02/payload/chemistry.log': '9e3d662a45940d2dbf867d98dd388b05c0213b4bc1da2973f556788b95cb60c8',
    'cpu-02/payload/dbmol-native.xml': 'ca8246b0d0976a0409e9915e094d49552f29fdb86905a80adf8107a2c5212d9f',
    'cpu-02/payload/dbmol-native.log': '25723fb6d2b940eddf4016910ea200bb69b8956d9ad02d251f2b5bf13afeb24c',
    'cpu-02/payload/dbmol-transport.xml': 'beda98b250233d30431319fe1fd43f2d7d4b5cbcfbc49e43a81cc0f20dba452a',
    'cpu-02/payload/dbmol-transport.log': 'e7e6f66f2563a27c789a6a970c4334e2d3d1f2df0d4e631e5a6f16e159c6563f'}
HISTORICAL_JOBS = {'cpu': '29798549', 'after-cpu': '29798550', 'cpu-02': '29798699', 'after-cpu-02': '29798700'}
PREVIOUS_FAILURE_NAMES = [
    'test_config_rejects_undeclared_policy_changes[extra]',
    'test_config_rejects_undeclared_policy_changes[seed]',
    'test_config_rejects_undeclared_policy_changes[steps]',
    'test_config_rejects_undeclared_policy_changes[temperature]',
    'test_peer_dispatch_waits_only_actual_authenticated_own_ended_callback_ids',
    'test_raw_checker_rejects_missing_wrong_or_nonfinite_derivative_paths[nonfinite-E_upper]',
    'test_raw_checker_rejects_missing_wrong_or_nonfinite_derivative_paths[nonfinite-X]']
TERMINAL = {
    'protenix': {'job': '29792670', 'status': 'passed', 'summary': (298580732, 'b94d4bae58b7c82b92f4cf0b6b0e9804734afff0f54ea0b162b65d825ec2661d'), 'stdout': (2201, '46f7ff9c942cf3ca9447e4d4d2630a3e604d8a6cd42504b56e8a0fcd6bb3170e'), 'submission': (1808, '5dc53846d086b2b6522d1d4aeb42c3461e4875f890b547b8362904793f33e58a')},
    'boltz2': {'job': '29792671', 'status': 'failed', 'summary': (301971481, '06a68008a0dfb8b6a66668447bcda4f01c7b54427d9b68f0d84e5926db278cc1'), 'stdout': (2199, 'b40c2741e3d500f74aafa8819a597d260b7372f5372eb2642aeb6c450341f5c0'), 'submission': (1798, 'ddc416a8ed853307b639e9011fefadbb484d2721dfd3e3f9d054045c1eee8df0')}}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def safe(name):
    p = PurePosixPath(name)
    require(p.parts and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name, 'Unsafe archive member')


def archive_members(path, expected_bytes, expected_sha, manifest_sha, count):
    require(path.stat().st_size == expected_bytes and digest(path.read_bytes()) == expected_sha, 'Local complete source archive differs: ' + str(path))
    members = {}
    with tarfile.open(path) as archive:
        for member in archive:
            safe(member.name)
            require(member.isfile() and member.name.startswith('source/'), 'Nonregular/foreign source member')
            name = member.name.removeprefix('source/')
            require(name not in members, 'Duplicate source member')
            members[name] = archive.extractfile(member).read()
    raw = members.pop('source_manifest.json')
    manifest = json.loads(raw)
    require(digest(raw) == manifest_sha and len(members) == count and set(members) == set(manifest['files']), 'Source manifest/member scope differs')
    for name, data in members.items():
        require(digest(data) == manifest['files'][name], 'Source member changed: ' + name)
    return raw, manifest, members


def scheduler(raw, job, queue, expected=None):
    compact = re.sub(r'\s+', '', raw)
    require(re.findall(r'Job<([^>]+)>', compact) == [job], 'Wrong/ambiguous actual scheduler job')
    require(re.findall(r'Queue<([^>]+)>', compact) == [queue], 'Wrong actual scheduler queue')
    states = re.findall(r'Status<([^>]+)>', compact)
    require(len(states) == 1, 'Missing unique actual scheduler state')
    if expected is not None:
        require(states == [expected], 'Wrong actual final scheduler state')
    return states[0]


def lsf(raw, job, passed, failure_code=1):
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', raw, re.M) == [(job, 'Done' if passed else 'Exited')], 'Wrong same-job final LSF subject')
    require(('Successfully completed.' in raw and 'Exited with exit' not in raw) if passed else bool(re.search(r'Exited with exit code\s+' + (r'\d+' if failure_code is None else str(failure_code)) + r'\b', raw)), 'Missing exact same-job LSF result')


def resources(phase):
    if phase == 'cpu':
        return ['-q', 'hpc', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '01:00']
    if phase in ('native', 'segment'):
        return ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00' if phase == 'native' else '23:00']
    require(phase in ('after-cpu', 'after-native', 'after-segment', 'dispatch'), 'Unknown actual phase')
    return ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']


def vector(label, phase, dependency, launcher_manifest, historical=False):
    parent = ROOT + '/' + label
    result = ['bsub', *resources(phase), '-J', 'mosaic-dbmol-' + label, '-oo', parent + '/stdout.%J.log', '-eo', parent + '/stderr.%J.log']
    if dependency:
        result += ['-w', dependency]
    if historical and label in ('cpu', 'after-cpu'):
        result += ['bash', '-l', SCIENCE + '/scripts/hpc/' + ('dbmol_cpu.lsf' if phase == 'cpu' else 'dbmol_stage.lsf'), SCIENCE, ROOT, MANIFEST]
    else:
        source = PREVIOUS_LAUNCHER if historical else LAUNCHER
        manifest = PREVIOUS_MANIFEST if historical else launcher_manifest
        result += ['bash', '-l', source + '/scripts/hpc/' + ('dbmol_storage_cpu.lsf' if phase == 'cpu' else 'dbmol_storage_stage.lsf'), source, manifest, SCIENCE, ROOT, MANIFEST]
    if phase != 'cpu':
        result += [phase, parent]
    return result


def declared_tests(data):
    tree = ast.parse(data)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
            names.add(node.name)
        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            names.update(child.name for child in node.body if isinstance(child, ast.FunctionDef) and child.name.startswith('test_'))
    return names


def cpu_groups(folder, science, launcher, files):
    counts, headers, coverage = {}, {}, {}
    wrapper = launcher['scripts/hpc/dbmol_storage_cpu.lsf'].decode()
    modules = set(re.findall(r'tests/(test_[A-Za-z0-9_]+)\.py', wrapper.split('contracts=(', 1)[1].split(')', 1)[0]))
    old_modules = {
        'hpc-contract': modules,
        'native-mapping': {'test_boltz_native_mapping', 'test_campaign25_native', 'test_campaign25_contract', 'test_pocket_boltz'},
        # The three inherited intentional slow/model deselections occupy these
        # two modules; they are absent from the239 allocated legacy rows.
        'legacy': {PurePosixPath(n).stem for n in science if n.startswith('tests/test_small_molecule_') and n.endswith('.py') and not PurePosixPath(n).stem.startswith('test_small_molecule_chemistry') and PurePosixPath(n).stem not in {'test_small_molecule_integration', 'test_small_molecule_contact_native'} and declared_tests(science[n])},
        'chemistry': {PurePosixPath(n).stem for n in science if n.startswith('tests/test_small_molecule_chemistry') and n.endswith('.py') and declared_tests(science[n])}}
    require('native=/work3/s234818/latentmol/env/boltz311/bin/python' in wrapper and 'python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python' in wrapper, 'Two declared CPU environments differ')
    require('"$native" -m pytest tests/test_dbmol_objective.py tests/test_dbmol_native_paths.py' in wrapper, 'Native dependency environment differs')
    require('"$python" -m pytest' in wrapper and '"$launcher/tests/test_dbmol_gradient_contract.py"' in wrapper and '"$launcher/tests/test_dbmol_pilot_contract.py"' in wrapper and 'tests/test_dbmol_bridge.py' in wrapper, 'Corrected transport fixture environments/paths differ')
    require('export CUDA_VISIBLE_DEVICES=\'\' JAX_PLATFORMS=cpu JAX_ENABLE_X64=0' in wrapper and 'export JAX_ENABLE_X64=1' in wrapper, 'Declared CPU/JAX modes differ')
    for group in (*OLD_COUNTS, *NEW_MODULES):
        name = 'run/cpu-03/payload/' + group + '.xml'
        require(name in files, 'Missing allocated raw XML: ' + group)
        tree = ET.parse(folder / name)
        cases = list(tree.iter('testcase'))
        require(cases and not any(list(tree.iter(kind)) for kind in ('failure', 'error', 'skipped')), 'Allocated failure/error/skip exists')
        count = len(cases)
        counts[group] = count
        headers[group] = sum(int(node.get('tests', '0')) for node in tree.iter('testsuite'))
        log = (folder / name).with_suffix('.log').read_text(encoding='utf8')
        require(re.search(r'\b' + str(count) + r'\s+passed\b', log), 'Actual XML row/log count mismatch')
        if group in OLD_COUNTS:
            require(count == OLD_COUNTS[group], 'Inherited832 exact scope differs')
        else:
            require(count == NEW_COUNTS[group], 'Immutable new fixture actual row scope differs')
        require(headers[group] == count + (3 if group == 'hpc-contract' else 0), 'XML header discrepancy exceeds inherited three unittest subtests')
        if group == 'legacy':
            require(re.search(r'\b3\s+deselected\b', log), 'Historical legacy deselections differ')
        else:
            require(not re.search(r'\b[1-9]\d*\s+deselected\b', log), 'Unexpected deselections')
        if group in old_modules:
            require(old_modules[group] <= {part for case in cases for part in case.get('classname', '').split('.')}, 'Inherited exact module attribution incomplete: ' + group)
            require(all(len(old_modules[group].intersection(case.get('classname', '').split('.'))) == 1 for case in cases), 'Foreign/ambiguous inherited testcase attribution: ' + group)
        if group in NEW_MODULES:
            wanted = set(NEW_MODULES[group]);coverage[group] = {}
            for module in wanted:
                corrected = module in ('fixture_storage_adoption', 'test_dbmol_gradient_contract', 'test_dbmol_pilot_contract')
                data = launcher['tests/' + module + '.py'] if corrected else science['tests/' + module + '.py']
                expected = declared_tests(data)
                selected = [case for case in cases if module in case.get('classname', '').split('.')]
                actual = {case.get('name', '').split('[', 1)[0] for case in selected}
                require(expected and selected and expected == actual, 'Source/function testcase coverage incomplete: ' + module)
                coverage[group][module] = {'source_functions': len(expected), 'actual_cases': len(selected)}
            require(all(len(wanted.intersection(case.get('classname', '').split('.'))) == 1 for case in cases), 'Ambiguous/foreign new module attribution')
    return counts, headers, coverage


def quota_from_raw(raw):
    pools = []
    for line in raw.splitlines():
        if not re.match(r'^\s*s234818\|306121\|\|', line):
            continue
        m = re.fullmatch(r'\s*s234818\|306121\|\|\s*([\d.]+)\s+(Byte|GiB)\|\s*([\d.]+)\s+(Byte|GiB)\|\|\s*(\d+)\|\s*(\d+)\s*', line)
        if not m:
            require('unlimited' in line, 'Malformed finite quota row')
            continue
        used = int((Decimal(m[1]) * (GIB if m[2] == 'GiB' else 1)).to_integral_value(rounding=ROUND_CEILING))
        hard = int((Decimal(m[3]) * (GIB if m[4] == 'GiB' else 1)).to_integral_value(rounding=ROUND_FLOOR))
        # The unchanged declared quota parser rounds used GiB upward and hard
        # GiB downward before multiplication, rather than rounding raw bytes.
        if m[2] == 'GiB': used = int(Decimal(m[1]).to_integral_value(rounding=ROUND_CEILING)) * GIB
        if m[4] == 'GiB': hard = int(Decimal(m[3]).to_integral_value(rounding=ROUND_FLOOR)) * GIB
        pools.append({'used_bytes_upper': used, 'hard_bytes_lower': hard, 'remaining_bytes': hard - used, 'remaining_files': int(m[6]) - int(m[5])})
    require(pools, 'No authenticated finite quota pool')
    return {'finite_pools': pools, 'available_bytes_lower': min(p['remaining_bytes'] for p in pools), 'available_files_lower': min(p['remaining_files'] for p in pools)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edition', default='01')
    parser.add_argument('--expect-failure', action='store_true')
    parser.add_argument('--admission-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(re.fullmatch(r'\d{2}', args.edition) and not (args.admission_only and args.expect_failure), 'Use one explicit scope and a fresh two-digit edition')
    base = REPO / 'results/hpc' / HERE.name
    folder = base / ('cpu-evidence-' + args.edition)
    output = args.output or HERE / ('admission-verification-' if args.admission_only else 'cpu-failure-verification-' if args.expect_failure else 'cpu-verification-')
    if args.output is None: output = output.with_name(output.name + args.edition + '.json')
    require(not output.exists(), 'Never overwrite earlier verification')
    metadata = read(folder / 'metadata.json');receipt = read(folder / 'delivery-receipt.json')
    archive = base / ('cpu-evidence-' + args.edition + '.tar.gz')
    require(Path(receipt['archive']).resolve() == archive.resolve() and archive.stat().st_size == receipt['bytes'] and digest(archive.read_bytes()) == receipt['sha256'], 'Compact archive path/length/hash differs')
    require(digest((folder / 'metadata.json').read_bytes()) == receipt['metadata_sha256'], 'Evidence metadata changed')
    files = metadata['files'];remote = {};seen = set()
    for name, record in files.items():
        safe(name)
        part = name.split('/', 1)
        parent = {'run': ROOT, 'science': SCIENCE, 'launcher': LAUNCHER, 'external': BASE}.get(part[0])
        require(parent and len(part) == 2 and record['path'] == parent + '/' + part[1] and record['path'] not in remote, 'Foreign/ambiguous remote proof path')
        data = (folder / name).read_bytes()
        require(len(data) == record['bytes'] and digest(data) == record['sha256'], 'Delivered proof bytes changed: ' + name)
        remote[record['path']] = record
    with tarfile.open(archive) as tar:
        for member in tar:
            safe(member.name)
            require(member.isfile() and member.name not in seen and member.name in set(files) | {'metadata.json'}, 'Unsafe/unexpected/duplicate compact member')
            seen.add(member.name)
            require(tar.extractfile(member).read() == (folder / member.name).read_bytes(), 'Archive/extracted file bytes differ')
    require(seen == set(files) | {'metadata.json'} and receipt['listed_files'] == len(files) and receipt['safe_members'] == len(seen), 'Incomplete compact evidence')
    package = read(HERE / 'package.json');launcher_sha = package['manifest_sha256']
    require(package['files'] == 7 and package['scientific_source_changed'] is False and package['scientific_source_manifest_sha256'] == MANIFEST, 'Launcher package changed scientific scope')
    lraw, lmanifest, launcher = archive_members(base / 'source01/source.tar.gz', package['archive_bytes'], package['archive_sha256'], launcher_sha, 7)
    science_base = REPO / 'results/hpc/20261023-boltz-dbmol-pilot-01'
    sraw, smanifest, science = archive_members(science_base / 'source01/source.tar.gz', 39676321, ARCHIVE, MANIFEST, 657)
    require(lmanifest['scientific_source_manifest_sha256'] == MANIFEST and lmanifest['version'] == 'mosaic-dbmol-fixture-adoption-v1' and lmanifest['previous_failed_launcher_manifest_sha256'] == package['previous_failed_launcher_manifest_sha256'] == PREVIOUS_MANIFEST, 'Wrong immutable fixture-only launcher adoption')
    require((folder / 'science/source_manifest.json').read_bytes() == sraw and (folder / 'launcher/source_manifest.json').read_bytes() == lraw, 'Delivered dual manifests differ from full local source archives')
    for label, count, sha in [('science', 657, MANIFEST), ('launcher', 7, launcher_sha)]:
        require(metadata['source_integrity'][label] == {'status': 'passed', 'manifest_sha256': sha, 'files': count, 'readonly': True}, 'Remote complete dual-source seal failed')
    for name, data in launcher.items():
        require((folder / 'launcher' / name).read_bytes() == data, 'Delivered launcher file differs')
    praw, pmanifest, previous_launcher = archive_members(REPO / 'results/hpc/20261023-boltz-dbmol-storage-02/source01/source.tar.gz', 20406, PREVIOUS_ARCHIVE, PREVIOUS_MANIFEST, 5)
    require((folder / 'external/boltz-dbmol-storage-release-02/source/source_manifest.json').read_bytes() == praw, 'Historical five-file launcher manifest changed')
    require(metadata['source_integrity']['previous_launcher'] == {'status': 'passed', 'manifest_sha256': PREVIOUS_MANIFEST, 'files': 5, 'readonly': True}, 'Historical launcher02 seal changed')
    for name, data in previous_launcher.items():
        require((folder / 'external/boltz-dbmol-storage-release-02/source' / name).read_bytes() == data, 'Historical launcher02 file changed')
    require(set(launcher) == set(previous_launcher) | {'tests/test_dbmol_gradient_contract.py', 'tests/test_dbmol_pilot_contract.py'}, 'Launcher03 member changes extend beyond two declared fixtures')
    require(launcher['scripts/hpc/dbmol_storage_stage.lsf'] == previous_launcher['scripts/hpc/dbmol_storage_stage.lsf'], 'Scientific stage wrapper changed during fixture adoption')
    for module in ('test_dbmol_gradient_contract', 'test_dbmol_pilot_contract'):
        require(declared_tests(launcher['tests/' + module + '.py']) == declared_tests(science['tests/' + module + '.py']), 'Fixture correction removed or added test functions')
    require(digest(science['examples/configs/dbmol_pilot.json']) == CONFIG, 'Pilot configuration changed')
    for name in ('examples/configs/dbmol_pilot.json', 'scripts/hpc/dbmol_cpu.lsf', 'scripts/hpc/dbmol_stage.lsf'):
        require((folder / 'science' / name).read_bytes() == science[name], 'Science config/wrapper changed')
    for key, (path, sha) in ASSETS.items():
        ref = metadata['asset_references'][key]
        require(ref['path'] == path and ref['sha256'] == sha and type(ref['bytes']) is int and ref['bytes'] > 0 and ref['downloaded'] is False, 'Frozen remote asset reference changed')
    for name, sha in OLD_FAILED.items():
        require(digest((folder / 'run' / name).read_bytes()) == sha, 'Historical failed CPU proof changed')
    for name, sha in PREVIOUS_FAILED.items():
        require(digest((folder / 'run' / name).read_bytes()) == sha, 'Historical failed CPU02 proof changed')
    require(read(folder / 'run/cpu/storage-cpu-entry/gate.json')['status'] == 'failed' and not any(name.startswith('run/cpu/payload/') and name.endswith('.xml') for name in files), 'Historical failed CPU was relabeled/reexecuted')
    for label, job in [('cpu', '29798549'), ('after-cpu', '29798550')]:
        lsf((folder / 'run' / label / ('stdout.' + job + '.log')).read_text(), job, False)
    for label, job in [('cpu-02', '29798699'), ('after-cpu-02', '29798700')]:
        lsf((folder / 'run' / label / ('stdout.' + job + '.log')).read_text(), job, False)
    require(read(folder / 'run/cpu-02/storage-cpu-entry/gate.json')['status'] == 'passed', 'CPU02 capacity pass changed')
    for group, count in {**OLD_COUNTS, 'dbmol-native': 83, 'dbmol-transport': 148}.items():
        tree = ET.parse(folder / ('run/cpu-02/payload/' + group + '.xml'));cases = list(tree.iter('testcase'))
        names = sorted(c.get('name') for c in cases if c.find('failure') is not None)
        require(len(cases) == count and not any(list(tree.iter(k)) for k in ('error', 'skipped')) and names == (PREVIOUS_FAILURE_NAMES if group == 'dbmol-transport' else []), 'Historical seven CPU02 fixture failures changed or were relabeled')
    require(re.search(r'\b7 failed, 141 passed\b', (folder / 'run/cpu-02/payload/dbmol-transport.log').read_text()) and 'run/cpu-02/payload/launcher-adoption.xml' not in files, 'Historical CPU02 failed/unexecuted scope changed')
    require(digest((folder / 'external/restart-lrreset-release-04/source/source_manifest.json').read_bytes()) == ORIGINAL_MANIFEST, 'Original scientific625 manifest changed')
    refs = metadata['remote_only_references']
    for arm, pin in TERMINAL.items():
        parent = BASE + '/restart-lrreset-04/run-' + arm
        original = metadata['original_terminal_metadata'][arm]
        require(original == {'status': pin['status'], 'cases': 1, 'case_status': 'passed', 'updates': 10000, 'states': 10001, 'large_original_bytes_downloaded': False}, 'Original terminal root metadata differs')
        for kind in ('summary', 'stdout', 'submission'):
            path = parent + ('/payload/study/summary.json' if kind == 'summary' else '/stdout.' + pin['job'] + '.log' if kind == 'stdout' else '/submission.json')
            rec = refs[path] if kind == 'summary' else remote[path]
            require((rec['bytes'], rec['sha256']) == pin[kind], 'Exact original terminal proof changed')
        text = (folder / ('external/restart-lrreset-04/run-' + arm + '/stdout.' + pin['job'] + '.log')).read_text()
        lsf(text, pin['job'], arm == 'protenix')
    control = read(folder / 'run/storage-adoption-03/started.json')
    require(control['launcher_manifest_sha256'] == launcher_sha and control['science_manifest_sha256'] == MANIFEST and control['archive_bytes'] == package['archive_bytes'] and control['archive_sha256'] == package['archive_sha256'], 'Actual staged launcher identity differs')
    submission_name = 'run/storage-adoption-03/submission-complete.json'
    record = read(folder / submission_name) if submission_name in files else None
    jobs = read(folder / 'run/jobs.json');observed = {};gates = {};budget_details = {}
    require(all(jobs.get(k) == v for k, v in HISTORICAL_JOBS.items()), 'Historical original/CPU02 IDs changed')
    for label, job in jobs.items():
        historical = label in ('cpu', 'after-cpu', 'cpu-02', 'after-cpu-02')
        require(historical or label in ('cpu-03', 'after-cpu-03') or re.fullmatch(r'(?:pilot-(?:native|after-native|segment|after-segment)|dispatch)-\d{4}', label), 'Foreign actual job parent')
        sub = read(folder / 'run' / label / 'submission.json')
        require(sub['job_id'] == job and sub['outcome'] == 'submitted' and sub['returncode'] == 0 and re.findall(r'Job <(\d+)>', sub['stdout']) == [job] and sub['args'] == vector(label, sub['phase'], sub['dependency'], launcher_sha, historical), 'Actual source/resource/job vector differs: ' + label)
        if sub['phase'] == 'after-cpu':
            producer = 'cpu' if label == 'after-cpu' else 'cpu-02' if label == 'after-cpu-02' else 'cpu-03'
            require(sub['dependency'] == 'ended(' + jobs[producer] + ')', 'CPU callback does not depend on own actual CPU')
        elif sub['phase'] in ('after-native', 'after-segment'):
            producer = label.replace('-after-native-', '-native-').replace('-after-segment-', '-segment-')
            require(sub['dependency'] == 'ended(' + jobs[producer] + ')', 'Native/segment own-ended callback differs')
        elif sub['phase'] in ('cpu', 'native', 'segment'):
            require(sub['dependency'] is None, 'Unexpected interarm/scientific dependency')
        query = metadata['scheduler'][label]
        require(query['job_id'] == job, 'Scheduler inventory ID differs')
        observed[label] = scheduler(query['raw'], job, 'gpul40s' if sub['phase'] in ('native', 'segment') else 'hpc') if query['returncode'] == 0 else 'unavailable_live_history'
    for name in files:
        if not name.endswith('/gate.json'): continue
        gate = read(folder / name)
        require(gate['reserve_bytes'] == 4 * GIB and gate['file_reserve'] == 100000, 'Byte/file reserve weakened')
        gates[name] = gate['status']
        if 'work3' not in gate: continue
        work = gate['work3'];quota = quota_from_raw(work['raw_quota'])
        require(work['quota'] == quota and work['projected_bytes'] == gate['projected_bytes'] and work['required_bytes'] == gate['projected_bytes'] + 4 * GIB, 'Raw quota/full byte/reserve arithmetic differs')
        require(work['available_bytes_conservative'] == min(quota['available_bytes_lower'] - work['stale_bytes_subtracted'], work['disk_free_bytes']), 'Stale-byte/disk conservative minimum differs')
        inode = work['filesystem_inode_observation'];values = tuple(inode[key] for key in ('f_files', 'f_ffree', 'f_favail'))
        if values == (0, 0, 0):
            require(work['filesystem_inode_reporting'] == 'unreported_all_zero' and work['available_files_lower'] == quota['available_files_lower'], 'All-zero inode scope/finite quota minimum differs')
        else:
            require(0 < values[2] <= values[1] <= values[0] and work['available_files_lower'] == min(quota['available_files_lower'], values[2]), 'Positive inode/finite quota minimum differs')
        if gate['status'] == 'passed':
            require(work['status'] == 'passed' and work['available_bytes_conservative'] >= work['required_bytes'] and work['available_files_lower'] >= 100000, 'Claimed storage pass lacks capacity')
        if 'terminal_original_evidence' in gate:
            expected_launcher = launcher_sha if '/storage-adoption-03/' in name or '/cpu-03/' in name or '/after-cpu-03/' in name or '/pilot-' in name or '/dispatch-' in name else PREVIOUS_MANIFEST
            require(gate['science_manifest_sha256'] == MANIFEST and gate['launcher_manifest_sha256'] == expected_launcher and set(gate['terminal_original_evidence']) == set(TERMINAL), 'Adopted terminal accounting scope differs')
            current = gate['original_current'];future = gate['original_future']
            require(future == {'protenix': 0, 'boltz2': 0, 'boltz2_chemistry': 5144021824 - current['boltz2_chemistry']} and future['boltz2_chemistry'] >= 0, 'Only exact completed originals may receive zero future charge')
            require(work['stale_bytes_subtracted'] >= sum(current.values()) + gate['own_current'] + gate['additional_noise_stale_bytes'], 'Current original/new/noise bytes not retained in stale charge')
            for arm, pin in TERMINAL.items():
                ev = gate['terminal_original_evidence'][arm]
                require(ev['status'] == 'terminal_original_authenticated' and ev['job_id'] == pin['job'] and ev['scientific_status'] == pin['status'] and ev['future_output_bytes'] == 0 and ev['live_scheduler_status_invented'] is False and ev['original_source_manifest_sha256'] == ORIGINAL_MANIFEST, 'Original failure/status rewritten by storage adoption')
                for kind, p in ev['proofs'].items():
                    require((p['bytes'], p['sha256']) == pin[kind], 'Gate original terminal proof pin differs')
            projected = sum(future.values()) + sum(p['remaining'] for p in gate['peers'].values()) + gate['own_full'] - gate['own_current'] + 2 * GIB + 512 * 1024 ** 2 + GIB
            require(gate['projected_bytes'] == projected and gate['additional_prospective_noisy_protenix_allowance'] == GIB, 'Peer/own/report/diagnostic/noise full projection arithmetic differs')
        budget_details[name] = {'status': gate['status'], 'required_bytes': work['required_bytes'], 'available_bytes_conservative': work['available_bytes_conservative'], 'stale_bytes_subtracted': work['stale_bytes_subtracted'], 'available_files_lower': work['available_files_lower'], 'full_long_admission': gate.get('full_long_admission')}
    base_value = {'files_verified': len(files), 'source_members_verified': {'science': 657, 'launcher': 7, 'preserved_previous_launcher': 5}, 'source_manifest_sha256': MANIFEST, 'launcher_manifest_sha256': launcher_sha, 'source_archive_sha256': ARCHIVE, 'launcher_archive_sha256': package['archive_sha256'], 'previous_launcher_manifest_sha256': PREVIOUS_MANIFEST, 'evidence_archive_sha256': receipt['sha256'], 'jobs': jobs, 'actual_scheduler': observed, 'storage_gates': gates, 'storage_budgets': budget_details, 'historical_failed_cpu_jobs_preserved': HISTORICAL_JOBS, 'historical_cpu02_fixture_failure_names': PREVIOUS_FAILURE_NAMES, 'original_scientific_status_preserved': {'protenix': 'passed', 'boltz2': 'failed'}, 'large_original_summary_bytes_downloaded': False, 'native_readiness_passed_by_this_audit': False, 'optimization_started_by_this_audit': False, 'scope': 'Independent stdlib bytes/XML/schema/source/resource/scheduler/LSF audit. No numerical/model execution. Storage observations are not reservations.'}
    if record is None:
        require(args.expect_failure and jobs == HISTORICAL_JOBS and gates.get('run/storage-adoption-03/storage-preflight/gate.json') == 'failed' and not any(name.startswith('run/cpu-03/') for name in files), 'No authenticated new submission or failed preflight; unknown outcome requires inspection')
        value = {**base_value, 'status': 'failed_new_preflight_evidence_verified', 'cpu_passed': False, 'new_cpu_submitted': False, 'cases_claimed_passed': None}
    else:
        require(record['status'] == 'submitted_not_passed' and record['root'] == ROOT and record['launcher_source'] == LAUNCHER and record['science_manifest_sha256'] == MANIFEST and record['launcher_manifest_sha256'] == launcher_sha and record['scientific_source_changed'] is False, 'Actual new submission-complete identity differs')
        for key, label in [('cpu', 'cpu-03'), ('callback', 'after-cpu-03')]:
            require(record[key] == read(folder / 'run' / label / 'submission.json') and record[key]['job_id'] == jobs[label], 'Current CPU ID not authenticated by actual submission-complete')
        require(record['preflight'] == read(folder / 'run/storage-adoption-03/storage-preflight/gate.json') and record['preflight']['status'] == 'passed', 'New preflight pass missing')
        for collection in (control['preserved_old_failed_cpu_proofs'], record['old_failed_cpu_proofs']):
            require({p['path'].removeprefix(ROOT + '/'): p['sha256'] for p in collection} == {**OLD_FAILED, **PREVIOUS_FAILED}, 'Historical failure adoption pins differ')
        if args.admission_only:
            value = {**base_value, 'status': 'passed_admission_evidence_verified', 'new_cpu_submitted': True, 'cpu_passed': False, 'cpu_acceptance_established_by_this_audit': False, 'cases_claimed_passed': None}
        else:
            failed = []
            for label in ('cpu-03', 'after-cpu-03'):
                raw = (folder / 'run' / label / ('stdout.' + jobs[label] + '.log')).read_text()
                is_failed = 'Exited with exit code' in raw
                lsf(raw, jobs[label], not is_failed, failure_code=None)
                require(observed[label] in (('EXIT', 'unavailable_live_history') if is_failed else ('DONE', 'unavailable_live_history')), 'Current CPU/callback remains RUN/PEND or has contradictory final status')
                if is_failed: failed.append(label)
            if args.expect_failure:
                require(failed, 'No actual NEW CPU/callback failure; old failures do not count')
                partial = {}
                for name in files:
                    if name.startswith('run/cpu-03/payload/') and name.endswith('.xml'):
                        try:
                            t = ET.parse(folder / name);partial[name] = {k: len(list(t.iter(k))) for k in ('testcase', 'failure', 'error', 'skipped')}
                        except ET.ParseError as e: partial[name] = {'incomplete_xml': repr(e)}
                value = {**base_value, 'status': 'failed_new_cpu_attempt_evidence_verified', 'failed_producers': failed, 'cpu_passed': False, 'partial_xml_counts': partial, 'cases_claimed_passed': None}
            else:
                require(not failed, 'New CPU/callback failed; preserve with explicit failed-attempt scope')
                certificate = read(folder / 'run/cpu-certificate.json')
                require(certificate['status'] == 'passed' and certificate['suite'] == SUITE and certificate['source_manifest_sha256'] == MANIFEST and certificate['launcher_manifest_sha256'] == launcher_sha and certificate['launcher_source'] == LAUNCHER, 'CPU certificate lost dual-source identity')
                for proof in certificate['proofs']:
                    require(proof['path'] in remote and remote[proof['path']]['sha256'] == proof['sha256'], 'CPU certificate proof omitted/changed')
                bound = {p['path']: p['sha256'] for p in certificate['proofs']}
                for name, sha in {**OLD_FAILED, **PREVIOUS_FAILED}.items(): require(bound.get(ROOT + '/' + name) == sha, 'CPU certificate omitted preserved original/CPU02 failure')
                for name, data in launcher.items(): require(bound.get(LAUNCHER + '/' + name) == digest(data), 'CPU certificate omitted a launcher member')
                require(bound.get(LAUNCHER + '/source_manifest.json') == launcher_sha, 'CPU certificate omitted launcher manifest')
                finals = [p for p in certificate['proofs'] if p['path'].startswith(ROOT + '/cpu-03/') and re.fullmatch(r'scheduler-final-\d+\.txt', PurePosixPath(p['path']).name)]
                require(len(finals) == 1, 'Missing unique certificate-bound newCPU final scheduler')
                scheduler((folder / ('run/' + finals[0]['path'].removeprefix(ROOT + '/'))).read_text(), jobs['cpu-03'], 'hpc', 'DONE')
                groups, headers, coverage = cpu_groups(folder, science, launcher, files)
                require(certificate['groups'] == groups and certificate['cases'] == sum(groups.values()) == 1135, 'Certificate actual seven-group1135 case counts differ')
                assessment = read(folder / 'run/cpu-03/payload/assessment.json')
                require(assessment['status'] == 'passed' and assessment['passed'] == 832 and assessment['source_manifest_sha256'] == MANIFEST, 'Inherited allocated832 assessment differs')
                require(digest((folder / 'run/cpu-03/payload/pocket_definition.json').read_bytes()) == POCKET and digest((folder / 'run/cpu-03/payload/composition_reference.json').read_bytes()) == REFERENCE, 'NewCPU original pocket/reference changed')
                require(gates.get('run/cpu-03/storage-cpu-entry/gate.json') == 'passed', 'Actual newCPU entrygate did not pass')
                value = {**base_value, 'status': 'passed', 'cpu_passed': True, 'cpu_cases': sum(groups.values()), 'inherited_cases': 832, 'new_cases': sum(groups[g] for g in NEW_MODULES), 'groups': groups, 'xml_header_counts': headers, 'source_function_coverage': coverage, 'xml_header_limitation': 'Actual testcase rows and raw pass logs are authoritative; the inherited hpc-contract header adds three unchanged unittest subtests, not extra passed cases.', 'cpu_certificate_sha256': digest((folder / 'run/cpu-certificate.json').read_bytes()), 'current_cpu_jobs': {'cpu': jobs['cpu-03'], 'callback': jobs['after-cpu-03']}, 'native_submission': {k: v for k, v in jobs.items() if k.startswith('pilot-native-') or k.startswith('pilot-after-native-')}}
    with output.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
