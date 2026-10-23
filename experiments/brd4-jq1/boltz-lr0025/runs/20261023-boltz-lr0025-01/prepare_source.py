"""Authenticate625/641 and prepare a new standalone baseline pair release.

Local execution is standard-library AST/JSON/byte packaging only. Initialize
canonical overlays once, review them, then explicitly seal; never overwrite
an existing source package or historical immutable input.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PARENT = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/source02/source.tar.gz'
PARENT_SHA = '4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e'
PARENT_MANIFEST = 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
OFFSET = REPO / 'results/hpc/20261023-protenix-extended-01/source02/source.tar.gz'
OFFSET_SHA = '8ba1c3370fee1f09b96af139dbf1403bb3c187c52d447ec4b0b23ebe4a65f63d'
OFFSET_MANIFEST = 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
TRAJECTORY = 'src/mosaic/small_molecule/chemistry/trajectory.py'
ALIASES = {
    'boltz_serial_protocol.py': 'scripts/boltz_serial_protocol.py',
    'boltz_serial_ops.py': 'scripts/boltz_serial_ops.py',
    'boltz_serial_driver.py': 'examples/brd4_25_atoms.py',
    'boltz_serial_contract.py': 'scripts/campaign25_contract.py',
    'boltz_serial_cpu.lsf': 'scripts/hpc/boltz_serial_cpu.lsf',
    'boltz_serial_stage.lsf': 'scripts/hpc/boltz_serial_stage.lsf',
    'fixture_boltz_serial.py': 'tests/test_boltz_serial.py',
    'boltz2.json': 'examples/configs/boltz_serial_boltz2.json',
    'boltz2_chemistry.json': 'examples/configs/boltz_serial_boltz2_chemistry.json',
    'README.md': 'docs/boltz_serial30k.md',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError((old, text.count(old)))
    return text.replace(old, new)


def source(path, archive_sha, manifest_sha, count):
    if digest(path.read_bytes()) != archive_sha:
        raise ValueError('Historical source archive bytes differ')
    files = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith('source/'):
                raise ValueError('Unexpected source archive member type/prefix')
            name = member.name.removeprefix('source/')
            if name in files or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe/duplicate source member')
            files[name] = archive.extractfile(member).read()
    raw = files.pop('source_manifest.json')
    manifest = json.loads(raw)
    if digest(raw) != manifest_sha or len(files) != count or set(files) != set(manifest['files']):
        raise ValueError('Historical manifest or complete member scope differs')
    for name, data in files.items():
        if digest(data) != manifest['files'][name]:
            raise ValueError('Historical member changed: ' + name)
    return files


def canonical(path, data):
    if path.exists():
        raise ValueError('Canonical initialization refuses overwrite: ' + str(path))
    path.write_bytes(data.replace(b'\r\n', b'\n'))


def initialize(files):
    contract = files['scripts/campaign25_contract.py'].decode()
    contract = replace(contract, 'def config25(config):', 'def legacy_config25(config):')
    contract = replace(contract, '\ndef update_parameters(config):', '''
from boltz_serial_protocol import VERSIONS as BOLTZ_SERIAL_VERSIONS, validate_config, require_boundary
FIXED25_VERSIONS = (*FIXED25_VERSIONS, *BOLTZ_SERIAL_VERSIONS)
def config25(config):
    if config.get('version') in BOLTZ_SERIAL_VERSIONS:
        return validate_config(config, legacy_config25)
    return legacy_config25(config)

def update_parameters(config):''')
    contract = contract.replace("(FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION):", "(FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, *BOLTZ_SERIAL_VERSIONS):")
    contract = replace(contract, "    boosted = config['version'] in (FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION)", "    boosted = config['version'] in (FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, *BOLTZ_SERIAL_VERSIONS)")
    contract = replace(contract, "            if config['version'] == FIXED25_LRRESET_VERSION:", "            if config['version'] in (FIXED25_LRRESET_VERSION, *BOLTZ_SERIAL_VERSIONS):")
    marker = '        verify_attempt_rates(pilot, config)'
    contract = replace(contract, marker, "        if config['version'] in BOLTZ_SERIAL_VERSIONS:\n            require_boundary(summary, assessment)\n" + marker)
    canonical(HERE / 'boltz_serial_contract.py', contract.encode())

    driver = files['examples/brd4_25_atoms.py'].decode()
    driver = replace(driver, '    args = parser.parse_args()', '''    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--segment-steps', type=int, default=10000)
    parser.add_argument('--resume-snapshot', type=Path)
    parser.add_argument('--resume-sha256')
    args = parser.parse_args()''')
    marker = '    config = config25(json.loads(args.config.read_text()))'
    driver = replace(driver, marker, marker + '''
    from boltz_serial_protocol import VERSIONS as BOLTZ_SERIAL_VERSIONS, EXECUTION, segment_scope, restore_logits
    serial = config['version'] in BOLTZ_SERIAL_VERSIONS
    execution = segment_scope(config, args.offset, args.segment_steps, args.resume_snapshot, args.resume_sha256) if serial else None
    if not serial and (args.offset or args.segment_steps != 10000 or args.resume_snapshot or args.resume_sha256):
        raise ValueError('Historical version cannot acquire continuation fields')''')
    marker = '        initial = logit_type.from_graph(initialize_graph(hard, config["initialization"]))'
    driver = replace(driver, marker, '''        if serial and args.resume_snapshot:
            initial = restore_logits(args.resume_snapshot, args.resume_sha256, logit_type, key, sha)
        else:
            initial = logit_type.from_graph(initialize_graph(hard, config['initialization']))
        if serial:
            report.update(serial_execution=execution, exact_boundary_replay='passed',
                initialization_execution={'version':EXECUTION,'jax_platform':'cpu',
                    'dtype':'float32','parameterization':'categorical_logits_v1','model_key':17})''')
    # Keep original, full normalized invalid chain assay and both endpoint pilots.
    driver = replace(driver, '                x = graph.X.at[1, 0].set(0.5 / 6).at[1, 4].set(0.5 + 0.5 / 6)', '''                if serial:
                    # Later segments no longer start at the uniform-chain row.
                    # Declare the entire normalized invalid assay row; never
                    # alter the saved optimization checkpoint to construct it.
                    x = graph.X.at[1].set(jnp.full((6,), 0.5 / 6)).at[1, 4].set(0.5 + 0.5 / 6)
                else:
                    x = graph.X.at[1, 0].set(0.5 / 6).at[1, 4].set(0.5 + 0.5 / 6)''')
    driver = replace(driver, 'fixed = prepare(probe, 0)', 'fixed = prepare(probe, args.offset if serial else 0)')
    driver = replace(driver, 'steps=10, config=opts, **common)', 'steps=10, config=opts, update_offset=args.offset if serial else None, **common)')
    driver = replace(driver, 'steps=10000, config=options, schedule=schedule, **common)', 'steps=args.segment_steps if serial else 10000, config=options, schedule=schedule, update_offset=args.offset if serial else None, **common)')
    marker = '            report["cases"] = [{"label": config["label"], "initialization": config["initialization"], **result}]'
    driver = replace(driver, marker, marker + '''
            if serial and (result['global_range'] != {'start':args.offset,'end':args.offset+args.segment_steps}
                    or [value['global_step'] for value in result['states']] != list(range(args.offset,args.offset+args.segment_steps+1))):
                raise ValueError('Global contiguous Boltz segment scope differs')''')
    marker = '        assessment.update(arm=arm, n_atoms=25, proofs=proofs)'
    driver = replace(driver, marker, marker + "\n        if serial: assessment['serial_execution'] = execution")
    canonical(HERE / 'boltz_serial_driver.py', driver.encode())

    cpu = files['scripts/hpc/fixed25_lrreset_cpu.lsf'].decode()
    start = cpu.index('"$python" - "$root" <<\'PY\'')
    end = cpu.index('\nmkdir "$root/cpu/payload"', start)
    cpu = cpu[:start] + '"$python" "$source/scripts/boltz_serial_ops.py" --phase cpu-entry --root "$root" --source "$source" --manifest "$manifest"\n' + cpu[end:]
    cpu = replace(cpu, 'tests/test_protenix_initialization.py)', 'tests/test_protenix_initialization.py tests/test_boltz_serial.py)')
    canonical(HERE / 'boltz_serial_cpu.lsf', cpu.encode())
    for arm in ['boltz2', 'boltz2_chemistry']:
        config = json.loads(files['examples/configs/brd4_jq1_25_fixed_lrreset_' + arm + '_10000.json'])
        config.update(version='mosaic-brd4-jq1-boltz-lr0025-30000-v1', steps=30000,
                      initial_step_size=.0025, extension={'version':'boltz-serial30k-v1',
                      'start_global_update':0,'target_global_update':30000,
                      'segment_updates':10000,'parent_job':None})
        if arm == 'boltz2_chemistry':
            config['repair_schedule']['total_updates'] = 30000
        config['label'] = 'Fresh Boltz-2 LR0.0025 / ' + ('chemistry0.001through5000 ->100at30000' if arm == 'boltz2_chemistry' else 'quantitative affinity only') + ' /30000updates'
        canonical(HERE / (arm + '.json'), (json.dumps(config, indent=2) + '\n').encode())


def seal(files, offset_files):
    original = dict(files)
    files[TRAJECTORY] = offset_files[TRAJECTORY]
    for local, target in ALIASES.items():
        files[target] = (HERE / local).read_bytes().replace(b'\r\n', b'\n')
    for name, data in files.items():
        if name.endswith('.py'):
            ast.parse(data, filename=name)
    changed_src = [name for name in original if name.startswith('src/') and files[name] != original[name]]
    if changed_src != [TRAJECTORY]:
        raise ValueError('Undeclared scientific-library changes')
    changed = {name:digest(data) for name,data in files.items() if original.get(name) != data}
    raw = (json.dumps({'files':{name:digest(data) for name,data in sorted(files.items())},
                      'parent_manifest_sha256':PARENT_MANIFEST,
                      'global_offset_source_manifest_sha256':OFFSET_MANIFEST,
                      'scope':'Two fresh Boltz30k/LR0.0025 chains; optional641globaloffset only; original states/jobs/releases unchanged'}, sort_keys=True, indent=2) + '\n').encode()
    output = REPO / 'results/hpc' / HERE.name / 'source01'
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as archive:
        for name, data in {**files, 'source_manifest.json':raw}.items():
            member = tarfile.TarInfo('source/' + name)
            member.mode = 0o444
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    (output / 'source_manifest.json').write_bytes(raw)
    package = {'tag':HERE.name, 'manifest_sha256':digest(raw), 'source_files':len(files),
               'archive_bytes':(output/'source.tar.gz').stat().st_size,
               'archive_sha256':digest((output/'source.tar.gz').read_bytes()),
               'parent_manifest_sha256':PARENT_MANIFEST,
               'changed_members':changed, 'scientific_src_change':changed_src,
               'execution':'Local stdlib packaging/AST only; no numerical validation'}
    (HERE / 'package.json').write_text(json.dumps(package, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(package, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    if args.initialize == args.seal:
        raise ValueError('Choose exactly one explicit preparation phase')
    files = source(PARENT, PARENT_SHA, PARENT_MANIFEST, 625)
    offset_files = source(OFFSET, OFFSET_SHA, OFFSET_MANIFEST, 641)
    if args.initialize:
        initialize(files)
    else:
        seal(files, offset_files)


if __name__ == '__main__':
    main()
