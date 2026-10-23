"""Exact byte restoration of the archived pose bank; no model imports or jobs."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
from datetime import datetime, timezone

WORK = Path('/work3/s234818/latentmol')
CONTROL = WORK / 'submissions/source138-context-restore-20261023'
ARCHIVE_SHA = 'ef82e950d0f1bb426d098592ba7d4b7a0584ef75fb8029babe93a57690c50d8b'
MANIFEST_SHA = '60765caddc95f881ed2207b12ac8ed69398f9b47c3635d7c71eae08cd32938e1'
BANK = WORK / 'runs/source056-random-start-study/20261005T225808Z_vae-random-starts-056_37735924/contexts'
CATALOG_SHA = 'a0ebed8bd67358b4027999901a8cb9e5be12731b4d77d68c6b9154513b2261eb'
PROTECTED = {
    WORK / 'runs/source121a002-chemistry-16core-24h-attempt002/variants/r2_async_w12_m128_h24/segments/001/20261018T175254Z_chemistry-train_89b58520/epoch_0036.pt':
        '3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f',
    WORK / 'runs/source137-epoch36-boltz-embedding-bridge-attempt002/gpu/fit/best-bridge.pt':
        '013834958d52f7aee999ca4c39496e14fc5333d69eb697d47433821f48290343',
    WORK / 'releases/source-136-valid-reference-epoch40-continuation-attempt001.tar.gz':
        '4833072e286c39e9233f4bd776cd169623bb1496a14e945c296116527cc0be69',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def quota():
    run = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, timeout=60, check=True)
    values = re.findall(r'\|\|\s*([0-9.]+)\s*GiB\|\s*([0-9.]+)\s*GiB', run.stdout)
    if not values:
        raise ValueError('Fresh quota unavailable')
    return {'stdout': run.stdout, 'free_gib': min(float(limit)-float(used) for used, limit in values)}


def protected():
    observed = {str(path): sha(path) for path in PROTECTED}
    if observed != {str(path): value for path, value in PROTECTED.items()}:
        raise ValueError('Protected VAE/mapping/continuation source identity changed')
    return observed


def main():
    archive = CONTROL / 'source056-prepared-contexts-exact-restore.tar.gz'
    if sha(archive) != ARCHIVE_SHA or sha(BANK / 'contexts.json') != CATALOG_SHA:
        raise ValueError('Restore archive/catalog identity mismatch')
    catalog = json.loads((BANK / 'contexts.json').read_text())['contexts']
    before = protected()
    with tarfile.open(archive) as stream:
        manifest_bytes = stream.extractfile('restore-manifest.json').read()
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA:
            raise ValueError('Restore manifest identity mismatch')
        manifest = json.loads(manifest_bytes)
        rows = manifest['files']
        expected = {row['relative_path'] for row in rows} | {'restore-manifest.json'}
        if len(rows) != 44 or len(expected) != 45 or set(stream.getnames()) != expected:
            raise ValueError('Restore capsule must contain precisely the 44 declared contexts')
        if any(not member.isfile() for member in stream.getmembers()):
            raise ValueError('Only regular-file restore members are allowed')
        for row in rows:
            target = Path(row['original_path']).resolve()
            match = re.fullmatch(r'count-([0-9]{2})/prepared\.pt', target.relative_to(BANK.resolve()).as_posix())
            if not match or (WORK / row['relative_path']).resolve() != target:
                raise ValueError('Restore target is outside the exact historical context allowlist')
            entry = catalog[str(int(match.group(1)))]
            if entry['sha256'] != row['sha256'] or (BANK / entry['path']).resolve() != target:
                raise ValueError('Cold preservation and live scientific catalog disagree')
            if stream.getmember(row['relative_path']).size != row['bytes']:
                raise ValueError('Restore size differs from authenticated member')
            if target.exists() and sha(target) != row['sha256']:
                raise FileExistsError('Refusing to overwrite different existing context: ' + str(target))
        before_quota = quota()
        if before_quota['free_gib'] < 5 + sum(row['bytes'] for row in rows) / 1024**3:
            raise RuntimeError('Insufficient fresh restoration headroom')
        write_new(CONTROL / 'intent.json', dict(archive_sha256=ARCHIVE_SHA, manifest_sha256=MANIFEST_SHA,
            catalog_sha256=CATALOG_SHA, files=rows, protected=before, quota_before=before_quota,
            timestamp_utc=datetime.now(timezone.utc).isoformat(), purpose='restore exact archived immutable pose inputs'))
        completed = []
        for row in rows:
            target = Path(row['original_path']).resolve()
            status = 'already_present_unchanged'
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name('prepared.pt.source138-restore.partial')
                h = hashlib.sha256()
                with temporary.open('xb') as destination, stream.extractfile(row['relative_path']) as source:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        destination.write(block)
                        h.update(block)
                    destination.flush()
                    os.fsync(destination.fileno())
                if h.hexdigest() != row['sha256'] or temporary.stat().st_size != row['bytes']:
                    raise ValueError('Restored temporary bytes failed authentication')
                os.link(temporary, target)  # Atomic no-overwrite publication.
                temporary.unlink()
                status = 'restored_original_bytes'
            if sha(target) != row['sha256']:
                raise ValueError('Published restoration failed authentication')
            completed.append(dict(path=str(target), sha256=row['sha256'], bytes=row['bytes'], status=status))
            with (CONTROL / 'journal.jsonl').open('a') as journal:
                journal.write(json.dumps(completed[-1]) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
    result = dict(status='exact_context_bank_restored', files=completed,
        protected_before=before, protected_after=protected(), quota_after=quota(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(), numerical_execution=False,
        jobs_changed=False, historical_archives_unchanged=True)
    write_new(CONTROL / 'completion.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_new(CONTROL / 'failure.json', dict(error_type=type(error).__name__, error=str(error)))
        raise
