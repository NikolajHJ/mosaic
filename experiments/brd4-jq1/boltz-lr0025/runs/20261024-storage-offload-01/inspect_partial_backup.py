"""Inspect bounded tar headers and opaque byte counts; never a backup pass."""
import gzip
import json
from pathlib import Path
import tarfile
import time

HERE = Path(__file__).resolve().parent
BACKUP = HERE.parents[4] / 'results/hpc/20261024-storage-offload-01/backup-01'


class Prefix:
    def __init__(self, stream, limit):
        self.stream, self.remaining = stream, limit

    def read(self, size=-1):
        size = self.remaining if size < 0 else min(size, self.remaining)
        value = self.stream.read(size)
        self.remaining -= len(value)
        return value


def main():
    path = BACKUP / 'legacy-offload-01.tar.gz'
    output = BACKUP / 'partial-header-inspection-01.json'
    assert not output.exists() and path.is_file() and not path.is_symlink()
    before = path.stat(); begin = time.monotonic()
    complete, logical, read_bytes, last, reason = 0, 0, 0, None, None
    try:
        with path.open('rb') as stream:
            bounded = Prefix(stream, before.st_size)
            with gzip.GzipFile(fileobj=bounded, mode='rb') as inflater:
                with tarfile.open(fileobj=inflater, mode='r|') as archive:
                    for member in archive:
                        assert member.isfile()
                        payload = archive.extractfile(member)
                        count = 0
                        while True:
                            block = payload.read(4 * 1024**2)
                            if not block: break
                            count += len(block); read_bytes += len(block)
                            if read_bytes >= 8 * 1024**3 or time.monotonic() - begin >= 90:
                                raise RuntimeError('Bounded metadata-only inspection limit')
                        assert count == member.size
                        complete += 1; logical += count; last = member.name
    except (EOFError, OSError, tarfile.TarError, RuntimeError) as error:
        reason = type(error).__name__ + ': ' + str(error)
    after = path.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino) and after.st_size >= before.st_size
    result = {'status': 'partial_prefix_inspected_no_verification_or_deletion_authority',
        'compressed_prefix_bytes': before.st_size, 'completed_regular_members_in_prefix': complete,
        'completed_member_logical_bytes': logical, 'opaque_payload_bytes_read': read_bytes,
        'last_complete_member': last, 'ended_reason': reason, 'elapsed_seconds': time.monotonic() - begin,
        'archive_still_may_be_growing': True, 'scientific_arrays_decoded': False,
        'full_archive_or_member_hashes_verified': False, 'deleted_files': 0}
    with output.open('x') as out: json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
