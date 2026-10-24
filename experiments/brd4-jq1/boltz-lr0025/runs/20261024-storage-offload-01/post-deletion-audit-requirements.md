# Post-deletion evidence requirements

The root operator owns all SSH, backup, removal and audit execution. The new
`post_deletion_audit.py` is an unexecuted, read-only standard-library collector;
it does not import a scientific package, decode arrays, restore results, delete
files or submit jobs. No existing source seal or evidence file was changed.

Before invoking it, retain the exact locally approved plan SHA, the remote
completed-stream receipt SHA and the independent laptop full-member-verification
SHA. The collector fetches the exact operation's raw receipts and newline-complete
journal as base64, rechecks their hashes and plan bindings, reconciles every
planned file with one journal outcome, and checks actual original-path absence
or retained changed-path metadata. Every recorded unlink needs its prior unique
matching durable intent. Already absent files have no removal-byte credit.

Supply a separately approved pin specification and its exact SHA:

```json
{
  "version": "mosaic-storage-offload-post-audit-pins-v1",
  "max_total_hash_bytes": 8589934592,
  "sources": [
    {
      "root": "/absolute/protected/source",
      "manifest_path": "/absolute/protected/source/source-manifest.json",
      "manifest_sha256": "exact manifest SHA256",
      "ignore_directory_names": [],
      "members": {"relative/file.py": "exact original member SHA256"}
    }
  ],
  "files": [
    {"path": "/absolute/protected/input", "bytes": 1, "sha256": "exact SHA256"}
  ]
}
```

Use the already reviewed complete current Mosaic scientific/launcher source
member maps. The independent active VAE/embedding releases use exact manifest
and selected execution/input file pins: their execution maps do not exhaust
physical prose/lock files, so they must not be supplied as false whole-source
inventories. Include both active source138 archives. Only declared `__pycache__` and `.pytest_cache`
directories may be excluded from a source inventory, matching its existing
authentication policy; all declared execution members and its manifest rehash.

The explicit file pins should include the retained seven root/proof metadata
files, current CPU/native/readiness certificates, original fixed Boltz context,
the required Protenix checkpoint/context and
continuation snapshot, and the independent VAE epoch36/source104 parents and
embedding bridge/samples/selected context assets. Copy exact original hashes
and lengths from the reviewed protection receipts; do not infer a new hash from
a basename. Whole protected environments/data/cache directories stay excluded
from removal. Dynamic current training outputs are allowed to advance and must
not be falsely compared to an obsolete snapshot. Rehashing huge current result
summaries is unnecessary for this disjoint old-run cleanup and is explicitly
rejected by this audit above 32 MiB.

The collector records a fresh supported `bjobs -noheader -o 'jobid stat'` query,
raw work3 quota and physical filesystem counters. These observations are separate
from the cleanup proof and from a fresh production admission gate. Record:

1. Removed *logical* bytes and exact path count from the authenticated journal.
2. Unique inode logical bytes and allocated blocks from the before inventory,
   without crediting incomplete or protected hardlink groups.
3. Fresh authenticated quota before/after, including refresh timestamps and
   concurrent active writes. A shared filesystem counter is not the user quota.

Do not call duplicated hardlink path bytes physical savings. Do not attribute a
quota change solely to cleanup while other jobs are writing, and do not claim a
quota pass, reservation or independent filesystem inode proof from all-zero
supplemental counters. All current experiments still need their unchanged fresh
admission checks before further model work.

## Approved, unexecuted caller

`post-audit-pin-spec-approval-01.json` approves only candidate SHA
`cb8e794d5f67731ff7389c99fd3827a7ad0874f2055879166d932760ad12a9f4`.
Its bounded 2 GiB pin scope covers seven full Mosaic member inventories and
270 explicit execution/input/proof pins. The 575,863,502-byte known upper bound
includes both active source138 archives. Redundant 4.348 GB cache-model
rehashes, huge saved summaries/JSONLs and advancing current outputs are
intentionally omitted; all corresponding paths remain protected and no fresh
cache-model digest is claimed.

`run_post_deletion_audit.py` is an unexecuted, one-shot root-only wrapper. After
the standalone archive verifier and separate stream/deletion receipt collectors
have completed, the root can supply their actual local paths with
`--verification`, `--stream-receipt` and `--deletion-receipt`. The wrapper first
authenticates the exact approval, code, operator03 source and 32-case allocated
proof, approved plan and all separately fetched backup/deletion authorities.
It stages only the approved audit code and pins in the exclusive new remote
`post-audit-01` directory. It then executes the unchanged read-only audit with
its 1,800-second bound and a 1,860-second parent timeout.

Full stdout/stderr, generated command and unknown outcomes remain in a fresh
ignored `backup-01/post-deletion-audit-01` result directory. A bootstrap record
and the unchanged audit JSON receive independent local byte/schema, complete
journal/path, source/input-pin and logical-accounting checks. The wrapper has
no delete, restore, optimization, job-submission or admission-pass action.
No wrapper, post-audit code or pin specification has been executed during
local preparation; only standard-library AST/compile/source/byte review ran.
