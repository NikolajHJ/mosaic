"""Verified local offload of one redundant compressed CCD download archive."""
import argparse
import json
import os
from pathlib import Path
import re
import stat

import cache_operator as guard

ROOT = guard.ROOT
TARGET = ROOT / "cache/boltz2/mols.tar.gz"
RETAIN = ROOT / "cache/boltz2/mols.tar"
NATIVE = ROOT / "env/boltz311/lib/python3.11/site-packages/boltz"


def dependencies():
    jobs = guard.scheduler()
    roots = {NATIVE}
    for job in jobs["active_jobs"]:
        roots.update(Path(name) for name in job["roots"] if Path(name).parent == guard.RELEASES)
        for match in re.findall(r"/work3/s234818/mosaic/[^\s;\"']+?/source(?:/|\s|$)", job["command"] + " "):
            roots.add(Path(match.rstrip("/ ")))
    seen, links, hits = {}, [], []
    for root in sorted(roots):
        guard.canonical(root)
        for folder, dirs, files in os.walk(root, followlinks=False):
            guard.deadline()
            dirs[:] = [name for name in dirs if name not in (".git", "__pycache__", ".pytest_cache")]
            for name in list(dirs) + files:
                path = Path(folder, name)
                if path.is_symlink() and path.resolve(strict=False) == TARGET:
                    links.append(str(path))
            dirs[:] = [name for name in dirs if not Path(folder, name).is_symlink()]
            for name in files:
                path = Path(folder, name)
                if path.is_symlink() or path.suffix not in (".py", ".sh", ".lsf", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"):
                    continue
                before = guard.fingerprint(path.stat())
                if before["size"] > 32 * 1024**2:
                    raise ValueError("Unbounded active metadata input: " + str(path))
                content = path.read_bytes()
                if guard.fingerprint(path.stat()) != before:
                    raise ValueError("Active source changed during audit")
                seen[str(path)] = len(content)
                if b"mols.tar.gz" in content:
                    hits.append(str(path))
    # Inspect environment symlinks without following them or loading packages.
    for base in (ROOT / "env", Path("/work3/s234818/mosaic/env")):
        if not base.is_dir():
            continue
        for folder, dirs, files in os.walk(base, followlinks=False):
            guard.deadline()
            for name in list(dirs) + files:
                path = Path(folder, name)
                if path.is_symlink() and path.resolve(strict=False) == TARGET:
                    links.append(str(path))
            dirs[:] = [name for name in dirs if not Path(folder, name).is_symlink()]
    if hits or links:
        raise ValueError("Archive has active source/symlink references: " + repr((hits, links)))
    return dict(scheduler=jobs, roots=[str(p) for p in sorted(roots)],
                inspected_text_files=len(seen), inspected_bytes=sum(seen.values()),
                direct_archive_references=hits, incoming_environment_links=links)


def target_check(proof):
    guard.canonical(TARGET)
    info = TARGET.lstat()
    expected = {key[3:]: value for key, value in proof["fingerprints"][str(TARGET)].items()}
    if (guard.fingerprint(info) != expected or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid() or info.st_nlink != 1
            or guard.sha(TARGET) != proof["compressed_sha256"]):
        raise ValueError("Redundant archive changed since byte proof")
    if (proof["decompressed_sha256"] != proof["retained_tar_sha256"]
            or proof["decompressed_bytes"] != RETAIN.stat().st_size
            or guard.sha(RETAIN) != proof["retained_tar_sha256"]):
        raise ValueError("Native uncompressed archive does not preserve exact payload")
    return expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("audit", "remove", "verify"))
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--protected-inputs", type=Path, required=True)
    p.add_argument("--audit-sha256")
    args = p.parse_args()
    control = guard.canonical(args.control)
    if control.parent != ROOT / "submissions" or control.name != "storage-cleanup-20261027-safe001":
        raise ValueError("Unexpected exact maintenance control directory")
    pins = guard.protected_inputs(args.protected_inputs)
    proof = json.loads((control / "archive-proof001.json").read_text())
    backup = json.loads((control / "archive-restoration-map.json").read_text())
    if (proof["remove"] != str(TARGET) or proof["retain"] != str(RETAIN)
            or backup["sha256"] != proof["compressed_sha256"]
            or backup["remote_original"] != str(TARGET)
            or backup["bytes"] != proof["fingerprints"][str(TARGET)]["st_size"]
            or str(TARGET) in pins or pins.get(str(RETAIN)) != proof["retained_tar_sha256"]):
        raise ValueError("Missing exact retained payload and local restore binding")
    before = guard.pin_check(pins)
    if args.mode == "audit":
        expected = target_check(proof)
        result = dict(status="passed_read_only_duplicate_archive_audit", utc=guard.now(),
            target=str(TARGET), retained_native_archive=str(RETAIN), stat=expected,
            protected=before, local_backup=backup, dependencies=dependencies(),
            reviewed_prefix_exception="Only this duplicate gzip file under cache/boltz2; retain native tar, extracted molecules and model weights",
            archive_proof_sha256=guard.sha(control / "archive-proof001.json"),
            restore_map_sha256=guard.sha(control / "archive-restoration-map.json"))
        guard.durable(control / "archive-audit.json", result)
    else:
        audit_path = control / "archive-audit.json"
        if guard.sha(audit_path) != args.audit_sha256:
            raise ValueError("Exact reviewed archive audit is required")
        audit = json.loads(audit_path.read_text())
        if (audit["protected"] != before or audit["local_backup"] != backup
                or audit["archive_proof_sha256"] != guard.sha(control / "archive-proof001.json")
                or audit["restore_map_sha256"] != guard.sha(control / "archive-restoration-map.json")):
            raise ValueError("Reviewed archive preservation binding changed")
        if args.mode == "remove":
            if (control / "archive-intent.json").exists() or (control / "archive-completion.json").exists():
                raise FileExistsError("Preserve previous removal; do not retry automatically")
            expected = target_check(proof)
            active = dependencies()
            guard.durable(control / "archive-intent.json", dict(utc=guard.now(), audit_sha256=args.audit_sha256,
                target=str(TARGET), stat=expected, sha256=proof["compressed_sha256"],
                local_backup=backup, protected=before, dependencies=active))
            parent = os.open(TARGET.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                if guard.fingerprint(os.stat(TARGET.name, dir_fd=parent, follow_symlinks=False)) != expected:
                    raise ValueError("Archive changed immediately before unlink")
                os.unlink(TARGET.name, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(parent)
            result = dict(status="completed_verified_duplicate_offload", utc=guard.now(),
                target=str(TARGET), removed_allocated_bytes=expected["blocks"] * 512,
                removed_logical_bytes=expected["size"], local_backup=backup,
                protected_after=guard.pin_check(pins), intent_sha256=guard.sha(control / "archive-intent.json"))
            guard.durable(control / "archive-completion.json", result)
        else:
            done = json.loads((control / "archive-completion.json").read_text())
            if (TARGET.exists() or TARGET.is_symlink() or done["protected_after"] != before
                    or done["intent_sha256"] != guard.sha(control / "archive-intent.json")):
                raise ValueError("Archive deletion/preservation verification failed")
            result = dict(status="passed_independent_archive_verification", utc=guard.now(),
                target_absent=True, protected_hashes=len(before),
                completion_sha256=guard.sha(control / "archive-completion.json"),
                removed_allocated_bytes=done["removed_allocated_bytes"], local_restore_copy=backup)
            guard.durable(control / "archive-verification.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
