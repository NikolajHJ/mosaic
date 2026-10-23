"""Independent post-seal stdlib byte/AST audit; never imports execution modules."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
EXPECTED = {
    "tag": "20261024-boltz-lr0025-03",
    "manifest_sha256": "c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6",
    "archive_bytes": 39628961,
    "archive_sha256": "d77d298b7cd2df41b4fb97fb22bf241a623c0d9825edab89226c54fa90b4b9ba",
    "source_files": 635,
}
PARENT = {
    "tag": "20261023-boltz-lr0025-02",
    "manifest_sha256": "3689071a4e7dd135071ff5b420c026543c4d10887c951077dff63d498d9f812c",
    "archive_sha256": "40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa",
    "source_files": 633,
}
ANCESTOR = {
    "tag": "20261022-brd4-jq1-25-fixed-lrreset-04",
    "manifest_sha256": "c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057",
    "archive_sha256": "4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e",
    "source_files": 625,
}
REVISION = "c74c659344123937608077e3e4078f75f09cd743"
ALIASES = {
    "boltz_serial_ops.py": "scripts/boltz_serial_ops.py",
    "boltz_serial_cpu.lsf": "scripts/hpc/boltz_serial_cpu.lsf",
    "boltz_serial_stage.lsf": "scripts/hpc/boltz_serial_stage.lsf",
    "boltz_serial_protocol.py": "scripts/boltz_serial_protocol.py",
    "boltz_serial_driver.py": "examples/brd4_25_atoms.py",
    "boltz_serial_contract.py": "scripts/campaign25_contract.py",
    "fixture_boltz_serial.py": "tests/test_boltz_serial.py",
    "fixture_protenix_initialization.py": "tests/test_protenix_initialization.py",
    "boltz2.json": "examples/configs/boltz_serial_boltz2.json",
    "boltz2_chemistry.json": "examples/configs/boltz_serial_boltz2_chemistry.json",
    "README.md": "docs/boltz_serial30k.md",
}
SHARED = {
    "scripts/hpc/startup/premodel_startup.py": "scripts/premodel_startup.py",
    "tests/core/fixture_premodel_startup.py": "tests/test_premodel_startup.py",
}
CHANGED = {
    "scripts/boltz_serial_ops.py", "scripts/hpc/boltz_serial_cpu.lsf",
    "docs/boltz_serial30k.md", *SHARED.values(),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def archive_for(spec, source="source01"):
    path = REPO / "results/hpc" / spec["tag"] / source / "source.tar.gz"
    raw_archive = path.read_bytes()
    require(sha(raw_archive) == spec["archive_sha256"], "Archive digest differs: " + spec["tag"])
    if "archive_bytes" in spec:
        require(len(raw_archive) == spec["archive_bytes"], "Archive length differs")
    files = {}
    with tarfile.open(path) as archive:
        for member in archive:
            require(member.isfile() and member.name.startswith("source/"), "Non-regular/outside member")
            name = member.name.removeprefix("source/")
            safe = PurePosixPath(name)
            require(name not in files and not safe.is_absolute() and ".." not in safe.parts
                    and "\\" not in name and ":" not in name and name, "Unsafe/duplicate member")
            require(member.mode & 0o222 == 0, "Writable sealed archive member")
            data = archive.extractfile(member).read()
            require(len(data) == member.size, "Truncated member")
            files[name] = data
    raw = files.pop("source_manifest.json")
    require(sha(raw) == spec["manifest_sha256"], "Manifest identity differs")
    manifest = json.loads(raw)
    require(len(files) == spec["source_files"] and set(files) == set(manifest["files"]), "Manifest/full scope differs")
    require(all(sha(data) == manifest["files"][name] for name, data in files.items()), "Member digest differs")
    return files, manifest, raw


def functions(data):
    tree = ast.parse(data)
    return {item.name: ast.dump(item, include_attributes=False) for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}


def fixture_cases(data):
    tree = ast.parse(data)
    cases = 0
    for item in tree.body:
        if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
            continue
        count = 1
        for decorator in item.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "parametrize"):
                count *= len(ast.literal_eval(decorator.args[1]))
        cases += count
    declared = next(ast.literal_eval(item.value) for item in tree.body
                    if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name)
                        and target.id == "STARTUP_CASES" for target in item.targets))
    require(cases == declared == 28, "Unexpected startup fixture AST row count")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "local-static-verification.json")
    out = parser.parse_args().output.resolve()
    require(not out.exists(), "Audit receipt already exists; inspect rather than replay")
    package = json.loads((HERE / "package.json").read_bytes())
    require(all(package[key] == value for key, value in EXPECTED.items()), "Package metadata differs")
    current, manifest, raw = archive_for(EXPECTED)
    previous, parent, _ = archive_for(PARENT)
    _, ancestor, _ = archive_for(ANCESTOR, "source02")
    require((REPO / "results/hpc" / EXPECTED["tag"] / "source01/source_manifest.json").read_bytes() == raw,
            "Sibling manifest bytes differ")
    require(parent.get("git_revision_reference", "ABSENT") == "ABSENT", "Original failed source was changed")
    require(manifest["git_revision_reference"] == ancestor["git_revision_reference"] == REVISION,
            "Corrected revision is not the exact authenticated ancestor value")
    require(manifest["parent_manifest_sha256"] == PARENT["manifest_sha256"]
            and manifest["git_revision_inheritance"] == package["git_revision_inheritance"], "Inheritance differs")
    changed = {name for name in current if previous.get(name) != current[name]}
    require(changed == CHANGED and not set(previous) - set(current), "Unexpected full source byte changes")
    require({name: sha(current[name]) for name in changed} == package["changed_members"], "Changed-member map differs")
    scientific = [name for name in previous if name.startswith("src/")]
    require(len(scientific) == 145 and all(current[name] == previous[name] for name in scientific), "Scientific bytes changed")
    protected = set(ALIASES.values()) - {"scripts/boltz_serial_ops.py", "scripts/hpc/boltz_serial_cpu.lsf", "docs/boltz_serial30k.md"}
    require(all(current[name] == previous[name] for name in protected), "Protected driver/protocol/config/fixture differs")
    for local, member in ALIASES.items():
        require((HERE / local).read_bytes().replace(b"\r\n", b"\n") == current[member], "Canonical alias differs: " + local)
    for local, member in SHARED.items():
        require((REPO / local).read_bytes().replace(b"\r\n", b"\n") == current[member], "Shared alias differs: " + local)
    ast_members = [name for name in current if name.endswith(".py")]
    for name in ast_members:
        ast.parse(current[name], filename=name)
    old_functions = functions(previous["scripts/boltz_serial_ops.py"])
    new_functions = functions(current["scripts/boltz_serial_ops.py"])
    changed_functions = sorted(name for name in old_functions if old_functions[name] != new_functions[name])
    require(changed_functions == sorted(["dbmol_budget", "storage_gate", "queue_native", "certify_cpu", "dispatch", "certify_segment", "main"]),
            "Inherited controller function changes differ")
    for name in ("produce", "native_assessment", "all_budgets"):
        require(old_functions[name] == new_functions[name], "Scientific/native/storage computation changed: " + name)
    startup_cases = fixture_cases(current["tests/test_premodel_startup.py"])
    result = {
        "status": "passed", "scope": "Post-seal local stdlib byte/JSON/AST verification only",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), **EXPECTED,
        "safe_members": len(current) + 1, "python_members_ast_parsed": len(ast_members),
        "parent_manifest_sha256": PARENT["manifest_sha256"], "ancestor_manifest_sha256": ANCESTOR["manifest_sha256"],
        "git_revision_reference": REVISION, "unchanged_scientific_members": len(scientific),
        "unchanged_protected_members": sorted(protected), "changed_members": package["changed_members"],
        "changed_inherited_controller_functions": changed_functions,
        "unchanged_produce_native_assessment_all_budgets_ast": True,
        "canonical_aliases_byte_matched": len(ALIASES), "shared_aliases_byte_matched": len(SHARED),
        "startup_fixture_declared_cases": startup_cases, "expected_allocated_cpu_cases": 799 + startup_cases,
        "numerical_tests_executed_locally": False, "allocated_cpu_pass_established": False,
        "actual_startup_smoke_pass_established": False, "native_readiness_pass_established": False,
        "remote_submission_established": False,
    }
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps({"status": "passed", "receipt": str(out), "sha256": sha(out.read_bytes()),
                      "source_files": len(current), "expected_allocated_cpu_cases": 799 + startup_cases}))


if __name__ == "__main__":
    main()
