"""Independent stdlib-only byte/XML/LSF audit of a precision CPU observation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def verify(observation, source, submission):
    observation, source, submission = map(Path, (observation, source, submission))
    raw = json.loads(observation.read_bytes())
    manifest_bytes = (source / "source_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    started = json.loads(submission.read_bytes())
    pin = hashlib.sha256(manifest_bytes).hexdigest()
    assert started["preparation"]["manifest_sha256"] == pin
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    files = {}
    for row in raw["files"]:
        data = row["text"].encode()
        assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
        assert row["path"] not in files
        files[row["path"]] = row
    job = started["cpu"]["job_id"]
    parent = Path(started["cpu"]["argv"][started["cpu"]["argv"].index("-oo") + 1]).parent.as_posix()
    own = json.loads(files[parent + "/submission.json"]["text"])
    assert own == started["cpu"]
    states = re.findall(r"Job <" + re.escape(job) + r">.*?Status\s*<([^>]+)>", raw["scheduler"]["stdout"], re.S)
    assert states in (["DONE"], ["EXIT"]), "Actual terminal CPU scheduler proof not present"
    log = files[parent + "/stdout." + job + ".log"]["text"]
    compact = "".join(log.split())
    name=own['argv'][own['argv'].index('-J')+1]
    assert name in ('mosaic-pro-precision-cpu','mosaic-pro-precision02-cpu',
                    'mosaic-pro-precision03-cpu','mosaic-pro-precision04-cpu','mosaic-pro-precision05-cpu',
                    'mosaic-pro-precision06-cpu','mosaic-pro-precision07-cpu')
    assert compact.count("Subject:Job") == 1 and re.findall(
        r'Subject:Job(\d+):<([^>]+)>incluster<([^>]+)>(Done|Exited)',compact)==[(job,name,'dcc','Done' if states==['DONE'] else 'Exited')]
    assert compact.count("inqueue<hpc>") == 1
    passed = states == ["DONE"]
    assert ("Successfullycompleted." in compact) == passed
    if not passed:
        assert "Exitedwithexitcode1." in compact
    tree = ET.fromstring(files[parent + "/payload/tests.xml"]["text"])
    cases = list(tree.iter("testcase"))
    counts = dict(Counter(case.get("classname") for case in cases))
    assert counts == manifest["cpu_counts"]
    assert len({(c.get("classname"), c.get("name")) for c in cases}) == len(cases)
    failed = [{"module": c.get("classname"), "name": c.get("name"),
               "outcomes": [{"type": child.tag, "message": child.get("message"), "text": child.text}
                            for child in c]} for c in cases if list(c)]
    assert bool(not failed) == passed, "Scheduler/fixture result mismatch"
    startup = json.loads(files[parent + "/payload/native-startup/endpoint-startup.json"]["text"])
    assert startup == {"status": "passed_metadata_only", "numerical_setup_entered": False, "updates": 0,
                       "plan_sha256": started["preparation"]["plan_sha256"], "job_id": job}
    return {"status": "independently_verified_cpu_pass" if passed else "independently_verified_cpu_failure",
            "captured_utc": raw["captured_utc"], "cpu_job_id": job, "scheduler_status": states[0],
            "source_manifest_sha256": pin, "files_byte_verified": len(files), "cases": len(cases),
            "module_counts": counts, "failed_cases": failed, "actual_startup_passed_metadata_only": True,
            "native_result_established": False, "optimizer_updates": 0,
            "observation_sha256": hashlib.sha256(observation.read_bytes()).hexdigest(),
            "original_results_changed": False, "local_scientific_calculations": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.observation, args.source, args.submission)
    with Path(args.output).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "failed_cases"}, indent=2))


if __name__ == "__main__":
    main()
