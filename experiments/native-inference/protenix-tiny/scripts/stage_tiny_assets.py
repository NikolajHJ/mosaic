"""Stage only the five pinned native Tiny assets, sequentially and with bounds."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    args.destination.mkdir(parents=True, exist_ok=True)
    receipts = []
    for asset in manifest["files"]:
        name = asset["rfilename"]
        if Path(name).name != name:
            raise ValueError("Only root-level assets are supported")
        path = args.destination / name
        expected_hash = (asset.get("lfs") or {}).get("sha256")
        if not path.exists():
            url = (f"https://huggingface.co/{manifest['repository']}/resolve/"
                   f"{manifest['revision']}/{name}")
            partial = path.with_name(name + ".part")
            # Unique setup attempt; retain an incomplete transfer instead of
            # silently overwriting it or repeatedly retrying a failed download.
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("xb") as handle:
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > asset["size"]:
                        raise RuntimeError(f"Asset exceeded its declared size: {name}")
                    handle.write(chunk)
            if count != asset["size"]:
                raise RuntimeError(f"Incomplete asset: {name}: {count} bytes")
            if expected_hash and sha256(partial) != expected_hash:
                raise RuntimeError(f"Asset checksum mismatch: {name}")
            partial.rename(path)
        actual_hash = sha256(path)
        if path.stat().st_size != asset["size"] or (expected_hash and actual_hash != expected_hash):
            raise RuntimeError(f"Existing asset does not match manifest: {name}")
        receipts.append({"file": name, "bytes": path.stat().st_size, "sha256": actual_hash})
        print(f"Verified {name}: {path.stat().st_size} bytes", flush=True)
        (args.destination / "asset_receipt.json").write_text(json.dumps({
            "repository": manifest["repository"], "revision": manifest["revision"],
            "verified": receipts,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
