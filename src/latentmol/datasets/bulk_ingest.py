"""Stream one raw molecule shard into immutable canonical SMILES and scaffolds."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import gzip
from itertools import islice
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from rdkit import rdBase

from latentmol.chemistry import MoleculeRejected, encode_smiles
from latentmol.datasets.chembl_audit import _ordered_results, _scaffold_and_size
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import GraphSpec


VERSION = "bulk-canonical-smiles-ingest-v1"
OUTPUT_FILE = "accepted.tsv.gz"
FORMATS = ("smiles", "pubchem")


def _rows(path, input_format):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        first = True
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if input_format == "pubchem":
                fields = raw.rstrip("\r\n").split("\t")
                if len(fields) != 2 or not fields[0].isascii() or not fields[0].isdigit():
                    yield None
                else:
                    yield fields[1].strip()
            else:
                smiles = line.split(None, 1)[0]
                if not (first and smiles.lower() == "smiles"):
                    yield smiles
            first = False


def _canonical_rows(batch, spec, min_atoms):
    results = []
    for text in batch:
        if text is None:
            results.append((None, None, 0, "malformed_input_row"))
            continue
        try:
            smiles = encode_smiles(text, spec)["smiles"]
        except MoleculeRejected as exc:
            results.append((None, None, 0, exc.reason))
            continue
        scaffold, heavy_atoms = _scaffold_and_size(smiles)
        if heavy_atoms < min_atoms:
            results.append((None, None, 0, "below_min_atoms"))
        else:
            results.append((smiles, scaffold, heavy_atoms, None))
    return results


def _existing(output, identity):
    """Authenticate a complete shard before skipping already committed work."""
    receipt_path = output / "manifest.json"
    accepted_path = output / OUTPUT_FILE
    if output.is_symlink() or not receipt_path.is_file() or not accepted_path.is_file():
        raise FileExistsError(f"Output exists without a complete ingest shard: {output}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if any(receipt.get(key) != expected for key, expected in identity.items() if key != "input"):
        raise ValueError(f"Existing ingest shard has a different policy or schema: {output}")
    if any(receipt["input"].get(key) != value for key, value in identity["input"].items() if key != "path"):
        raise ValueError(f"Existing ingest shard has different input bytes or format: {output}")
    output_metadata = receipt["output"]
    if (output_metadata["path"] != OUTPUT_FILE or accepted_path.stat().st_size != output_metadata["bytes"]
            or file_sha256(accepted_path) != output_metadata["sha256"]):
        raise ValueError(f"Existing accepted shard bytes differ from the receipt: {output}")
    return receipt


def ingest(input_path, output, *, source, input_format, spec=None, workers=1,
           input_sha256=None, min_atoms=1, limit=None):
    """Normalize one raw shard and publish its data and receipt together.

    The plain SMILES reader accepts an optional first-line SMILES header and
    ignores comments and blank lines. PubChem rows require a numeric CID and
    exactly one tab-separated SMILES field. Limits count parsed records before
    chemistry filters. Global deduplication and split assignment are separate.
    """
    spec = spec if spec is not None else GraphSpec()
    if (input_format not in FORMATS or not isinstance(source, str) or not source.strip()
            or source != source.strip() or any(char in source for char in "\t\r\n")):
        raise ValueError("Require a nonempty source label and a supported input format")
    if (type(workers) is not int or workers < 1 or type(min_atoms) is not int or not 1 <= min_atoms <= spec.max_nodes
            or limit is not None and (type(limit) is not int or limit < 1)):
        raise ValueError("Invalid worker, minimum atom count or row limit")
    started = time.monotonic()
    source_path = Path(input_path).resolve(strict=True)
    output = Path(output).absolute()
    start_stat = source_path.stat()
    digest = file_sha256(source_path)
    if input_sha256 is not None and digest != input_sha256:
        raise ValueError("Raw shard SHA-256 differs from the expected snapshot")
    identity = dict(version=VERSION, source=source, spec_fingerprint=spec.fingerprint(), spec=spec.to_dict(),
                    rdkit_version=rdBase.rdkitVersion, min_atoms=min_atoms, limit=limit,
                    input=dict(path=str(source_path), sha256=digest, bytes=start_stat.st_size, format=input_format,
                               compression="gzip" if source_path.suffix.lower() == ".gz" else "none"))
    if os.path.lexists(output):
        return _existing(output, identity)

    # The final directory is never visible partially. Separate scratch names
    # also let an interrupted job resume without cleaning another writer's work.
    stage_root = output.parent / ".ingest-stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    accepted = Counter()
    rejected = Counter()
    rows_seen = 0
    with TemporaryDirectory(dir=stage_root, prefix=output.name + ".") as scratch:
        stage = Path(scratch)
        path = stage / OUTPUT_FILE
        executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
        with executor as pool, path.open("xb") as raw, gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", compresslevel=3, mtime=0) as stream:
            rows = islice(_rows(source_path, input_format), limit)
            for rows_seen, (smiles, scaffold, atoms, reason) in enumerate(
                    _ordered_results(rows, _canonical_rows, pool, workers, spec, min_atoms), 1):
                if reason is None:
                    stream.write(f"{smiles}\t{scaffold}\n".encode("utf-8"))
                    accepted[atoms] += 1
                else:
                    rejected[reason] += 1
                if rows_seen % 100_000 == 0:
                    print(json.dumps(dict(stage="bulk_ingest", source=source, rows=rows_seen,
                                          accepted=accepted.total())), flush=True)
        end_stat = source_path.stat()
        if ((start_stat.st_size, start_stat.st_mtime_ns) != (end_stat.st_size, end_stat.st_mtime_ns)
                or not rows_seen or rows_seen != accepted.total() + rejected.total()):
            raise ValueError("Input changed or no source records could be read")
        receipt = dict(identity, counts=dict(source_rows=rows_seen, accepted=accepted.total(), rejected=rejected.total()),
                       rejections=dict(rejected), accepted_heavy_atom_counts=dict(sorted(accepted.items())),
                       output=dict(path=OUTPUT_FILE, bytes=path.stat().st_size, sha256=file_sha256(path)),
                       elapsed_seconds=time.monotonic() - started, workers=workers)
        write_json(stage / "manifest.json", receipt)
        if os.path.lexists(output):
            return _existing(output, identity)
        try:
            stage.rename(output)
        except OSError:
            if os.path.lexists(output):
                return _existing(output, identity)
            raise
    # JSON stores atom-count histogram keys as strings. Return the same shape
    # on a fresh run and an authenticated resume.
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, dest="input_path")
    parser.add_argument("--format", choices=FORMATS, required=True, dest="input_format")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--input-sha256")
    parser.add_argument("--min-atoms", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(ingest(**vars(args)), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
