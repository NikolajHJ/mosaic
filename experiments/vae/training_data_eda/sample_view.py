"""Census connected-graph cycle ranks and sample canonical training SMILES."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def manifest(path, expected, version):
    data = json.loads((path / "manifest.json").read_bytes())
    payload = {k: v for k, v in data.items() if k not in {"created_at", "dataset_sha256"}}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    if (hashlib.sha256(raw).hexdigest() != expected or data["dataset_sha256"] != expected
            or data["version"] != version):
        raise ValueError(f"Unexpected manifest at {path}")
    return data


def extract(view_root, expected_sha, output, panel_size, seed, chunk=131072):
    view = manifest(view_root, expected_sha, "balanced-packed-sparse-view-v1")
    parent_root = Path(view["provenance"]["parent_cache"])
    parent_sha = view["provenance"]["parent_cache_sha256"]
    parent = manifest(parent_root, parent_sha, "packed-sparse-graph-cache-v1")
    total = view["counts"]["train"]
    n_parent = parent["counts"]["train"]
    max_atoms = view["spec"]["max_nodes"]
    if parent["spec"] != view["spec"] or not 1 <= panel_size <= total or chunk < 1:
        raise ValueError("Incompatible schemas or invalid sample/chunk size")
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    index_path = view_root / "train.u32"
    data_path = parent_root / "train.graphs.bin"
    offsets_path = parent_root / "train.offsets.bin"
    for path, record in ((index_path, view["files"]), (data_path, parent["files"]),
                         (offsets_path, parent["files"])):
        if path.stat().st_size != record[path.name]["bytes"]:
            raise ValueError(f"Stored file length differs for {path.name}")
    if index_path.stat().st_size != total * 4 or offsets_path.stat().st_size != (n_parent + 1) * 8:
        raise ValueError("Wrong train index or offsets length")
    with index_path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != view["files"][index_path.name]["sha256"]:
            raise ValueError("Training membership digest mismatch")
    indices = np.memmap(index_path, dtype="<u4", mode="r")
    for start in range(0, total - 1, chunk):
        stop = min(start + chunk, total - 1)
        if np.any(indices[start:stop] >= indices[start + 1:stop + 1]):
            raise ValueError("Non-increasing training index")
    if int(indices[-1]) >= n_parent:
        raise ValueError("Training index exceeds parent length")

    positions = np.sort(np.random.Generator(np.random.PCG64(seed)).choice(
        total, size=panel_size, replace=False))
    # A connected simple graph has rank E - V + 1, at most (V-1)(V-2)/2.
    rank_bins = (max_atoms - 1) * (max_atoms - 2) // 2 + 1
    histogram = np.zeros((max_atoms + 1, rank_bins), dtype=np.int64)
    graph_hash = hashlib.sha256()
    offset_hash = hashlib.sha256()
    written = cursor = byte_position = 0
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".view-scan-") as temporary:
        stage = Path(temporary) / "inputs"
        stage.mkdir()
        with (gzip.open(stage / "panel.tsv.gz", "wt", encoding="utf-8", newline="") as panel,
              data_path.open("rb") as graphs, offsets_path.open("rb") as offsets):
            writer = csv.writer(panel, delimiter="\t", lineterminator="\n")
            writer.writerow(["view_row", "parent_row", "heavy_atoms", "cycle_rank", "smiles"])
            sentinel = offsets.read(8)
            if sentinel != bytes(8):
                raise ValueError("Sparse offsets must begin at zero")
            offset_hash.update(sentinel)
            for start in range(0, n_parent, chunk):
                stop = min(start + chunk, n_parent)
                raw_offsets = offsets.read((stop - start) * 8)
                offset_hash.update(raw_offsets)
                ends = np.frombuffer(raw_offsets, dtype="<u8")
                if len(ends) != stop - start or ends[0] <= byte_position or np.any(ends[1:] <= ends[:-1]):
                    raise ValueError(f"Invalid parent offsets at row {start}")
                end = int(ends[-1])
                raw = graphs.read(end - byte_position)
                if len(raw) != end - byte_position:
                    raise ValueError("Truncated parent graphs")
                graph_hash.update(raw)
                next_cursor = int(indices.searchsorted(stop))
                rows = indices[cursor:next_cursor].astype(np.int64) - start
                # Offsets cover all parent rows, while the census selects only view members.
                starts = np.empty(len(ends), dtype=np.int64)
                starts[0] = 0
                starts[1:] = ends[:-1] - byte_position
                head = starts[rows]
                block = np.frombuffer(raw, dtype=np.uint8)
                size = block[head].astype(np.int64)
                edges = block[head + 1].astype(np.int64) + 256 * block[head + 2].astype(np.int64)
                rank = edges - size + 1
                smiles_start = head + 3 + 2 * size + 3 * edges
                row_end = ends[rows].astype(np.int64) - byte_position
                if (np.any(size < 1) or np.any(size > max_atoms) or np.any(rank < 0)
                        or np.any(edges > size * (size - 1) // 2) or np.any(smiles_start >= row_end)):
                    raise ValueError(f"Invalid packed graph header in parent block {start}")
                flat = size * rank_bins + rank
                histogram += np.bincount(flat, minlength=histogram.size).reshape(histogram.shape)
                next_written = int(positions.searchsorted(next_cursor))
                for pos in positions[written:next_written]:
                    i = int(pos) - cursor
                    smiles = raw[smiles_start[i]:row_end[i]].decode("utf-8")
                    writer.writerow([pos, start + rows[i], size[i], rank[i], smiles])
                cursor, written, byte_position = next_cursor, next_written, end
            if graphs.read(1) or offsets.read(1):
                raise ValueError("Unexpected trailing data")
        if (graph_hash.hexdigest() != parent["files"][data_path.name]["sha256"]
                or offset_hash.hexdigest() != parent["files"][offsets_path.name]["sha256"]):
            raise ValueError("Parent training data digest mismatch")
        expected_counts = view["active_atom_counts"]["train"]
        sizes = histogram.sum(axis=1)
        if (written != panel_size or cursor != total or sizes.sum() != total
                or any(sizes[i] != expected_counts[str(i)] for i in range(1, max_atoms + 1))):
            raise ValueError("Scanned membership or size distribution differs from manifest")
        joint = [dict(heavy_atoms=int(n), cycle_rank=int(r), count=int(histogram[n, r]))
                 for n, r in zip(*np.nonzero(histogram))]
        result = dict(dataset_sha256=expected_sha, parent_dataset_sha256=parent_sha, split="train",
                      parent_rows=n_parent, train_rows=total, sample_size=written, sample_seed=seed,
                      sampling="numpy.random.Generator(PCG64).choice without replacement, sorted view rows",
                      size_counts=expected_counts, cycle_rank_counts={str(i): int(v) for i, v in
                      enumerate(histogram.sum(axis=0)) if v}, joint_counts=joint,
                      exact_size_histogram_matches_manifest=True, verified_train_file_sha256=True,
                      rdkit_preparation_version=view["rdkit_version"], numpy_version=np.__version__,
                      job_id=os.environ.get("LSB_JOBID"))
        (stage / "census.json").write_text(json.dumps(result, indent=2) + "\n")
        if os.path.lexists(output):
            raise FileExistsError(output)
        stage.rename(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    result = extract(args.view, args.dataset_sha256, args.output, args.sample_size, args.seed)
    print(f"Verified {result['train_rows']} training graphs and saved {result['sample_size']} molecules.")
