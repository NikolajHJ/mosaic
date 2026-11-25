import gzip

import pytest

from latentmol.datasets.bulk_ingest import ingest
from latentmol.experiments import file_sha256


def _accepted(path):
    with gzip.open(path / "accepted.tsv.gz", "rt", encoding="utf-8") as stream:
        return [line.rstrip("\n").split("\t") for line in stream]


def test_pubchem_gzip_matches_serial_parallel_and_resumes(tmp_path):
    source = tmp_path / "cid-smiles.gz"
    source.write_bytes(gzip.compress(b"1\tCCO\n2\tOCC\n3\tc1ccccc1\n4\tC\n5\t[Na+]\n6\tC.C\nmissing cid\n"))
    kwargs = dict(source="pubchem", input_format="pubchem", min_atoms=2, input_sha256=file_sha256(source))
    first, second = tmp_path / "serial", tmp_path / "parallel"
    serial = ingest(source, first, **kwargs)
    parallel = ingest(source, second, workers=2, **kwargs)
    assert serial["counts"] == {"source_rows": 7, "accepted": 3, "rejected": 4}
    assert serial["accepted_heavy_atom_counts"] == {"3": 2, "6": 1}
    assert serial["rejections"] == {"below_min_atoms": 1, "unsupported_element": 1,
                                    "disconnected": 1, "malformed_input_row": 1}
    assert _accepted(first) == [["CCO", ""], ["CCO", ""], ["c1ccccc1", "c1ccccc1"]]
    assert parallel["output"] == serial["output"]
    manifest = first / "manifest.json"
    before = manifest.stat().st_mtime_ns
    assert ingest(source, first, workers=2, **kwargs) == serial
    assert manifest.stat().st_mtime_ns == before


def test_plain_smiles_limit_and_collision_checks(tmp_path):
    source = tmp_path / "input.smi"
    source.write_text("# sample\n\nSMILES Name\nOCC ethanol\nCCC propane\nCCN amine\n", encoding="utf-8")
    output = tmp_path / "normalized"
    kwargs = dict(source="zinc", input_format="smiles")
    receipt = ingest(source, output, limit=2, **kwargs)
    assert _accepted(output) == [["CCO", ""], ["CCC", ""]]
    assert receipt["counts"] == {"source_rows": 2, "accepted": 2, "rejected": 0}
    with pytest.raises(ValueError, match="different policy or schema"):
        ingest(source, output, **kwargs)
    with (output / "accepted.tsv.gz").open("ab") as stream:
        stream.write(b"invalid")
    with pytest.raises(ValueError, match="bytes differ"):
        ingest(source, output, limit=2, **kwargs)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        ingest(source, tmp_path / "missing", input_sha256="0" * 64, **kwargs)


def test_incomplete_raw_gzip_can_be_retried_without_partial_output(tmp_path):
    source = tmp_path / "input.gz"
    compressed = gzip.compress(b"1\tCCO\n2\tCCC\n")
    source.write_bytes(compressed[:-3])
    output = tmp_path / "normalized"
    kwargs = dict(source="pubchem", input_format="pubchem")
    with pytest.raises(EOFError):
        ingest(source, output, **kwargs)
    assert not output.exists()
    source.write_bytes(compressed)
    assert ingest(source, output, **kwargs)["counts"]["accepted"] == 2
