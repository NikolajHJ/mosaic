import json
from pathlib import Path
import subprocess

import pytest

from latentmol.evaluation.vina_docking import VinaDockingScorer


def test_native_pose_and_stereo_receipt(tmp_path, monkeypatch):
    pytest.importorskip("meeko")
    receptor, box = tmp_path / "r.pdbqt", tmp_path / "box.json"
    receptor.touch()
    box.write_text(json.dumps({"center": [28, 15, -2], "size": [20, 19, 21]}))
    monkeypatch.setenv("VINA_EXECUTABLE", "vina-test")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "Vina test")
    scorer = VinaDockingScorer(receptor, box)
    affinity = "-6.5"

    def run(args, **kwargs):
        assert "--score_only" not in args and "--local_only" not in args
        assert args[args.index("--center_z") + 1] == "-2.0"
        assert args[args.index("--size_z") + 1] == "21.0"
        Path(args[args.index("--out") + 1]).write_text(f"REMARK VINA RESULT: {affinity}  0.0  0.0\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    pose = tmp_path / "best.pdbqt"
    result = scorer.score("CC(O)Cl", pose_path=pose)
    assert result["status"] == "ok" and result["affinity_kcal_mol"] == -6.5
    assert result["unspecified_stereo_count"] == 1
    assert "@" in result["embedded_isomeric_smiles"]
    assert "@" not in result["canonical_smiles"]
    with pytest.raises(FileExistsError):
        scorer.score("CC(O)Cl", pose_path=pose)
    affinity = "nan"
    bad = scorer.score("CC")
    assert bad["status"] == "failed" and bad["affinity_kcal_mol"] is None
    assert scorer.score("C.C")["status"] == "failed"
