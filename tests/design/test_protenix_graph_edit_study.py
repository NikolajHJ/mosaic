import json
from pathlib import Path
from types import SimpleNamespace

import torch

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.design import protenix_graph_edit_study as study
from latentmol.design.protenix_restart_parents import RestartParent
from latentmol.design.protenix_restarts import Candidate, write_json
from latentmol.representation import GraphSpec, HardGraphBatch


def candidate(smiles, spec, step=0):
    row = encode_smiles(smiles, spec)
    hard = HardGraphBatch(*(torch.tensor(row[key], dtype=torch.long)[None]
                            for key in ("atom_types", "charges", "bond_types")))
    return Candidate(row["smiles"], torch.zeros((1, 3)), hard, 0.0, "fixture", step)


def test_one_hop_graph_edits_are_seeded_connected_and_unique():
    spec = GraphSpec(max_nodes=8)
    parent = candidate("CCCOC", spec)
    pool, counts = study.graph_edit_pool(parent, spec, seed=18, budget=80)
    assert (pool, counts) == study.graph_edit_pool(parent, spec, seed=18, budget=80)
    assert counts["checked"] <= 80
    assert counts["checked"] == len(pool) + sum(counts["rejected"].values())
    assert counts["rejected"]["disconnected"] > 0
    assert {row["edit"]["kind"] for row in pool} == {"atom_substitution", "bond_reattachment"}
    assert len({row["smiles"] for row in pool}) == len(pool)
    assert parent.smiles not in {row["smiles"] for row in pool}
    excluded = {pool[0]["smiles"]}
    filtered, _ = study.graph_edit_pool(parent, spec, seed=18, budget=80, excluded=excluded)
    assert not excluded.intersection(row["smiles"] for row in filtered)

    original = parent.hard
    for row in pool:
        atoms, charges, bonds = original.atoms[0].clone(), original.charges[0], original.bonds[0].clone()
        edit = row["edit"]
        if edit["kind"] == "atom_substitution":
            atoms[edit["slot"]] = edit["atom_label"]
            assert torch.count_nonzero(atoms != original.atoms[0]) == 1
        else:
            left, right = edit["removed"]
            bonds[left, right] = bonds[right, left] = 0
            left, right = edit["added"]
            bonds[left, right] = bonds[right, left] = spec.bond_names.index("SINGLE")
            assert torch.count_nonzero(bonds.triu(1) != original.bonds[0].triu(1)) == 2
        assert torch.equal(atoms != 0, original.atoms[0] != 0)
        assert inspect_graph(atoms, charges, bonds, spec).smiles == row["smiles"]


def test_graph_study_matches_actual_fresh_attempts_and_reports_success(tmp_path, monkeypatch):
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=8)
    parents = [
        RestartParent(row, row.hard.atoms[0] != 0, 2.6, {"kind": "fixture"})
        for row in (candidate(smiles, decoder.spec) for smiles in ("CCCOC", "CCCCC"))
    ]
    calls = {}

    def latent_branch(_decoder, _score, design, branch, _point, _active, *,
                      select_shortlist, shortlist_limit, **_kwargs):
        names = ("CNCNC", "CNCCN") if "parent-00" in branch.name else ("CNCNN",)
        candidates = [candidate(smiles, decoder.spec, step=index + 1)
                      for index, smiles in enumerate(names)]
        branch.mkdir()
        write_json(branch / "summary.json", {"updates": design.updates})
        return select_shortlist(candidates, shortlist_limit)

    def fresh_score(root, records):
        arm = root.name.removeprefix("fresh-")
        calls[arm] = records
        assert len({row["smiles"] for row in records}) == len(records)
        return [dict(row, loss=None, status="embedding_failed") if arm == "graph" and index == 0
                else dict(row, loss=2.2, status="scored") for index, row in enumerate(records)]

    config_path = (Path(__file__).resolve().parents[2] / "experiments" / "latent-design"
                   / "protenix-vae" / "restart-graph-edit.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tasks"]["seed-a"]["parents"] = 2
    config["fresh_attempts"] = 2
    monkeypatch.setattr(study, "load_parent_panel", lambda *_args, **_kwargs: parents)
    monkeypatch.setattr(study, "propose_start", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(study, "run_branch", latent_branch)
    scorer = SimpleNamespace(score_at=lambda *_args, **_kwargs: None, close=lambda: None)
    output = tmp_path / "run"
    result = study.run_study(decoder, lambda *_args: scorer, fresh_score, config,
                             "seed-a", output, "fixture")

    assert result["status"] == "completed"
    assert result["by_parent"]["parent-00"]["fresh_attempt_shortfall"] == 0
    assert result["by_parent"]["parent-01"]["fresh_attempt_shortfall"] == 1
    assert [row["start"] for row in calls["latent"]] == [row["start"] for row in calls["graph"]]
    assert result["by_arm"]["graph"]["fresh_attempts"] == result["by_arm"]["latent"]["fresh_attempts"] == 3
    assert result["by_arm"]["graph"]["fresh_scored"] == 2
    assert result["by_arm"]["latent"]["fresh_scored"] == 3
    assert result["by_arm"]["graph"]["optimization_updates"] == 0
    assert result["by_arm"]["latent"]["optimization_updates"] == 1000
    assert {row["edit"]["kind"] for row in calls["graph"][:2]} == {"atom_substitution", "bond_reattachment"}
    assert result["wall_seconds"] > 0
    assert json.loads((output / "result.json").read_text(encoding="utf-8")) == result
