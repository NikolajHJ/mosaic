import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae"
          / "restart-operations/evaluate_boltz_direct.py")
spec = importlib.util.spec_from_file_location("boltz_direct_readout", SCRIPT)
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)


class MockDocking:
    def __init__(self, founder_failed):
        self.calls = []
        self.founder_failed = founder_failed

    def score_many(self, names):
        self.calls.append(names)
        scores = {"CC": None if self.founder_failed else -3., "CCO": None, "CC=O": -5., "C1CC1": -4.}
        return {s: dict(status="failed", reason="embedding", affinity_kcal_mol=None) if scores[s] is None
                else dict(status="ok", affinity_kcal_mol=scores[s]) for s in names}

    def accounting(self):
        return {"unique_attempts": sum(map(len, self.calls))}


@pytest.mark.parametrize("founder_failed", [False, True])
def test_direct_readout_scores_distinct_top_two_and_keeps_failures(tmp_path, founder_failed):
    def arm(smiles):
        return dict(seed=12, founder_smiles="CC", status="completed", fresh_attempts=len(smiles),
                    fresh_successes=len(smiles), top20=[dict(smiles=s, loss=v, step=i) for i, (s, v) in enumerate(smiles)])

    source = dict(status="completed", optimization_space="relaxed_graph_logits", seed=12, founder_smiles="CC",
                  variants=dict(control=arm([("CC", -7), ("OCC", -5), ("CCO", -6), ("C1CC1", -3), ("CCC", -2)]),
                                motif=arm([("CCO", -5), ("CC=O", -3)]), triangle=arm([]), both=arm([("CCO", -4)])))
    folder = tmp_path / "control"
    folder.mkdir()
    (folder / "metrics.jsonl").write_text("\n".join(json.dumps({"smiles": s}) for s in
                                                  ["CC", "CCO", "OCC", "C1CC1", "CCC", "CC=O", None]))
    mock = MockDocking(founder_failed)
    result = readout.evaluate(tmp_path, source, mock)
    assert mock.calls == [sorted(["CC", "CCO", "CC=O", "C1CC1"])]
    assert result["repeated_slots"] == 2
    assert result["variants"]["control"]["selected"][0]["hard_dbmol_loss"] == -6
    assert result["variants"]["control"]["selected"][0]["vina"]["reason"] == "embedding"
    assert result["variants"]["motif"]["best_minus_founder_kcal_mol"] == (None if founder_failed else -2.)
    assert result["variants"]["both"]["docked"] == 0 and result["variants"]["both"]["best_vina"] is None
    assert result["variants"]["triangle"]["selected"] == []
    chemistry = result["variants"]["control"]["chemistry"]
    assert (chemistry["scope"], chemistry["distinct_new"], chemistry["with_carbonyl"]) == ("all_valid_steps", 4, 1)
