from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latentmol.guidance.protenix_pocket import map_explicit_pocket, validate_pocket


CONFIG = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae/restart-multitarget.json"


def test_all_seven_target_mappings_and_native_token_order():
    config = json.loads(CONFIG.read_text())
    expected_lengths = [254, 258, 505, 350, 501, 219, 263]
    for arm, length in zip(config["arms"].values(), expected_lengths):
        pocket = validate_pocket(arm["target"]["pocket"])
        assert len(pocket["protein_sequence"]) == length
        # Native token positions need not be sorted in input sequence order.
        protein = np.arange(length)[::-1]
        context = SimpleNamespace(protein_token_indices=protein,
            base_features=dict(residue_index=np.arange(1, length + 1), asym_id=np.ones(length)),
            identities=dict(protein_sequence=pocket["protein_sequence"]))
        assert map_explicit_pocket(context, pocket) == tuple(r["sequence_position"]-1 for r in pocket["residues"])
    assert config["arms"]["er-alpha-y537s"]["target"]["pocket"]["protein_sequence"][240] == "S"
    assert config["arms"]["vdr-h305f"]["target"]["pocket"]["protein_sequence"][140] == "F"
    assert config["arms"]["p53-y220c"]["target"]["pocket"]["protein_sequence"][126] == "C"


def test_pocket_rejects_author_number_and_sequence_mapping_errors():
    pocket = json.loads(CONFIG.read_text())["arms"]["glucocerebrosidase"]["target"]["pocket"]
    broken = deepcopy(pocket)
    broken["residues"][0]["sequence_position"] -= 2
    with pytest.raises(ValueError, match="mapping"):
        validate_pocket(broken)
    broken = deepcopy(pocket)
    broken["residues"].append(broken["residues"][0])
    with pytest.raises(ValueError, match="duplicated"):
        validate_pocket(broken)
    length = len(pocket["protein_sequence"])
    context = SimpleNamespace(protein_token_indices=np.arange(length),
        identities=dict(protein_sequence=pocket["protein_sequence"]),
        base_features=dict(residue_index=np.arange(length), asym_id=np.ones(length)))
    with pytest.raises(ValueError, match="token/residue"):
        map_explicit_pocket(context, pocket)
    context.identities["protein_sequence"] = "A" * length
    with pytest.raises(ValueError, match="sequences"):
        map_explicit_pocket(context, pocket)
