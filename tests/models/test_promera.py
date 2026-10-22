
# Shared test import paths; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__, include_tests=True)

import jax
import numpy as np

from mosaic.structure_prediction import PolymerType, TargetChain


def test_structure_writer_repacks_binder_and_preserves_rna():
    from jpromera.features import featurize

    from mosaic.models.promera import _schema, _StructureWriter

    chains = [
        TargetChain("XX", use_msa=False),
        TargetChain("ACG", polymer_type=PolymerType.RNA, use_msa=False),
    ]
    schema = _schema(chains)
    _features, placeholder_structure = featurize(schema, build_msa=False)
    writer = _StructureWriter(placeholder_structure, schema)

    designed_schema = {chain_id: dict(chain) for chain_id, chain in schema.items()}
    designed_schema["A"]["sequence"] = "AG"
    designed_features, _designed_structure = featurize(
        designed_schema,
        build_msa=False,
    )
    pssm = jax.nn.one_hot(np.asarray([0, 7]), 20)
    structure = writer(
        np.asarray(designed_features.ref_pos[0]),
        binder_pssm=pssm,
    )

    assert [chain.name for chain in structure[0]] == ["A", "B"]
    assert [[residue.name for residue in chain] for chain in structure[0]] == [
        ["ALA", "GLY"],
        ["A", "C", "G"],
    ]
