import pytest
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

from latentmol.evaluation.synthesis_accessibility import audited_sa_score


def test_audited_sa_preserves_easy_score_and_excludes_upper_tail_outlier():
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    assert audited_sa_score(aspirin) == pytest.approx(sascorer.calculateScore(aspirin))
    archived = Chem.MolFromSmiles("CC1(C)C2C3C4OCC5CC(C21)N1C(O)NNC5(C2C5CSC1C52)C43")
    assert audited_sa_score(archived) == pytest.approx(8.001070858627374)


def test_audited_sa_rejects_an_empty_molecule():
    with pytest.raises(ValueError, match="empty molecule"):
        audited_sa_score(Chem.Mol())
