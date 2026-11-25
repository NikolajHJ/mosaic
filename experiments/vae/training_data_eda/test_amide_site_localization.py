from experiments.vae.training_data_eda.amide_site_localization import auc, blinded_source
from latentmol.representation import GraphSpec


def test_two_ground_truths_with_same_precursor_present_identical_blinded_sites():
    spec = GraphSpec()
    amide = blinded_source("CN(C)C(C)=O", spec)
    aminoalcohol = blinded_source("CC(O)N(C)C", spec)
    assert amide["positive"] and not aminoalcohol["positive"]
    assert amide["clean"] != aminoalcohol["clean"]
    assert amide["precursor"] == aminoalcohol["precursor"]
    assert amide["sites"] == aminoalcohol["sites"]
    assert amide["examined_site"] == aminoalcohol["examined_site"]


def test_tied_scores_credit_half():
    assert auc([1, 3], [1, 2]) == 0.625
    assert auc([1], []) is None
