"""The next-epoch teacher swap preserves inherited objective/RNG metadata."""
import copy
import json

import pytest

from latentmol.vae_neighborhood_continuation import STEP, structural_sampler_start


def fixture(tmp_path):
    source=tmp_path/'source-study.json';source.write_text(json.dumps({'archive_sha256':'d'*64}))
    neighborhood={'version':'local-latent-chemistry-v1','fraction':.25,'noise_scale':.05,'ramp_updates':100}
    study={'structural_adoption':{'source_study':str(source),'source_study_sha256':'b'*64},
           'probe_checkpoint':{'sha256':'c'*64},'neighborhood':neighborhood,'base_plan_sha256':'e'*64}
    base={'protocol':{'prior_ratio':2,'repair_weight':.5}}
    payload={'epoch':44,'source_archive_sha256':'d'*64,'continuation':{'updates':290224},
        'chemistry_phase':{'plan_sha256':'b'*64,'protocol':base['protocol'],'optimizer_step_control':STEP,
            'neighborhood_continuation':{'study_sha256':'b'*64,'base_plan_sha256':'e'*64,
                                         'policy':neighborhood,'start_update':283628}}}
    return study,base,payload


def test_exact44_adoption_preserves_original_local_ramp_start(tmp_path):
    study,base,payload=fixture(tmp_path);before=copy.deepcopy(payload)
    assert structural_sampler_start(study,base,payload,'c'*64)==283628
    assert payload==before


@pytest.mark.parametrize('field',['epoch','protocol','optimizer','archive','ramp','checkpoint'])
def test_structural_adoption_rejects_changed_inherited_policy(tmp_path,field):
    study,base,payload=fixture(tmp_path);sha='c'*64
    if field=='epoch':payload['epoch']=43
    elif field=='protocol':payload['chemistry_phase']['protocol']={}
    elif field=='optimizer':payload['chemistry_phase']['optimizer_step_control']={}
    elif field=='archive':payload['source_archive_sha256']='f'*64
    elif field=='ramp':payload['chemistry_phase']['neighborhood_continuation']['start_update']=290224
    else:sha='f'*64
    with pytest.raises(ValueError,match='exact completed epoch44'):
        structural_sampler_start(study,base,payload,sha)
