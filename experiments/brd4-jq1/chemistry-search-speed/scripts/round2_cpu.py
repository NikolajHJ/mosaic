"""Allocated real-input exact hybrid comparisons and separate native profiling."""
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry import prepare_failure_guidance
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    from exact_reuse import install_reuse
    from hybrid_search import install_hybrid
    from profile_oracle import profile_oracle
    from round2_data import cases,evaluation
    plan=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    data=cases(plan);search=ProbabilityRepairConfig(**plan['search']);progress=ProgressSearchConfig(**plan['progress_search'])
    baseline={};rows=[];report={'status':'running','rows':rows,'started_utc':datetime.now(timezone.utc).isoformat(),
                              'original_runs_changed':False,'scope':'Exact CPU search adapters; no optimizer/model execution'}
    try:
        for _,z,_,_ in data:
            d=z.decoded();jax.block_until_ready((d,jax.nn.softmax(d.atoms,-1),jax.nn.softmax(d.charges,-1),jax.nn.softmax(d.symmetric_bond_logits,-1)))
        for mode in ['baseline','combined','hybrid4','hybrid8']:
            ctx=nullcontext({}) if mode=='baseline' else install_reuse(mode='combined') if mode=='combined' else install_hybrid(workers=int(mode[-1]))
            with ctx as stats:
                for index,(row,z,old,spec) in enumerate(data):
                    times=[]
                    for repeat in range(3):
                        start=time.perf_counter();p=prepare_failure_guidance(z.decoded(),spec,update=index,search=search,progress_search=progress)
                        jax.block_until_ready(p.arrays);times.append(time.perf_counter()-start)
                        # Exact target and all receipt fields, not just molecule argmax.
                        assert p.identity==old.identity
                        payload=json.dumps({'receipt':p.receipt,'arrays':{k:np.asarray(v).tolist() for k,v in p.arrays.items()}},sort_keys=True)
                        if mode=='baseline' and repeat==0:baseline[row['label']]=payload
                        assert payload==baseline[row['label']],(mode,row['label'],'prepared payload')
                    loss,grad=evaluation(z,p,spec)
                    if mode=='baseline':baseline[row['label']+'-loss']=loss
                    assert loss==baseline[row['label']+'-loss'],(mode,row['label'],'loss/grad')
                    detail={'mode':mode,'input':row,'first_encounter_seconds':times[0],'seconds':times,
                            'warm_median_seconds':statistics.median(times[1:]),'preparation_identity':p.identity,
                            'loss':loss,'original_checks':p.receipt['original_checks'],'additional_checks':p.receipt['additional_checks'],
                            'adapter_stats':dict(stats),'exact_targets_receipts_loss_gradients':True}
                    rows.append(detail);(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps({k:detail[k] for k in ['mode','first_encounter_seconds','warm_median_seconds','adapter_stats']}),flush=True)
        profiles=[]
        for mode in ['baseline','combined']:
            ctx=nullcontext({}) if mode=='baseline' else install_reuse(mode='combined')
            with ctx:
                with profile_oracle() as measured:
                    _,z,_,spec=data[0]
                    p=prepare_failure_guidance(z.decoded(),spec,update=0,search=search,progress_search=progress)
                assert p.identity==data[0][2].identity
                profiles.append({'mode':mode,'oracle_profile':measured,'preparation_identity':p.identity})
        report.update(status='passed_exact_hybrid_saved_input_comparison',oracle_profiles=profiles,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      limitations=['Warm repetitions and adjacent saved inputs are distinct','No full native-model runtime or production adoption','Worker tasks are hard-graph bundles, not counts of individual RDKit calls'])
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        (out/'assessment.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
