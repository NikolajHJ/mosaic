"""Allocated real-GPU kernel parity/FD and synchronized component timings."""
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

TOLERANCE=2e-5
EPSILONS=(.01,.003,.001)


def check_case(case,method):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpu_matrix import install_matrix_loss,require_backend
    from round2_data import evaluation
    from mosaic.small_molecule.chemistry.loss import failure_aware_repair_loss
    require_backend('gpu')
    row,z,prepared,spec=case
    original,old_gradient=evaluation(z,prepared,spec)
    with install_matrix_loss(method):
        observed,gradient=evaluation(z,prepared,spec)
        np.testing.assert_allclose([observed['value'],observed['primal']],
                                   [original['value'],original['primal']],atol=TOLERANCE,rtol=TOLERANCE)
        assert observed['components'].keys()==original['components'].keys()
        np.testing.assert_allclose(list(observed['components'].values()),list(original['components'].values()),
                                   atol=TOLERANCE,rtol=TOLERANCE)
        differences={}
        directions=[]
        for field in ['atoms','bonds']:
            a=np.asarray(getattr(gradient,field));b=np.asarray(getattr(old_gradient,field))
            np.testing.assert_allclose(a,b,atol=TOLERANCE,rtol=TOLERANCE)
            assert np.isfinite(a).all()
            differences[field]=float(np.max(np.abs(a-b)))
            if not np.any(b):
                # Valid/raw routes can legitimately have no term in a field.
                assert not np.any(a)
                directions.append({'field':field,'status':'not_applicable_exact_zero_original_gradient'})
                continue
            index=np.unravel_index(np.argmax(np.abs(b)),b.shape)
            ad=float(a[index]);assert abs(ad)>1e-8,'No resolved signal for required FD'
            probes=[]
            for epsilon in EPSILONS:
                values=[]
                for sign in [-1,1]:
                    changed=getattr(z,field).at[index].add(sign*epsilon)
                    from mosaic.small_molecule.chemistry.state import GraphLogits
                    trial=GraphLogits(changed,z.bonds) if field=='atoms' else GraphLogits(z.atoms,changed)
                    value=failure_aware_repair_loss(trial.decoded(),prepared,spec)[0]
                    jax.block_until_ready(value);values.append(float(value))
                fd=(values[1]-values[0])/(2*epsilon)
                error=abs(ad-fd)/max(abs(ad),abs(fd),1e-12)
                probes.append({'epsilon':epsilon,'primal_minus':values[0],'primal_plus':values[1],
                               'analytic':ad,'finite_difference':fd,'relative_error':error,'passed':error<=.15})
            assert any(probes[k]['passed'] and probes[k+1]['passed'] for k in range(2)),(row['label'],method,field,probes)
            directions.append({'field':field,'coordinate':list(map(int,index)),'status':'passed','probes':probes})
    return {'input':row['label'],'method':method,'status':'passed','preparation_identity':prepared.identity,
            'original':original,'observed':observed,'max_abs_gradient_error':differences,
            'finite_differences':directions,'tolerance':TOLERANCE,
            'original_failure_or_readiness_status_changed':False}


def check_features(case):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpu_matrix import matrix_features,require_backend
    require_backend('gpu')
    row,z,_,spec=case
    decoded=z.decoded()
    host=[np.asarray(jnp.argmax(v,-1),dtype=np.int32) for v in
          [decoded.atoms,decoded.charges,decoded.symmetric_bond_logits]]
    with jax.default_device(jax.devices('cpu')[0]):
        cpu=matrix_features(*host,spec)
        jax.block_until_ready(cpu)
    gpu=matrix_features(*host,spec,require_gpu=True)
    jax.block_until_ready(gpu)
    for key in cpu:
        np.testing.assert_array_equal(np.asarray(cpu[key]),np.asarray(gpu[key]))
    assert 'valid' not in gpu
    return {'input':row['label'],'status':'passed_exact_integer_cpu_gpu_features','keys':list(gpu),
            'native_oracle_replaced':False}


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpu_matrix import (require_backend,matrix_features,local_valence_scan,local_valence_matrix)
    from mosaic.small_molecule.chemistry import loss
    from round2_data import cases
    require_backend('gpu');assert len(jax.devices('gpu'))==1
    plan=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    data=cases(plan)
    report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
            'devices':[{'platform':d.platform,'kind':d.device_kind,'id':d.id} for d in jax.devices()],
            'scope':'GPU matrix/local-loss components only; RDKit/search remain CPU; no optimizer or native model execution',
            'saved_runs_changed':False,'loss_checks':[],'feature_checks':[],'timings':[]}
    try:
        for case in data:
            report['feature_checks'].append(check_features(case))
            for method in ['scan','matrix']:
                report['loss_checks'].append(check_case(case,method))
        _,z,prepared,spec=data[0]
        decoded=z.decoded();indices=tuple(spec.bond_names.index(k) for k in ['NONE','SINGLE','DOUBLE','TRIPLE','AROMATIC'])
        original_local=loss._local
        for batch in [1,32,256]:
            host=[np.tile(np.asarray(jnp.argmax(v,-1),dtype=np.int32),(batch,)+(1,)*(v.ndim-2))
                  for v in [decoded.atoms,decoded.charges,decoded.symmetric_bond_logits]]
            start=time.perf_counter();device=[jax.device_put(v,jax.devices('gpu')[0]) for v in host]
            jax.block_until_ready(device);transfer=time.perf_counter()-start
            start=time.perf_counter();features=matrix_features(*device,spec,require_gpu=True)
            jax.block_until_ready(features);first=time.perf_counter()-start
            times=[]
            for _ in range(20):
                start=time.perf_counter();value=matrix_features(*device,spec,require_gpu=True);jax.block_until_ready(value);times.append(time.perf_counter()-start)
            report['timings'].append({'component':'integer_matrix_features','batch':batch,'transfer_seconds':transfer,
                                      'first_compile_and_call_seconds':first,'warm_seconds':times,'warm_median_seconds':statistics.median(times)})
            log_bonds=jnp.tile(jax.nn.log_softmax(decoded.symmetric_bond_logits,-1),(batch,1,1,1))
            active=jnp.tile(prepared.arrays['active'],(batch,1));caps=jnp.tile(prepared.arrays['caps'],(batch,1))
            reference=None
            for method,function in [('original',original_local),('scan',local_valence_scan),('matrix',local_valence_matrix)]:
                # New function identity prevents accidentally timing a cached old kernel.
                def scalar(v,f=function):return jnp.mean(f(v,active,caps,indices))
                kernel=jax.jit(jax.value_and_grad(scalar))
                start=time.perf_counter();value,gradient=kernel(log_bonds);jax.block_until_ready((value,gradient));first=time.perf_counter()-start
                if reference is None:reference=(np.asarray(value),np.asarray(gradient))
                np.testing.assert_allclose(np.asarray(value),reference[0],atol=TOLERANCE,rtol=TOLERANCE)
                np.testing.assert_allclose(np.asarray(gradient),reference[1],atol=TOLERANCE,rtol=TOLERANCE)
                assert np.isfinite(np.asarray(gradient)).all()
                times=[]
                for _ in range(20):
                    start=time.perf_counter();result=kernel(log_bonds);jax.block_until_ready(result);times.append(time.perf_counter()-start)
                report['timings'].append({'component':'local_risk_value_and_gradient','method':method,'batch':batch,
                                          'first_compile_and_call_seconds':first,'warm_seconds':times,
                                          'warm_median_seconds':statistics.median(times),'tolerance':TOLERANCE})
                print(json.dumps(report['timings'][-1]),flush=True)
        report.update(status='passed_separate_gpu_matrix_benchmark',completed_utc=datetime.now(timezone.utc).isoformat(),
                      limitations=['FP32 association is tolerance-tested, not bitwise identity','No new native chemistry/GPU search oracle','Kernel speed is not full optimization speed','No production adoption'])
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        (out/'assessment.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
