"""Independent combined FP64 probes against unchanged Protenix production AD.

The original native checker, loss, optimizer and fixed teacher are untouched.
Only the validation measurement uses a separately authenticated full-double
native worker and a private CPU64 copy of the original chemistry function.
The original represented FP32 logit stencil and prospective acceptance policy
remain in force. Every point records both precisions; they are not bit-identical.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

VERSION='protenix-fixed-chemistry-full-reference-v1'


def require(value,message):
    if not value:raise ValueError(message)


def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def validated_config(config):
    require(config.get('repair_weight',0)>0 and config.get('element_js_weight')==0
            and config.get('objective_mode')=='native_plus_chemistry'
            and 'confidence_margin' not in config and 'neighbor_environment' not in config,
            'Only the unchanged positive fixed chemistry/composition-off objective is supported')


def compare_direction(production,reference,relative_tolerance=.15):
    """Declared useful-contrast comparison; no unit-scale absolute floor."""
    import math
    require(relative_tolerance==.15,'Original directional relative tolerance is fixed')
    require(math.isfinite(production) and math.isfinite(reference),'Nonfinite reference contrast')
    magnitude=max(abs(production),abs(reference))
    return magnitude>1e-7 and production*reference>0 and abs(production-reference)<=relative_tolerance*magnitude


def reference_loss_config(private,config):
    # A private execution defines its own dataclass identity. Copy only the
    # original immutable field values; never pass another module's class object.
    return private.FailureGuidanceConfig(**vars(config))


def independent_scalar(value):
    """Actual full-double evaluator output, not a promoted FP32 scalar."""
    import numpy as np
    array=np.asarray(value)
    require(array.shape==() and array.dtype==np.float64 and bool(np.isfinite(array)),
            'Authenticated combined reference must actually return finite FP64')
    return {'value':float(array),'dtype':'float64','resolution_ulp':float(abs(np.spacing(array))),
            'resolution_scope':'independent_full_FP64_binding_and_chemistry_not_FP32_wrapper'}


def reference_logit_evaluate(values,primal,autodiff,frozen,*,seal_callback=None):
    """Keep the exact comparator/planner; declare the actual reference precision.

    Legacy _scalar deliberately charges at least FP32 resolution. That is right
    for wrapped production forwards, but false for a genuine complete FP64
    reference. Specialize only that typed measurement adapter, never epsilons,
    signal multipliers, pair selection, relative tolerance or pass criteria.
    Caller must retain authenticated full-worker and chemistry precision proof.
    """
    import logit_admission
    private=types.SimpleNamespace(**vars(logit_admission.a));private._scalar=independent_scalar
    namespace={**vars(logit_admission),'a':private}
    exec(compile(inspect.getsource(logit_admission.evaluate),logit_admission.__file__,'exec'),namespace)
    result=namespace['evaluate'](values,primal,autodiff,frozen,seal_callback=seal_callback)
    result['reference_measurement_precision_policy']='authenticated-full-FP64-scalar-v1'
    return result


def combined(self, original, *, owner, reference_module):
    def wrapped(loss,evaluation,prepared,spec,reference,*,key,config,directory=None):
        import numpy as np
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        import logit_admission
        from mosaic.small_molecule.chemistry.trajectory import evaluate_combined
        from mosaic.small_molecule.chemistry.state import GraphLogits
        from mosaic.small_molecule.chemistry import loss as chemistry
        validated_config(config)
        require(self.name=='protenix' and directory is not None,'Protenix combined-only reference')
        directory=Path(directory)
        require(directory.is_relative_to(self.output),'Combined reference escaped current owner')
        directory.mkdir(exist_ok=False)
        values={f:np.asarray(getattr(evaluation.logits,f)).copy() for f in ('atoms','bonds')}
        require(all(v.dtype==np.float32 for v in values.values()),'Original production logits must remain FP32')
        identity=prepared.identity;prepared.verify(spec)
        # Re-execute exact sealed source, without the production GPU bridge patches.
        private=types.ModuleType('_combined_independent_CPU64_chemistry')
        private.__package__=chemistry.__package__;private.__file__=chemistry.__file__
        sys.modules[private.__name__]=private
        exec(compile(Path(chemistry.__file__).read_bytes(),chemistry.__file__,'exec'),private.__dict__)
        reference_config=reference_loss_config(private,config['loss'])
        fixed_config=json.dumps(config,sort_keys=True,default=lambda v:vars(v))
        original_frozen=self._frozen(loss,directory=directory)
        count=0;production_records=[];reference_records=[]
        def production(point):
            nonlocal count
            state=evaluation.logits
            for f in ('atoms','bonds'):
                state=eqx.tree_at(lambda v:getattr(v,f),state,jax.device_put(point[f],getattr(state,f).device))
            value=evaluate_combined(loss,state,prepared,spec,reference,key=key,config=config)
            gradients={f:np.asarray(getattr(value.gradients,f)).copy() for f in values}
            graph=state.graph();path=directory/('production-'+str(count).zfill(4)+'.npz');count+=1
            np.savez_compressed(path,**point,X=np.asarray(graph.X),E_upper=np.asarray(graph.E_upper),
                primal=np.float64(value.primal),AD_value=np.float64(value.value),
                **{'gradient_'+f:g for f,g in gradients.items()})
            record={'primal':float(value.primal),'AD_value':float(value.value),
                'gradients':{f:hashlib.sha256(g.tobytes()).hexdigest() for f,g in gradients.items()},
                'raw':reference_module.binding(path),'frozen':original_frozen()}
            production_records.append(record)
            return value,gradients
        before,_=production(values)
        cfg=deepcopy(owner['reference_configuration']['arms']['protenix'])
        cfg['command_prefix']=reference_module.expanded_command(cfg,owner['_sha256'])
        session=None;result=None
        try:
            session=reference_module.Session(cfg,directory/'reference-session',reference_module._key_data(key))
            def frozen():
                prepared.verify(spec)
                require(prepared.identity==identity and fixed_config==json.dumps(config,sort_keys=True,default=lambda v:vars(v)),
                        'Detached teacher/coefficient mutated during reference trials')
                return {'production':original_frozen(),'reference':session.identity['frozen'],'preparation':identity}
            def independent(point,gradient):
                # Widen exact represented original logits; softmax64 probabilities
                # are explicitly different from the actual FP32 production graph.
                with jax.enable_x64(True),jax.default_device(jax.devices('cpu')[0]):
                    state=GraphLogits(**{f:jnp.asarray(point[f],dtype=jnp.float64) for f in values})
                    graph=state.graph();raw={f:np.asarray(getattr(graph,f)) for f in ('X','E_upper')}
                    native=session.evaluate(raw,gradient)
                    def chemical(v):
                        return np.float64(config['repair_weight'])*private.failure_aware_repair_loss(v.decoded(),prepared,spec,reference_config)[0]
                    if gradient:
                        scalar,raw_gradient=native
                        chemical_value,cg=jax.value_and_grad(chemical)(state)
                        pulled={}
                        for f,field in (('atoms','X'),('bonds','E_upper')):
                            probability=getattr(graph,field);g=jnp.asarray(raw_gradient[field],dtype=jnp.float64)
                            pulled[f]=np.asarray(probability*(g-jnp.sum(probability*g,-1,keepdims=True))+getattr(cg,f))
                    else:
                        scalar=native;chemical_value=chemical(state);pulled=None
                    total=np.float64(scalar)+np.float64(chemical_value)
                    require(np.asarray(chemical_value).dtype==np.float64 and np.isfinite(total)
                            and (pulled is None or all(g.dtype==np.float64 and np.isfinite(g).all() for g in pulled.values())),
                            'Independent combined reference must retain actual FP64 scalar and gradient operations')
                    path=directory/('combined-reference-'+str(len(reference_records)).zfill(4)+'.npz')
                    np.savez_compressed(path,**point,**raw,total=total,native=scalar,weighted_chemistry=np.asarray(chemical_value),
                                        **({} if pulled is None else {'gradient_'+f:g for f,g in pulled.items()}))
                    reference_records.append({'raw':reference_module.binding(path),'gradient':gradient})
                    return (total,pulled) if gradient else total
            def ad(point):
                live,g=production(point);scalar,independent_gradient=independent(point,True)
                require(abs(float(live.primal)-float(scalar))<=2e-5,'Independent full-objective value parity failed')
                for f in values:
                    row=int(np.ptp(g[f],axis=-1).argmax());source=int(g[f][row].argmin());target=int(g[f][row].argmax())
                    p=float(g[f][row,target]-g[f][row,source]);r=float(independent_gradient[f][row,target]-independent_gradient[f][row,source])
                    require(compare_direction(p,r),'Independent production/reference logit contrast disagreement: '+f)
                # FD measures the independent FP64 forward; expected displacement
                # still uses the actual unchanged production gradient.
                return scalar,g
            result=reference_logit_evaluate(values,lambda p:independent(p,False),ad,frozen,
                seal_callback=lambda seal:write(directory/'pair-seal.json',seal))
            session.finish();production(values)
            require(all(v==production_records[0] for v in [
                {**v,'raw':production_records[0]['raw']} for v in production_records]),
                'Original production changed across combined reference co-residency')
            result.update(preparation_identity=identity,repair_weight=float(config['repair_weight']),
                reference_policy=VERSION,original_optimizer_or_reference_changed=False,
                precision_scope='FP64 independent combined forward against original FP32 production AD at represented FP32 logits',
                softmax_probabilities_bitidentical_between_precisions=False,
                production_records=production_records,reference_records=reference_records,
                original_failure_preserved=True)
            write(directory/'report.json',result)
            # Retain original coarse checks as distinct observations.
            legacy=directory/'original-coarse'
            try:original(loss,evaluation,prepared,spec,reference,key=key,config=config,directory=legacy)
            except AssertionError as error:
                require(str(error) in ('Combined frozen-reference atoms derivative check failed','Combined frozen-reference bonds derivative check failed'),
                        'Unexpected original coarse error')
            self.logit_records.append({'report':reference_module.binding(directory/'report.json'),'status':result['status'],
                'version':logit_admission.VERSION,'reference_policy':VERSION,
                'original_coarse_report':reference_module.binding(legacy/'report.json')})
            require(result['status']=='passed' and result['can_admit_new_child'] and result['frozen_unchanged'],
                    'Independent combined reference/production logit gate failed')
            return result['directions']
        finally:
            if session is not None and not session.closed:session.abort()
            if result is not None and not (directory/'report.json').exists():write(directory/'report.json',result)
    return wrapped


@contextmanager
def install(chemistry_hooks,owner,reference_module):
    previous=chemistry_hooks.combined_method
    def replacement(original):
        def method(state,coarse):return combined(state,coarse,owner=owner,reference_module=reference_module)
        return method
    chemistry_hooks.combined_method=replacement
    try:yield
    finally:chemistry_hooks.combined_method=previous
