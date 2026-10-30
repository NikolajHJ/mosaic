"""Independent full-FP64 reference; never imported by production optimizers."""
import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import types
import sys


def require(value,message):
    if not value:raise ValueError(message)


def reference_tree(raw,filename):
    """Only reference dtype admission changes; every mathematical AST stays exact."""
    original=ast.parse(raw,filename=filename);tree=deepcopy(original);counts={}
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef) or node.name not in ('validate_graph','contact_distogram_float64'):continue
        count=0
        for child in ast.walk(node):
            if isinstance(child,ast.Attribute) and isinstance(child.value,ast.Name) and child.value.id=='torch' and child.attr=='float32':
                child.attr='float64';count+=1
        counts[node.name]=count
    require(counts=={'validate_graph':2,'contact_distogram_float64':3},'Original dtype-only admission guards differ')
    restored=deepcopy(tree)
    for node in restored.body:
        if isinstance(node,ast.FunctionDef) and node.name in counts:
            for child in ast.walk(node):
                if isinstance(child,ast.Attribute) and isinstance(child.value,ast.Name) and child.value.id=='torch' and child.attr=='float64':
                    # These two original functions contain no explicit float64 type constants.
                    child.attr='float32'
    require(ast.dump(restored)==ast.dump(original),'Reference adapter changed more than dtype admission')
    return ast.fix_missing_locations(tree)


def private_dbmol(original,centers):
    raw=Path(original.__file__).read_bytes();tree=reference_tree(raw,original.__file__)
    private=types.ModuleType('_complete_FP64_DBMol_reference');private.__package__=original.__package__
    private.__file__=original.__file__;sys.modules[private.__name__]=private
    exec(compile(tree,original.__file__,'exec'),private.__dict__)
    import torch
    require(centers.dtype==torch.float32 and centers.shape==(64,),'Original FP32 native bin constants required')
    wide=centers.double();partition=(centers<8).clone()
    def fixed_bins(logits):
        require(logits.dtype==torch.float64 and logits.shape[-1]==64 and logits.device==wide.device,
                'FP64 reference distogram/bin placement differs')
        require(torch.equal(wide<8,partition),'Reference changed native contact partition')
        return wide
    private.native_bin_centers=fixed_bins
    return private,{'version':'complete-DBMol-FP64-reference-v1','original_source_sha256':hashlib.sha256(raw).hexdigest(),
        'dtype_guards_only':{'validate_graph':2,'contact_distogram_float64':3},'mathematical_AST_reversal_exact':True,
        'native_FP32_distance_partition_preserved':True,'production_code_changed':False}


def frozen_state(models,context,digest):
    return {'models':[digest(m.state_dict()) for m in models],
        'context':digest(context['features']|{'coordinates':context['coordinates']})}


class Evaluator:
    """Original FP32 producer and independently loaded FP64 reference coexist."""
    def __init__(self,plan,composite,precision,baseline):
        import torch
        from mosaic.small_molecule.boltz2 import native,dbmol
        self.torch=torch;self.native=native;self.precision=precision;self.composite=composite
        require(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'L40S' in torch.cuda.get_device_name(),
                'One actual L40S required')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        path=plan['context']['path'];require(native.file_digest(path)==plan['context']['sha256'],'Original context bytes changed')
        self.context=torch.load(path,map_location='cuda',weights_only=True);native.validate_context(self.context)
        def load():
            models=[native.load_model(plan['affinity_checkpoint']['path']).to('cuda')]
            if composite:models.append(dbmol.load_structure_model(plan['structure_checkpoint']['path']).to('cuda'))
            return models
        self.models=load();self.original=frozen_state(self.models,self.context,native.tensor_digest)
        self.producer=(dbmol.DBMolScorer(*self.models,self.context) if composite else
            lambda x,e:{'dbmol_affinity_loss':self.affinity(self.models[0],self.context,x,e)})
        self.check_objective='dbmol_total_loss' if composite else 'dbmol_affinity_loss'
        self.before=self.measure(baseline,self.check_objective,True)
        self.ref_models=load()
        require(self.original==frozen_state(self.ref_models,self.context,native.tensor_digest),'Reference initial model copy differs')
        self.promotions=[precision.promote_model(m,native.tensor_digest) for m in self.ref_models]
        self.reference_context=precision.promote_context(self.context)
        if composite:
            centers=dbmol.native_bin_centers(torch.empty(1,1,1,64,dtype=torch.float32,device='cuda'))
            private,self.derivation=private_dbmol(dbmol,centers)
            self.reference=private.DBMolScorer(*self.ref_models,self.reference_context)
        else:
            self.reference=lambda x,e:{'dbmol_affinity_loss':self.affinity(self.ref_models[0],self.reference_context,x,e)}
            self.derivation={'version':'original-full-FP64-affinity-reference','production_code_changed':False}
        self.reference_frozen=frozen_state(self.ref_models,self.reference_context,native.tensor_digest)
        self.during=self.measure(baseline,self.check_objective,True)
        require(self.same(self.before,self.during),'Exact production changed during reference co-residency')

    @staticmethod
    def same(left,right):
        return (left['value']==right['value'] and left['dtype']==right['dtype'] and left['components']==right['components']
            and all(left['gradients'][f].dtype==right['gradients'][f].dtype
                    and left['gradients'][f].tobytes()==right['gradients'][f].tobytes() for f in ('X','E_upper')))

    def affinity(self,model,context,x,e):
        # Plain Boltz optimizes the quantitative head alone. DBMol's separate
        # composite scorer retains its own pred-minus-logit affinity definition.
        values=self.native.score(model,context,x,e);return values[0]

    def measure(self,values,objective,gradient,reference=False):
        import contextlib
        import numpy as np
        torch=self.torch;dtype=torch.float64 if reference else torch.float32
        points=[torch.tensor(values[f],device='cuda',dtype=dtype,requires_grad=gradient) for f in ('X','E_upper')]
        mode=self.precision.precision_mode() if reference else contextlib.nullcontext()
        original_default=torch.get_default_dtype()
        try:
            if reference:torch.set_default_dtype(torch.float64)
            with mode,torch.set_grad_enabled(gradient):
                outputs=(self.reference if reference else self.producer)(*points);loss=outputs[objective]
                require(loss.dtype==(torch.float64 if reference or self.composite else torch.float32)
                        and bool(torch.isfinite(loss)),'Wrong/nonfinite scalar precision')
                result={'value':float(loss.detach().cpu()),'dtype':str(loss.dtype),
                    'components':{k:float(v.detach().cpu()) for k,v in outputs.items() if k.startswith('dbmol_') and v.numel()==1}}
                if gradient:
                    gradients=torch.autograd.grad(loss,points)
                    require(all(g.dtype==dtype and bool(g.isfinite().all()) for g in gradients),'Wrong/detached raw gradients')
                    result['gradients']={f:g.detach().cpu().numpy().copy() for f,g in zip(('X','E_upper'),gradients)}
        finally:torch.set_default_dtype(original_default)
        if reference:result['operation_audit']=mode.assessment()
        return result

    def verify(self):
        require(frozen_state(self.models,self.context,self.native.tensor_digest)==self.original,'Production frozen values changed')
        require(frozen_state(self.ref_models,self.reference_context,self.native.tensor_digest)==self.reference_frozen,
                'Reference frozen values changed')
        return {'production':self.original,'reference':self.reference_frozen,'all_castbacks_exact':True}

    def close(self,baseline):
        identities=self.verify();del self.reference;self.ref_models.clear()
        self.torch.cuda.empty_cache();after=self.measure(baseline,self.check_objective,True)
        require(self.same(self.before,after),'Exact production changed after reference exit')
        require(frozen_state(self.models,self.context,self.native.tensor_digest)==self.original,'Production frozen identity changed')
        return identities,after
