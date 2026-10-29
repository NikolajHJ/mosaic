"""Reuse qualified endpoint algorithms with explicit chemistry metadata scope.

No native/repair/optimizer arithmetic changes. The inherited nonchemistry hook
is specialized only for a positive, frozen chemistry coefficient and both model
names. Protenix's native checker retains the independent full-FP64 reference;
the combined logit checker remains the previously authorized prospective v1.
"""

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import chemistry_continuation_contract as c


def expression(tree, before, after, count=1):
    wanted=ast.dump(ast.parse(before,mode='eval').body);replacement=ast.parse(after,mode='eval').body
    class Replace(ast.NodeTransformer):
        changed=0
        def visit(self,node):
            if isinstance(node,ast.expr) and ast.dump(node)==wanted:
                self.changed+=1;return ast.copy_location(deepcopy(replacement),node)
            return super().visit(node)
    visitor=Replace();result=visitor.visit(tree)
    c.require(visitor.changed==count,'Original authenticated metadata expression differs: '+before)
    return ast.fix_missing_locations(result)


def combined_method(function):
    """Specialize only the coefficient/name guard and recorded coefficient."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    old="self.name in ('boltz','dbmol') and directory is not None and config['repair_weight']==0 and config['element_js_weight']==0"
    new="self.name in ('protenix','boltz') and directory is not None and config['repair_weight']>0 and config['element_js_weight']==0 and config.get('objective_mode')=='native_plus_chemistry' and 'confidence_margin' not in config"
    tree=expression(tree,old,new)
    replacements=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='update':
            for keyword in node.keywords:
                if keyword.arg=='repair_weight' and isinstance(keyword.value,ast.Constant) and keyword.value.value==0.:
                    keyword.value=ast.parse("float(config['repair_weight'])",mode='eval').body;replacements+=1
    c.require(replacements==1,'Original fixed-weight report shape differs')
    namespace={};exec(compile(ast.fix_missing_locations(tree),function.__code__.co_filename,'exec'),
                      dict(function.__globals__),namespace)
    return namespace['combined']


@contextmanager
def installed(base, precision, plan, arm, output):
    """Private class/metadata adaptation; originals restored on every exit."""
    record=c.configured_arm(plan,arm);root=Path(output)
    inherited=base.EndpointHooks;combined=combined_method(inherited.combined)
    class ChemistryHooks(inherited):
        def combined(self, original):
            delegated=combined(self,original)
            def fixed(*args,**kwargs):
                config=kwargs['config'];prepared=args[2]
                before=json.dumps(config,sort_keys=True,default=lambda x:vars(x))
                identity=prepared.identity
                result=delegated(*args,**kwargs)
                c.require(prepared.identity==identity and json.dumps(config,sort_keys=True,default=lambda x:vars(x))==before,
                          'Chemistry targets/coefficient/config changed within a derivative cohort')
                return result
            return fixed
    factory=SimpleNamespace(EndpointHooks=ChemistryHooks)
    old_configuration,old_contrasts=precision._configuration,precision._original_contrasts
    def configuration(current,settings):
        c.require(set(settings)=={'arms'} and set(settings['arms'])=={'protenix'},'Pro-only reference config required')
        cfg=settings['arms']['protenix']
        c.require(cfg['historical_global']==10000 and cfg['historical_original_directions']=={'X':[],'E_upper':[]}
                  and cfg['policy_kind']=='protenix-original-fp64' and cfg['max_session_bytes']==precision.SESSION_CAP,
                  'Chemistry first-boundary/reference policy differs')
        c.require(cfg==current['reference_configuration']['arms']['protenix'], 'Frozen reference configuration differs')
        return settings
    def contrasts(state,point,cfg):
        c.require(state.name=='protenix' and cfg['historical_global']==10000,'Unknown chemistry reference scope')
        # The historical failure was in LOGIT coordinates. Do not invent raw
        # probability contrasts from that different-coordinate report.
        return {'X':[],'E_upper':[]},'not_preexisting_at_new_endpoint'
    precision._configuration=configuration;precision._original_contrasts=contrasts
    try:
        if arm=='protenix':
            with precision.install_precision_hooks(factory,plan,plan['reference_configuration']):
                state=factory.EndpointHooks(record,arm,root,record['start'])
                with base.install_hooks(state):
                    import mosaic.small_molecule.chemistry.checks as checks
                    original=checks.check_combined_directions
                    checks.check_combined_directions=state.combined(original)
                    try:yield state
                    finally:checks.check_combined_directions=original
        else:
            state=ChemistryHooks(record,arm,root,record['start'])
            with base.install_hooks(state):yield state
    finally:precision._configuration=old_configuration;precision._original_contrasts=old_contrasts


def protocol_function(function, arm, name, steps):
    """Only adapt the explicit per-job count/range metadata, never computation."""
    c.require(arm in c.ARMS and steps==c.ARMS[arm]['remaining'],'Full remaining horizon required')
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    if arm=='protenix' and name=='segment_scope':
        tree=expression(tree,'steps != 10000',f'steps != {steps}')
        tree=expression(tree,'range(start, 50000, 10000)','(10000,)')
    elif arm=='protenix' and name=='require_boundary':
        tree=expression(tree,"execution.get('updates') != 10000",f"execution.get('updates') != {steps}")
    elif arm=='boltz' and name=='segment_scope':
        tree=expression(tree,'steps != SEGMENT',f'steps != {steps}')
        tree=expression(tree,'offset not in (0, 10000, 20000)','offset != 10000')
    elif arm=='boltz' and name=='require_boundary':
        tree=expression(tree,"execution.get('updates') != SEGMENT",f"execution.get('updates') != {steps}")
        tree=expression(tree,"execution.get('start') not in (0, 10000, 20000)","execution.get('start') != 10000")
        tree=expression(tree,"execution['start'] + SEGMENT",f"execution['start'] + {steps}")
    else:raise ValueError('Undeclared original scope adaptation')
    namespace={};exec(compile(tree,function.__code__.co_filename,'exec'),dict(function.__globals__),namespace)
    return namespace[name]


def driver_tree(raw,filename,arm):
    """Original optimizer already uses CLI steps; adapt its fixed count guard."""
    tree=ast.parse(raw,filename=filename)
    if arm in ('protenix','boltz'):
        tree=expression(tree,'result["status"] != "passed" or len(result["updates"]) != 10000 or [r["step"] for r in result["states"]] != list(range(10001))',
                        'result["status"] != "passed" or len(result["updates"]) != args.segment_steps or [r["step"] for r in result["states"]] != list(range(args.segment_steps + 1))')
    else:raise ValueError('Unknown original chemistry driver')
    return tree
