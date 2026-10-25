"""Allocated call-site integration against the exact original sealed launchers.

These fixtures execute original metadata AST call sites, not an invented
signature. Synthetic proof records below are engineering fixtures only. No
model is loaded, optimizer update performed or scientific gate called passed.
"""
import ast
from contextlib import contextmanager
import copy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import production_contract as c
import production_ops as op
import production_run as run


class OriginalCallSiteTests(unittest.TestCase):
    @contextmanager
    def routes(self,arm):
        # Every source byte is authenticated before importing the original
        # metadata-only module. Numerical imports in its functions stay lazy.
        original=run.import_original(arm)
        spec=c.profile(arm);index=0;pin='a'*64
        producer=op.phase_parent('segment',arm,index)
        parent=op.phase_parent('after-segment',arm,index)
        guard={'status':'passed','original_gate_changed':False,'reserve_bytes':4*c.GIB,
               'file_reserve':100000,'required_bytes':123456,'available_bytes':123456,
               'chemistry':{arm:{'full':c.projection(1234,arm)}}}
        def row(path):return {'path':str(path),'bytes':23,'sha256':'b'*64}
        def read(path):
            if Path(path)==parent/'payload/production-storage-after-segment-entry.json':return copy.deepcopy(guard)
            if Path(path)==parent/'terminal-proofs.json':return {'proofs':[row(parent/'actual-scheduler.json')]}
            raise AssertionError('Unexpected metadata input '+str(path))
        with patch.object(op,'read',side_effect=read),patch.object(op,'binding',side_effect=row),\
             patch.object(op,'check_binding',side_effect=lambda value:value),\
             patch.object(run,'require_startup',return_value={'status':'synthetic_startup_fixture_only'}) as startup:
            transitions=run.install_terminal_metadata_routes(original,arm,index,pin,parent,producer,row(producer/'stdout.1.log'))
            yield original,spec,index,pin,parent,producer,transitions,startup

    def original_calls(self,original,function,name):
        tree=ast.parse(Path(original.__file__).read_text())
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==function]
        self.assertEqual(len(functions),1)
        return [node for node in ast.walk(functions[0]) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Name) and node.func.id==name]

    def evaluate_original(self,call,original,**values):
        namespace={**original.__dict__,**values}
        expression=ast.fix_missing_locations(ast.Expression(copy.deepcopy(call)))
        return eval(compile(expression,original.__file__+'[actual-metadata-call-fixture]','eval'),namespace)

    def test_protenix_actual_five_argument_startup_call(self):
        with self.routes('protenix') as (original,spec,index,pin,parent,producer,transitions,startup):
            calls=self.original_calls(original,'certify_segment','verify_gpu_startup')
            self.assertEqual(len(calls),1);self.assertEqual(len(calls[0].args),5)
            self.evaluate_original(calls[0],original,arm=spec['arm'],index=index,
                                   manifest=spec['manifest'],parent=producer)
            startup.assert_called_once_with('protenix','segment',0,pin)

    def test_boltz_actual_four_argument_startup_call(self):
        with self.routes('boltz') as (original,spec,index,pin,parent,producer,transitions,startup):
            calls=self.original_calls(original,'main','verify_startup')
            self.assertEqual(len(calls),1);self.assertEqual(len(calls[0].args),4)
            self.evaluate_original(calls[0],original,arm=spec['arm'],phase='segment',index=index,
                                   args=SimpleNamespace(manifest=spec['manifest']))
            startup.assert_called_once_with('boltz','segment',0,pin)

    def test_boltz_terminal_has_no_hidden_original_startup_call(self):
        with self.routes('boltz') as (original,*unused):
            self.assertEqual(self.original_calls(original,'certify_segment','verify_startup'),[])
            source=ast.parse(Path(run.__file__).read_text())
            terminal=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='terminal')
            calls=[node for node in ast.walk(terminal) if isinstance(node,ast.Call)
                   and isinstance(node.func,ast.Name) and node.func.id=='require_startup']
            self.assertEqual(len(calls),1)

    def test_protenix_actual_command_path_call(self):
        with self.routes('protenix') as (original,spec,index,pin,parent,producer,transitions,startup):
            call,=self.original_calls(original,'certify_segment','command_path')
            result=self.evaluate_original(call,original,arm=spec['arm'],index=index)
            self.assertEqual(result,producer/'startup-command.json')

    def test_protenix_actual_final_job_call_uses_legacy_proof_schema(self):
        with self.routes('protenix') as (original,spec,index,pin,parent,producer,transitions,startup):
            call,=self.original_calls(original,'certify_segment','final_job')
            rows=self.evaluate_original(call,original,parent=producer)
            self.assertTrue(rows)
            self.assertTrue(all(set(row)=={'path','sha256'} for row in rows))

    def test_boltz_actual_final_job_call_uses_producer_not_callback(self):
        with self.routes('boltz') as (original,spec,index,pin,parent,producer,transitions,startup):
            call,=self.original_calls(original,'certify_segment','final_job')
            self.evaluate_original(call,original,producer=producer)
            with self.assertRaises(ValueError):original.final_job(parent,'gpul40s')

    def test_protenix_actual_next_storage_call(self):
        with self.routes('protenix') as (original,spec,index,pin,parent,producer,transitions,startup):
            call,=self.original_calls(original,'certify_segment','storage_gate')
            value=self.evaluate_original(call,original,arm=spec['arm'],index=index)
            self.assertEqual(value['status'],'passed')

    def test_boltz_actual_next_storage_call(self):
        with self.routes('boltz') as (original,spec,index,pin,parent,producer,transitions,startup):
            call,=self.original_calls(original,'certify_segment','storage_gate')
            value=self.evaluate_original(call,original,parent=parent)
            self.assertEqual(value['status'],'passed')

    def test_protenix_actual_next_phase_intent_serializes_path(self):
        with self.routes('protenix') as (original,spec,index,pin,parent,producer,transitions,startup):
            call,=self.original_calls(original,'certify_segment','queue_native')
            self.evaluate_original(call,original,arm=spec['arm'],index=index,manifest=spec['manifest'])
            self.assertEqual(transitions,[{'args':[spec['arm'],1,spec['manifest'],str(parent)],
                                           'kwargs':{},'submission_performed':False}])

    def test_boltz_actual_next_phase_intent(self):
        with self.routes('boltz') as (original,spec,index,pin,parent,producer,transitions,startup):
            call,=self.original_calls(original,'certify_segment','queue_native')
            self.evaluate_original(call,original,arm=spec['arm'],index=index,manifest=spec['manifest'],parent=parent)
            self.assertEqual(len(transitions),1);self.assertFalse(transitions[0]['submission_performed'])

    def test_wrong_original_startup_manifest_rejected(self):
        with self.routes('protenix') as (original,spec,index,pin,parent,producer,transitions,startup):
            with self.assertRaises(ValueError):original.verify_gpu_startup(spec['arm'],'segment',0,'c'*64,producer)
            startup.assert_not_called()

    def test_wrong_original_next_boundary_rejected(self):
        with self.routes('boltz') as (original,spec,index,pin,parent,producer,transitions,startup):
            with self.assertRaises(ValueError):original.queue_native(spec['arm'],2,spec['manifest'],parent)
            self.assertEqual(transitions,[])

    def test_protenix_actual_startup_helper_signature(self):
        source=Path(c.profile('protenix')['source'])
        op.verify(source,c.profile('protenix')['manifest'])
        helper=op.module(source/'scripts/extended_startup.py','_production_fixture_pro_startup')
        inspect.signature(helper.run_startup_child).bind(['actual-argv'],source,c.profile('protenix')['manifest'],Path('/tmp/owned'))
        self.assertEqual('ProtenixPremodelBoundaryReached: '+helper.MESSAGE,
            'ProtenixPremodelBoundaryReached: Intentional Protenix GPU startup-only boundary; no native model loaded')

    def test_boltz_actual_startup_helper_signature(self):
        source=Path(c.profile('boltz')['source'])
        op.verify(source,c.profile('boltz')['manifest'])
        helper=op.module(source/'scripts/premodel_startup.py','_production_fixture_bol_startup')
        inspect.signature(helper.execute_generated_command).bind(['actual-argv'],source,c.profile('boltz')['manifest'],arm='boltz2_chemistry')
        self.assertEqual('PremodelBoundaryReached: '+helper.MESSAGE,
            'PremodelBoundaryReached: Intentional startup-only boundary; no native model loaded')
