"""Portable startup interface regressions; no arrays or model imports."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT=Path(os.environ.get('DBMOL_TEST_SOURCE',Path(__file__).resolve().parents[2]))


def check_inputs(native):
    """Execute the actual constructor's input-pin loop without loading models."""
    tree=ast.parse((ROOT/'src/latentmol/design/latent_boltz.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BoltzLatentScorer')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    loop=next(n for n in init.body if isinstance(n,ast.For))
    tree=ast.parse((ROOT/'src/latentmol/experiments.py').read_text())
    digest=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='file_sha256')
    namespace=dict(Path=Path,hashlib=hashlib,native=native)
    exec(compile(ast.Module(body=[digest,loop],type_ignores=[]),'<actual native pin loop>','exec'),namespace)


class TensorRecord:
    shape=(1,1,2)
    def __getitem__(self,index):return self
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return [[0.25,0.75]]


class StartupInterfaces(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def inputs(self):
        values={}
        for key in ('prepared','affinity','confidence'):
            p=self.root/key;p.write_bytes(key.encode())
            values[key]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        return json.loads(json.dumps(values))

    def test_json_string_paths_are_accepted(self):
        check_inputs(self.inputs())

    def test_bad_input_hash_is_still_rejected(self):
        values=self.inputs();values['affinity']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Native input changed: affinity'):
            check_inputs(values)

    def capture(self,**kwargs):
        path=ROOT/'src/latentmol/reporting/capture.py'
        spec=importlib.util.spec_from_file_location('exact_capture_interface',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        graph=SimpleNamespace(**{name:TensorRecord() for name in ('atoms','charges','bonds')})
        schema=SimpleNamespace(max_nodes=1,to_dict=lambda:dict(max_nodes=1))
        p=module.capture_shared_graph(self.root,0,graph,schema,active_slots=[0],temperature=.7,**kwargs)
        return json.loads(p.read_text())

    def test_explicit_graph_scope_is_accepted_and_saved(self):
        value=self.capture(scope='Fixture: current shared probabilities')
        self.assertEqual(value['scope'],'Fixture: current shared probabilities')
        self.assertEqual(value['atoms'],[[0.25,0.75]])

    def test_default_scope_and_atomic_export_remain(self):
        value=self.capture()
        self.assertIn('Saved canonical shared graph',value['scope'])
        self.assertFalse(list(self.root.glob('*.tmp')))


if __name__=='__main__':unittest.main()
