"""Allocated filesystem/actual-adapter-entry regressions, not native passes."""
import ast
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import production_ops as op
import production_contract as c
import production_run as run


class AccelerationStartupTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(str(os.environ.get('LSB_JOBID','')).isdigit(), 'Allocated HPC fixtures only')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patch = patch.object(op,'ROOT',self.root)
        self.patch.start();self.addCleanup(self.patch.stop)
        self.parent = self.root/'segment';self.parent.mkdir()

    def test_owned_acceleration_parent_created_before_gpu_child(self):
        (self.parent/'payload').mkdir()
        directory = run.prepare_acceleration_output(self.parent)
        self.assertEqual(directory,self.parent/'payload/acceleration')
        self.assertTrue(directory.is_dir())
        self.assertFalse((self.parent/'payload/study').exists())

    def test_absent_payload_is_not_silently_recreated(self):
        with self.assertRaises(ValueError):run.prepare_acceleration_output(self.parent)

    def test_unknown_existing_adapter_tree_rejected(self):
        (self.parent/'payload').mkdir()
        run.prepare_acceleration_output(self.parent)
        with self.assertRaises(FileExistsError):run.prepare_acceleration_output(self.parent)

    def test_existing_file_cannot_become_acceleration_tree(self):
        (self.parent/'payload').mkdir()
        (self.parent/'payload/acceleration').write_text('preserve')
        with self.assertRaises(FileExistsError):run.prepare_acceleration_output(self.parent)
        self.assertEqual((self.parent/'payload/acceleration').read_text(),'preserve')

    def test_external_root_rejected(self):
        with patch.object(op,'ROOT',self.root/'other'):
            with self.assertRaises(ValueError):run.prepare_acceleration_output(self.parent)

    def test_linked_payload_rejected(self):
        destination=self.root/'outside';destination.mkdir()
        (self.parent/'payload').symlink_to(destination,target_is_directory=True)
        with self.assertRaises(ValueError):run.prepare_acceleration_output(self.parent)

    def test_linked_parent_rejected(self):
        linked=self.root/'linked';linked.symlink_to(self.parent,target_is_directory=True)
        with self.assertRaises(ValueError):run.prepare_acceleration_output(linked)

    def test_actual_sealed_adapter_entry_requires_and_accepts_parent(self):
        source=Path(c.PILOTS['boltz']['source'])
        op.verify(source,'05e316a7b1bcceaf286b03231200a2d1f66e3f6a7ce927d4c1e73f08a24f86f9')
        tree=ast.parse((source/'scripts/gpu_local_bridge.py').read_bytes())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LocalGPUClient')
        ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        self.assertEqual(ast.unparse(ctor.body[0]),'directory = Path(directory)')
        self.assertEqual(ast.unparse(ctor.body[1]),'directory.mkdir(exist_ok=False)')
        ctor.body=ctor.body[:2]+[ast.Return(ast.Constant(None))]
        module=ast.fix_missing_locations(ast.Module(body=[ctor],type_ignores=[]))
        namespace={'Path':Path};exec(compile(module,'<actual-native-GPU-client-filesystem-prefix>','exec'),namespace)
        actual_prefix=namespace['__init__']
        tree=ast.parse((source/'scripts/full_model_pilot.py').read_bytes())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='combined_adapters')
        module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        namespace={'contextmanager':contextmanager,'ExitStack':ExitStack,'Path':Path}
        @contextmanager
        def search(**kwargs):yield {'unit_only':True}
        @contextmanager
        def offloaded(directory):
            actual_prefix(SimpleNamespace(),directory)
            yield {'unit_only_filesystem_prefix':True}
        with patch.dict(sys.modules,{'hybrid_search':SimpleNamespace(install_hybrid=search),
                                    'gpu_local_bridge':SimpleNamespace(install_offloaded_matrix_loss=offloaded)}):
            exec(compile(module,'<actual-sealed-combined-adapter-entry>','exec'),namespace)
            (self.parent/'payload').mkdir()
            with self.assertRaises(FileNotFoundError):
                with namespace['combined_adapters']('boltz-chem30k',self.parent/'payload/acceleration'):pass
            directory=run.prepare_acceleration_output(self.parent)
            with namespace['combined_adapters']('boltz-chem30k',directory) as evidence:
                self.assertTrue(evidence['matrix_policy']['unit_only_filesystem_prefix'])
            self.assertTrue((directory/'gpu-chemistry').is_dir())
            self.assertFalse((self.parent/'payload/study').exists())
