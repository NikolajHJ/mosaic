"""Allocated Boltz-only recovery ownership and budget metadata contracts."""
import ast
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import production_ops as op
import production_recovery as r


class RecoveryOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(str(os.environ.get('LSB_JOBID','')).isdigit(),'Allocated HPC fixtures only')

    def test_only_control_and_boltz_execution_scopes_allowed(self):
        for phase in ('cpu','after-cpu'):r.execution_scope(phase,'control')
        for phase in ('native','segment','after-native','after-segment'):r.execution_scope(phase,'boltz')

    def test_existing_protenix_cannot_be_executed_or_submitted(self):
        for phase in ('native','segment','after-native','after-segment'):
            with self.assertRaises(ValueError):r.execution_scope(phase,'protenix')

    def test_rejected_protenix_submission_creates_no_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new'
            with patch.object(op,'ROOT',root):
                with self.assertRaises(ValueError):op.submit('segment','a'*64,'protenix',0)
            self.assertFalse(root.exists())

    def test_budget_roots_preserve_live_protenix_and_new_boltz(self):
        root=Path('/new/recovery')
        self.assertEqual(r.budget_root('protenix',root),r.OLD_ROOT/'protenix')
        self.assertEqual(r.budget_root('boltz',root),root/'boltz')
        with self.assertRaises(ValueError):r.budget_root('unknown',root)

    def test_old_control_bytes_retained_separately_from_arm_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'run';root.mkdir();source=base/'release/source';source.mkdir(parents=True)
            (source/'proof.json').write_bytes(b'123')
            (root/'jobs.json').write_bytes(b'12345')
            for arm in ('protenix','boltz'):
                (root/arm).mkdir();(root/arm/'artifact').write_bytes(b'not a control byte')
            with patch.object(r,'OLD_ROOT',root),patch.object(r,'OLD_SOURCE',source):
                self.assertEqual(r.old_control_bytes(op),8)

    def test_unknown_recovery_seal_rejected_before_old_artifacts(self):
        def must_not_read(*args):raise AssertionError('unknown scope reached artifact reader')
        fake=SimpleNamespace(require=op.require,verify=must_not_read)
        with self.assertRaises(ValueError):r.authenticate(fake,{'recovery':{'version':'other'}})

    def test_live_old_control_scan_retries_whole_snapshot_not_individual_file(self):
        attempts=[]
        def complete_scan(module):
            attempts.append(module)
            if len(attempts)<3:raise FileNotFoundError('Atomic metadata rename during whole scan')
            return 123
        with patch.object(r,'_old_control_bytes_once',complete_scan):
            self.assertEqual(r.old_control_bytes(op),123)
        self.assertEqual(attempts,[op,op,op])

    def test_fresh_dispatch_uses_only_explicit_recovery_arms(self):
        tree=ast.parse(Path(op.__file__).read_bytes())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        loops=[n for n in ast.walk(fn) if isinstance(n,ast.For)
               and any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='queue_pair' for v in ast.walk(n))]
        self.assertEqual(len(loops),1)
        self.assertEqual(ast.unparse(loops[0].iter),'recovery.DISPATCH_ARMS')
        self.assertEqual(r.DISPATCH_ARMS,('boltz',))
