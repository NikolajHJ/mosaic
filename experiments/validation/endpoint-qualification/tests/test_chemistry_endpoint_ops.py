"""Allocated metadata contracts for exact diagnostic ownership and dispatch."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import chemistry_endpoint_ops as op


class ChemistryEndpointOpsTests(unittest.TestCase):
    def test_only_diagnostic_phases_exist(self):
        self.assertEqual(op.PHASES,('cpu','after-cpu','gpu-protenix','gpu-boltz'))

    def test_segment_dispatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'No optimization'):op.command('segment','a'*64)

    def test_native_protenix_resources(self):
        args=op.command('gpu-protenix','a'*64)
        self.assertEqual(args[args.index('-q')+1],'gpul40s')
        self.assertEqual(args[args.index('-W')+1],'02:00')
        self.assertEqual(args[args.index('-n')+1],'4')
        self.assertIn('rusage[mem=8GB]',args)
        self.assertEqual(args[args.index('-gpu')+1],'num=1:mode=exclusive_process')

    def test_native_boltz_resources(self):
        args=op.command('gpu-boltz','a'*64)
        self.assertIn('mosaic-chem-endpoint02-gpu-boltz',args)
        self.assertNotIn('-w',args)

    def test_cpu_resources(self):
        args=op.command('cpu','a'*64)
        self.assertEqual(args[args.index('-q')+1],'hpc')
        self.assertEqual(args[args.index('-W')+1],'01:00')
        self.assertNotIn('-gpu',args)

    def test_callback_only_own_dependency(self):
        args=op.command('after-cpu','a'*64,'ended(1234)')
        self.assertEqual(args[args.index('-w')+1],'ended(1234)')
        self.assertEqual(args[args.index('-n')+1],'1')

    def test_gpu_dependency_rejected(self):
        with self.assertRaises(ValueError):op.command('gpu-protenix','a'*64,'done(1234)')

    def test_callback_requires_dependency(self):
        with self.assertRaises(ValueError):op.command('after-cpu','a'*64)

    def test_full_new_group_budget(self):
        self.assertEqual(op.CAP,576*1024**2)

    def test_actual_job_mismatch_rejected(self):
        with patch.object(op,'read',return_value={'status':'submitted','source_manifest_sha256':'a'*64,
                'job_id':'1234','argv':op.command('cpu','a'*64)}),patch.dict(os.environ,{'LSB_JOBID':'9999','LSB_QUEUE':'hpc'}):
            with self.assertRaisesRegex(ValueError,'Actual own'):op.own('cpu','a'*64)

    def test_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'bad.json';path.write_text('{"value":NaN}')
            with self.assertRaises(ValueError):op.read(path)

    def test_evidence_write_exclusive(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'failed.json';op.write(path,{'status':'failed'})
            with self.assertRaises(FileExistsError):op.write(path,{'status':'passed'})

    def test_xml_rejects_skipped_case(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'tests.xml';path.write_text('<testsuite><testcase classname="x" name="n"><skipped/></testcase></testsuite>')
            with self.assertRaises(ValueError):op.xml_cases(path,{'x':1})

    def test_xml_rejects_missing_module(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'tests.xml';path.write_text('<testsuite><testcase classname="x" name="n"/></testsuite>')
            with self.assertRaises(ValueError):op.xml_cases(path,{'x':1,'y':1})

    def test_xml_rejects_duplicate_case(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'tests.xml';path.write_text('<testsuite><testcase classname="x" name="n"/><testcase classname="x" name="n"/></testsuite>')
            with self.assertRaises(ValueError):op.xml_cases(path,{'x':2})

    def test_xml_counts_actual_cases(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'tests.xml';path.write_text('<testsuite tests="99"><testcase classname="x" name="n"/></testsuite>')
            self.assertEqual(op.xml_cases(path,{'x':1}),1)

    def test_sealed_generated_counts_use_actual_package_attribution(self):
        manifest=op.read(Path(__file__).resolve().parents[1]/'source_manifest.json')
        self.assertEqual(set(manifest['cpu_counts']),{
            'tests.test_chemistry_endpoint_diagnostic.ChemistryEndpointDiagnosticTests',
            'tests.test_chemistry_endpoint_ops.ChemistryEndpointOpsTests'})
