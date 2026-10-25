"""Allocated routing/actual-parser/terminal-ownership metadata regressions."""
import ast
import contextlib
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import production_contract as c
import production_ops as op
import production_run as run


class ProductionRoutingTests(unittest.TestCase):
    def per_arm_plan(self):
        return {'version':c.VERSION,'pilots':{
            arm:{**copy.deepcopy(c.PILOTS[arm]),'manifest_sha256':('a' if arm=='protenix' else 'c')*64,
                 'cpu_certificate_sha256':('b' if arm=='protenix' else 'd')*64,
                 'job_id':'29820001' if arm=='protenix' else '29820002'} for arm in c.PROFILES}}

    def test_control_parent_fixed(self):
        self.assertEqual(op.phase_parent('cpu'),op.ROOT/'cpu')

    def test_control_cannot_impersonate_arm(self):
        with self.assertRaises(ValueError):op.phase_parent('cpu','protenix',0)

    def test_initial_production_root_independent(self):
        self.assertEqual(op.phase_parent('segment','protenix',0),op.ROOT/'protenix/random-segment-0000')
        self.assertNotIn('protenix-extended-03',str(op.phase_parent('segment','protenix',0)))

    def test_original_chem_boltz_root_not_mutated(self):
        self.assertEqual(op.phase_parent('segment','boltz',0),op.ROOT/'boltz/boltz2_chemistry-segment-0000')

    def test_complete_actual_command_has_source_pin(self):
        command=op.argv('segment','a'*64,'boltz',0)
        self.assertIn(str(op.SOURCE/'scripts/production_stage.lsf'),command)
        self.assertEqual(command[-5:],[str(op.SOURCE),'a'*64,'segment','boltz','0'])

    def test_callback_uses_only_own_ended_dependency(self):
        command=op.argv('after-segment','a'*64,'boltz',0,'ended(29820001)')
        self.assertEqual(command[command.index('-w')+1],'ended(29820001)')

    def test_shell_dependency_injection_rejected(self):
        with self.assertRaises(ValueError):op.argv('after-cpu','a'*64,dependency='ended(1); true')

    def test_failed_submission_outcome_never_duplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent=root/'cpu';parent.mkdir()
            with patch.object(op,'ROOT',root),patch.object(op,'phase_parent',return_value=parent),patch.object(op.subprocess,'run') as execute:
                with self.assertRaises(FileExistsError):op.submit('cpu','a'*64)
                execute.assert_not_called()
            callback=root/'after-cpu';callback.mkdir()
            for arm in c.PROFILES:
                (root/arm).mkdir();op.write(root/arm/'adoption.json',{'status':'synthetic_validated_adoption_fixture'})
            gate=callback/'production-storage-first-dispatch.json'
            op.write(gate,{'status':'failed','required_bytes':200,'available_bytes':100})
            with patch.object(op,'ROOT',root),patch.object(op,'SOURCE',root/'absent-release/source'):
                self.assertTrue(op.preserve_storage_block(callback,{'job_id':'29820003'},'a'*64,ValueError('capacity')))
                waiting=op.read(root/'waiting-storage.json')
                self.assertEqual(waiting['status'],'blocked_storage_production_not_submitted')
                self.assertFalse(waiting['production_submissions'])
                self.assertEqual(waiting['failed_storage_receipts'],[op.binding(gate)])
                with self.assertRaises(FileExistsError):
                    op.preserve_storage_block(callback,{'job_id':'29820003'},'a'*64,ValueError('capacity'))
                (root/'boltz'/c.stage_label('boltz','segment',0)).mkdir()
                with self.assertRaises(ValueError):
                    op.preserve_storage_block(callback,{'job_id':'29820003'},'a'*64,ValueError('capacity'))

    def test_exclusive_receipt_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'proof.json';op.write(path,{'state':'failed'})
            with self.assertRaises(FileExistsError):op.write(path,{'state':'passed'})
            self.assertEqual(op.read(path),{'state':'failed'})

    def test_rejected_actual_startup_is_captured_before_assertion(self):
        with tempfile.TemporaryDirectory() as temp:
            payload=Path(temp);expected='Original Pro pilot raw LSF Exited with exit code 1'
            result=SimpleNamespace(returncode=1,stdout='original captured stdout\n',stderr=expected)
            original_require=op.require;observed=[]
            def require_after_capture(value,message):
                if message.startswith('Actual production wrapper/parser metadata startup rejected'):
                    path=payload/'startup-attempt-protenix.json'
                    captured=op.read(path)
                    self.assertEqual(captured['stderr'],expected)
                    self.assertEqual(captured['returncode'],1)
                    observed.append(path)
                return original_require(value,message)
            with patch.object(op.subprocess,'run',return_value=result) as execute,\
                 patch.object(op,'require',side_effect=require_after_capture):
                with self.assertRaisesRegex(ValueError,'captured original reason'):
                    op.metadata_startups(payload,'a'*64)
            self.assertEqual(execute.call_count,1)
            self.assertEqual(len(observed),1)
            summary=op.read(payload/'startup.json')
            self.assertEqual(summary['status'],'failed_metadata_only')
            self.assertFalse(summary['gpu_or_gradient_passed'])
            self.assertEqual(summary['optimizer_updates'],0)
            self.assertEqual(summary['commands'][0]['stderr'],expected)
            self.assertEqual(summary['commands'][0]['captured_attempt'],op.binding(observed[0]))
            self.assertFalse((payload/'startup-attempt-boltz.json').exists())

    def test_actual_startup_timeout_retains_partial_output_and_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            payload=Path(temp)
            failure=op.subprocess.TimeoutExpired(['original','command'],180,
                output=b'original partial stdout\n',stderr=b'original partial traceback\n')
            with patch.object(op.subprocess,'run',side_effect=failure) as execute:
                with self.assertRaises(op.subprocess.TimeoutExpired):op.metadata_startups(payload,'a'*64)
            self.assertEqual(execute.call_count,1)
            captured=op.read(payload/'startup-attempt-protenix.json')
            self.assertIsNone(captured['returncode'])
            self.assertTrue(captured['timed_out_or_interrupted'])
            self.assertEqual(captured['stdout'],'original partial stdout\n')
            self.assertEqual(captured['stderr'],'original partial traceback\n')
            self.assertIn('TimeoutExpired',captured['error'])
            summary=op.read(payload/'startup.json')
            self.assertEqual(summary['status'],'failed_metadata_only')
            self.assertEqual(summary['commands'][0]['captured_attempt'],
                             op.binding(payload/'startup-attempt-protenix.json'))

    def test_both_actual_startup_commands_keep_raw_receipts_without_gpu_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            payload=Path(temp);results=[SimpleNamespace(returncode=0,stdout='Pro metadata only\n',stderr=''),
                SimpleNamespace(returncode=0,stdout='Bol metadata only\n',stderr='')]
            with patch.object(op.subprocess,'run',side_effect=results) as execute:
                result=op.metadata_startups(payload,'a'*64)
            self.assertEqual(execute.call_count,2)
            self.assertEqual(result,op.read(payload/'startup.json'))
            self.assertEqual(result['status'],'passed_metadata_only')
            self.assertFalse(result['gpu_or_gradient_passed'])
            self.assertEqual(result['optimizer_updates'],0)
            self.assertEqual([row['arm'] for row in result['commands']],list(c.PROFILES))
            for name,row in zip(c.PROFILES,result['commands']):
                self.assertEqual(row['captured_attempt'],op.binding(payload/('startup-attempt-'+name+'.json')))
                self.assertEqual(row['argv'][-7:],['--arm',name,'--phase','segment','--index','0','--metadata-startup'])

    def test_exact_bound_file_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'proof.json';path.write_text('{}')
            old=op.binding(path);path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):op.check_binding(old)
            (Path(temp)/'protenix').mkdir();(Path(temp)/'protenix/data').write_bytes(b'arm-projection-owns-these-bytes'*20)
            expected=path.stat().st_size
            with patch.object(op,'ROOT',Path(temp)),patch.object(op,'SOURCE',Path(temp)/'absent-release/source'):
                self.assertEqual(op.control_bytes(),expected)
                with patch.object(c,'CONTROL_CAP',expected):self.assertEqual(op.enforce_control_cap(),expected)
                with patch.object(c,'CONTROL_CAP',expected-1):
                    with self.assertRaises(ValueError):op.enforce_control_cap()
        with patch.object(op,'_tree_bytes_once',side_effect=[FileNotFoundError('atomic rename'),17]) as scan:
            self.assertEqual(op.tree_bytes('/unused'),17);self.assertEqual(scan.call_count,2)
        with patch.object(op,'_tree_bytes_once',side_effect=FileNotFoundError('still changing')) as scan:
            with self.assertRaises(FileNotFoundError):op.tree_bytes('/unused')
            self.assertEqual(scan.call_count,3)
        with patch.object(op,'_control_bytes_once',side_effect=[FileNotFoundError('atomic job map'),19]) as scan:
            self.assertEqual(op.control_bytes(),19);self.assertEqual(scan.call_count,2)
        with patch.object(op,'_control_bytes_once',side_effect=FileNotFoundError('still changing')) as scan:
            with self.assertRaises(FileNotFoundError):op.control_bytes()
            self.assertEqual(scan.call_count,3)

    def test_nan_json_is_not_a_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'proof.json';path.write_text('{"n":NaN}')
            with self.assertRaises(ValueError):op.read(path)
            with patch.object(op,'ROOT',Path(temp)),patch.object(op,'SOURCE',Path(temp)/'absent-release/source'),\
                 patch.object(c,'CONTROL_CAP',path.stat().st_size+1):
                result=Path(temp)/'assessment.json'
                with self.assertRaises(ValueError):op.finish_assessment(result,{'status':'passed','stage':'cpu'})
                self.assertEqual(op.read(result)['status'],'failed')
                self.assertTrue(op.read(result)['control_cap_exceeded'])

    def test_lsf_unique_actual_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'stdout.123.log').write_text(
                'Subject: Job 123: <my-job> in cluster <dcc> Done\nDispatched in queue <gpul40s>\nSuccessfully completed.\n')
            value=op.lsf_success(root,{'job_id':'123','argv':['bsub','-J','my-job','-q','gpul40s']})
            self.assertEqual(value['path'],str(root/'stdout.123.log'))

    def test_lsf_failure_cannot_be_relabelled_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'stdout.123.log').write_text(
                'Subject: Job 123: <my-job> in cluster <dcc> Exited\nDispatched in queue <gpul40s>\nExited with exit code 1.\n')
            with self.assertRaises(ValueError):op.lsf_success(root,{'job_id':'123','argv':['bsub','-J','my-job','-q','gpul40s']})

    def test_different_lsf_job_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'stdout.123.log').write_text(
                'Subject: Job 124: <my-job> in cluster <dcc> Done\nDispatched in queue <gpul40s>\nSuccessfully completed.\n')
            with self.assertRaises(ValueError):op.lsf_success(root,{'job_id':'123','argv':['bsub','-J','my-job','-q','gpul40s']})

    def test_lsf_wrapped_job_name_compacts_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'stdout.123.log').write_text(
                'Subject: Job 123: <my-\n job> in cluster <dcc> Done\nDispatched in queue <gpu\n l40s>\nSuccessfully completed.\n')
            op.lsf_success(root,{'job_id':'123','argv':['bsub','-J','my-job','-q','gpul40s']})

    def test_duplicate_lsf_subject_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);text='Subject: Job 123: <my-job> in cluster <dcc> Done\n'
            (root/'stdout.123.log').write_text(text+text+'Dispatched in queue <gpul40s>\nSuccessfully completed.\n')
            with self.assertRaises(ValueError):op.lsf_success(root,{'job_id':'123','argv':['bsub','-J','my-job','-q','gpul40s']})

    def test_exact_raw_xml_module_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'tests.xml';path.write_text('<testsuite><testcase classname="tests.foo" name="test_one"/></testsuite>')
            self.assertEqual(op.xml_cases(path,{'tests.foo':1}),1)

    def test_failed_cpu_xml_stops_adoption(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'tests.xml';path.write_text('<testsuite><testcase classname="tests.foo" name="test_one"><failure/></testcase></testsuite>')
            with self.assertRaises(ValueError):op.xml_cases(path,{'tests.foo':1})

    def test_duplicate_xml_rows_cannot_inflate_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'tests.xml';path.write_text('<testsuite>'+2*'<testcase classname="tests.foo" name="test_one"/>'+'</testsuite>')
            with self.assertRaises(ValueError):op.xml_cases(path,{'tests.foo':2})

    def test_original_parser_ast_accepts_complete_command(self):
        source='''import argparse\nfrom pathlib import Path\ndef main():\n    parser=argparse.ArgumentParser()\n    parser.add_argument('--mode',choices=('validate','run'),required=True)\n    parser.add_argument('--output',type=Path,required=True)\n    args=parser.parse_args()\n    raise RuntimeError('model must not run')\n'''
        with tempfile.TemporaryDirectory() as temp:
            driver=Path(temp)/'driver.py';driver.write_text(source)
            value=run.parse_original(driver,['python',str(driver),'--mode','run','--output','/tmp/result'])
            self.assertEqual((value.mode,value.output),('run',Path('/tmp/result')))

    def test_original_parser_unknown_argument_rejected(self):
        source='''def main():\n    parser=argparse.ArgumentParser()\n    args=parser.parse_args()\n'''
        with tempfile.TemporaryDirectory() as temp:
            driver=Path(temp)/'driver.py';driver.write_text(source)
            with contextlib.redirect_stderr(None),self.assertRaises(SystemExit):run.parse_original(driver,['python',str(driver),'--skip-gradients'])

    def test_original_parser_prefix_cannot_hide_execution(self):
        source='''def main():\n    parser=argparse.ArgumentParser()\n    import torch\n    args=parser.parse_args()\n'''
        with tempfile.TemporaryDirectory() as temp:
            driver=Path(temp)/'driver.py';driver.write_text(source)
            with self.assertRaises(ValueError):run.parse_original(driver,['python',str(driver)])

    def test_missing_original_parser_endpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            driver=Path(temp)/'driver.py';driver.write_text('def main():\n    pass\n')
            with self.assertRaises(ValueError):run.parse_original(driver,['python',str(driver)])

    def test_no_numerical_import_in_metadata_controller(self):
        tree=ast.parse(Path(op.__file__).read_text())
        imports={alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
        imports|={node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module}
        self.assertTrue(imports.isdisjoint({'jax','torch','numpy','equinox'}))

    def test_per_arm_metadata_has_distinct_namespaces_and_roots(self):
        plan=self.per_arm_plan();calls=[]
        def load(path,name):
            calls.append((Path(path),name))
            arm=('protenix' if Path(path)==Path(c.PILOTS['protenix']['source'])/'scripts/full_model_ops.py' else 'boltz')
            return SimpleNamespace(ROOT=Path(c.PILOTS[arm]['root']),SOURCE=Path(c.PILOTS[arm]['source']))
        with patch.object(op,'verify'),patch.object(op,'module',side_effect=load):
            pro=op.pilot_metadata('protenix',plan);bol=op.pilot_metadata('boltz',plan)
        self.assertIsNot(pro,bol)
        self.assertNotEqual(calls[0][1],calls[1][1])
        self.assertEqual(calls[0][0],Path(c.PILOTS['protenix']['source'])/'scripts/full_model_ops.py')
        self.assertEqual(calls[1][0],Path(c.PILOTS['boltz']['source'])/'scripts/full_model_ops.py')

    def test_per_arm_loaded_metadata_cannot_redirect_root(self):
        plan=self.per_arm_plan()
        with patch.object(op,'verify'),patch.object(op,'module',return_value=SimpleNamespace(
                ROOT=Path(c.PILOTS['boltz']['root']),SOURCE=Path(c.PILOTS['protenix']['source']))):
            with self.assertRaises(ValueError):op.pilot_metadata('protenix',plan)

    def test_identical_numerical_adapters_required_across_sources(self):
        plan=self.per_arm_plan();files={name:'a'*64 for name in c.ADAPTER_FILES}
        with patch.object(op,'verify',return_value={'files':files}):
            self.assertEqual(op.adapter_parity(plan),files)

    def test_changed_numerical_adapter_cannot_be_hidden_by_metadata_retry(self):
        plan=self.per_arm_plan();left={name:'a'*64 for name in c.ADAPTER_FILES};right=dict(left)
        right['scripts/gpu_matrix.py']='b'*64
        with patch.object(op,'verify',side_effect=[{'files':left},{'files':right}]):
            with self.assertRaises(ValueError):op.adapter_parity(plan)

    def test_missing_numerical_member_cannot_be_adopted(self):
        plan=self.per_arm_plan();files={name:'a'*64 for name in c.ADAPTER_FILES}
        del files['scripts/gpu_local_worker.py']
        with patch.object(op,'verify',return_value={'files':files}):
            with self.assertRaises(ValueError):op.adapter_parity(plan)

    def test_metadata_only_retry_can_differ_without_scientific_drift(self):
        plan=self.per_arm_plan();left={name:'a'*64 for name in c.ADAPTER_FILES};right=dict(left)
        left['scripts/full_model_ops.py']='b'*64;right['scripts/full_model_ops.py']='c'*64
        with patch.object(op,'verify',side_effect=[{'files':left},{'files':right}]):
            self.assertEqual(set(op.adapter_parity(plan)),set(c.ADAPTER_FILES))
