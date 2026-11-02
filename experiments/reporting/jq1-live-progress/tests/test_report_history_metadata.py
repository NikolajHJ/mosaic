"""Portable history metadata controls; opaque snapshot bytes are never evaluated."""
import json
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import quick_inputs as inputs
import report_history as history


class RetainedEndpointMetadataTests(unittest.TestCase):
    def capture(self, root, retained=True, latest_step=6):
        study=root/'study';case=study/'case-0';case.mkdir(parents=True)
        def save(path,value):
            path.write_bytes(inputs.encoded(value));return str(path)
        endpoint=case/'endpoint.npz';endpoint.write_bytes(b'opaque retained endpoint')
        latest=case/'latest-state.npz';latest.write_bytes(b'opaque older rolling checkpoint')
        rows=[{'step':i,'global_step':i,'snapshot_retained':i==0 or (retained and i==7),
               'snapshot':'endpoint.npz' if i==7 and retained else None,
               'snapshot_sha256':inputs.sha(endpoint) if i==7 and retained else None} for i in range(8)]
        (case/'states.jsonl').write_bytes(b''.join(inputs.encoded(r) for r in rows))
        (case/'updates.jsonl').write_bytes(b''.join(inputs.encoded({'step':i,'global_step':i}) for i in range(7)))
        save(case/'report.json',{'states':rows})
        save(case/'latest-state.json',{'step':latest_step,'snapshot_sha256':inputs.sha(latest)})
        cfg=Path(save(root/'config.json',{'steps':10}))
        manifest=Path(save(root/'manifest.json',{'files':{'config.json':inputs.sha(cfg)}}))
        submission=save(root/'submission.json',{'job_id':'123','status':'submitted'})
        stdout=root/'stdout';stdout.write_text('Subject: Job 123: <original> in cluster <dcc> Exited\nExited with signal termination: 9.\n')
        stderr=root/'stderr';stderr.write_text('preserved original interruption')
        return history.capture_segment({'run':str(study),'config':str(cfg),'job_id':'123',
            'science_manifest':str(manifest),'source_manifest_sha256':inputs.sha(manifest),
            'submission':submission,'stdout':str(stdout),'stderr':str(stderr)},
            {'allowed_roots':[str(root)]})[0]

    def test_actual_retained_endpoint_takes_precedence_over_older_latest(self):
        with tempfile.TemporaryDirectory() as d:
            row=self.capture(Path(d))
            self.assertEqual(row['expected_counts'],{'committed_updates':7,'scalar_states':8,'retained_frames':2})
            self.assertEqual(row['latest_checkpoint_role'],'older_auxiliary_not_endpoint')
            self.assertEqual(row['retained_endpoint_snapshot']['sha256'],inputs.sha(Path(d)/'study/case-0/endpoint.npz'))
            self.assertEqual(inputs.load(row['latest_metadata']['path'])['step'],6)

    def test_stale_latest_without_retained_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(ValueError,'committed endpoint'):
            self.capture(Path(d),retained=False)

    def test_future_latest_cannot_substitute_for_committed_endpoint(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(ValueError,'committed endpoint'):
            self.capture(Path(d),latest_step=8)

    def test_named_partial_can_be_selected_without_a_terminal_summary(self):
        with tempfile.TemporaryDirectory() as d:
            row=self.capture(Path(d));study=Path(d)/'study'
            self.assertFalse((study/'summary.json').exists())
            selection={'version':inputs.VERSION,'edition':'edition-0001',
                       'runs':[{'id':'interrupted','segments':[{'run':str(study)}]}]}
            with patch.object(inputs,'seed_catalog',return_value=({},{})),patch.object(
                    inputs,'capture_segment',return_value=(row,{'steps':10,'initial_step_size':0.3},'fixed')):
                result=inputs.capture(selection,{'history_renderer_sha256':'a'*64,
                                                'exploratory_renderer_sha256':'b'*64})
            self.assertEqual(result['arms'][0]['id'],'interrupted')
            self.assertEqual(result['arms'][0]['expected_committed_updates'],7)

    def test_packaged_adapter_identities_follow_actual_source_bytes(self):
        import report as cli
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=b'# unchanged scientific fixture\n'
            manifest=inputs.encoded({'files':{'scripts/report_data.py':hashlib.sha256(raw).hexdigest()}})
            archive=root/'base.tar.gz'
            with tarfile.open(archive,'w:gz') as bundle:
                for name,value in {'scripts/report_data.py':raw,'source_manifest.json':manifest}.items():
                    member=tarfile.TarInfo(name);member.size=len(value);bundle.addfile(member,io.BytesIO(value))
            storage=root/'scripts/hpc/storage_accounting.py';storage.parent.mkdir(parents=True);storage.write_text('# metadata fixture\n')
            profile={'base_archive':{'path':str(archive),'bytes':archive.stat().st_size,
                                    'sha256':inputs.sha(archive),'manifest_sha256':hashlib.sha256(manifest).hexdigest()},
                     'validated_renderer_sha256':hashlib.sha256(raw).hexdigest()}
            names=('chemistry_screen','history','exploratory','defog')
            profile.update({name+'_renderer_sha256':'a'*64 for name in names})
            with patch.object(cli,'REPO',root):_,engine,packed=cli.package_engine(profile)
            with tarfile.open(fileobj=io.BytesIO(packed)) as bundle:
                runtime=json.load(bundle.extractfile('profile.json'))
            for name in names:
                self.assertEqual(runtime[name+'_renderer_sha256'],engine['files']['scripts/report_'+name+'.py'])
                self.assertEqual(profile[name+'_renderer_sha256'],'a'*64)


if __name__=='__main__':unittest.main()
