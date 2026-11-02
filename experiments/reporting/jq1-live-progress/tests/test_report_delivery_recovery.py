"""Portable persisted-terminal metadata controls; no arrays/models."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_delivery_recovery as recovery


class TerminalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.value = {'state':'SUCCESS_PERSISTED','job_id':'123',
                      'live':{'stdout':'','stderr':'Job <123> is not found\n'},
                      'history':{'returncode':0,'stdout':'Job <123>, User <s234818>\nCompleted <done>\n'}}
        self.lsf = 'Subject: Job 123: fixture\nin queue <hpc>, as user <s234818>\nSuccessfully completed.\n'

    def test_real_persisted_success_does_not_require_fabricated_live_DONE(self):
        recovery.verify_terminal(self.value,self.lsf,'123')
        self.assertEqual(self.value['state'],'SUCCESS_PERSISTED')
        self.assertEqual(self.value['live']['stdout'],'')

    def test_LSF_Done_successfully_history_format(self):
        self.value['history']['stdout']='Job <123>, User <s234818>\nDone successfully. The CPU time used is 517.9 seconds;\nPost job process done successfully;'
        recovery.verify_terminal(self.value,self.lsf,'123')

    def test_wrong_live_DONE_job_rejected(self):
        self.value['live']['stdout']='Job <124>, Status <DONE>'
        with self.assertRaises(ValueError):recovery.verify_terminal(self.value,self.lsf,'123')

    def test_active_failed_or_ambiguous_live_job_rejected(self):
        for state in ('RUN','PEND','EXIT','PSUSP'):
            value=deepcopy(self.value);value['live']['stdout']='Job <123>, Status <'+state+'>'
            with self.subTest(state=state),self.assertRaises(ValueError):
                recovery.verify_terminal(value,self.lsf,'123')

    def test_unavailable_scheduler_is_not_itself_persisted_success(self):
        value=deepcopy(self.value);value['live']['stderr']='connection error'
        with self.assertRaises(ValueError):recovery.verify_terminal(value,self.lsf,'123')

    def test_wrong_job_or_failure_history_rejected(self):
        for text in ('Job <124> Completed <done>','Job <123> Completed <exit>',
                     'Job <123> Exited with exit code 1 Completed <done>'):
            value=deepcopy(self.value);value['history']['stdout']=text
            with self.subTest(text=text),self.assertRaises(ValueError):
                recovery.verify_terminal(value,self.lsf,'123')

    def test_wrong_job_queue_or_user_in_LSF_rejected(self):
        for text in (self.lsf.replace('Job 123:','Job 124:'),self.lsf.replace('<hpc>','<gpul40s>'),
                     self.lsf.replace('<s234818>','<another>')):
            with self.subTest(text=text),self.assertRaises(ValueError):
                recovery.verify_terminal(self.value,text,'123')

    def test_missing_or_duplicate_LSF_success_rejected(self):
        for text in (self.lsf.replace('Successfully completed.',''),self.lsf+'Successfully completed.'):
            with self.assertRaises(ValueError):recovery.verify_terminal(self.value,text,'123')


if __name__=='__main__':unittest.main()
