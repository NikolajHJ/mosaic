"""Objective-configured shared frozen-VAE campaign, target50k."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from base_campaign import Campaign as BaseCampaign, read, save, digest, require_done


def remaining_budget(rate, remaining, *, max_hours=72):
    if not math.isfinite(rate) or rate<=0 or type(remaining) is not int or remaining<=0:
        raise ValueError('Missing finite measured runtime or remaining budget')
    count=min(remaining,math.floor((max_hours*3600-1200)/(1.3*rate)))
    if count<1:raise ValueError('One measured update exceeds allocation')
    minutes=min(max_hours*60,15*math.ceil((1200+1.3*rate*count)/900))
    return count,str(minutes//60)+':'+str(minutes%60).zfill(2)


class Campaign(BaseCampaign):
    def __init__(self,path,sha):
        self.path,self.sha=Path(path),sha
        if digest(path)!=sha:raise ValueError('Plan changed')
        self.plan=read(path)
        if self.plan['version'] not in ('vae-dbmol-campaign-v1','vae-latent-campaign-v1') or self.plan['target_updates']!=50000:
            raise ValueError('Unknown experiment or endpoint')
        if self.plan.get('scorer_backend','dbmol') not in ('dbmol','protenix_tiny'):
            raise ValueError('Unknown scorer backend')
        self.control=Path(self.plan['control']);self.root=Path(self.plan['campaign']);self.release=Path(self.plan['release'])
        self.receipts=self.control/'receipts'
        for name,expected in self.plan['input_sha256'].items():
            if digest(name)!=expected:raise ValueError('Control/input changed: '+name)

    def source(self):
        value=self.plan['checkpoint']
        if self.scratch(value['path'],'hash')!=value['sha256']:raise ValueError('Decoder checkpoint changed')
        return value

    def capacity(self):
        from storage_accounting import assess_output
        self.root.mkdir(parents=True,exist_ok=True)
        return assess_output(self.root,self.plan['output_cap_bytes'],self.plan['output_file_cap'])

    def launch(self,stage,updates,resume=None,wall_seconds=1000):
        worker=[self.plan['torch_python'],'-u',str(self.control/'campaign.py'),
            '--plan',str(self.path),'--plan-sha256',self.sha,'--mode','run-stage',
            '--stage',stage,'--updates',str(updates),'--wall-seconds',str(wall_seconds)]
        if resume:worker+=['--resume',str(resume)]
        return ['/usr/bin/python3',str(self.control/'hpc_launch.py'),
            '--archive',str(self.release)+'.tar.gz','--archive-sha256',self.plan['archive_sha256'],
            '--release',str(self.release),'--receipt',str(self.control/(stage+'-launch.json')),
            '--startup-seconds','900','--run-seconds',str(wall_seconds+180),
            '--','env','PYTHONPATH='+str(self.release/'src'),'bash',
            'scripts/hpc/protenix_latent_run.sh' if self.plan.get('scorer_backend')=='protenix_tiny' else 'scripts/hpc/latent_boltz_run.sh',*worker]

    def queue_segment(self,stage,count,resume=None,walltime='0:20'):
        save(self.control/(stage+'-capacity.json'),self.capacity())
        hours,minutes=map(int,walltime.split(':'));seconds=hours*3600+minutes*60-300
        job=self.submit(stage,self.launch(stage,count,resume,seconds),gpu=True,walltime=walltime)
        callback=self.submit('after-'+stage,self.cli('advance','--stage',stage),dependency='ended('+job+')')
        result=dict(stage=stage,job_id=job,callback=callback,updates=count,resume=resume,walltime=walltime)
        save(self.control/(stage+'-queued.json'),result)
        return result

    def run_stage(self,stage,updates,resume,wall_seconds):
        if not os.environ.get('LSB_JOBID'):raise ValueError('Allocated GPU required')
        save(self.control/(stage+'-entry-capacity.json'),self.capacity())
        source=self.source()
        protenix=self.plan.get('scorer_backend')=='protenix_tiny'
        argv=[self.plan['torch_python'],'-u','-m',
            'latentmol.design.latent_protenix' if protenix else 'latentmol.design.latent_boltz',
            '--config',self.plan['full_config'],
            '--checkpoint',source['path'],'--checkpoint-sha256',source['sha256'],
            '--output',str(self.root/stage),'--max-updates',str(updates),'--wall-seconds',str(wall_seconds-60)]
        if protenix:argv+=['--mosaic-python',self.plan['mosaic_python'],'--mosaic-config',self.plan['native_config']]
        else:argv+=['--native-config',self.plan['native_config']]
        if resume:argv+=['--resume',resume]
        if stage=='startup':
            tests=[self.plan['torch_python'],'-m','pytest','-q','tests/design/test_latent_boltz.py',
                   '--junitxml='+str(self.control/'startup-tests.xml')]
            result=subprocess.run(tests)
            if result.returncode:raise RuntimeError('Focused adapter controls failed')
        result=subprocess.run(argv,env=dict(os.environ,LATENTMOL_SOURCE_SHA256=self.plan['archive_sha256'],LATENTMOL_CAMPAIGN_SHA256=self.sha))
        if result.returncode:raise RuntimeError('Native stage failed; preserve its checkpoint and error')

    def advance(self,stage):
        require_done(read(self.receipts/(stage+'.json'))['job_id'])
        value=self.scratch(self.root/stage/'result.json')
        keys=('finite_required_binding_path','finite_chemistry_path','frozen_decoder_unchanged',
              'frozen_native_close_completed','same_shared_graph_and_temperature')
        if value['status'] not in ('completed','segment_completed') or not all(value['technical_checks'].get(k) is True for k in keys):
            raise ValueError('No finite/frozen shared-graph stage success')
        if value['checkpoint_sha256']!=self.plan['checkpoint']['sha256'] or value['updates_this_segment']<=0:
            raise ValueError('Input identity or applied progress differs')
        left=50000-value['completed_step']
        if left<0:raise ValueError('Exceeded authorized endpoint')
        if not left:
            result=dict(status='completed_50000',completed_step=50000,state=value['state'])
            save(self.control/'completion.json',result);return result
        rate=value.get('warm_mean_update_seconds') or value['mean_update_seconds']
        count,wall=remaining_budget(rate,left,max_hours=self.plan.get('max_gpu_hours',72))
        index=1 if stage=='startup' else int(stage.split('-')[1])+1
        return self.queue_segment('main-'+str(index).zfill(4),count,value['state'],wall)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--mode',choices=('queue','run-stage','advance'),required=True)
    p.add_argument('--stage');p.add_argument('--updates',type=int);p.add_argument('--resume');p.add_argument('--wall-seconds',type=int)
    a=p.parse_args();campaign=Campaign(a.plan,a.plan_sha256)
    if a.mode=='queue':result=campaign.queue_segment(campaign.plan.get('initial_stage','startup'),
        campaign.plan.get('initial_updates',20),walltime=campaign.plan.get('initial_walltime','0:20'))
    elif a.mode=='run-stage':result=campaign.run_stage(a.stage,a.updates,a.resume,a.wall_seconds)
    else:result=campaign.advance(a.stage)
    print(json.dumps(result))


if __name__=='__main__':main()
