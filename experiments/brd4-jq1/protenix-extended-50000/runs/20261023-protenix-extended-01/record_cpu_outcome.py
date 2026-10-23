"""Record independently verified CPU/adoption/submission metadata, no calculations."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def main():
    verification=json.loads((HERE/'cpu-verification.json').read_bytes())
    assert verification['status']=='passed' and verification['cpu_cases']==832
    cpath=HERE/'campaign.json';campaign=json.loads(cpath.read_bytes())
    assert campaign['status']=='submitted_not_passed','Inspect existing outcome before duplicate recording'
    campaign.update(status='cpu_passed_native_queued',jobs=verification['jobs'],cpu_cases=832,new_cases=56,
        cpu_certificate_sha256=verification['cpu_certificate_sha256'],
        cpu_verification_sha256=hashlib.sha256((HERE/'cpu-verification.json').read_bytes()).hexdigest(),
        original_protenix_adoption='passed_original10000_unchanged',new_native_passed=False,
        new_segments_submitted=False,full_extended_storage_passed=False)
    cpath.write_bytes((json.dumps(campaign,indent=2)+'\n').encode())
    text='''

## CPU/adoption verified; both fresh native validations queued, 23October11:46Paris

CPU29798115 and callback29798116 actualDONE/hpc/same-job LSFsuccess. Full832 raw
cases passed:500contracts(56new)+239legacy+34mapping+59chemistry, zero failures,
errors or skips;3 unchanged legacy deselections. Raw contract XML header503
versus500 rows is the inherited3 unittest-subtest surplus, not three extra tests.
New allocated checks include exact split/uninterrupted FP32 logits/graph/key/raw
gradients, altered checkpoint rejection and real chemistry preparation/schedule
global offsets across the5000 ramp boundary. Other144scientific library members
remain exact parent631. Local checks were stdlib AST/JSON/byte/layout only.

Callback rehashed/authenticated original29792670 full summary/case/committed
JSONL/terminal native success/frozen/raw/parity/attempt rates and full-frame
collector. Exact saved terminal checkpoint14226bytes,
SHAedf15f06ba1b6591d62cbc3e04e19ecd93b553a0627da38c370fdeed3c1b780f,
is the plain continuation's immutable input; original10000 results preserved.
CPUcertificatec0f784d523d358dfd09d7f57fcfe333fbb5c069d93074681ea35562a5f423171.
Independent45-file byte/XML/JSON/LSF/source/config/actualresource/gate audit passed;
all641 source members and642 safe archive members independently rehashed.

Actual fresh native submissions: random29798159/callback29798160 and plain
continuation29798161/callback29798162. AllfourPEND at09:46:30UTC. No new native
pass or optimization segment ID yet. Both initial readiness checks, empirical
pilot runtime and whole50k/40k byte/file projections must pass before either
independent chain begins. No full extended-output storage pass is claimed from
the smaller validation allowance. Each later segment requires authenticated
predecessor scientific completion, exact saved-state replay, fresh native and
remaining-capacity/runtime gates. Failed/uncompleted segments never authorize
successors. No gate waiver, additional cleanup or heartbeat unpause.

Random50k keepsLR.3/Dirichlet1 seed17, .001through5000 then geometric100at50000.
Plain29792670 keeps native-onlyLR.3 and global10000->50000, no new chemistry.
All old runs/report editions/Boltz endpoint diagnostics remain unchanged. A new
50k report must pin actual new segment IDs and join global saved frames with
duplicate boundary states authenticated exactly; old contracts/editions cannot
be retargeted. All rendering/array/gradient checks stay allocated DTU HPC only.
'''
    with (HERE/'README.md').open('a',encoding='utf-8') as f:f.write(text)
    with (REPO/'docs/experiment_ledger.md').open('a',encoding='utf-8') as f:f.write(text.replace('## CPU/adoption verified;','### Protenix50k CPU/adoption verified;'))
    overview=HERE.parents[1]/'README.md'
    with overview.open('a',encoding='utf-8') as f:f.write('\n\n23October11:46Paris: source641 staged/read-only, CPU29798115/callback29798116\nDONE/full832 and original plain completion adoption passed. Independent45-file\naudit passed. Fresh randomnative29798159 and continuationnative29798161 queued;\nno newnative pass, full extended-storage pass or optimization segment yet.\n')
    print('Recorded832 CPU/adoption pass and exact new queued native IDs.')
if __name__=='__main__':main()
