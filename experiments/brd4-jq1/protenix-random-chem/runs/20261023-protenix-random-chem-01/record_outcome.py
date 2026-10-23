"""Record actual new source/CPU/submission evidence; standard-library metadata only."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def main():
    folder=REPO/'results/hpc'/HERE.name;cpu=json.loads((HERE/'cpu-verification.json').read_bytes())
    raw=(folder/'submission-stdout.bin').read_bytes().decode().split('Authenticated 625 immutable source files\n',1)[1].strip();submission=json.loads(raw)
    record={'status':'cpu_passed_native_queued','source_manifest':cpu['manifest'],'source':submission['source'],'root':submission['root'],'package':json.loads((HERE/'package.json').read_bytes()),'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),'jobs':cpu['jobs'],'cpu_cases':776,'cpu_certificate_sha256':cpu['cpu_certificate_sha256'],'cpu_verification_sha256':hashlib.sha256((HERE/'cpu-verification.json').read_bytes()).hexdigest(),'preflight':submission['preflight'],'new_native_passed':False,'new_long_submitted':False,'existing_results_unchanged':True,'heartbeat_remains_paused':True}
    (HERE/'campaign.json').write_text(json.dumps(record,indent=2)+'\n')
    with (HERE/'README.md').open('a',encoding='utf-8') as f:f.write('''

## Actual submission and CPU outcome, 23 October 11:09 Paris

Source631 staged read-only at08:54UTC, manifest
bb3ac9cae836b59f9a62fb33d4775c0b4388f3b200a7422ce8a620d2d405950f;
archive39609007 bytes SHAc005429dc2581c929662698e80da154fabc0f3365615facfa7582641fa7fcddc.
All145 scientific library members equal parent625. New validation preflight passed
14371611825 required versus21144155255 conservative available bytes, preserving
all old remaining projections, existing analysis allowance, new1GiB validation
and4GiB/100000file reserves. Finite user file headroom19567808; allzero filesystem
inode tuple is supplemental/unreported. Not a full new pilot/long capacity pass.

CPU29796731 and callback29796732 actually DONE/LSF success. Full776 raw cases:
444 contracts including48 new,239 legacy,34 mapping,59 chemistry; zero failures,
errors or skips,3 unchanged legacy deselections. The contract XML header447
versus444 rows is the inherited3 unittest-subtest surplus, not three extra tests;
collected444/pass444 match actual rows. Independent35-file byte/XML/source/config/
CPU/proof/submission/storage audit passed; see cpu-verification.json.
CPU certificate7adde8e09679637c597b72c6806aa4fc51cec590ba494da029b457e8d22e1a5d.

Fresh native validation29796780 and own-ended callback29796781 were actually
submitted and remain PEND at09:09:36UTC. No native pass or optimization long ID
yet. Never duplicate. All fresh raw/parity/null/primal/frozen/random identity,
nonzero chemistry/low-high coupled/BOTH10pilot/runtime gates and fresh combined
full output projection/reserves must pass before one independent long. No old
scientific run/result, cleanup or heartbeat schedule changed.
''')
    with (HERE.parents[1]/'README.md').open('a',encoding='utf-8') as f:f.write('''

Latest23October11:09Paris: source631 authenticated/read-only, CPU29796731 and
callback29796732 DONE/full776 cases and independent35-file audit passed. Fresh
native29796780/callback29796781 queued. No native pass or10000-update optimization
submission yet. Old results and independent endpoint/report work are unchanged.
These launchers are account-bound historical execution entries; their constants
are not portable account defaults.
''')
    with (REPO/'experiments/brd4-jq1/README.md').open('a',encoding='utf-8') as f:f.write('''

Additional23October user-authorized [random Protenix plus delayed chemistry](protenix-random-chem/README.md):
freshLR0.3/Dirichlet1 seed17/chemistry0.001-to100, independent of saved studies.
''')
    with (REPO/'docs/experiment_ledger.md').open('a',encoding='utf-8') as f:f.write('''

### 2026-10-23 11:09 Paris — random Protenix chemistry776CPU passed; fresh native queued

New source631 bb3ac9cae836b59f9a62fb33d4775c0b4388f3b200a7422ce8a620d2d405950f
authenticated/read-only. CPU29796731/callback29796732 actualDONE/LSF/full776 rows:
444 contracts(48new)+239legacy+34mapping+59chemistry, zero failures/errors/skips,
3 unchanged legacy deselections. Raw XML contract header surplus3 is inherited
unittest subtests; actual collected/pass444 matches444 rows. Independent35-file
byte/XML/source/config/CPU/proof/actualsubmission/quota audit passed, certificate
7adde8e09679637c597b72c6806aa4fc51cec590ba494da029b457e8d22e1a5d.
Initial full old3remaining+analysis+new1GiBvalidation+4GiB/100000files gate passed
14371611825 required/21144155255 available; not new full pilot/long admission.
Actual fresh native29796780/callback29796781 queued, no native pass or long ID.
Same declared LR0.3/randomDirichlet1 seed17/chemistry0.001-to100 and original
frozen context/model/pocket/key/checks. Old results, separate Boltz diagnostic
and report unchanged; no cleanup or automation unpause.
''')
    print('Recorded actual776 CPU pass, independent audit and queued native validation.')
if __name__=='__main__':main()
