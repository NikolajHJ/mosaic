"""Record verified publication/browser evidence; metadata only, no numerical work."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    folder=REPO/'results/hpc'/HERE.name;latest=read(folder/'viewer/latest.json')
    assert read(HERE/'delivery-verification-interim-0001.json')['status']=='passed'
    http=read(HERE/'http-verification-edition-0001-02.json');assert http['served_files_verified']==413
    qa={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'url':'http://127.0.0.1:8779/report.html','edition':'edition-0001','tab':'1','html_sha256':latest['html_sha256'],'checks':{
      'protenix_numeric_then_next':[1500,1501],'protenix_midpoint':5000,'protenix_latest':10000,'actual_slider_drag_interior':7263,
      'boltz_playback_observed_update':2,'manual_during_playback':{'chosen':321,'paused_button':'Play','stable_after_zoom':321},'boltz_previous':320,'boltz_latest':10000,
      'zoom':'150% then restored100%','probability_table_data_rows':[25,300],
      'protenix_terminal_raw_invalid_disconnected_visible':True,'boltz_terminal_sanitized_carbon_chain_visible':True,
      'all_three_selectors':True,'chemistry_empty':{'graph_children':0,'metrics':'','heatmap_children':0,'all_eight_frame_controls_disabled':True,'number':'No saved snapshot'},
      'stable_wrapper_pins_edition':True,'diagnostic_panel_pending_other_two_not_requested':True},
      'scope':'Actual browser presentation only; no scientific pass inferred','preview_sha256':sha(folder/'report-preview-01.png')}
    (HERE/'browser-qa-edition-0001.json').write_text(json.dumps(qa,indent=2)+'\n')
    campaign=read(HERE/'campaign.json');campaign.update(status='report_published_diagnostic_queued',report_url=qa['url'],published_edition=latest,published_milestones=latest['milestone_progress'],delivery_verification_sha256=sha(HERE/'delivery-verification-interim-0001.json'),browser_qa_sha256=sha(HERE/'browser-qa-edition-0001.json'),http_complete_verified_files=413,diagnostic_findings=None)
    (HERE/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    with (HERE/'README.md').open('a',encoding='utf-8') as f:f.write('''

## Verified delivery, 23 October 11:09 Paris

Report job 29796556 actually finished DONE/hpc with same-job LSF success and a
passed presentation assessment. Collectors 29796553/54 verified all 10,001 original
frames and 10,000 updates per arm. Protenix remains scientifically passed; Boltz
remains failed. Original chemistry is queued with zero captured states.

Archive: 90,878,067 bytes, SHA9551b2e0b0a821e606fa740ac09df432ec6575f562dfe51390c84629d6009ae7,
all 441 files verified. The first transfer timed out and stopped, preserving its
12,713,984-byte prefix. Exact missing-range recovery authenticated every prefix,
range and full archive SHA. Two stalled range-transfer SSH children were verified
and stopped; no scientific or scheduler process was changed. All partial files,
timeouts and recovery receipts remain. Local audit corrections for the canonical
archive prefix and UTF8 decoding retained their failed attempts; no source or
scientific assessment changed. Large original summaries/NPZs remain explicit HPC
references; the complete local report assets and compact proofs are verified.

New immutable edition-0001: http://127.0.0.1:8779/report.html. Captured at
08:48:51.533606 UTC and published at 09:03:14.202985 UTC. States are
10001/10001/0 and milestone floors 10000/10000/0. Initial HTTP audit started
before publisher completion and covered 410 files; a fresh stable-file-set audit
after completion passed ALL 413 served files. Actual browser QA passed numeric,
arrows, midpoint/latest, drag to 7263, Boltz playback/manual321 pause/stability,
previous320, both 25-atom/300-pair tables, raw invalid Protenix graph and REAL empty
chemistry graph/metrics/heatmap clearing with all eight frame controls disabled.
Stable wrapper checked; screenshot report-preview-01.png. Server PID53268 is hidden
and bound to localhost8779; inspect process/port before any recovery, no duplicate.

At 09:09:36 UTC diagnostic29796551/capture29796552 remain PEND, with no finding
or established unique cause. Existing interim29796557 and final29796558 remain
submitted, never duplicate. Source17 pins chemistry as unstarted: its later start
requires a new immutable report source with actual entry/driver pins before that
arm is published. Original saved runs are unchanged. The existing heartbeat is
user-paused; do not silently unpause it.
''')
    with (REPO/'docs/experiment_ledger.md').open('a',encoding='utf-8') as f:f.write('''

### 2026-10-23 11:09 Paris — completed lower-rate P/B report delivered

Report29796556 actual DONE/hpc/LSF and passed presentation assessment; collectors
29796553/54 audited all10001 original frames/10000 updates each. Pro passed/Bol
failed scientific outcomes preserved. Archive90878067 bytes SHA9551b2e0b0a821e606fa740ac09df432ec6575f562dfe51390c84629d6009ae7,
all441 files verified after authenticated partial/range recovery; all timeouts,
stopped exact byte-transfer processes and partial evidence retained. No scientific
process/result mutation. Independent delivery audit passed; local archive-prefix/
UTF8 audit failures preserved before correction. New edition0001 at8779/report.html
captures10001/10001/0 states and milestone floors10000/10000/0. First HTTP audit
covered410 files while publication finished; subsequent stable-set audit passed
ALL413. Actual browser QA passed drag7263/numeric1500/next1501/mid5000/terminal,
Bol play2/manual321 pause and stability/prev320/terminal/25atom300pair tables,
all selectors, raw graphs and actual empty-chemistry clearing/disabled controls.
Screenshot, QA, source and publication hashes recorded in analysis03; server53268.
At09:09:36UTC current chemistry29792672 and separate Boltz diagnostic29796551 remain
PEND. No root-cause finding or retroactive gradient pass exists. Existing interim
29796557/final29796558 unchanged; future started chemistry requires fresh immutable
entry pins. Current report is delivered, endpoint investigation remains pending.
Heartbeat stays paused. All older viewers/editions/results remain unchanged.
''')
    print('Recorded verified immutable report, complete HTTP audit and actual browser QA.')
if __name__=='__main__':main()
