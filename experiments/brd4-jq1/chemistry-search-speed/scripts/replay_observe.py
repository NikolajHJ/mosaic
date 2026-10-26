"""Fresh single-connection compact read-only replay-diagnostic observation.

Uses the reviewed client byte-capture helper with an explicit independent root;
there is no invocation of a submission, recovery, cleanup or numerical path.
"""
from pathlib import Path
import production_observe as capture

REPO=Path(__file__).resolve().parents[4]
capture.LOCAL=REPO/'results/hpc/20261026-chemistry-replay-02'
capture.PROGRAM=capture.PROGRAM.replace('chemistry-production-observation-v1','chemistry-replay-observation-v1')
capture.PROGRAM=capture.PROGRAM.replace("base/'chemistry-production-02'","base/'chemistry-replay-02'")
capture.PROGRAM=capture.PROGRAM.replace("base/'chemistry-production-release-02/source'","base/'chemistry-replay-release-02/source'")


if __name__=='__main__':capture.main()
