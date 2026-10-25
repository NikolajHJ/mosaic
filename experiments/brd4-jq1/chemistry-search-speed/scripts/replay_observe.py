"""Fresh single-connection compact read-only replay-diagnostic observation.

Uses the reviewed client byte-capture helper with an explicit independent root;
there is no invocation of a submission, recovery, cleanup or numerical path.
"""
from pathlib import Path
import production_observe as capture

REPO=Path(__file__).resolve().parents[4]
capture.LOCAL=REPO/'results/hpc/20261025-chemistry-replay-01'
capture.PROGRAM=capture.PROGRAM.replace('chemistry-production-observation-v1','chemistry-replay-observation-v1')
capture.PROGRAM=capture.PROGRAM.replace("base/'chemistry-production-01'","base/'chemistry-replay-01'")
capture.PROGRAM=capture.PROGRAM.replace("base/'chemistry-production-release-01/source'","base/'chemistry-replay-release-01/source'")


if __name__=='__main__':capture.main()
