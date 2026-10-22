# BRD4 frozen Boltz-2 affinity optimization: scripts

This folder belongs to [the experiment protocol](../README.md).

- [assess_boltz_arm.py](assess_boltz_arm.py): Adopt a passed arm from a failed aggregate readiness, without rerunning science.
- [assess_boltz_cpu.py](assess_boltz_cpu.py): Authenticate allocated CPU test outcomes before native GPU readiness.
- [assess_boltz_native.py](assess_boltz_native.py): Seal actual native gates/pilots for both requested initializations.
- [brd4_boltz_affinity.py](brd4_boltz_affinity.py): Allocated Boltz affinity readiness or a fresh Mosaic delayed-chemistry arm.
- [render_boltz_affinity.py](render_boltz_affinity.py): Allocated saved-only Boltz affinity report with authenticated raw trajectories.
- [submit_boltz_arms.py](submit_boltz_arms.py): Allocated controller: submit independent arms only after actual native success.
