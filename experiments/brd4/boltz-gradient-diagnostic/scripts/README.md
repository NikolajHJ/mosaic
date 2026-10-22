# BRD4 Boltz gradient diagnostics: scripts

This folder belongs to [the experiment protocol](../README.md).

- [analyze_boltz_balance.py](analyze_boltz_balance.py): Allocated saved-evidence analysis; never rerun a model or chemistry evaluator.
- [analyze_boltz_gradient_diagnostic.py](analyze_boltz_gradient_diagnostic.py): Allocated NumPy analysis of authenticated saved native diagnostic records.
- [check_boltz_diagnostic_cpu.py](check_boltz_diagnostic_cpu.py): Authenticate persisted CPU certificates without depending on live LSF retention.
- [diagnose_boltz_random_gradient.py](diagnose_boltz_random_gradient.py): Allocated, evidence-only replay of the failed random Boltz bond derivative.
