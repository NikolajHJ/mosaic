# Allocated correctness fixtures

Exact original-versus-accelerated preparation, loss/gradient and validation/cache
contracts. Run only in the allocated DTU HPC benchmark job, never locally.

Round2 CPU modules have143planned cases; `test_gpu_execution.py` has19separate
GPU-only cases and fails on CPU fallback. Counts are disjoint and must be
authenticated from raw allocated XML, never inferred from this README.

`test_gpu_adoption_terminal.py` has 19 additional allocated metadata fixtures
for the distinct admission03 capsule. They check raw terminal-proof handling,
not chemistry or GPU arithmetic, and do not replace the original 143 cases or
the 19 actual-GPU-only cases.

Full-model fixtures separately cover exact original configuration/provenance,
real generated-command parser dispatch before model setup, adapter restoration,
finite fixed-step bookkeeping and the CPU-to-GPU VJP transport with a declared
fake client. Fake-client fixtures do not establish GPU or native-model parity;
the subsequent full-model allocated pilots must supply that evidence.

`test_gpu_process_admission.py` adds allocated CPU metadata/fake-process fixtures
for resolved whole-job GPU exclusivity, actual Default compute mode, strict CUDA
device scope, canary identity and owned-process cleanup. These fixtures do not
exercise a GPU; the real two-process canary and original native-model gradient
checks execute only inside the declared L40S pilot allocation.

The production metadata suite declares94allocated cases:50policy,30routing,14exact original metadata call-site integrations. Fixtures do not certify native gradients, and local AST checks do not count as executed tests.
