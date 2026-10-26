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

`test_softmax_replay_diagnostic.py`30 and `test_replay_ops.py`30 form the isolated
replay capsule metadata suite. All60 passed on allocated CPU29816497; no native
model/optimizer evaluation follows from fixtures. Actual GPU controls require
all ten successful processes, byte-bound saved inputs and preserved failures.

Replay02 retains30controller fixtures and extends diagnostic metadata30→37;
all67 passed allocatedCPU29818483. New regressions run a REAL audit hook in an
isolated stdlib subprocess: archived-source hashing succeeds while a subsequent
direct checkpoint read stays blocked. Error restoration, path/write rejection
and exact compiler-control environment/22receipt scope are covered. This CPU
pass does not establish a GPU finding or native full-model acceptance.

Current fullmodel05 declares408cases: retained388 plus20 compiler-qualification/
context-metadata fixtures. CPU29818536 passed every rawcase,0fail/errors/skips;
failed04's407/408 fixture NameError remains preserved. Production02 adds6exact
context-transport fixtures to50policy+33routing+14actual-callsite =103cases;
those are queued on CPU29818556, not yet passed. No local test or scientific
array execution. Replay22 GPU controls are a separate isolated diagnostic;
only the fresh full-model pilot can establish new native acceptance.
