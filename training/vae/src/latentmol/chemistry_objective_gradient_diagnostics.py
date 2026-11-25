"""Opt-in full logical decoder-gradient observations for one chemistry update.

The caller supplies already weighted scalar terms on the existing microbatch
tapes. This observer runs VJPs but never forwards a model, samples, writes .grad,
steps an optimizer or selects a training update. Parameter displacement is the
actual applied decoder movement, including any declared optimizer-step policy.
First-order dot products and cosines do not certify finite loss or validity.
"""
from copy import deepcopy
import math


VERSION = "full-logical-chemistry-decoder-gradients-v1"
BASE_TERMS = frozenset(("reconstruction", "kl", "repair", "composition", "occupancy"))


class ObjectiveGradientDiagnostics:
    """Wrap exactly one caller-owned update and pass this object as its observer.

    Usage: ``with ObjectiveGradientDiagnostics(model, ...) as observer: ...``;
    call ``observer.receipt()`` after exit. The caller remains responsible for
    full model/optimizer/RNG restoration of its disposable experiment. Retained
    gradients use native dtype; float64 reductions need only one parameter at a
    time. All gradient/snapshot tensors are released on context exit.
    """

    def __init__(self, model, *, real_points=256, prior_points=512, microbatch_size=128):
        if any(type(value) is not int or value < 1 for value in (real_points, prior_points, microbatch_size)):
            raise ValueError("Gradient diagnostics require positive exact logical counts")
        self.model = model
        self.points = dict(real=real_points, prior=prior_points)
        self.microbatch_size = microbatch_size
        self.schedule = [branch for branch, count in self.points.items()
                         for _ in range(math.ceil(count / microbatch_size))]
        self._entered = self._closed = False
        self._handles, self._values, self._native = [], {}, {}
        self._before, self._parameters, self._terms, self._observed = {}, [], None, []
        self._result = None

    def __enter__(self):
        import torch
        if self._entered or self._closed:
            raise ValueError("Each gradient observer owns exactly one update")
        self._entered = True
        self._parameters = [("decoder." + name, p) for name, p in self.model.decoder.named_parameters()
                            if p.requires_grad]
        if not self._parameters or any(not p.is_leaf or not p.is_floating_point() for _, p in self._parameters):
            raise ValueError("Require trainable floating decoder leaf parameters")
        try:
            for name, p in self._parameters:
                if not bool(torch.isfinite(p).all()):
                    raise FloatingPointError("Nonfinite starting decoder parameters")
                self._before[name] = p.detach().clone()
                # autograd.grad does not invoke post-accumulate hooks. These
                # copies observe actual backward accumulation before clipping.
                self._handles.append(p.register_post_accumulate_grad_hook(
                    lambda parameter, name=name: self._capture_native(name, parameter)))
        except BaseException:
            self._release()
            raise
        return self

    def _capture_native(self, name, parameter):
        if parameter.grad is not None:
            self._native[name] = parameter.grad.detach().clone()

    def __call__(self, branch, terms):
        import torch
        if not self._entered or self._closed or len(self._observed) >= len(self.schedule):
            raise ValueError("Gradient observation outside its one logical update")
        keys = frozenset(terms) if type(terms) is dict else frozenset()
        if (branch != self.schedule[len(self._observed)] or keys not in (BASE_TERMS, BASE_TERMS | {"valid_self"}, BASE_TERMS | {"valid_reference"})
                or (self._terms is not None and keys != self._terms)):
            raise ValueError("Gradient observer lost the ordered complete microbatch/term schedule")
        self._terms = keys
        for term, scalar in terms.items():
            if (not isinstance(scalar, torch.Tensor) or scalar.ndim != 0
                    or not scalar.is_floating_point() or not bool(torch.isfinite(scalar))):
                raise ValueError("Gradient observation requires finite already weighted scalar terms")
            destination = self._values.setdefault(branch + "/" + term, {})
            if not scalar.requires_grad:
                continue
            gradients = torch.autograd.grad(scalar, tuple(p for _, p in self._parameters),
                                            retain_graph=True, allow_unused=True)
            for (name, _), value in zip(self._parameters, gradients):
                if value is None:
                    continue
                value = value.detach()
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite weighted component gradient")
                # Do not retain model-sized zero arrays for unused decoder
                # heads or graph-connected zero terms.
                if not bool(value.count_nonzero()):
                    continue
                if name in destination:
                    destination[name].add_(value)
                else:
                    destination[name] = value.clone()
        self._observed.append(branch)

    @staticmethod
    def _metric(dot, norm_square, delta_square):
        if not all(math.isfinite(v) for v in (dot, norm_square, delta_square)):
            raise FloatingPointError("Nonfinite full logical gradient observation")
        norm, delta = math.sqrt(norm_square), math.sqrt(delta_square)
        return dict(gradient_norm=norm, gradient_dot_displacement=dot,
                    displacement_norm=delta, cosine=dot / (norm * delta) if norm and delta else None)

    def _finish(self):
        import torch
        if self._observed != self.schedule:
            raise ValueError("Gradient receipt requires every logical microbatch")
        current = [("decoder." + name, p) for name, p in self.model.decoder.named_parameters() if p.requires_grad]
        if [(n, id(p)) for n, p in current] != [(n, id(p)) for n, p in self._parameters]:
            raise ValueError("Decoder parameter identity changed during observation")
        groups = {key: [key] for key in self._values}
        groups.update({term: [key for key in self._values if key.endswith("/" + term)]
                       for term in sorted(self._terms)})
        groups["full_objective"] = list(self._values)
        accumulators = {key: [0., 0.] for key in groups}
        pairs = [("reconstruction", "repair"), ("reconstruction", "valid_self"),
                 ("repair", "valid_self"), ("reconstruction", "prior/repair"),
                 ("reconstruction", "valid_reference"), ("repair", "valid_reference")]
        pair_dots = {a + "__" + b: 0. for a, b in pairs if a in groups and b in groups}
        delta_square, maximum_error = 0., 0.
        with torch.no_grad():
            for name, parameter in self._parameters:
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError("Nonfinite actual decoder proposal")
                delta = parameter.detach().double() - self._before[name].double()
                delta_square += float(delta.square().sum())
                vectors = {}
                for key, members in groups.items():
                    parts = [self._values[member][name] for member in members if name in self._values[member]]
                    vector = sum((part.double() for part in parts), torch.zeros_like(delta))
                    vectors[key] = vector
                    accumulators[key][0] += float((vector * delta).sum())
                    accumulators[key][1] += float(vector.square().sum())
                native = self._native.get(name)
                native = torch.zeros_like(delta) if native is None else native.double()
                recovered = vectors["full_objective"]
                error = float((recovered - native).abs().max())
                maximum_error = max(maximum_error, error)
                rtol, atol = (1e-9, 1e-11) if parameter.dtype == torch.float64 else (1e-4, 2e-7)
                if not torch.allclose(recovered, native, rtol=rtol, atol=atol):
                    raise ValueError("Weighted terms do not reconstruct actual pre-clip decoder backward")
                for a, b in pairs:
                    key = a + "__" + b
                    if key in pair_dots:
                        pair_dots[key] += float((vectors[a] * vectors[b]).sum())
        metrics = {key: self._metric(dot, square, delta_square) for key, (dot, square) in accumulators.items()}
        cosines = {}
        for key, dot in pair_dots.items():
            a, b = key.split("__")
            denominator = metrics[a]["gradient_norm"] * metrics[b]["gradient_norm"]
            cosines[key] = dot / denominator if denominator else None
        self._result = dict(version=VERSION, status="observed_not_readiness", complete_logical_update=True,
            real_points=self.points["real"], prior_points=self.points["prior"], microbatch_size=self.microbatch_size,
            observed_microbatches={branch: self._observed.count(branch) for branch in self.points},
            weighted_terms=sorted(self._terms), metrics=metrics, term_gradient_cosines=cosines,
            maximum_gradient_reconstruction_error=maximum_error,
            component_sum_matches_native_backward=True, observer_model_forwards=0,
            observer_optimizer_steps=0, observer_random_samples=0,
            displacement_scope="actual applied decoder parameter delta; caller owns optimizer step",
            gradient_scope="full logical already weighted terms, before clipping; decoder parameters only",
            interpretation="Negative gradient dot displacement is first-order downhill; finite loss and validity are separate")

    def _release(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._before.clear()
        self._values.clear()
        self._native.clear()
        self._parameters.clear()
        self._closed = True

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                self._finish()
        finally:
            self._release()
        return False

    def receipt(self):
        if self._result is None:
            raise ValueError("A failed or unfinished observer cannot issue a full logical receipt")
        return deepcopy(self._result)
