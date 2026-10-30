"""Fresh full validation of the declared Protenix execution profile; no segments."""
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import sys

CAP = 512 * 1024**2
SUCCESS = 'COMPLETE_ORIGINAL_VALIDATION_RETURNED'


def require(value, message):
    if not value:
        raise ValueError(message)


def validation_tree(source, kind):
    """Remove diagnostic stopping only; run the original VALIDATE branch fully."""
    require(kind in ('plain', 'chemistry'), 'Only the two Protenix validation paths')
    tree = ast.parse(source)
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
            and tree.body[0].name == kind, 'Exact inherited validation function required')
    fn = tree.body[0]
    if kind == 'plain':
        vectors = [n for n in ast.walk(fn) if isinstance(n, ast.List)
                   and any(isinstance(v, ast.Constant) and v.value == '--phase' for v in n.elts)]
        require(len(vectors) == 1, 'One original plain command required')
        values = vectors[0].elts
        at = next(i for i, v in enumerate(values) if isinstance(v, ast.Constant) and v.value == '--phase')
        require(isinstance(values[at + 1], ast.Constant) and values[at + 1].value == 'validate',
                'Production commands are forbidden in qualification')
    else:
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'execute']
        require(len(calls) == 1 and isinstance(calls[0].args[2], ast.Constant)
                and calls[0].args[2].value == 'validate', 'Chemistry production branch forbidden')
    changes = 0
    class Full(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal changes
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'stopped_tree':
                require(len(node.args) == 1 and not node.keywords, 'Diagnostic stop shape differs')
                changes += 1
                return node.args[0]
            return node
        def visit_ExceptHandler(self, node):
            require(isinstance(node.type, ast.Name) and node.type.id == 'BaseException',
                    'Unexpected inherited exception handling')
            node.body = [ast.Raise()]
            return node
    tree = Full().visit(tree)
    require(changes == 1 and isinstance(fn.body[-1], ast.Raise), 'One original stop and final sentinel required')
    fn.body[-1] = ast.Return(ast.Tuple([ast.Name('records', ast.Load()), ast.Constant(SUCCESS)], ast.Load()))
    return ast.fix_missing_locations(tree)


def reference_startup_tree(source):
    """Keep the actual original reference CLI checks, after the new producer startups."""
    tree = ast.parse(source)
    require(len(tree.body) == 1 and tree.body[0].name == 'premodel', 'Original startup body required')
    fn = tree.body[0]
    selected = [n for n in fn.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Tuple)
                and [v.id if isinstance(v, ast.Name) else None for v in n.target.elts] == ['kind', 'action']]
    require(len(selected) == 1, 'One original producer startup loop required')
    fn.body.remove(selected[0])
    return ast.fix_missing_locations(tree)


def functions(helper):
    namespace = dict(vars(helper))
    helper.CAP = CAP
    for kind in ('plain', 'chemistry'):
        exec(compile(validation_tree(inspect.getsource(getattr(helper, kind)), kind),
                     helper.__file__, 'exec'), namespace)
    exec(compile(reference_startup_tree(inspect.getsource(helper.premodel)), helper.__file__, 'exec'), namespace)
    return namespace


def validate_policy(plan):
    require(plan['qualification_arms'] == ['plain', 'chemistry']
            and plan['qualification_policy'] == {'phase': 'validate', 'optimizer_submission_allowed': False,
                'both_complete_chemistry_pilots_required': True, 'replicas_per_native_checkpoint': 16},
            'Closed full-validation-only policy required')
    require(plan['diagnostic_storage'] == {'group_cap_bytes': CAP,
        'reference_metadata_cap_bytes': 128 * 1024**2, 'inherited_reference_cap_bytes': 64 * 1024**2,
        'all_output_retained': True, 'optimizer_admission_allowed': False}, 'Funded metadata limits differ')


def startup(helper, plan, output):
    output.mkdir()
    actions = functions(helper)
    rows = []
    for kind in plan['qualification_arms']:
        child = output / kind
        child.mkdir()
        value = actions[kind](plan, child, startup=True)
        require(value['status'] == 'passed_metadata_only' and value['numerical_setup_entered'] is False
                and value['updates'] == 0 and value['job_id'] == os.environ['LSB_JOBID'],
                'Actual qualification producer startup failed')
        rows.append({'arm': kind, 'startup': value})
    actions['premodel'](plan, output / 'reference-checks')
    refs = helper.read(output / 'reference-checks/assessment.json')['reference_metadata']
    helper.write(output / 'assessment.json', {'status': 'passed_actual_full_validation_startups',
        'startups': rows, 'reference_metadata': refs, 'optimizer_updates': 0,
        'can_submit_optimizer': False, 'model_constructed': False})


def full_child(helper, plan, output, kind):
    output.mkdir()
    result = {'status': 'failed', 'arm': kind, 'optimizer_continuation_updates': 0,
              'can_submit_optimizer': False, 'original_failures_preserved': True}
    try:
        records, status = functions(helper)[kind](plan, output)
        require(status == SUCCESS, 'Original complete validation did not return')
        summary = helper.read(output / 'payload/study/summary.json')
        require(summary['status'] == 'passed' and summary['job_id'] == os.environ['LSB_JOBID'],
                'Original complete native/pilot/runtime validation failed')
        summary_path = output / 'payload/study/summary.json'
        result.update(status='passed_complete_original_validation_pending_independent_capture',
                      measurements=records, original_summary={'path': str(summary_path),
                          'bytes': summary_path.stat().st_size, 'sha256': helper.sha(summary_path)},
                      execution_profile=plan['execution_profile'])
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['output_bytes'] = helper.bounded(output)
        helper.write(output / 'qualification-assessment.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--mode', required=True, choices=('startup', 'plain', 'chemistry'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Import the helper only after authenticating every inherited file in the wrapper.
    import failure_diagnostic_ops as ops
    own = ops.read(args.plan)
    require(ops.sha(Path(args.plan)) == args.pin and os.environ.get('LSB_JOBID', '').isdigit(),
            'Authenticated plan and actual allocation required')
    validate_policy(own)
    helper = ops.module(Path(own['helper_source']['path']) / 'scripts/continuation_failure_diagnostics.py',
                        '_authenticated_full_validation_metadata_helpers')
    helper.CAP = CAP
    helper.guard_candidate_environment(os.environ) if args.mode != 'startup' else None
    if args.mode == 'startup':
        startup(helper, own, args.output)
    else:
        full_child(helper, own, args.output, args.mode)


if __name__ == '__main__':
    main()
