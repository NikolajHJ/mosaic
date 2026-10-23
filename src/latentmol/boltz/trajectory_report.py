"""Artifact-only retention reports; native seeds/draws are not new trajectories.

Both heads are reported independently. Fresh minus proxy is descriptive because
hardening, preprocessing, pose, recycling and precision can all change together.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from latentmol.experiments import write_json


def distribution(values):
    return {'n': len(values), 'mean': mean(values) if values else None,
            'sd': pstdev(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def build_report(selection, results, seeds):
    """Every planned member/seed remains visible, including unavailable starts."""
    requested = {r['id'] for r in selection['requests']}
    if set(results) - requested:
        raise ValueError('Unexpected native result outside sealed request list')
    roles = {r['id']: r for r in selection['roles']}
    members = {(r['role'], r['step'], r['draw']): r for r in selection['members']}
    detailed = []
    groups = defaultdict(list)
    for member in selection['members']:
        baseline = members[(member['role'], 0, 0)]
        draw_baseline = members[(member['role'], 0, member['draw'])]
        role = roles[member['role']]
        for seed in seeds:
            result = results.get(member['requests'].get(str(seed)), {})
            start = results.get(baseline['requests'].get(str(seed)), {})
            draw_start = results.get(draw_baseline['requests'].get(str(seed)), {})
            eligible = result.get('retention_eligible') is True
            paired = eligible and start.get('retention_eligible') is True
            paired_draw = eligible and draw_start.get('retention_eligible') is True
            affinity = result.get('affinity') or {}
            initial = start.get('affinity') or {}
            original_draw = draw_start.get('affinity') or {}
            objective = role.get('objective')
            signed = None
            if paired and objective:
                key = objective['output']
                signed = (-1 if objective['direction'] == 'maximize' else 1) * objective.get('weight', 1) * (affinity[key] - initial[key]) / objective.get('scale', 1)
            candidate = member.get('candidate', {})
            row = {'member': member['id'], 'role': member['role'], 'cohort': member['cohort'],
                   'target': member['target'], 'arm': role['arm'], 'count': role.get('count'),
                   'initialization_seed': role.get('seed'), 'step': member['step'], 'draw': member['draw'],
                   'kind': member['kind'], 'native_seed': seed, 'candidate_status': member['status'],
                   'parent_status': role.get('parent_status'), 'pilot_status': role.get('pilot_status'),
                   'full_budget_verified': role.get('full_budget_verified', False),
                   'smiles': candidate.get('smiles'), 'heavy_atoms': candidate.get('heavy_atoms'),
                   'request_id': member['requests'].get(str(seed)), 'native_status': result.get('status', 'not_attempted'),
                   'retention_eligible': eligible, 'paired_eligible': paired,
                   'failure': member['failure'] or result.get('failure') or (result.get('identity') or {}).get('failure'),
                   'same_smiles_as_start': candidate.get('smiles') == baseline.get('candidate', {}).get('smiles')
                   if candidate.get('smiles') and baseline.get('candidate', {}).get('smiles') else None,
                   'native_binary': affinity.get('affinity_probability_binary'),
                   'native_continuous': affinity.get('affinity_pred_value'),
                   'delta_binary': affinity['affinity_probability_binary'] - initial['affinity_probability_binary'] if paired else None,
                   'delta_continuous': affinity['affinity_pred_value'] - initial['affinity_pred_value'] if paired else None,
                   'delta_signed_objective': signed,
                   'draw_start_delta_binary': affinity['affinity_probability_binary'] - original_draw['affinity_probability_binary'] if paired_draw else None,
                   'draw_start_delta_continuous': affinity['affinity_pred_value'] - original_draw['affinity_pred_value'] if paired_draw else None,
                   'proxy_binary': (member.get('proxy') or {}).get('binary'),
                   'proxy_context_sha256': (member.get('proxy') or {}).get('context_sha256')}
            detailed.append(row)
            groups[(member['role'], member['step'], member['kind'])].append(row)
    grouped = []
    for (role, step, kind), rows in sorted(groups.items()):
        sample_members = [r for r in selection['members'] if r['role'] == role and r['step'] == step and r['kind'] == kind]
        unique = {r.get('candidate', {}).get('smiles') for r in sample_members if r.get('candidate', {}).get('smiles')}
        grouped.append({'role': role, 'cohort': roles[role]['cohort'], 'target': rows[0]['target'],
                        'step': step, 'kind': kind, 'planned_draws': len(sample_members),
                        'valid_draws': sum(r['status'] == 'valid' for r in sample_members),
                        'invalid_draws': sum(r['status'] == 'invalid' for r in sample_members),
                        'unavailable_draws': sum(r['status'] == 'unavailable' for r in sample_members),
                        'unverified_draws': sum(r['status'] == 'unverified' for r in sample_members),
                        'unique_valid_smiles': len(unique), 'planned_pose_memberships': len(rows),
                        'native_eligible': sum(r['retention_eligible'] for r in rows),
                        'paired_eligible': sum(r['paired_eligible'] for r in rows),
                        'binary': distribution([r['native_binary'] for r in rows if r['retention_eligible']]),
                        'continuous': distribution([r['native_continuous'] for r in rows if r['retention_eligible']]),
                        'delta_binary': distribution([r['delta_binary'] for r in rows if r['paired_eligible']]),
                        'delta_continuous': distribution([r['delta_continuous'] for r in rows if r['paired_eligible']]),
                        'delta_signed_objective': distribution([r['delta_signed_objective'] for r in rows if r['delta_signed_objective'] is not None]),
                        'binary_gain_memberships': sum(r['delta_binary'] > 0 for r in rows if r['paired_eligible']),
                        'continuous_lower_memberships': sum(r['delta_continuous'] < 0 for r in rows if r['paired_eligible']),
                        'both_heads_favorable_memberships': sum(r['delta_binary'] > 0 and r['delta_continuous'] < 0
                                                               for r in rows if r['paired_eligible'])})
    summary = {'planned_roles': len(roles), 'planned_candidate_memberships': len(selection['members']),
               'planned_pose_memberships': len(detailed), 'unique_native_requests': len(selection['requests']),
               'terminal_requests': len(results), 'request_status': dict(Counter(r['status'] for r in results.values())),
               'candidate_status': dict(Counter(r['status'] for r in selection['members'])),
               'all_requests_terminal': set(results) == requested,
               'baseline': 'Same native seed on each role\'s step-0 argmax. Secondary draw-start deltas use saved draw index; no new draws.',
               'interpretation': 'Binary larger is favorable; continuous smaller is favorable for known-active lead optimization. These carbon/prior starts are not known actives.',
               'limits': ['No measured binding, synthesis or selectivity.',
                          'Native seeds/draws, duplicate SMILES, matched arms and related chain lengths are correlated; no independence or significance claim.',
                          'Partial parents remain partial even when some native candidates score.',
                          'Fresh scoring changes hardening, preprocessing, pose, recycling and precision together; it does not isolate a causal source of signal loss.',
                          'A score on changed processed chemical identity is excluded from retention, but retained in raw artifacts.'],
               'groups': grouped}
    return summary, detailed


def write_report(root, selection, results, seeds):
    summary, detailed = build_report(selection, results, seeds)
    write_json(root / 'summary.json', summary)
    temporary = root / 'memberships.csv.tmp'
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed[0]) if detailed else ['member'])
        writer.writeheader(); writer.writerows(detailed)
    temporary.replace(root / 'memberships.csv')
    lines = ['# Fresh-native trajectory evaluation', '',
             f"Native requests terminal: {summary['terminal_requests']}/{summary['unique_native_requests']}; "
             f"planned candidate memberships: {summary['planned_candidate_memberships']}.", '',
             summary['baseline'], '', summary['interpretation'], '',
             'See memberships.csv for every planned checkpoint/draw/pose outcome and summary.json for grouped distributions.', '',
             *[f'- {text}' for text in summary['limits']]]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary
