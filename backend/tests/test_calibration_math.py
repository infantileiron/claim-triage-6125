import numpy as np
import pytest

from src.services.calibration import CalibrationError, estimate, hash_snapshot


def snapshot(opinions, gold, n_claims=None):
    n = n_claims or (max([o['claim_id'] for o in opinions] + [g['claim_id'] for g in gold]) if opinions or gold else 0)
    snap = {'opinions': opinions,
            'gold': gold,
            'claims': [{'id': i + 1, 'code': f'C{i + 1}', 'label': None} for i in range(n)]}
    snap['hash'] = hash_snapshot(snap)
    return snap


def simulate(seed=3, n=100, annotators=('A', 'B', 'C', 'D'), coverage=.6,
             sens=None, spec=None, truth=None, gold_ids=None, reverse=()):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n) if truth is None else np.asarray(truth)
    sens_map = {'A': .85, 'B': .85, 'C': .85, 'D': .85}
    spec_map = {'A': .85, 'B': .85, 'C': .85, 'D': .85}
    if sens:
        sens_map.update(sens)
    if spec:
        spec_map.update(spec)
    if isinstance(coverage, (int, float)):
        coverage = {a: coverage for a in annotators}
    ops = []
    for i in range(n):
        for a in annotators:
            if rng.random() < coverage[a]:
                p = sens_map[a] if z[i] else 1 - spec_map[a]
                y = (int(rng.random() < sens_map[a]) if z[i]
                     else int(rng.random() < (1 - spec_map[a])))
                if a in reverse:
                    y = 1 - y
                ops.append({'claim_id': i + 1, 'annotator': a, 'label': y, 'revision': 1, 'note': ''})
    gids = list(range(0, n, 15))[:6] if gold_ids is None else gold_ids
    gold = [{'claim_id': int(i + 1), 'label': int(z[i])} for i in gids]
    return snapshot(ops, gold), z


def test_recovers_sensitivity_and_specificity():
    snap, z = simulate(seed=3, n=140, coverage=.7)
    r = estimate(snap)
    assert abs(r['prevalence'] - z.mean()) < .12
    for a in r['annotators']:
        assert a['status'] in ('reliable', 'unstable')
        assert abs(a['sensitivity'] - .85) < .15
        assert abs(a['specificity'] - .85) < .15
    assert r['evidence']['gold'] >= 2


def test_per_claim_error_is_small_when_confident():
    snap, z = simulate(seed=5, n=140, coverage=.7)
    r = estimate(snap)
    est = [c for c in r['claims'] if c['status'] == 'estimated']
    assert len(est) > 30
    errs = [abs(c['probability'] - z[c['claim_id'] - 1]) for c in est]
    assert np.mean(errs) < .2
    for c in r['claims']:
        if c['status'] == 'insufficient':
            assert c['probability'] is None and c['reason']


def test_reversed_annotator_is_flagged_not_trusted():
    snap, z = simulate(seed=5, n=140, coverage=.7, reverse=('D',))
    r = estimate(snap)
    status = {a['annotator']: a for a in r['annotators']}
    assert status['D']['status'] == 'reversed'
    assert status['D']['sensitivity'] < .35 and status['D']['specificity'] < .35
    for a in ('A', 'B', 'C'):
        assert status[a]['status'] in ('reliable', 'unstable')
    est = [c for c in r['claims'] if c['status'] == 'estimated']
    assert est
    acc = np.mean([int(c['probability'] >= .5) == z[c['claim_id'] - 1] for c in est])
    assert acc > .8


def test_disjoint_population_without_gold_is_excluded():
    rng = np.random.default_rng(11)
    z = rng.integers(0, 2, 120)
    ops = []
    for i in range(120):
        group = ['A', 'B'] if i < 70 else ['C', 'D']
        for a in group:
            if rng.random() < .8:
                y = int(rng.random() < .85) if z[i] else int(rng.random() < .15)
                ops.append({'claim_id': i + 1, 'annotator': a, 'label': y,
                            'revision': 1, 'note': ''})
    gold = [{'claim_id': int(i + 1), 'label': int(z[i])} for i in (5, 20, 35, 50, 65)]
    r = estimate(snapshot(ops, gold))
    status = {a['annotator']: a for a in r['annotators']}
    assert status['A']['status'] in ('reliable', 'unstable')
    assert status['C']['status'] == 'unanchored' and status['C']['sensitivity'] is None
    assert r['excluded_groups']
    excluded_claims = {c for g in r['excluded_groups'] for c in g['claims']}
    assert {71, 80, 120} <= excluded_claims
    for c in r['claims']:
        if c['claim_id'] in excluded_claims:
            assert c['probability'] is None and c['status'] == 'insufficient'


def test_refuses_without_enough_evidence():
    snap, _ = simulate(seed=9, n=20, coverage=.3, gold_ids=[1])
    snap['opinions'] = snap['opinions'][:4]
    with pytest.raises(CalibrationError) as exc:
        estimate(snap)
    assert exc.value.reasons and exc.value.checks


def test_refuses_without_gold():
    snap, _ = simulate(seed=9, n=80, coverage=.6, gold_ids=[])
    with pytest.raises(CalibrationError) as exc:
        estimate(snap)
    assert '金标准' in ' '.join(exc.value.reasons)


def test_gold_is_fixed_and_not_estimated():
    snap, z = simulate(seed=13, n=90, coverage=.6)
    r = estimate(snap)
    gold = {g['claim_id']: g['label'] for g in snap['gold']}
    for c in r['claims']:
        if c['claim_id'] in gold:
            assert c['status'] == 'gold'
            assert c['probability'] == float(gold[c['claim_id']])


def test_snapshot_hash_excludes_codes_but_detects_label_change():
    snap, _ = simulate(seed=17, n=40, coverage=.6)
    h = snap['hash']
    snap2 = snapshot([dict(o) for o in snap['opinions']], [dict(g) for g in snap['gold']], n_claims=40)
    assert snap2['hash'] == h
    snap2['opinions'][0]['label'] = 1 - snap2['opinions'][0]['label']
    assert hash_snapshot(snap2) != h
