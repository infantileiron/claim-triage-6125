"""标注可靠性校准。

历史人工标签可能把多人共同误判当成真相。这里用 Dawid-Skene 风格的二分类潜变量
模型，把「票据真实状态」当作未知量，用不完整重叠的独立意见与少量金标准一起估计：

    z_i ~ Bernoulli(pi)                 票据真实状态
    P(y_ij = 1 | z_i = 1) = sensitivity_j
    P(y_ij = 0 | z_i = 0) = specificity_j

由此得到每位财务的漏报率 1-sensitivity、误报率 1-specificity，以及每笔票据的异常
后验概率。金标准票据的 z 固定为已确认值，用于锚定方向；多数票只作为对照展示，
不作为真值。数据不足或人群缺少锚点时拒绝估计并列出原因，不做虚假的高置信度输出。
"""
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

SEED = 3106
MIN_OPINIONS = 12        # 批次最少独立意见条数
MIN_ANNOTATORS = 2       # 至少两名财务
MIN_PER_ANNOTATOR = 4    # 每人最少意见条数，低于此只标记不拒批
MIN_SHARED = 2           # 每人至少 2 条重叠或落在金标准上的意见
MIN_GOLD = 2             # 至少 2 笔金标准
MIN_COMPONENT_GOLD = 2   # 每个意见人群至少 2 笔金标准锚点
MAX_ITER = 400
TOL = 1e-7
PRIOR = 1.0              # sensitivity/specificity 的 Beta(1,1) 平滑
BOOTSTRAP = 40
CONFIDENT_WIDTH = .4     # 后验区间宽度不超过此值才算可靠
UNCERTAIN_WIDTH = .7     # 超过此值视为证据不足，不进入软标签训练


class CalibrationError(ValueError):
    """证据不足，拒绝估计。reasons 与 checks 用于页面解释。"""

    def __init__(self, reasons, checks=None):
        self.reasons = list(reasons)
        self.checks = checks or []
        super().__init__('；'.join(self.reasons))


def _now():
    return datetime.now(timezone.utc).isoformat()


def snapshot_core(snapshot):
    """只保留参与估计的数据，用于计算版本哈希（不含创建时间）。"""
    return {
        'opinions': [
            {'claim_id': o['claim_id'], 'annotator': o['annotator'],
             'label': int(o['label']), 'revision': int(o.get('revision', 1))}
            for o in sorted(snapshot['opinions'], key=lambda x: (x['claim_id'], x['annotator']))
        ],
        'gold': [
            {'claim_id': g['claim_id'], 'label': int(g['label'])}
            for g in sorted(snapshot['gold'], key=lambda x: x['claim_id'])
        ],
    }


def hash_snapshot(snapshot):
    payload = json.dumps(snapshot_core(snapshot), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_snapshot(session):
    """在当前事务里冻结一版意见与金标准。批次和重试都只使用这一版。"""
    from src.models.entities import Claim, GoldStandard, Opinion
    claims = {c.id: c for c in session.query(Claim).order_by(Claim.id).all()}
    opinions = [
        {'id': o.id, 'claim_id': o.claim_id, 'claim_code': getattr(claims.get(o.claim_id), 'code', str(o.claim_id)),
         'annotator': o.annotator, 'label': int(o.label), 'note': o.note or '',
         'revision': int(o.revision or 1), 'entered_by': o.entered_by or '', 'updated': o.updated or ''}
        for o in session.query(Opinion).order_by(Opinion.claim_id, Opinion.annotator).all()
    ]
    gold = [
        {'id': g.id, 'claim_id': g.claim_id, 'claim_code': getattr(claims.get(g.claim_id), 'code', str(g.claim_id)),
         'label': int(g.label), 'note': g.note or '', 'confirmed_by': g.confirmed_by or ''}
        for g in session.query(GoldStandard).order_by(GoldStandard.claim_id).all()
    ]
    snapshot = {
        'opinions': opinions,
        'gold': gold,
        'claims': [{'id': c.id, 'code': c.code, 'label': c.label} for c in claims.values()],
        'created': _now(),
    }
    snapshot['hash'] = hash_snapshot(snapshot)
    return snapshot


def _index(snapshot):
    opinions = snapshot['opinions']
    annotators = sorted({o['annotator'] for o in opinions})
    claim_ids = sorted({int(o['claim_id']) for o in opinions})
    codes = {int(c['id']): c['code'] for c in snapshot.get('claims', [])}
    for o in opinions:
        codes.setdefault(int(o['claim_id']), o.get('claim_code', str(o['claim_id'])))
    ai = {a: j for j, a in enumerate(annotators)}
    ci = {c: i for i, c in enumerate(claim_ids)}
    Y = np.full((len(claim_ids), len(annotators)), np.nan)
    for o in opinions:
        Y[ci[int(o['claim_id'])], ai[o['annotator']]] = float(o['label'])
    gold = {int(g['claim_id']): int(g['label']) for g in snapshot['gold']}
    return claim_ids, codes, annotators, Y, gold


def _components(n, Y):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for j in range(Y.shape[1]):
        idx = np.where(~np.isnan(Y[:, j]))[0]
        for x in idx[1:]:
            ra, rb = find(int(idx[0])), find(int(x))
            if ra != rb:
                parent[rb] = ra
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _em(Y, gold_mask, gold_z, pi0, sens0, spec0, max_iter=MAX_ITER, tol=TOL):
    n, m = Y.shape
    obs = ~np.isnan(Y)
    y = np.where(obs, Y, 0.0)
    sens = np.clip(np.asarray(sens0, dtype=float), 1e-4, 1 - 1e-4)
    spec = np.clip(np.asarray(spec0, dtype=float), 1e-4, 1 - 1e-4)
    pi = float(np.clip(pi0, 1e-3, 1 - 1e-3))
    q = np.full(n, .5)
    it = 0
    for it in range(1, max_iter + 1):
        a = (obs * y) @ np.log(sens) + (obs * (1 - y)) @ np.log1p(-sens)
        b = (obs * y) @ np.log1p(-spec) + (obs * (1 - y)) @ np.log(spec)
        diff = np.log(pi) - np.log1p(-pi) + a - b
        qn = 1.0 / (1.0 + np.exp(-np.clip(diff, -60, 60)))
        if gold_mask.any():
            qn[gold_mask] = gold_z[gold_mask]
        change = float(np.max(np.abs(qn - q))) if n else 0.0
        q = qn
        pos = q @ obs
        neg = (1 - q) @ obs
        tp = q @ (obs * y)
        tn = (1 - q) @ (obs * (1 - y))
        sens = np.clip((tp + PRIOR) / (pos + 2 * PRIOR), 1e-4, 1 - 1e-4)
        spec = np.clip((tn + PRIOR) / (neg + 2 * PRIOR), 1e-4, 1 - 1e-4)
        pi = float(np.clip((q.sum() + PRIOR) / (n + 2 * PRIOR), 1e-3, 1 - 1e-3))
        if change < tol:
            break
    # 用最终参数计算观测数据对数似然，用于多初值选优
    a = (obs * y) @ np.log(sens) + (obs * (1 - y)) @ np.log1p(-sens)
    b = (obs * y) @ np.log1p(-spec) + (obs * (1 - y)) @ np.log(spec)
    ll = np.logaddexp(np.log(pi) + a, np.log1p(-pi) + b)
    if gold_mask.any():
        ll = ll.copy()
        ll[gold_mask] = np.where(gold_z[gold_mask] == 1,
                                 (np.log(pi) + a)[gold_mask],
                                 (np.log1p(-pi) + b)[gold_mask])
    return {'q': q, 'sens': sens, 'spec': spec, 'pi': pi,
            'll': float(ll.sum()), 'iters': it}


def _gold_init(Y, gold_mask, gold_z):
    m = Y.shape[1]
    sens = np.full(m, .75)
    spec = np.full(m, .75)
    obs = ~np.isnan(Y)
    y = np.where(obs, Y, 0.0)
    if gold_mask.any():
        z = gold_z[gold_mask]
        for j in range(m):
            observed = obs[gold_mask, j]
            yj = y[gold_mask, j]
            pos = observed & (z == 1)
            neg = observed & (z == 0)
            tp = np.sum(yj[pos] == 1)
            tn = np.sum(yj[neg] == 0)
            sens[j] = (tp + PRIOR) / (pos.sum() + 2 * PRIOR) if pos.any() else .75
            spec[j] = (tn + PRIOR) / (neg.sum() + 2 * PRIOR) if neg.any() else .75
        pi0 = float(np.clip(z.mean(), .05, .95))
    else:
        pi0 = .35
    return pi0, sens, spec


def _fit_best(Y, gold_mask, gold_z):
    inits = [_gold_init(Y, gold_mask, gold_z)]
    obs = ~np.isnan(Y)
    y = np.where(obs, Y, 0.0)
    vote = y.sum(0) / np.maximum(obs.sum(0), 1)
    pi_vote = float(np.clip(y[obs].mean() if obs.any() else .35, .05, .95))
    for level in (.75, .85, .6):
        inits.append((pi_vote, np.full(Y.shape[1], level), np.full(Y.shape[1], level)))
    # 反向标注的可能性：允许初始灵敏度低于 0.5，由似然选优
    inits.append((pi_vote, np.full(Y.shape[1], .3), np.full(Y.shape[1], .7)))
    best = None
    seen = set()
    for pi0, sens0, spec0 in inits:
        key = (round(float(pi0), 3), tuple(np.round(sens0, 3)), tuple(np.round(spec0, 3)))
        if key in seen:
            continue
        seen.add(key)
        fit = _em(Y, gold_mask, gold_z, pi0, sens0, spec0)
        if best is None or fit['ll'] > best['ll']:
            best = fit
    return best


def _ci(samples):
    samples = [s for s in samples if s is not None and np.isfinite(s)]
    if len(samples) < 6:
        return None
    return [float(np.percentile(samples, 5)), float(np.percentile(samples, 95))]


def _check_evidence(claim_ids, annotators, Y, gold):
    n_opinions = int(np.sum(~np.isnan(Y)))
    observed_per_annotator = (~np.isnan(Y)).sum(0)
    well_observed = int(np.sum(observed_per_annotator >= MIN_PER_ANNOTATOR))
    gold_with_opinion = [g for g in gold if g in set(claim_ids)]
    comps = _components(len(claim_ids), Y)
    claims_by_index = claim_ids
    groups = []
    for comp in comps:
        comp_claims = {claims_by_index[i] for i in comp}
        comp_annotators = sorted({annotators[j] for j in range(Y.shape[1])
                                  if any(not np.isnan(Y[i, j]) for i in comp)})
        anchors = sorted(comp_claims & set(gold))
        groups.append({'claims': sorted(comp_claims), 'annotators': comp_annotators,
                       'gold_anchors': anchors, 'opinions': int(np.sum(~np.isnan(Y[comp]))) })
    anchored = [g for g in groups if len(g['gold_anchors']) >= MIN_COMPONENT_GOLD]
    checks = [
        {'key': 'opinions', 'ok': n_opinions >= MIN_OPINIONS,
         'detail': f'独立意见 {n_opinions} 条（下限 {MIN_OPINIONS}）'},
        {'key': 'annotators', 'ok': len(annotators) >= MIN_ANNOTATORS,
         'detail': f'标注者 {len(annotators)} 人（下限 {MIN_ANNOTATORS}）'},
        {'key': 'observed_annotators', 'ok': well_observed >= MIN_ANNOTATORS,
         'detail': f'意见不少于 {MIN_PER_ANNOTATOR} 条的标注者 {well_observed} 人（下限 {MIN_ANNOTATORS}）'},
        {'key': 'gold', 'ok': len(gold_with_opinion) >= MIN_GOLD,
         'detail': f'带意见的金标准 {len(gold_with_opinion)} 笔（下限 {MIN_GOLD}）'},
        {'key': 'anchored_groups', 'ok': len(anchored) >= 1,
         'detail': f'{len(groups)} 个互不重叠的意见人群，其中 {len(anchored)} 个含至少 {MIN_COMPONENT_GOLD} 笔金标准锚点'},
        {'key': 'coverage', 'ok': True,
         'detail': f'可估计意见 {sum(g["opinions"] for g in anchored)} / {n_opinions} 条'},
    ]
    return checks, groups, anchored


def estimate(snapshot):
    """估计标注者漏报/误报率与逐票异常概率。证据不足抛 CalibrationError。"""
    claim_ids, codes, annotators, Y, gold = _index(snapshot)
    if not claim_ids or not annotators:
        raise CalibrationError(['没有可用的独立意见'], [
            {'key': 'opinions', 'ok': False, 'detail': '独立意见 0 条'},
        ])
    checks, groups, anchored = _check_evidence(claim_ids, annotators, Y, gold)
    failed = [c for c in checks if not c['ok']]
    if failed:
        raise CalibrationError([c['detail'] for c in failed], checks)

    ci_index = {cid: i for i, cid in enumerate(claim_ids)}
    included_ids = sorted({cid for g in anchored for cid in g['claims']})
    included = [ci_index[cid] for cid in included_ids]
    included_pos = {claim_ids[i]: k for k, i in enumerate(included)}
    Yin = Y[included]
    gm = np.array([claim_ids[i] in gold for i in included], dtype=bool)
    gz = np.array([float(gold[claim_ids[i]]) if claim_ids[i] in gold else 0.0 for i in included])
    fit = _fit_best(Yin, gm, gz)

    # 自助法给出区间，避免用点估计冒充高置信度
    rng = np.random.default_rng(SEED + 7)
    n = len(included)
    q_samples = defaultdict(list)
    sens_samples = defaultdict(list)
    spec_samples = defaultdict(list)
    valid = 0
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        gmb = gm[idx]
        if int(gmb.sum()) < MIN_COMPONENT_GOLD:
            continue
        try:
            bfit = _em(Yin[idx], gmb, gz[idx], fit['pi'], fit['sens'], fit['spec'],
                       max_iter=120, tol=1e-6)
        except Exception:
            continue
        valid += 1
        for local, original in enumerate(idx):
            q_samples[int(original)].append(float(bfit['q'][local]))
        for j in range(Y.shape[1]):
            sens_samples[j].append(float(bfit['sens'][j]))
            spec_samples[j].append(float(bfit['spec'][j]))

    # 每位财务的意见数、重叠与金标准命中；统计基于全部意见，参数只对含锚点人群有效
    full_obs = ~np.isnan(Y)
    full_y = np.where(full_obs, Y, 0.0)
    per_annotator = []
    for j, name in enumerate(annotators):
        observed_all = full_obs[:, j]
        n_j = int(observed_all.sum())
        shared = 0
        for i in np.where(observed_all)[0]:
            others = np.sum(full_obs[i]) - 1
            if others > 0 or claim_ids[i] in gold:
                shared += 1
        gold_hits = int(np.sum(observed_all & np.array([cid in gold for cid in claim_ids])))
        gold_correct = int(np.sum(observed_all & np.array([cid in gold for cid in claim_ids])
                               & (full_y[:, j] == np.array([gold.get(cid, 0) for cid in claim_ids]))))
        in_anchor = any(name in g['annotators'] for g in anchored)
        notes = []
        sens = spec = acc = None
        s_ci = p_ci = None
        if not in_anchor:
            status = 'unanchored'
            notes.append('所属意见人群缺少金标准锚点，无法校准其漏报/误报率')
        else:
            sens = float(fit['sens'][j])
            spec = float(fit['spec'][j])
            acc = float(fit['pi'] * sens + (1 - fit['pi']) * spec)
            s_ci = _ci(sens_samples[j])
            p_ci = _ci(spec_samples[j])
            status = 'reliable'
            if n_j < MIN_PER_ANNOTATOR:
                status = 'insufficient'
                notes.append(f'意见仅 {n_j} 条，少于 {MIN_PER_ANNOTATOR} 条')
            elif shared < MIN_SHARED:
                status = 'insufficient'
                notes.append(f'与他人或金标准重叠仅 {shared} 条')
            elif sens < .5 and spec < .5:
                status = 'reversed'
                notes.append('与金标准方向相反：可能把正常/异常颠倒录入')
            elif acc < .5:
                status = 'reversed'
                notes.append('综合准确率低于随机，方向可疑')
            elif (s_ci and s_ci[0] < .5) or (p_ci and p_ci[0] < .5):
                status = 'unstable'
                notes.append('区间跨过随机水平，质量尚不稳定')
            if sens < .6:
                notes.append('漏报偏高')
            if spec < .6:
                notes.append('误报偏高')
        per_annotator.append({
            'annotator': name, 'opinions': n_j, 'overlap': shared,
            'gold_opinions': gold_hits, 'gold_correct': gold_correct,
            'sensitivity': sens,
            'false_negative_rate': (1 - sens) if sens is not None else None,
            'specificity': spec,
            'false_positive_rate': (1 - spec) if spec is not None else None,
            'accuracy': acc,
            'sensitivity_ci': s_ci, 'specificity_ci': p_ci,
            'status': status, 'note': '；'.join(notes),
        })

    # 逐票概率；多数票只做对照
    per_claim = []
    for i, cid in enumerate(claim_ids):
        if cid not in included_pos:
            comp = next(g for g in groups if cid in g['claims'])
            reason = ('所属人群缺少至少 %d 笔金标准锚点，未给出概率' % MIN_COMPONENT_GOLD)
            per_claim.append({'claim_id': cid, 'claim_code': codes.get(cid, str(cid)),
                              'opinions': int(np.sum(~np.isnan(Y[i]))),
                              'annotators': [annotators[j] for j in range(Y.shape[1]) if not np.isnan(Y[i, j])],
                              'majority_label': _majority(Y[i]), 'probability': None,
                              'low': None, 'high': None, 'ci_width': None, 'label': None,
                              'status': 'insufficient', 'reason': reason, 'gold': cid in gold,
                              'group_gold_anchors': len(comp['gold_anchors'])})
            continue
        local = included_pos[cid]
        p = float(fit['q'][local])
        if cid in gold:
            per_claim.append({'claim_id': cid, 'claim_code': codes.get(cid, str(cid)),
                              'opinions': int(np.sum(~np.isnan(Y[i]))),
                              'annotators': [annotators[j] for j in range(Y.shape[1]) if not np.isnan(Y[i, j])],
                              'majority_label': _majority(Y[i]), 'probability': float(gold[cid]),
                              'low': float(gold[cid]), 'high': float(gold[cid]), 'ci_width': 0.0,
                              'label': int(gold[cid]), 'status': 'gold',
                              'reason': '金标准已确认，不参与留出评估', 'gold': True, 'group_gold_anchors': None})
            continue
        ci = _ci(q_samples[local])
        low, high = (ci if ci else (None, None))
        width = (high - low) if ci else None
        if ci is None:
            status, reason = 'uncertain', '自助法有效样本不足，未给出可靠区间'
            shown_probability, shown_label = p, None
        elif width > UNCERTAIN_WIDTH:
            # 区间过宽时不展示点估计，避免把不确定的结果当成可靠标签
            status, reason = 'insufficient', '后验区间过宽，证据不足以给出可靠标签'
            shown_probability, shown_label = None, None
        elif width > CONFIDENT_WIDTH:
            status, reason = 'uncertain', '后验区间较宽，标签仅供参考'
            shown_probability, shown_label = p, int(p >= .5)
        else:
            status, reason = 'estimated', '区间较窄，可用于软标签训练'
            shown_probability, shown_label = p, int(p >= .5)
        per_claim.append({'claim_id': cid, 'claim_code': codes.get(cid, str(cid)),
                          'opinions': int(np.sum(~np.isnan(Y[i]))),
                          'annotators': [annotators[j] for j in range(Y.shape[1]) if not np.isnan(Y[i, j])],
                          'majority_label': _majority(Y[i]), 'probability': shown_probability,
                          'posterior': p,
                          'low': low, 'high': high, 'ci_width': width,
                          'label': shown_label, 'status': status, 'reason': reason,
                          'gold': False, 'group_gold_anchors': None})

    excluded = [{'claims': g['claims'], 'annotators': g['annotators'],
                 'gold_anchors': len(g['gold_anchors']), 'opinions': g['opinions']}
                for g in groups if len(g['gold_anchors']) < MIN_COMPONENT_GOLD]
    majority_diff = sum(1 for c in per_claim
                        if c['majority_label'] is not None and c['probability'] is not None
                        and c['status'] != 'gold' and c['majority_label'] != int(c['probability'] >= .5))
    return {
        'method': 'dawid_skene_beta_em',
        'seed': SEED,
        'iterations': int(fit['iters']),
        'log_likelihood': float(fit['ll']),
        'prevalence': float(fit['pi']),
        'bootstrap_valid': valid,
        'evidence': {
            'opinions': int(np.sum(~np.isnan(Y))),
            'annotators': len(annotators),
            'claims': len(claim_ids),
            'gold': len([g for g in gold if g in set(claim_ids)]),
            'gold_total': len(gold),
            'included_opinions': int(np.sum(~np.isnan(Yin))),
            'included_claims': len(included),
            'groups': [{'claims': len(g['claims']), 'annotators': g['annotators'],
                        'gold_anchors': len(g['gold_anchors']), 'opinions': g['opinions']} for g in groups],
        },
        'checks': checks,
        'annotators': per_annotator,
        'claims': per_claim,
        'excluded_groups': excluded,
        'majority_disagreements': majority_diff,
        'warning': ('概率来自潜变量模型，不是多数票，也不是人工确认。金标准只用于锚定方向与评估标注者，'
                    '不参与留出评估。区间较宽或人群无锚点时不要当成高置信度标签。'
                    '反向标注者应单独复核其意见，而不是简单多数否决。'),
    }


def _majority(row):
    vals = row[~np.isnan(row)]
    if len(vals) == 0:
        return None
    frac = float(vals.mean())
    if frac > .5:
        return 1
    if frac < .5:
        return 0
    return None
