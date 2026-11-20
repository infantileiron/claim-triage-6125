import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 3106
FEATURES = ['金额', '里程', '节假日', '每公里金额']
SOFT_STATUSES = ('estimated',)


def features(row):
    return [row['amount'], row['distance'], int(row['holiday']), row['amount'] / (row['distance'] + 1)]


def metrics(rows, threshold):
    tp = sum(r['label'] == 1 and r['score'] >= threshold for r in rows)
    fp = sum(r['label'] == 0 and r['score'] >= threshold for r in rows)
    fn = sum(r['label'] == 1 and r['score'] < threshold for r in rows)
    tn = sum(r['label'] == 0 and r['score'] < threshold for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0,
                accuracy=(tp + tn) / len(rows) if rows else 0)


def usable_soft(calibration):
    """校准结果中可用于训练的软标签：claim_id -> 条目。排除金标准与宽区间。"""
    if not calibration or calibration.get('state') != 'done' or not calibration.get('result'):
        return {}
    gold_ids = {int(g['claim_id']) for g in (calibration.get('snapshot') or {}).get('gold', [])}
    out = {}
    for c in calibration['result'].get('claims', []):
        if c.get('status') in SOFT_STATUSES and c.get('probability') is not None:
            cid = int(c['claim_id'])
            if cid not in gold_ids:
                out[cid] = c
    return out


def train(rows, threshold, calibration=None):
    """逻辑回归训练；可选把校准后的软标签并入训练集。

    先按原始人工标签分层划分留出集（金标准排除在外），再在训练集里用软标签替换
    对应样本的目标；留出集始终保留原始人工标签，既不使用金标准也不使用软标签，
    避免用估计值评估估计。校准批次是冻结快照，事后意见更正不会改变本版本。
    """
    soft = usable_soft(calibration)
    gold_labels = {}
    if calibration and calibration.get('snapshot'):
        gold_labels = {int(g['claim_id']): int(g['label'])
                       for g in calibration['snapshot'].get('gold', [])}

    hard = [r for r in rows if r['label'] is not None]
    holdout_pool = [r for r in hard if int(r['id']) not in gold_labels]
    counts = {str(v): sum(int(r['label']) == v for r in holdout_pool) for v in (0, 1)}
    if len(holdout_pool) < 40 or min(counts.values()) < 10:
        raise ValueError('至少40个已标注样本且留出集正常/异常各至少10个；请先补充人工标签')

    train_rows, test_rows = train_test_split(
        holdout_pool, test_size=.25, random_state=SEED,
        stratify=[int(r['label']) for r in holdout_pool])
    gold_rows = [r for r in rows if int(r['id']) in gold_labels]

    X, y, w, used_soft = [], [], [], []
    for r in train_rows:
        cid = int(r['id'])
        if cid in soft:
            c = soft[cid]
            p = float(c['probability'])
            width = c.get('ci_width')
            narrow = 1.0 if width is None else max(.15, min(1.0, 1 - width / .7))
            certainty = max(.05, abs(2 * p - 1)) * narrow
            # 软标签用两条加权样本表达，等价于加权交叉熵
            X.append(features(r)); y.append(1); w.append(certainty * p)
            X.append(features(r)); y.append(0); w.append(certainty * (1 - p))
            used_soft.append({'claim_id': cid, 'code': r['code'], 'probability': p,
                              'status': c.get('status'), 'ci_width': width, 'weight': certainty})
        else:
            X.append(features(r)); y.append(int(r['label'])); w.append(1.0)
    for r in gold_rows:
        X.append(features(r)); y.append(gold_labels[int(r['id'])]); w.append(1.0)

    model = Pipeline([('scale', StandardScaler()),
                      ('logit', LogisticRegression(random_state=SEED, max_iter=1000, C=1.0))])
    model.fit(X, y, logit__sample_weight=w)
    scored = [{'id': r['id'], 'code': r['code'], 'label': int(r['label']),
               'score': float(p)} for r, p in zip(
        test_rows, model.predict_proba([features(r) for r in test_rows])[:, 1])]
    pending = [r for r in rows if r['label'] is None]
    predictions = [{'id': r['id'], 'code': r['code'], 'score': float(p)} for r, p in zip(
        pending, model.predict_proba([features(r) for r in pending])[:, 1])] if pending else []
    holdout_soft_overlap = sum(1 for r in test_rows if int(r['id']) in soft)
    holdout_gold_overlap = sum(1 for r in test_rows if int(r['id']) in gold_labels)
    calibration_note = ''
    if calibration:
        calibration_note = (f"软标签来自校准 #{calibration.get('id')}（{len(used_soft)} 笔进入训练集），"
                            f"金标准 {len(gold_rows)} 笔仅参与训练、不进入留出评估。")
    return dict(
        train_n=len(train_rows), test_n=len(test_rows),
        label_counts=counts, metrics=metrics(scored, threshold),
        threshold_comparison=[metrics(scored, t) for t in (.3, .5, .7)],
        holdout=scored, pending=predictions,
        coefficients=[{'feature': f, 'coefficient': float(c)}
                      for f, c in zip(FEATURES, model.named_steps['logit'].coef_[0])],
        train_codes=[r['code'] for r in train_rows], test_codes=[r['code'] for r in test_rows],
        scaler_mean=model.named_steps['scale'].mean_.tolist(), seed=SEED,
        checksum=hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        calibration_usage={'id': calibration.get('id') if calibration else None,
                     'snapshot_hash': calibration.get('snapshot_hash') if calibration else None,
                     'soft_labels': used_soft,
                     'gold_train_only': [int(r['id']) for r in gold_rows],
                     'holdout_soft_overlap': holdout_soft_overlap,
                     'holdout_gold_overlap': holdout_gold_overlap,
                     'note': calibration_note},
        warning='评分只用于排定人工复核顺序。标签不进入特征；标准化只在训练集拟合。'
                '留出集为演示评估集，反复调阈值后不再是最终独立验收集。'
                '金标准不参与留出评估；软标签是模型估计，不等于人工确认。'
                '留出集保留原始人工标签，可能与校准意见重叠，指标仅作演示。'
                '系数为标准化关联，不代表因果。')
