import json
import os

import redis
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import IntegrityError

from src.config.database import db
from src.controllers.business import ensure_calibration, ensure_training
from src.middlewares.auth import admin, current, editor
from src.models.entities import (Audit, Calibration, Claim, GoldStandard, Opinion,
                                 OpinionRevision, Review, Training, User)
from src.schemas.business import (CalibrationInput, ClaimInput, GoldInput, Login,
                                  OpinionInput, ReviewInput, TrainingInput)
from src.services.business import audit, dump
from src.services.calibration import build_snapshot, hash_snapshot
from src.utils.security import token, verify

router = APIRouter()


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _calibration_dump(row, include_snapshot=False):
    out = dump(row)
    if not include_snapshot:
        out.pop('snapshot', None)
    return out


@router.post('/auth/login')
def login(data: Login, s=Depends(db)):
    u = s.query(User).filter_by(email=data.email).first()
    if not u or not verify(data.password, u.password):
        raise HTTPException(401, '账号或密码错误')
    return {'token': token(u), 'user': dump(u)}


@router.get('/me')
def me(u=Depends(current)):
    return dump(u)


@router.get('/claims')
def claims(u=Depends(current), s=Depends(db)):
    return [dump(r) for r in s.query(Claim).order_by(Claim.id).all()]


@router.post('/claims')
def create_claim(data: ClaimInput, u=Depends(editor), s=Depends(db)):
    r = Claim(**data.model_dump())
    s.add(r)
    try:
        s.commit()
    except IntegrityError:
        s.rollback()
        raise HTTPException(409, '报销编号重复')
    audit(s, u, '录入报销 ' + r.code)
    return dump(r)


@router.put('/claims/{i}')
def edit_claim(i: int, data: ClaimInput, u=Depends(editor), s=Depends(db)):
    r = s.get(Claim, i)
    if not r:
        raise HTTPException(404, '报销样本不存在')
    for k, v in data.model_dump().items():
        setattr(r, k, v)
    try:
        s.commit()
    except IntegrityError:
        s.rollback()
        raise HTTPException(409, '报销编号重复')
    audit(s, u, '修订报销 ' + str(i))
    return dump(r)


# ---------- 独立意见 ----------

@router.get('/opinions')
def opinions(u=Depends(current), s=Depends(db)):
    return [dump(r) for r in s.query(Opinion).order_by(Opinion.claim_id, Opinion.annotator).all()]


@router.get('/claims/{i}/opinions')
def claim_opinions(i: int, u=Depends(current), s=Depends(db)):
    if not s.get(Claim, i):
        raise HTTPException(404, '报销样本不存在')
    rows = s.query(Opinion).filter_by(claim_id=i).order_by(Opinion.annotator).all()
    revisions = s.query(OpinionRevision).filter_by(claim_id=i).order_by(OpinionRevision.id.desc()).all()
    return {'opinions': [dump(r) for r in rows], 'revisions': [dump(r) for r in revisions]}


@router.post('/opinions')
def put_opinion(data: OpinionInput, u=Depends(editor), s=Depends(db)):
    """同一(票据, 标注者)只保留一版意见；每次更正追加一条修订记录。"""
    claim = s.get(Claim, data.claim_id)
    if not claim:
        raise HTTPException(404, '报销样本不存在')
    annotator = data.annotator.strip()
    if not annotator:
        raise HTTPException(422, '标注者不能为空')
    now = _now()
    row = s.query(Opinion).filter_by(claim_id=data.claim_id, annotator=annotator).first()
    if row:
        if int(row.label) == int(data.label) and (row.note or '') == (data.note or ''):
            return {'opinion': dump(row), 'revision': None, 'changed': False}
        row.label = int(data.label)
        row.note = data.note
        row.revision = int(row.revision or 1) + 1
        row.entered_by = u.email
        row.updated = now
        rev = row.revision
    else:
        row = Opinion(claim_id=data.claim_id, annotator=annotator, label=int(data.label),
                      note=data.note, revision=1, entered_by=u.email, created=now, updated=now)
        s.add(row)
        s.flush()
        rev = 1
    s.add(OpinionRevision(opinion_id=row.id, claim_id=row.claim_id, annotator=annotator,
                          label=row.label, note=row.note, revision=rev,
                          entered_by=u.email, created=now))
    s.commit()
    audit(s, u, f'录入意见 {claim.code} / {annotator} = {row.label} (r{rev})')
    return {'opinion': dump(row), 'revision': rev, 'changed': True}


# ---------- 金标准 ----------

@router.get('/gold')
def gold(u=Depends(current), s=Depends(db)):
    return [dump(r) for r in s.query(GoldStandard).order_by(GoldStandard.claim_id).all()]


@router.post('/gold')
def put_gold(data: GoldInput, u=Depends(admin), s=Depends(db)):
    """金标准由负责人确认，只用于锚定方向与评估标注者，不参与留出评估。"""
    claim = s.get(Claim, data.claim_id)
    if not claim:
        raise HTTPException(404, '报销样本不存在')
    now = _now()
    row = s.query(GoldStandard).filter_by(claim_id=data.claim_id).first()
    if row:
        row.label = int(data.label)
        row.note = data.note
        row.confirmed_by = u.email
        row.updated = now
    else:
        row = GoldStandard(claim_id=data.claim_id, label=int(data.label), note=data.note,
                           confirmed_by=u.email, created=now, updated=now)
        s.add(row)
    s.commit()
    audit(s, u, f'确认金标准 {claim.code} = {row.label}')
    return dump(row)


# ---------- 校准批次 ----------

@router.get('/calibrations')
def calibrations(u=Depends(current), s=Depends(db)):
    return [_calibration_dump(r) for r in s.query(Calibration).order_by(Calibration.id.desc()).all()]


@router.get('/calibrations/{i}')
def calibration_detail(i: int, u=Depends(current), s=Depends(db)):
    row = ensure_calibration(s, i)
    return _calibration_dump(row, include_snapshot=True)


@router.post('/calibrations')
def create_calibration(data: CalibrationInput, u=Depends(editor), s=Depends(db)):
    snap = build_snapshot(s)
    now = _now()
    row = Calibration(name=data.name, state='queued', snapshot=snap,
                      snapshot_hash=snap['hash'], attempts=0, created=now, updated=now)
    s.add(row)
    s.commit()
    try:
        redis.Redis.from_url(os.environ['REDIS_URL']).rpush('claim:calibrations', row.id)
    except redis.RedisError:
        row.state = 'failed'
        row.error = '队列暂不可用，请稍后重试；意见与金标准已冻结在本批次'
        row.updated = _now()
        s.commit()
        raise HTTPException(503, row.error)
    audit(s, u, f'提交标注校准 {row.id}')
    return _calibration_dump(row)


@router.post('/calibrations/{i}/retry')
def retry_calibration(i: int, u=Depends(editor), s=Depends(db)):
    """失败批次沿用创建时冻结的同一版意见，只重试估计，不重复累计意见。"""
    row = ensure_calibration(s, i)
    if row.state in ('queued', 'running'):
        raise HTTPException(409, '批次仍在等待或估计中，不能重复提交')
    if row.state != 'failed':
        raise HTTPException(409, '仅失败批次可以重试')
    row.state = 'queued'
    row.error = ''
    row.attempts = int(row.attempts or 0) + 1
    row.updated = _now()
    s.commit()
    try:
        redis.Redis.from_url(os.environ['REDIS_URL']).rpush('claim:calibrations', row.id)
    except redis.RedisError:
        row.state = 'failed'
        row.error = '队列暂不可用，请稍后重试；意见与金标准仍沿用本批次冻结版本'
        row.updated = _now()
        s.commit()
        raise HTTPException(503, row.error)
    audit(s, u, f'重试标注校准 {row.id}')
    return _calibration_dump(row)


@router.get('/calibrations/{i}/export')
def export_calibration(i: int, u=Depends(current), s=Depends(db)):
    row = ensure_calibration(s, i)
    if row.state != 'done':
        raise HTTPException(409, '仅成功校准可导出')
    payload = _calibration_dump(row, include_snapshot=True)
    return Response(json.dumps(payload, ensure_ascii=False, indent=2), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename="annotation-calibration-{i}.json"'})


# ---------- 训练版本 ----------

def _calibration_payload(s, calibration_id):
    row = ensure_calibration(s, calibration_id)
    if row.state != 'done' or not row.result:
        raise HTTPException(409, '只有成功完成的校准才能用于训练')
    # 冻结完整一版：后续意见更正或重新校准都不改变本训练版本
    return {'id': row.id, 'state': 'done', 'result': row.result,
            'snapshot': row.snapshot, 'snapshot_hash': row.snapshot_hash}


@router.get('/trainings')
def trainings(u=Depends(current), s=Depends(db)):
    return [dump(r) for r in s.query(Training).order_by(Training.id.desc()).all()]


@router.post('/trainings')
def train(data: TrainingInput, u=Depends(editor), s=Depends(db)):
    rows = [dump(r) for r in s.query(Claim).order_by(Claim.id).all()]
    payload = data.model_dump()
    calibration = None
    if data.calibration_id is not None:
        calibration = _calibration_payload(s, data.calibration_id)
    frozen = {'snapshot': rows, 'calibration': calibration}
    r = Training(name=data.name, threshold=data.threshold, result=frozen)
    s.add(r)
    s.commit()
    try:
        redis.Redis.from_url(os.environ['REDIS_URL']).rpush('claim:trainings', r.id)
    except redis.RedisError:
        r.state = 'failed'
        r.error = '队列暂不可用，请稍后重新提交'
        s.commit()
        raise HTTPException(503, r.error)
    audit(s, u, f'提交训练 {r.id}' + (f'（校准 #{data.calibration_id}）' if data.calibration_id else '（原始标签）'))
    return dump(r)


@router.get('/trainings/{i}')
def detail(i: int, u=Depends(current), s=Depends(db)):
    return dump(ensure_training(s, i))


@router.get('/trainings/{i}/export')
def export(i: int, u=Depends(current), s=Depends(db)):
    r = ensure_training(s, i)
    if r.state != 'done':
        raise HTTPException(409, '仅成功训练可导出')
    return Response(json.dumps(dump(r), ensure_ascii=False, indent=2), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename="claim-triage-{i}.json"'})


@router.post('/trainings/{i}/reviews')
def generate(i: int, u=Depends(editor), s=Depends(db)):
    r = ensure_training(s, i)
    if r.state != 'done':
        raise HTTPException(409, '训练完成后才能生成复核任务')
    created = 0
    for p in r.result['pending']:
        if p['score'] >= r.threshold and not s.query(Review).filter_by(training_id=i, claim_id=p['id']).first():
            s.add(Review(training_id=i, claim_id=p['id'], score=p['score']))
            created += 1
    s.commit()
    audit(s, u, '生成风险复核 ' + str(i))
    return {'created': created}


@router.get('/reviews')
def reviews(u=Depends(current), s=Depends(db)):
    return [dump(r) for r in s.query(Review).order_by(Review.id.desc()).all()]


@router.put('/reviews/{i}')
def review(i: int, data: ReviewInput, u=Depends(editor), s=Depends(db)):
    r = s.get(Review, i)
    if not r:
        raise HTTPException(404, '复核任务不存在')
    if r.status == 'approved':
        raise HTTPException(409, '已签核复核不可修改')
    r.decision = data.decision
    r.note = data.note
    s.commit()
    audit(s, u, '人工复核 ' + str(i))
    return dump(r)


@router.post('/reviews/{i}/approve')
def approve(i: int, u=Depends(admin), s=Depends(db)):
    r = s.get(Review, i)
    if not r:
        raise HTTPException(404, '复核任务不存在')
    if r.decision == 'pending' or r.status != 'draft':
        raise HTTPException(409, '先填写人工判断，且只签核一次')
    r.status = 'approved'
    s.commit()
    audit(s, u, '签核复核 ' + str(i))
    return dump(r)


@router.get('/audits')
def audits(u=Depends(admin), s=Depends(db)):
    return [dump(r) for r in s.query(Audit).order_by(Audit.id.desc()).limit(100)]
