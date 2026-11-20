import math
import random

from src.config.database import Base, engine, Session
from src.models.entities import Claim, GoldStandard, Opinion, OpinionRevision, User

ANNOTATORS = ['财务-沈', '财务-陆', '财务-周', '财务-程', '财务-顾问A', '财务-顾问B']
# 前四位覆盖共同样本；后两位只在部分票据上发表意见，形成不完整重叠。
COVERAGE = [.62, .58, .55, .5, .3, .26]
SENS = [.86, .82, .78, .74, .8, .7]
SPEC = [.88, .84, .8, .86, .82, .76]


def seed():
    from src.utils.security import hash_password
    Base.metadata.create_all(engine)
    rng = random.Random(3106)
    with Session() as s:
        for role in ('admin', 'analyst', 'viewer'):
            email = role + '@demo.local'
            if not s.query(User).filter_by(email=email).first():
                s.add(User(email=email, role=role, password=hash_password('Demo123!')))
        created = []
        for i in range(300):
            code = f'HX-1115-{i + 1:03}'
            distance = round(rng.uniform(5, 800), 1)
            holiday = rng.random() < .25
            amount = round(distance * rng.uniform(1.2, 5.5) + rng.uniform(50, 900), 2)
            risk = -4.7 + amount / 700 + 1.1 * holiday + amount / (distance + 1) / 8
            truth = int(rng.random() < 1 / (1 + math.exp(-risk))) if i < 260 else None
            label = truth
            # 历史标签沿用一条有偏的人工结果：多数服从真值，但个别票据被多人共同误判。
            if truth is not None:
                noise = rng.random()
                if noise < .12:
                    label = 1 - truth  # 历史标签被共同误判
            row = s.query(Claim).filter_by(code=code).first()
            if not row:
                row = Claim(code=code, amount=amount, distance=distance, holiday=holiday,
                            label=label, note='历史人工复核标签；仅用于模拟训练，不连接真实报销')
                s.add(row)
                s.flush()
            created.append((row, truth))
        s.commit()
        # 为前 240 笔历史样本生成不完整重叠的独立意见
        for idx, (claim, truth) in enumerate(created):
            if truth is None or idx >= 240:
                continue
            for j, annotator in enumerate(ANNOTATORS):
                if rng.random() >= COVERAGE[j]:
                    continue
                if s.query(Opinion).filter_by(claim_id=claim.id, annotator=annotator).first():
                    continue
                # 财务-程 每 20 笔出现方向性偏差；财务-顾问B 整体把正常/异常颠倒录入，
                # 用于演示反向标注者被识别而不是被多数票淹没。
                p_correct = SENS[j] if truth == 1 else SPEC[j]
                if idx % 20 == 0 and j == 3:
                    p_correct = .18
                if annotator == '财务-顾问B':
                    p_correct = 1 - p_correct
                y = truth if rng.random() < p_correct else 1 - truth
                now = '2026-11-15T09:00:00+00:00'
                op = Opinion(claim_id=claim.id, annotator=annotator, label=int(y), note='',
                             revision=1, entered_by='seed', created=now, updated=now)
                s.add(op)
                s.flush()
                s.add(OpinionRevision(opinion_id=op.id, claim_id=claim.id, annotator=annotator,
                                      label=int(y), note='', revision=1, entered_by='seed', created=now))
        # 少量已确认金标准，优先覆盖不同人群
        gold_targets = [created[i] for i in (3, 47, 96, 141, 188, 233)]
        for claim, truth in gold_targets:
            if truth is None:
                continue
            if not s.query(GoldStandard).filter_by(claim_id=claim.id).first():
                s.add(GoldStandard(claim_id=claim.id, label=int(truth), note='已核对原始票据与审批流',
                                   confirmed_by='admin@demo.local', created='2026-11-15T10:00:00+00:00',
                                   updated='2026-11-15T10:00:00+00:00'))
        s.commit()
        print({'users': s.query(User).count(), 'claims': s.query(Claim).count(),
               'opinions': s.query(Opinion).count(), 'gold': s.query(GoldStandard).count()})


if __name__ == '__main__':
    seed()
