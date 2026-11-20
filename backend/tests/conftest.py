import os
import tempfile

import pytest

# 必须在导入 src.* 之前设置环境变量：数据库引擎在模块导入时创建。
_TMP = tempfile.mkdtemp(prefix='claim-calibration-')
os.environ.setdefault('DATABASE_URL', 'sqlite:///' + os.path.join(_TMP, 'test.db'))
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/0')
os.environ.setdefault('JWT_SECRET', 'test-secret')
os.environ.setdefault('SEED_DEMO', 'false')


@pytest.fixture()
def app_ctx():
    from src.main import app  # 导入即注册全部 ORM 模型
    from src.config.database import Base, engine
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return app


class FakeRedis:
    def __init__(self):
        self.queue = []

    def rpush(self, key, value):
        self.queue.append((key, value))

    def blpop(self, key, timeout=0):
        for i, (k, v) in enumerate(self.queue):
            if k == key:
                return self.queue.pop(i)
        return None


@pytest.fixture()
def client(app_ctx, monkeypatch):
    import src.routes.api as api
    fake = FakeRedis()
    monkeypatch.setattr(api.redis.Redis, 'from_url', lambda url: fake)
    from fastapi.testclient import TestClient
    c = TestClient(app_ctx)
    c.fake_redis = fake
    yield c


def login(client, email):
    r = client.post('/auth/login', json={'email': email, 'password': 'Demo123!'})
    assert r.status_code == 200, r.text
    return {'Authorization': 'Bearer ' + r.json()['token']}


@pytest.fixture()
def users(client):
    from src.config.database import Session
    from src.models.entities import User
    from src.utils.security import hash_password
    with Session() as s:
        for role in ('admin', 'analyst', 'viewer'):
            if not s.query(User).filter_by(email=role + '@demo.local').first():
                s.add(User(email=role + '@demo.local', role=role, password=hash_password('Demo123!')))
        s.commit()
    return {'admin': login(client, 'admin@demo.local'),
            'analyst': login(client, 'analyst@demo.local'),
            'viewer': login(client, 'viewer@demo.local')}


def seed_claims(client, headers, n=60):
    ids = []
    for i in range(n):
        r = client.post('/claims', json={'code': f'T-{i + 1:03}', 'amount': 100 + i * 10,
                                         'distance': 40 + i, 'holiday': i % 3 == 0,
                                         'label': 1 if i % 2 else 0, 'note': ''}, headers=headers)
        assert r.status_code == 200, r.text
        ids.append(r.json()['id'])
    return ids


def add_opinion(client, headers, claim_id, annotator, label):
    r = client.post('/opinions', json={'claim_id': claim_id, 'annotator': annotator,
                                       'label': label, 'note': ''}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def add_gold(client, headers, claim_id, label):
    r = client.post('/gold', json={'claim_id': claim_id, 'label': label,
                                   'note': '已核对原始票据'}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def run_calibration(cid):
    """同步执行 worker 中的校准逻辑。"""
    from src.config.database import Session
    from src.models.entities import Calibration
    from src.services.calibration import CalibrationError, estimate
    with Session() as s:
        row = s.get(Calibration, cid)
        assert row.state == 'queued'
        row.state = 'running'
        s.commit()
        try:
            row.result = estimate(row.snapshot)
            row.state = 'done'
        except CalibrationError as e:
            row.state = 'failed'
            row.error = str(e)
            row.result = {'refused': True, 'reasons': e.reasons, 'checks': e.checks}
        s.commit()
        return row.state


def run_training(tid):
    from src.config.database import Session
    from src.models.entities import Training
    from src.services.analysis import train
    with Session() as s:
        row = s.get(Training, tid)
        frozen = row.result or {}
        row.state = 'running'
        s.commit()
        try:
            row.result = {**train(frozen['snapshot'], row.threshold, frozen['calibration']),
                          'snapshot': frozen['snapshot'], 'calibration_source': frozen['calibration']}
            row.state = 'done'
        except Exception as e:
            row.state = 'failed'
            row.error = str(e)
        s.commit()
        return row.state
