import json

from src.models.entities import Calibration, Training


def test_worker_retry_uses_same_snapshot_and_no_double_count(client, users, monkeypatch, tmp_path):
    from tests.conftest import add_gold, add_opinion, seed_claims
    import src.worker as worker
    monkeypatch.setattr(worker, 'ARTIFACTS', tmp_path)

    ids = seed_claims(client, users['analyst'], 50)
    truth = [i % 2 for i in range(50)]
    for i, cid in enumerate(ids):
        for who in ('沈', '陆', '周'):
            add_opinion(client, users['analyst'], cid, who, truth[i])
    batch = client.post('/calibrations', json={'name': '无金标准'},
                        headers=users['analyst']).json()
    assert batch['state'] == 'queued'
    worker.run_calibration(batch['id'])
    detail = client.get(f"/calibrations/{batch['id']}", headers=users['analyst']).json()
    assert detail['state'] == 'failed'
    frozen_hash = detail['snapshot_hash']
    frozen_opinions = len(detail['snapshot']['opinions'])

    # 重试前补金标准、更正意见，但重试仍只用冻结的那一版
    for i in (1, 17, 33):
        add_gold(client, users['admin'], ids[i], truth[i])
    add_opinion(client, users['analyst'], ids[0], '沈', 1 - truth[0])
    retry = client.post(f"/calibrations/{batch['id']}/retry", headers=users['analyst']).json()
    assert retry['attempts'] == 1 and retry['snapshot_hash'] == frozen_hash
    worker.run_calibration(batch['id'])
    again = client.get(f"/calibrations/{batch['id']}", headers=users['analyst']).json()
    assert again['state'] == 'failed'
    assert again['snapshot_hash'] == frozen_hash
    assert len(again['snapshot']['opinions']) == frozen_opinions
    assert again['error']  # 仍因冻结版本缺金标准而拒绝


def test_worker_recovery_requeues_and_skips_done(client, users, monkeypatch, tmp_path):
    from tests.conftest import add_gold, add_opinion, seed_claims
    import src.worker as worker
    from src.config.database import Session
    monkeypatch.setattr(worker, 'ARTIFACTS', tmp_path)

    ids = seed_claims(client, users['analyst'], 50)
    truth = [i % 2 for i in range(50)]
    for i, cid in enumerate(ids):
        for who in ('沈', '陆', '周'):
            add_opinion(client, users['analyst'], cid, who, truth[i])
    for i in (2, 18, 34, 46):
        add_gold(client, users['admin'], ids[i], truth[i])
    batch = client.post('/calibrations', json={'name': '恢复'},
                        headers=users['analyst']).json()
    client.fake_redis.queue.clear()

    class Conn:
        def __init__(self):
            self.pushed = []

        def rpush(self, key, value):
            self.pushed.append((key, value))

    conn = Conn()
    with Session() as s:
        row = s.get(Calibration, batch['id'])
        row.state = 'running'
        s.commit()
    worker.recover(conn)
    assert (worker.CALIBRATION_QUEUE, batch['id']) in conn.pushed
    with Session() as s:
        assert s.get(Calibration, batch['id']).state == 'queued'

    # 已完成的批次不会被恢复逻辑重复入队
    worker.run_calibration(batch['id'])
    conn.pushed.clear()
    worker.recover(conn)
    assert not any(v == batch['id'] and k == worker.CALIBRATION_QUEUE for k, v in conn.pushed)
    # 同一 id 重复投递只执行一次
    worker.run_calibration(batch['id'])
    with Session() as s:
        assert s.get(Calibration, batch['id']).state == 'done'
    assert json.loads((tmp_path / f"calibration-{batch['id']}.json").read_text())
