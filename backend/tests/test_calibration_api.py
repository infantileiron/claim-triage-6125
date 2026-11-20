import json

from tests.conftest import (add_gold, add_opinion, run_calibration, run_training,
                            seed_claims)


def _seed_annotations(client, users, ids, pattern):
    """pattern: (claim_index, annotator, label) 列表；同时设置金标准。"""
    for idx, who, lab in pattern:
        add_opinion(client, users['analyst'], ids[idx], who, lab)


def _batch(client, users, name='批次'):
    r = client.post('/calibrations', json={'name': name}, headers=users['analyst'])
    assert r.status_code == 200, r.text
    return r.json()


def _healthy_pattern(ids, truth):
    ops = []
    for i, cid in enumerate(ids):
        for who in ('沈', '陆', '周'):
            ops.append((i, who, truth[i]))
    return ops


def test_viewer_cannot_write_but_can_read(client, users):
    ids = seed_claims(client, users['analyst'], 3)
    assert client.post('/opinions', json={'claim_id': ids[0], 'annotator': '沈', 'label': 1},
                       headers=users['viewer']).status_code == 403
    assert client.post('/calibrations', json={'name': '拒绝'}, headers=users['viewer']).status_code == 403
    assert client.post('/gold', json={'claim_id': ids[0], 'label': 1, 'note': '原始票据'},
                       headers=users['viewer']).status_code == 403
    assert client.post('/gold', json={'claim_id': ids[0], 'label': 1, 'note': '原始票据'},
                       headers=users['analyst']).status_code == 403
    assert client.get('/calibrations', headers=users['viewer']).status_code == 200
    assert client.get('/opinions', headers=users['viewer']).status_code == 200


def test_opinion_revision_keeps_one_version_and_audits(client, users):
    ids = seed_claims(client, users['analyst'], 2)
    first = add_opinion(client, users['analyst'], ids[0], '沈', 1)
    assert first['revision'] == 1 and first['changed'] is True
    same = add_opinion(client, users['analyst'], ids[0], '沈', 1)
    assert same['changed'] is False and same['revision'] is None
    second = add_opinion(client, users['analyst'], ids[0], '沈', 0)
    assert second['revision'] == 2 and second['changed'] is True
    detail = client.get(f'/claims/{ids[0]}/opinions', headers=users['analyst']).json()
    assert len(detail['opinions']) == 1
    assert [r['revision'] for r in detail['revisions']] == [2, 1]
    assert detail['revisions'][0]['label'] == 0


def test_calibration_refuses_and_saves_reasons(client, users):
    ids = seed_claims(client, users['analyst'], 6)
    add_opinion(client, users['analyst'], ids[0], '沈', 1)
    batch = _batch(client, users, '证据不足')
    assert batch['state'] == 'queued'
    assert run_calibration(batch['id']) == 'failed'
    detail = client.get(f"/calibrations/{batch['id']}", headers=users['analyst']).json()
    assert detail['result']['refused'] is True
    assert detail['result']['reasons']
    assert '金标准' in detail['error']
    # 失败批次可以重试，且沿用同一快照哈希
    before = detail['snapshot_hash']
    r = client.post(f"/calibrations/{batch['id']}/retry", headers=users['analyst'])
    assert r.status_code == 200 and r.json()['state'] == 'queued'
    assert r.json()['attempts'] == 1
    assert r.json()['snapshot_hash'] == before


def test_retry_reuses_snapshot_even_after_opinion_change(client, users):
    ids = seed_claims(client, users['analyst'], 50)
    truth = [i % 2 for i in range(50)]
    for i, who, lab in _healthy_pattern(ids, truth):
        add_opinion(client, users['analyst'], ids[i], who, lab)
    # 提交时没有金标准，冻结的快照会拒绝估计
    batch = _batch(client, users, '首次')
    assert run_calibration(batch['id']) == 'failed'
    # 之后补金标准并更正意见，再重试：仍沿用同一份冻结快照
    for i in (3, 19, 33):
        add_gold(client, users['admin'], ids[i], truth[i])
    add_opinion(client, users['analyst'], ids[0], '沈', 1 - truth[0])
    r = client.post(f"/calibrations/{batch['id']}/retry", headers=users['analyst']).json()
    assert r['snapshot_hash'] == batch['snapshot_hash']
    assert r['attempts'] == 1
    # 该批次仍会因冻结时无金标准而失败，证明没有偷偷混入新意见/新金标准
    assert run_calibration(batch['id']) == 'failed'


def test_calibration_success_persists_independent_rates(client, users):
    ids = seed_claims(client, users['analyst'], 60)
    truth = [(i * 7) % 3 == 0 for i in range(60)]
    truth = [int(t) for t in truth]
    for i, cid in enumerate(ids):
        for who in ('沈', '陆', '周'):
            add_opinion(client, users['analyst'], cid, who, truth[i])
        # 程是反向标注者
        add_opinion(client, users['analyst'], cid, '程', 1 - truth[i])
    for i in (0, 11, 22, 33, 44, 55):
        add_gold(client, users['admin'], ids[i], truth[i])
    batch = _batch(client, users, '反向识别')
    assert run_calibration(batch['id']) == 'done'
    detail = client.get(f"/calibrations/{batch['id']}", headers=users['analyst']).json()
    assert detail['state'] == 'done'
    by = {a['annotator']: a for a in detail['result']['annotators']}
    assert by['程']['status'] == 'reversed'
    assert by['程']['sensitivity'] < .5
    assert by['沈']['status'] in ('reliable', 'unstable')
    assert all(0 <= a['false_negative_rate'] <= 1 for a in detail['result']['annotators']
               if a['false_negative_rate'] is not None)
    gold_status = [c for c in detail['result']['claims'] if c['status'] == 'gold']
    assert len(gold_status) == 6
    assert client.get(f"/calibrations/{batch['id']}/export", headers=users['viewer']).status_code == 200


def test_training_uses_soft_labels_and_excludes_gold_from_holdout(client, users):
    ids = seed_claims(client, users['analyst'], 60)
    truth = [int((i * 11) % 4 < 2) for i in range(60)]
    for i, cid in enumerate(ids):
        for who in ('沈', '陆', '周', '程'):
            # 程的意见有高漏报，制造与多数不同的方向
            lab = truth[i] if who != '程' else int(i % 5 != 0)
            add_opinion(client, users['analyst'], cid, who, lab)
    gold_ids = [0, 9, 18, 27, 36, 45]
    for i in gold_ids:
        add_gold(client, users['admin'], ids[i], truth[i])
    batch = _batch(client, users, '软标签')
    assert run_calibration(batch['id']) == 'done'
    r = client.post('/trainings', json={'name': '软标签版本', 'threshold': .5,
                                        'calibration_id': batch['id']}, headers=users['analyst'])
    assert r.status_code == 200, r.text
    tid = r.json()['id']
    assert run_training(tid) == 'done'
    detail = client.get(f'/trainings/{tid}', headers=users['analyst']).json()
    usage = detail['result']['calibration_usage']
    assert usage['id'] == batch['id']
    assert usage['soft_labels']
    # 金标准不参与留出评估；留出样本保留原始人工标签
    assert usage['holdout_gold_overlap'] == 0
    assert all(h['label'] in (0, 1) for h in detail['result']['holdout'])
    assert set(usage['gold_train_only']) == {ids[i] for i in gold_ids}
    # 留出集全部是原始人工标签
    test_codes = set(detail['result']['test_codes'])
    codes = {r['id']: r['code'] for r in detail['result']['snapshot']}
    assert not (test_codes & {codes[ids[i]] for i in gold_ids})


def test_training_rejects_unknown_or_pending_calibration(client, users):
    seed_claims(client, users['analyst'], 50)
    assert client.post('/trainings', json={'name': '无此校准', 'threshold': .5,
                                           'calibration_id': 9999},
                       headers=users['analyst']).status_code == 404
    ids = client.get('/claims', headers=users['analyst']).json()
    truth = [int(i % 2) for i in range(50)]
    for i in range(50):
        add_opinion(client, users['analyst'], ids[i]['id'], '沈', truth[i])
    for i in (3, 13, 23, 33):
        add_gold(client, users['admin'], ids[i]['id'], truth[i])
    batch = _batch(client, users, '未执行')
    r = client.post('/trainings', json={'name': '等待中', 'threshold': .5,
                                        'calibration_id': batch['id']}, headers=users['analyst'])
    assert r.status_code == 409


def test_correction_during_training_uses_complete_frozen_version(client, users):
    """意见更正与训练并发时，训练只读取提交时的完整一批校准。"""
    ids = seed_claims(client, users['analyst'], 60)
    truth = [int((i * 3) % 5 < 2) for i in range(60)]
    for i, cid in enumerate(ids):
        for who in ('沈', '陆', '周'):
            add_opinion(client, users['analyst'], cid, who, truth[i])
    for i in (2, 17, 32, 47):
        add_gold(client, users['admin'], ids[i], truth[i])
    batch = _batch(client, users, '并发前')
    assert run_calibration(batch['id']) == 'done'
    r = client.post('/trainings', json={'name': '并发训练', 'threshold': .5,
                                        'calibration_id': batch['id']}, headers=users['analyst'])
    tid = r.json()['id']
    frozen_hash = client.get(f'/trainings/{tid}', headers=users['analyst']).json()['result']['calibration']['snapshot_hash']
    # 训练排队后、执行前更正意见并重新校准
    add_opinion(client, users['analyst'], ids[0], '沈', 1 - truth[0])
    new_batch = _batch(client, users, '并发后')
    assert run_calibration(new_batch['id']) == 'done'
    # 旧校准批次不受后来更正影响
    old = client.get(f"/calibrations/{batch['id']}", headers=users['analyst']).json()
    assert old['snapshot_hash'] == frozen_hash
    assert run_training(tid) == 'done'
    detail = client.get(f'/trainings/{tid}', headers=users['analyst']).json()
    assert detail['result']['calibration_usage']['snapshot_hash'] == frozen_hash
    assert detail['result']['calibration_usage']['id'] == batch['id']
    # 新校准批次反映了更正后的意见
    assert new_batch['snapshot_hash'] != frozen_hash


def test_retry_rejected_while_queued_or_done(client, users):
    ids = seed_claims(client, users['analyst'], 40)
    batch = _batch(client, users, '排队中')
    assert client.post(f"/calibrations/{batch['id']}/retry", headers=users['analyst']).status_code == 409
    assert client.post('/calibrations/9999/retry', headers=users['analyst']).status_code == 404
    assert batch['snapshot_hash'] == client.get(f"/calibrations/{batch['id']}",
                                                headers=users['analyst']).json()['snapshot_hash']
