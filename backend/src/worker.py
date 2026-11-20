import json
import logging
import os
import time
from pathlib import Path

import redis

from src.config.database import Session
from src.models.entities import Calibration, Training
from src.services.analysis import train
from src.services.calibration import CalibrationError, estimate

CALIBRATION_QUEUE = 'claim:calibrations'
TRAINING_QUEUE = 'claim:trainings'
ARTIFACTS = Path(os.environ.get('ARTIFACTS_DIR', '/artifacts'))


def recover(conn):
    """启动恢复：等待中/估计中的批次重新排队。快照已冻结，不会重复累计意见。"""
    with Session() as s:
        for row in s.query(Calibration).filter(Calibration.state.in_(['queued', 'running'])).all():
            row.state = 'queued'
            conn.rpush(CALIBRATION_QUEUE, row.id)
        for row in s.query(Training).filter(Training.state.in_(['queued', 'running'])).all():
            row.state = 'queued'
            conn.rpush(TRAINING_QUEUE, row.id)
        s.commit()


def _write_artifact(name, payload):
    try:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / name).write_text(json.dumps(payload, ensure_ascii=False))
    except OSError:
        logging.exception('artifact write failed')


def run_calibration(item):
    with Session() as s:
        row = s.get(Calibration, int(item))
        if not row or row.state != 'queued':
            return
        row.state = 'running'
        s.commit()
        try:
            result = estimate(row.snapshot)
            row.result = result
            row.error = ''
            row.state = 'done'
            _write_artifact('calibration-' + str(row.id) + '.json', result)
        except CalibrationError as e:
            # 证据不足是明确的业务结论：保存可解释原因，而不是制造高置信度
            row.state = 'failed'
            row.error = str(e)
            row.result = {'refused': True, 'reasons': e.reasons, 'checks': e.checks}
        except Exception as e:
            logging.exception('calibration')
            row.state = 'failed'
            row.error = '校准执行失败：' + str(e)
        s.commit()


def run_training(item):
    with Session() as s:
        row = s.get(Training, int(item))
        if not row or row.state != 'queued':
            return
        frozen = row.result or {}
        snapshot = frozen.get('snapshot', [])
        calibration = frozen.get('calibration')
        row.state = 'running'
        s.commit()
        try:
            result = train(snapshot, row.threshold, calibration)
            row.result = {**result, 'snapshot': snapshot, 'calibration_source': calibration}
            row.state = 'done'
            _write_artifact(str(row.id) + '.json', row.result)
        except Exception as e:
            row.state = 'failed'
            row.error = str(e)
        s.commit()


def cycle(conn, timeout=1):
    """处理一个任务；返回是否处理了任务。"""
    item = conn.blpop(CALIBRATION_QUEUE, timeout=timeout)
    if item:
        run_calibration(item[1])
        return True
    item = conn.blpop(TRAINING_QUEUE, timeout=max(timeout, 3))
    if item:
        run_training(item[1])
        return True
    return False


def main():
    conn = redis.Redis.from_url(os.environ['REDIS_URL'])
    recover(conn)
    while True:
        Path('/tmp/heartbeat').touch()
        try:
            cycle(conn)
        except Exception:
            logging.exception('worker')
            time.sleep(2)


if __name__ == '__main__':
    main()
