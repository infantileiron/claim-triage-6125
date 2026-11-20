from fastapi import HTTPException

from src.models.entities import Calibration, Training


def ensure_training(s, i):
    row = s.get(Training, i)
    if not row:
        raise HTTPException(404, '训练版本不存在')
    return row


def ensure_calibration(s, i):
    row = s.get(Calibration, i)
    if not row:
        raise HTTPException(404, '校准批次不存在')
    return row
