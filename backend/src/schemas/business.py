from typing import Literal, Optional

from pydantic import BaseModel, Field


class Login(BaseModel):
    email: str
    password: str


class ClaimInput(BaseModel):
    code: str = Field(min_length=1, max_length=60)
    amount: float = Field(ge=0, le=100000, allow_inf_nan=False)
    distance: float = Field(ge=0, le=10000, allow_inf_nan=False)
    holiday: bool = False
    label: Optional[Literal[0, 1]] = None
    note: str = Field(default='', max_length=1000)


class TrainingInput(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    threshold: float = Field(default=.5, ge=.1, le=.9, allow_inf_nan=False)
    calibration_id: Optional[int] = Field(default=None, ge=1)


class ReviewInput(BaseModel):
    decision: Literal['valid', 'flagged']
    note: str = Field(min_length=6, max_length=2000)


class OpinionInput(BaseModel):
    claim_id: int = Field(ge=1)
    annotator: str = Field(min_length=1, max_length=80)
    label: Literal[0, 1]
    note: str = Field(default='', max_length=1000)


class GoldInput(BaseModel):
    claim_id: int = Field(ge=1)
    label: Literal[0, 1]
    note: str = Field(default='', min_length=4, max_length=1000)


class CalibrationInput(BaseModel):
    name: str = Field(min_length=2, max_length=100)
