from sqlalchemy import (Boolean, Column, Float, ForeignKey, Integer, JSON, String,
                        Text, UniqueConstraint)
from src.config.database import Base


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(100), unique=True)
    password = Column(String(200))
    role = Column(String(20))


class Claim(Base):
    __tablename__ = 'claims'
    id = Column(Integer, primary_key=True)
    code = Column(String(60), unique=True, nullable=False)
    amount = Column(Float)
    distance = Column(Float)
    holiday = Column(Boolean)
    label = Column(Integer, nullable=True)
    note = Column(Text, default='')


class Training(Base):
    __tablename__ = 'trainings'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    threshold = Column(Float)
    state = Column(String(20), default='queued')
    result = Column(JSON)
    error = Column(Text, default='')


class Review(Base):
    __tablename__ = 'reviews'
    __table_args__ = (UniqueConstraint('training_id', 'claim_id', name='uq_review_source'),)
    id = Column(Integer, primary_key=True)
    training_id = Column(Integer, ForeignKey('trainings.id'))
    claim_id = Column(Integer, ForeignKey('claims.id'))
    score = Column(Float)
    decision = Column(String(30), default='pending')
    note = Column(Text, default='')
    status = Column(String(20), default='draft')


class Audit(Base):
    __tablename__ = 'audits'
    id = Column(Integer, primary_key=True)
    actor = Column(String(100))
    action = Column(String(200))
    created = Column(String(40))


class Opinion(Base):
    """每位财务对每笔票据的当前独立意见；同一(票据, 标注者)只保留一版。"""
    __tablename__ = 'opinions'
    __table_args__ = (UniqueConstraint('claim_id', 'annotator', name='uq_opinion_claim_annotator'),)
    id = Column(Integer, primary_key=True)
    claim_id = Column(Integer, ForeignKey('claims.id'), nullable=False)
    annotator = Column(String(80), nullable=False)
    label = Column(Integer, nullable=False)
    note = Column(Text, default='')
    revision = Column(Integer, default=1, nullable=False)
    entered_by = Column(String(100), default='')
    created = Column(String(40), default='')
    updated = Column(String(40), default='')


class OpinionRevision(Base):
    """意见的追加式版本记录，供更正审计与批次快照追溯。"""
    __tablename__ = 'opinion_revisions'
    id = Column(Integer, primary_key=True)
    opinion_id = Column(Integer, ForeignKey('opinions.id'), nullable=False)
    claim_id = Column(Integer, ForeignKey('claims.id'), nullable=False)
    annotator = Column(String(80), nullable=False)
    label = Column(Integer, nullable=False)
    note = Column(Text, default='')
    revision = Column(Integer, nullable=False)
    entered_by = Column(String(100), default='')
    created = Column(String(40), default='')


class GoldStandard(Base):
    """少量已确认金标准；不参与留出评估，只用于锚定方向与评估标注者。"""
    __tablename__ = 'gold_standards'
    id = Column(Integer, primary_key=True)
    claim_id = Column(Integer, ForeignKey('claims.id'), unique=True, nullable=False)
    label = Column(Integer, nullable=False)
    note = Column(Text, default='')
    confirmed_by = Column(String(100), default='')
    created = Column(String(40), default='')
    updated = Column(String(40), default='')


class Calibration(Base):
    __tablename__ = 'calibrations'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    state = Column(String(20), default='queued')
    result = Column(JSON)
    error = Column(Text, default='')
    snapshot = Column(JSON)
    snapshot_hash = Column(String(64), default='')
    attempts = Column(Integer, default=0, nullable=False)
    created = Column(String(40), default='')
    updated = Column(String(40), default='')
