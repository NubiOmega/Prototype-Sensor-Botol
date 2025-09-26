"""QC rule management and evaluation utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

from sqlalchemy.orm import Session

from .db import QCRule, session_scope

DetectionLike = Mapping[str, object]


@dataclass(slots=True)
class RuleDefinition:
    id: int
    name: str
    reject_if: List[str]
    min_conf: float
    version: str
    created_at: datetime

    @classmethod
    def from_model(cls, model: QCRule) -> "RuleDefinition":
        payload: MutableMapping[str, object] = {}
        if isinstance(model.json, dict):
            payload.update(model.json)
        reject_if = [str(item) for item in payload.get("reject_if", [])]
        min_conf = float(payload.get("min_conf", 0.35))
        created = model.created_at or datetime.utcnow()
        version = f"rule-{model.id}-{created.strftime('%Y%m%d%H%M%S')}"
        return cls(
            id=model.id,
            name=model.name,
            reject_if=reject_if,
            min_conf=min_conf,
            version=version,
            created_at=created,
        )

    def evaluate(self, detections: Sequence[DetectionLike]) -> str:
        if self.should_reject(detections):
            return "FAIL"
        return "PASS"

    def should_reject(self, detections: Sequence[DetectionLike]) -> bool:
        for det in detections:
            label = str(det.get("label", ""))
            if label not in self.reject_if:
                continue
            conf_value = det.get("confidence")
            try:
                conf = float(conf_value) if conf_value is not None else 0.0
            except (TypeError, ValueError):
                conf = 0.0
            if conf >= self.min_conf:
                return True
        return False


def list_rules(session: Session) -> List[RuleDefinition]:
    rules = session.query(QCRule).order_by(QCRule.created_at.desc()).all()
    return [RuleDefinition.from_model(rule) for rule in rules]


def load_active_rule(session: Session) -> Optional[RuleDefinition]:
    rule = session.query(QCRule).filter(QCRule.active.is_(True)).order_by(QCRule.created_at.desc()).first()
    if not rule:
        return None
    return RuleDefinition.from_model(rule)


def set_active_rule(session: Session, rule_id: int) -> Optional[RuleDefinition]:
    rule = session.get(QCRule, rule_id)
    if not rule:
        return None
    session.query(QCRule).filter(QCRule.id != rule_id).update({QCRule.active: False})
    rule.active = True
    session.flush()
    return RuleDefinition.from_model(rule)


def save_rule(
    session: Session,
    *,
    name: str,
    payload: Mapping[str, object],
    rule_id: Optional[int] = None,
    make_active: bool = False,
) -> RuleDefinition:
    serialisable = json.loads(json.dumps(payload))
    if rule_id:
        rule = session.get(QCRule, rule_id)
        if not rule:
            raise ValueError(f"Rule with id {rule_id} not found")
        rule.name = name
        rule.json = serialisable
        if make_active:
            set_active_rule(session, rule.id)
        session.flush()
        return RuleDefinition.from_model(rule)
    rule = QCRule(name=name, json=serialisable, active=False)
    session.add(rule)
    session.flush()
    if make_active:
        set_active_rule(session, rule.id)
    return RuleDefinition.from_model(rule)


def ensure_rule_exists() -> RuleDefinition:
    with session_scope() as session:
        rule = load_active_rule(session)
        if rule:
            return rule
        default_payload = {
            "reject_if": [
                "crack_hairline",
                "crack_major",
                "chip_mouth",
                "chip_body",
                "no_cap",
                "label_torn",
                "other_defect",
            ],
            "min_conf": 0.35,
        }
        new_rule = save_rule(session, name="Default", payload=default_payload, make_active=True)
        return new_rule


__all__ = [
    "ensure_rule_exists",
    "list_rules",
    "load_active_rule",
    "RuleDefinition",
    "save_rule",
    "set_active_rule",
]
