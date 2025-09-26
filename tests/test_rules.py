from datetime import datetime

from app.qc_rules import RuleDefinition


def test_rule_evaluation():
    rule = RuleDefinition(
        id=1,
        name="Reject crack",
        reject_if=["crack"],
        min_conf=0.35,
        version="rule-1",
        created_at=datetime.utcnow(),
    )
    detections = [{"label": "crack", "confidence": 0.4}]
    assert rule.evaluate(detections) == "FAIL"

    detections = [{"label": "crack", "confidence": 0.2}]
    assert rule.evaluate(detections) == "PASS"

    detections = [{"label": "ok", "confidence": 0.9}]
    assert rule.evaluate(detections) == "PASS"
