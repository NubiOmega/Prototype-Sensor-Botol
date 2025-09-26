from pathlib import Path

from app import db
from app.db import DetectionPayload


def test_db_init_and_insert(tmp_path):
    db_path = tmp_path / "qc_test.db"
    db.init_db(db_path)

    with db.session_scope() as session:
        model = db.register_model(
            session,
            name="TestModel",
            path="weights.pt",
            size=123,
            file_hash="abc123",
            set_active=True,
        )
        batch = db.upsert_batch(
            session,
            lot_id="LOT1",
            line="L1",
            shift="A",
            operator="OP1",
        )
        inspection = db.record_inspection(
            session,
            batch_id=batch.id,
            frame_path="frame.jpg",
            pass_fail="PASS",
            rule_version="rule-1",
            inference_ms=25.0,
            fps=30.0,
            model_id=model.id,
            detections=[
                DetectionPayload(cls="ok", conf=0.9, bbox=(0.0, 0.0, 10.0, 10.0))
            ],
        )
        db.close_batch(session, batch.id)

        assert inspection.id is not None
        assert db.get_active_model(session).id == model.id
