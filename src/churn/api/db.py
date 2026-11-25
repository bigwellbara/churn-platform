"""Prediction log in Postgres (any SQLAlchemy URL works; tests use SQLite).

Every prediction is stored with its full input features. That log is the raw material
for the drift check (are today's customers different from the training data?) and for
measuring real-world accuracy once actual churn outcomes are known.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String, create_engine, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    customer_id: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    # JSONB on Postgres (indexable, queryable), plain JSON elsewhere.
    features: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    churn_probability: Mapped[float]
    churn_prediction: Mapped[bool]
    threshold: Mapped[float]


class PredictionStore:
    def __init__(self, url: str):
        # pool_pre_ping: transparently replace connections the database has dropped.
        self.engine = create_engine(url, pool_pre_ping=True)
        # Simple schema management for a single table. A multi-table schema that evolves
        # over time would use Alembic migrations instead.
        Base.metadata.create_all(self.engine)

    def save(self, rows: list[dict[str, Any]]) -> None:
        with Session(self.engine) as session, session.begin():
            session.add_all([PredictionLog(**row) for row in rows])

    def is_healthy(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def count(self) -> int:
        with Session(self.engine) as session:
            return session.scalar(select(func.count()).select_from(PredictionLog)) or 0

    def dispose(self) -> None:
        self.engine.dispose()
