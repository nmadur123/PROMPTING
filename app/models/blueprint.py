"""Saqlangan tahlil va yaratilgan prompt."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Blueprint(Base):
    """Bitta startup uchun to'liq tahlil + yaratilgan prompt.

    Havola bilan ulashish uchun `id` qisqa uuid. Foydalanuvchi hisobi hozircha
    yo'q — natija havolani bilgan har kimga ochiq (ataylab, do'stga yuborish uchun).
    """

    __tablename__ = "blueprints"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    description: Mapped[str] = mapped_column(Text)
    lang: Mapped[str] = mapped_column(String(4), default="uz")
    region: Mapped[str] = mapped_column(String(8), default="uz")
    monthly_users: Mapped[int] = mapped_column(Integer, default=1000)

    project_type: Mapped[str] = mapped_column(String(40), default="")
    confidence: Mapped[int] = mapped_column(Integer, default=0)  # foizda

    target_model: Mapped[str] = mapped_column(String(120), default="")
    analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    prompt: Mapped[str] = mapped_column(Text, default="")
    used_llm: Mapped[int] = mapped_column(Integer, default=0)
