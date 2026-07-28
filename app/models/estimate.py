"""Saqlangan hisob-kitob.

Har bir `/api/analyze` chaqiruvi shu yerga yoziladi. Tarix, kvota va profil
raqamlari — hammasi shu jadvaldan hisoblanadi, ya'ni interfeysdagi hech bir
son qo'lda yozilgan emas.

To'liq tahlil JSON ustunida saqlanadi: sxema hali o'zgarib turadi, va uni
ustunlarga yoyish har o'zgarishda migratsiya talab qilardi. Ro'yxatda
ko'rsatiladigan maydonlar (sarlavha, narx, stack) alohida ustunlarda —
tarixni ochish uchun butun JSON'ni o'qish shart bo'lmasin.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex[:16]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Estimate(Base):
    __tablename__ = "estimates"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    # Foydalanuvchi yozgan matn va undan olingan sarlavha.
    description: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(160), default="")

    project_type: Mapped[str] = mapped_column(String(40), default="")
    project_title: Mapped[str] = mapped_column(String(80), default="")
    confidence: Mapped[int] = mapped_column(Integer, default=0)  # foizda
    lang: Mapped[str] = mapped_column(String(4), default="uz")
    region: Mapped[str] = mapped_column(String(8), default="uz")
    monthly_users: Mapped[int] = mapped_column(Integer, default=0)

    # Ro'yxatda ko'rsatiladigan qiymatlar.
    monthly_usd: Mapped[float] = mapped_column(default=0.0)
    stack: Mapped[list] = mapped_column(JSON, default=list)

    analysis: Mapped[dict] = mapped_column(JSON, default=dict)


# Tarix so'rovi doim "shu foydalanuvchi, eng yangisi birinchi" shaklida.
Index("ix_estimates_user_created", Estimate.user_id, Estimate.created_at.desc())
