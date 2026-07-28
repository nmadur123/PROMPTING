"""Foydalanuvchi va kirish hodisalari.

`LoginEvent` alohida jadval, chunki superadmin paneliga "kirdi / chiqdi /
kunlik faol" kabi savollar kerak, ularni esa `users.last_login` kabi bitta
ustundan olib bo'lmaydi — u faqat oxirgi holatni biladi, tarixni emas.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _uuid() -> str:
    return f"usr_{uuid.uuid4().hex}{uuid.uuid4().hex[:16]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    USER = "user"
    SUPERADMIN = "superadmin"


class Plan(str, enum.Enum):
    SKETCH = "sketch"
    DRAFTING = "drafting"
    STUDIO = "studio"
    ENTERPRISE = "enterprise"


# Oyiga nechta hisob-kitob. None — cheksiz.
#
# Cheksiz faqat ENTERPRISE da qoldi va uni superadmin qo'lda beradi. Ilgari
# STUDIO cheksiz edi — ya'ni tarifni sotib olgan har kim cheksiz ishlatardi va
# OpenRouter hisobi nazoratdan chiqib ketishi mumkin edi.
PLAN_QUOTA: dict[Plan, int | None] = {
    Plan.SKETCH: 3,
    Plan.DRAFTING: 20,
    Plan.STUDIO: 100,
    # Enterprise ham CHEKLANGAN. Ilgari bu `None` (cheksiz) edi va superadmin
    # kimnidir Enterprise ga o'tkazsa, u cheksiz ishlata olardi. Cheksizlik
    # faqat bitta hisobga tegishli bo'lishi kerak — u `is_vip()` orqali
    # beriladi va tarifga umuman bog'liq emas.
    Plan.ENTERPRISE: 1000,
}

# Bitta analiz narxi. Superadmin kredit berganda daromad shu bo'yicha yoziladi.
PRICE_PER_ANALYSIS_USD = 1.0

# Oylik tarif narxlari — YAGONA manba.
#
# Narx ilgari ikkita frontend faylida takrorlanardi va biri o'zgarganda
# ikkinchisi jimgina eski narxni ko'rsatib qolardi. Endi u shu yerda va
# `/api/pricing-info` orqali beriladi: o'zgartirish uchun frontendni qayta
# yig'ish shart emas.
#
# None — narx belgilanmagan (individual kelishuv).
PLAN_PRICE_USD: dict[Plan, float | None] = {
    Plan.SKETCH: 0.0,
    Plan.DRAFTING: 9.99,
    Plan.STUDIO: 29.99,
    Plan.ENTERPRISE: None,
}

# Cheksiz huquqqa ega yagona hisob. Ro'yxatdan o'tishda ham, huquq
# tekshiruvida ham shu yagona manba ishlatiladi — email ikki joyda
# qattiq yozilib, keyin bittasi unutilib qolmasin.
VIP_EMAIL = "mansurovislombek130@gmail.com"


def is_vip(email: str) -> bool:
    return (email or "").strip().lower() == VIP_EMAIL


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    # Google/GitHub orqali kirgan foydalanuvchida parol bo'lmaydi. Bo'sh xesh
    # hech qanday parol bilan mos kelmaydi, ya'ni bunday hisobga parol orqali
    # kirib bo'lmaydi — bu ataylab shunday.
    password_hash: Mapped[str] = mapped_column(String(255), default="")
    # Firebase `sub` — bitta odam emailini o'zgartirsa ham shu o'zgarmaydi.
    firebase_uid: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, default=None)
    provider: Mapped[str] = mapped_column(String(32), default="password")
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.USER)
    plan: Mapped[Plan] = mapped_column(Enum(Plan), default=Plan.SKETCH)
    lang: Mapped[str] = mapped_column(String(4), default="uz")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    # Umrbod hisoblagich. Oylik qolgan kvota `estimates` jadvalidan sanaladi,
    # chunki bu ustunni har oy nolga tushirish uchun alohida vazifa kerak
    # bo'lardi va u ishlamay qolsa kvota jimgina noto'g'ri bo'lib qolardi.
    estimates_used: Mapped[int] = mapped_column(default=0)
    # Oylik limitdan tashqari sotib olingan analizlar. Oy oxirida yonmaydi —
    # foydalanuvchi pul to'lagan, muddat qo'yish uni yo'qotish bo'lardi.
    credits: Mapped[int] = mapped_column(default=0)
    # Telegram username (@siz). Superadmin to'lovni shu orqali topadi.
    telegram: Mapped[str] = mapped_column(String(64), default="")


class EventKind(str, enum.Enum):
    REGISTER = "register"
    LOGIN = "login"
    LOGOUT = "logout"
    ESTIMATE = "estimate"


class LoginEvent(Base):
    """Bitta harakat yozuvi. Faollik statistikasi shu jadvaldan hisoblanadi."""

    __tablename__ = "login_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[EventKind] = mapped_column(Enum(EventKind))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    # Superadmin panelida "qayerdan kirdi" degan savol chiqadi; IP saqlanmaydi,
    # faqat brauzer satri qisqartirilib olinadi — bu shaxsni aniqlamaydi.
    user_agent: Mapped[str] = mapped_column(String(200), default="")


# DAU/WAU so'rovlari doim (kind, created_at) bo'yicha filtrlaydi.
Index("ix_events_kind_created", LoginEvent.kind, LoginEvent.created_at)
