"""To'lov va tarif yozuvlari.

Ikki jadval:

`CreditGrant` — daromad daftari. Superadmin kredit berganda bitta yozuv
qo'shiladi va u hech qachon o'zgarmaydi. Jami daromadni `users.credits`
ustunidan hisoblab bo'lmaydi: u ishlatilgani sari kamayadi, ya'ni "qancha pul
tushdi" savoliga javob bermaydi. Shuning uchun alohida o'zgarmas daftar.

`UpgradeRequest` — foydalanuvchining tarif so'rovi. Foydalanuvchi email va
Telegram usernameni qoldiradi, superadmin panelda ko'radi va tasdiqlaydi.
So'rovni saqlash shart: aks holda odam Telegramga yozgach superadmin uni
qaysi hisob bilan bog'lashni faqat xotirasidan topishi kerak bo'ladi.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.user import utcnow


def _grant_id() -> str:
    return f"grn_{uuid.uuid4().hex[:24]}"


def _req_id() -> str:
    return f"req_{uuid.uuid4().hex[:24]}"


class CreditGrant(Base):
    """Superadmin bergan kreditlar — o'zgarmas daromad yozuvi."""

    __tablename__ = "credit_grants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_grant_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # Kim berdi. Superadmin o'chirilsa ham yozuv qolsin — SET NULL.
    granted_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), default=None
    )
    credits: Mapped[int] = mapped_column(default=0)
    # Necha dollar deb yozildi. Chegirmali paket berilsa kredit soniga
    # to'g'ridan-to'g'ri teng bo'lmasligi mumkin, shuning uchun alohida maydon.
    amount_usd: Mapped[float] = mapped_column(Float, default=0.0)
    note: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class RequestStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class UpgradeRequest(Base):
    """Foydalanuvchining tarif/kredit so'rovi."""

    __tablename__ = "upgrade_requests"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_req_id)
    # Hisob topilmasa ham so'rov qabul qilinadi: odam boshqa email bilan
    # ro'yxatdan o'tgan bo'lishi mumkin va buni superadmin qo'lda hal qiladi.
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), default=None, index=True
    )
    email: Mapped[str] = mapped_column(String(255), index=True)
    telegram: Mapped[str] = mapped_column(String(64), default="")
    plan: Mapped[str] = mapped_column(String(32), default="")
    credits: Mapped[int] = mapped_column(default=0)
    message: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[RequestStatus] = mapped_column(
        Enum(RequestStatus), default=RequestStatus.PENDING, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    handled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    handled_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), default=None
    )


# Panel doim "kutayotgan so'rovlar, yangisi birinchi" tartibida so'raydi.
Index("ix_requests_status_created", UpgradeRequest.status, UpgradeRequest.created_at)
