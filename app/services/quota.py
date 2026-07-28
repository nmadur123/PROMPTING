"""Oylik kvota hisobi.

Kvota `estimates` jadvalidagi joriy oy yozuvlaridan sanaladi, foydalanuvchidagi
hisoblagichdan emas. Sabab: hisoblagichni har oy nolga tushirish uchun rejali
vazifa kerak, u ishlamay qolsa kvota jimgina noto'g'ri bo'lib qoladi va buni
hech kim sezmaydi. Jadvaldan sanash esa hech qachon eskirmaydi.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.estimate import Estimate
from app.models.user import PLAN_QUOTA, User, is_vip


@dataclass
class Quota:
    plan: str
    limit: Optional[int]      # oylik limit; None — cheksiz
    used: int                 # shu oyda ishlatilgan
    remaining: Optional[int]  # oylik limitdan qolgani; None — cheksiz
    resets_at: str            # keyingi oy boshi, ISO
    credits: int = 0          # oylik limitdan tashqari sotib olingan analizlar
    unlimited: bool = False   # VIP yoki Enterprise

    @property
    def total_remaining(self) -> Optional[int]:
        """Jami qolgan analiz: oylik qoldiq + kredit. None — cheksiz."""
        if self.unlimited or self.remaining is None:
            return None
        return self.remaining + self.credits

    @property
    def exhausted(self) -> bool:
        total = self.total_remaining
        return total is not None and total <= 0


def _month_start(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month(now: Optional[datetime] = None) -> datetime:
    start = _month_start(now)
    return start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)


async def get_quota(db: AsyncSession, user: User) -> Quota:
    # VIP hisob tarifdan qat'i nazar cheksiz — bu yagona istisno.
    unlimited = is_vip(user.email)
    limit = None if unlimited else PLAN_QUOTA.get(user.plan)
    used = (
        await db.execute(
            select(func.count(Estimate.id)).where(
                Estimate.user_id == user.id,
                Estimate.created_at >= _month_start(),
            )
        )
    ).scalar() or 0

    return Quota(
        plan=user.plan.value,
        limit=limit,
        used=used,
        remaining=None if limit is None else max(0, limit - used),
        resets_at=_next_month().isoformat(),
        credits=max(0, user.credits or 0),
        unlimited=unlimited,
    )


async def already_charged(db: AsyncSession, user: User, description: str, minutes: int = 60) -> bool:
    """Shu matn uchun yaqinda hisobdan yechilganmi?

    Landing sahifada bitta g'oya uchun avval `/analyze`, keyin `/generate`
    chaqiriladi — ikkalasi ham yechsa, bitta analiz uchun ikki marta pul
    ketardi. Chat sahifasida esa faqat `/generate` chaqiriladi.

    Shuning uchun yechishdan oldin shu foydalanuvchining aynan shu matn bo'yicha
    yaqin oradagi yozuvi bor-yo'qligi tekshiriladi.
    """
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    found = (
        await db.execute(
            select(Estimate.id).where(
                Estimate.user_id == user.id,
                Estimate.description == description,
                Estimate.created_at >= since,
            ).limit(1)
        )
    ).scalar()
    return found is not None


async def consume(db: AsyncSession, user: User, quota: Quota) -> None:
    """Bitta analizni hisobdan yechadi.

    Avval oylik limit ishlatiladi, u tugagach kreditdan olinadi. Tartib muhim:
    kredit yonmaydi, oylik limit esa oy oxirida baribir yo'qoladi — shuning
    uchun avval yo'qoladiganini sarflash foydalanuvchi foydasiga.

    Chaqiruvchi `db.commit()` ni o'zi qiladi — bu funksiya bitta tranzaksiyaning
    bir qismi bo'lishi kerak.
    """
    user.estimates_used += 1

    if quota.unlimited or quota.limit is None:
        return

    # Oylik limitda joy bormi? `used` bu chaqiruvdan OLDINGI holat.
    if quota.used < quota.limit:
        return

    if user.credits > 0:
        user.credits -= 1
