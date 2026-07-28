"""Superadmin statistikasi.

Barcha raqamlar `login_events` jadvalidan hisoblanadi. Faol foydalanuvchi —
oynada kamida bitta hodisasi bor **noyob** foydalanuvchi; hodisalar sonini
sanash odamlar sonidan katta chiqadi va ko'rsatkichni shishiradi.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import EventKind, LoginEvent, Role, User
from app.models.billing import CreditGrant, RequestStatus, UpgradeRequest
from app.models.user import PRICE_PER_ANALYSIS_USD
from app.routers.auth import require_superadmin

router = APIRouter(prefix="/api/admin", tags=["Superadmin"])


class SeriesPoint(BaseModel):
    date: str
    registered: int
    active: int
    estimates: int


class RecentUser(BaseModel):
    id: str
    email: str
    name: str
    role: str
    plan: str
    telegram: str
    credits: int
    created_at: str
    last_seen_at: str | None
    estimates_used: int


class StatsOut(BaseModel):
    users_total: int
    users_new_today: int
    users_new_7d: int
    superadmins: int
    dau: int
    wau: int
    mau: int
    logins_today: int
    logouts_today: int
    estimates_total: int
    estimates_today: int
    series: list[SeriesPoint]
    recent: list[RecentUser]
    # Daromad — berilgan kreditlar daftaridan. `users.credits` dan hisoblab
    # bo'lmaydi: u ishlatilgani sari kamayadi va "qancha tushdi" ni bilmaydi.
    revenue_total_usd: float
    revenue_today_usd: float
    revenue_30d_usd: float
    credits_sold: int
    credits_outstanding: int
    pending_requests: int
    price_per_analysis_usd: float
    generated_at: str


def _day_start(offset_days: int = 0) -> datetime:
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start - timedelta(days=offset_days)


async def _distinct_actors(db: AsyncSession, since: datetime) -> int:
    stmt = select(func.count(func.distinct(LoginEvent.user_id))).where(LoginEvent.created_at >= since)
    return (await db.execute(stmt)).scalar() or 0


async def _events_since(db: AsyncSession, kind: EventKind, since: datetime) -> int:
    stmt = select(func.count(LoginEvent.id)).where(
        LoginEvent.kind == kind, LoginEvent.created_at >= since
    )
    return (await db.execute(stmt)).scalar() or 0


@router.get("/stats", response_model=StatsOut, summary="Panel ko'rsatkichlari")
async def stats(
    days: int = Query(14, ge=7, le=90, description="Grafik uchun kunlar soni"),
    _: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
) -> StatsOut:
    today = _day_start()
    week = _day_start(7)
    month = _day_start(30)

    users_total = (await db.execute(select(func.count(User.id)))).scalar() or 0
    users_new_today = (
        await db.execute(select(func.count(User.id)).where(User.created_at >= today))
    ).scalar() or 0
    users_new_7d = (
        await db.execute(select(func.count(User.id)).where(User.created_at >= week))
    ).scalar() or 0
    superadmins = (
        await db.execute(select(func.count(User.id)).where(User.role == Role.SUPERADMIN))
    ).scalar() or 0

    estimates_total = (
        await db.execute(select(func.count(LoginEvent.id)).where(LoginEvent.kind == EventKind.ESTIMATE))
    ).scalar() or 0

    # Kunlik qator. SQLite va Postgres sanani turlicha kesadi, shuning uchun
    # oraliqlar Python tomonda beriladi — bu ikkala bazada ham bir xil ishlaydi
    # va 90 kungacha bo'lgan oynada so'rovlar soni ham qabul qilinarli.
    series: list[SeriesPoint] = []
    for offset in range(days - 1, -1, -1):
        start = _day_start(offset)
        end = start + timedelta(days=1)
        registered = (
            await db.execute(
                select(func.count(User.id)).where(User.created_at >= start, User.created_at < end)
            )
        ).scalar() or 0
        active = (
            await db.execute(
                select(func.count(func.distinct(LoginEvent.user_id))).where(
                    LoginEvent.created_at >= start, LoginEvent.created_at < end
                )
            )
        ).scalar() or 0
        est = (
            await db.execute(
                select(func.count(LoginEvent.id)).where(
                    LoginEvent.kind == EventKind.ESTIMATE,
                    LoginEvent.created_at >= start,
                    LoginEvent.created_at < end,
                )
            )
        ).scalar() or 0
        series.append(SeriesPoint(date=start.date().isoformat(), registered=registered, active=active, estimates=est))

    rows = (
        await db.execute(select(User).order_by(User.created_at.desc()).limit(20))
    ).scalars().all()

    async def _revenue(since: datetime | None) -> float:
        stmt = select(func.coalesce(func.sum(CreditGrant.amount_usd), 0.0))
        if since is not None:
            stmt = stmt.where(CreditGrant.created_at >= since)
        return float((await db.execute(stmt)).scalar() or 0.0)

    credits_sold = int(
        (await db.execute(select(func.coalesce(func.sum(CreditGrant.credits), 0)))).scalar() or 0
    )
    credits_outstanding = int(
        (await db.execute(select(func.coalesce(func.sum(User.credits), 0)))).scalar() or 0
    )
    pending_requests = int(
        (
            await db.execute(
                select(func.count(UpgradeRequest.id)).where(
                    UpgradeRequest.status == RequestStatus.PENDING
                )
            )
        ).scalar()
        or 0
    )

    return StatsOut(
        users_total=users_total,
        users_new_today=users_new_today,
        users_new_7d=users_new_7d,
        superadmins=superadmins,
        dau=await _distinct_actors(db, today),
        wau=await _distinct_actors(db, week),
        mau=await _distinct_actors(db, month),
        logins_today=await _events_since(db, EventKind.LOGIN, today),
        logouts_today=await _events_since(db, EventKind.LOGOUT, today),
        estimates_total=estimates_total,
        estimates_today=await _events_since(db, EventKind.ESTIMATE, today),
        series=series,
        recent=[
            RecentUser(
                id=u.id,
                email=u.email,
                name=u.name,
                role=u.role.value,
                created_at=u.created_at.isoformat(),
                last_seen_at=u.last_seen_at.isoformat() if u.last_seen_at else None,
                estimates_used=u.estimates_used,
                plan=u.plan.value,
                telegram=u.telegram or "",
                credits=u.credits or 0,
            )
            for u in rows
        ],
        revenue_total_usd=round(await _revenue(None), 2),
        revenue_today_usd=round(await _revenue(today), 2),
        revenue_30d_usd=round(await _revenue(month), 2),
        credits_sold=credits_sold,
        credits_outstanding=credits_outstanding,
        pending_requests=pending_requests,
        price_per_analysis_usd=PRICE_PER_ANALYSIS_USD,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
