"""Superadmin: foydalanuvchilarni boshqarish, kredit berish, tarif so'rovlari.

To'lovning o'zi tizim ichida emas — odam Telegram orqali to'laydi, superadmin
esa panelda kredit qo'shadi yoki tarifni almashtiradi. Shuning uchun bu yerda
to'lov provayderi yo'q, faqat qo'lda boshqaruv va daromad daftari.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.billing import CreditGrant, RequestStatus, UpgradeRequest
from app.models.user import (
    PLAN_PRICE_USD,
    PLAN_QUOTA,
    PRICE_PER_ANALYSIS_USD,
    Plan,
    User,
    is_vip,
    utcnow,
)
from app.routers.auth import optional_user, require_superadmin
from app.services.quota import get_quota

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Admin billing"])


# --------------------------------------------------------------------------- #
# Sxemalar
# --------------------------------------------------------------------------- #


class AdminUserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str
    plan: str
    telegram: str
    credits: int
    estimates_used: int
    quota_used: int
    quota_limit: int | None
    total_remaining: int | None
    unlimited: bool
    created_at: str
    last_seen_at: str | None


class GrantIn(BaseModel):
    credits: int = Field(ge=1, le=100_000, description="Nechta analiz qo'shilsin")
    # Chegirmali paket berilsa summa kredit soniga teng bo'lmasligi mumkin.
    # Bo'sh qoldirilsa 1 analiz = 1 dollar hisobida yoziladi.
    amount_usd: float | None = Field(default=None, ge=0, le=1_000_000)
    telegram: str = Field(default="", max_length=64)
    note: str = Field(default="", max_length=300)


class PlanIn(BaseModel):
    plan: str = Field(description="sketch | drafting | studio | enterprise")


class UserPatchIn(BaseModel):
    """Tahrirlanadigan maydonlar. Berilmagani o'zgarmaydi."""

    name: str | None = Field(default=None, max_length=120)
    telegram: str | None = Field(default=None, max_length=64)
    credits: int | None = Field(default=None, ge=0, le=100_000)
    is_active: bool | None = None


class UpgradeRequestIn(BaseModel):
    email: EmailStr
    telegram: str = Field(default="", max_length=64)
    plan: str = Field(default="", max_length=32)
    credits: int = Field(default=0, ge=0, le=100_000)
    message: str = Field(default="", max_length=1000)


class UpgradeRequestOut(BaseModel):
    id: str
    email: str
    telegram: str
    plan: str
    credits: int
    message: str
    status: str
    created_at: str
    handled_at: str | None
    user_found: bool


# --------------------------------------------------------------------------- #
# Yordamchilar
# --------------------------------------------------------------------------- #


def _clean_telegram(value: str) -> str:
    """@ belgisi va havola prefikslarini olib tashlaydi — bazada bir xil ko'rinish."""
    v = (value or "").strip()
    for prefix in ("https://t.me/", "http://t.me/", "t.me/", "@"):
        if v.lower().startswith(prefix.lower()):
            v = v[len(prefix):]
    return v.strip()[:64]


async def _to_admin_user(db: AsyncSession, u: User) -> AdminUserOut:
    q = await get_quota(db, u)
    return AdminUserOut(
        id=u.id,
        email=u.email,
        name=u.name,
        role=u.role.value,
        plan=u.plan.value,
        telegram=u.telegram or "",
        credits=u.credits or 0,
        estimates_used=u.estimates_used,
        quota_used=q.used,
        quota_limit=q.limit,
        total_remaining=q.total_remaining,
        unlimited=q.unlimited or q.limit is None,
        created_at=u.created_at.isoformat(),
        last_seen_at=u.last_seen_at.isoformat() if u.last_seen_at else None,
    )


# --------------------------------------------------------------------------- #
# Foydalanuvchilar
# --------------------------------------------------------------------------- #


@router.get("/admin/users", response_model=list[AdminUserOut], summary="Foydalanuvchilar ro'yxati")
async def list_users(
    q: str = Query(default="", max_length=120, description="Email, ism yoki Telegram bo'yicha qidiruv"),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_superadmin),
) -> list[AdminUserOut]:
    stmt = select(User)
    term = q.strip()
    if term:
        like = f"%{term.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(User.email).like(like),
                func.lower(User.name).like(like),
                func.lower(User.telegram).like(like),
            )
        )
    stmt = stmt.order_by(User.created_at.desc()).limit(limit)
    rows = (await db.execute(stmt)).scalars().all()
    return [await _to_admin_user(db, u) for u in rows]


@router.post("/admin/users/{user_id}/grant", response_model=AdminUserOut, summary="Kredit berish")
async def grant_credits(
    user_id: str,
    body: GrantIn,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superadmin),
) -> AdminUserOut:
    """Foydalanuvchiga analiz krediti qo'shadi va daromad daftariga yozadi."""
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Foydalanuvchi topilmadi")

    amount = body.amount_usd if body.amount_usd is not None else body.credits * PRICE_PER_ANALYSIS_USD

    user.credits = (user.credits or 0) + body.credits
    if body.telegram:
        user.telegram = _clean_telegram(body.telegram)

    db.add(CreditGrant(
        user_id=user.id,
        granted_by=admin.id,
        credits=body.credits,
        amount_usd=round(amount, 2),
        note=body.note.strip()[:300],
    ))
    await db.commit()
    await db.refresh(user)
    return await _to_admin_user(db, user)


@router.post("/admin/users/{user_id}/plan", response_model=AdminUserOut, summary="Tarifni almashtirish")
async def change_plan(
    user_id: str,
    body: PlanIn,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_superadmin),
) -> AdminUserOut:
    try:
        plan = Plan(body.plan)
    except ValueError:
        allowed = ", ".join(p.value for p in Plan)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Noma'lum tarif. Ruxsat etilgan: {allowed}")

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Foydalanuvchi topilmadi")

    user.plan = plan
    await db.commit()
    await db.refresh(user)
    return await _to_admin_user(db, user)


@router.patch("/admin/users/{user_id}", response_model=AdminUserOut, summary="Foydalanuvchini tahrirlash")
async def update_user(
    user_id: str,
    body: UserPatchIn,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_superadmin),
) -> AdminUserOut:
    """Ism, Telegram, kredit va faollikni o'zgartiradi.

    Kredit bu yerda TO'G'RIDAN-TO'G'RI o'rnatiladi (tuzatish uchun) va daromad
    daftariga yozilmaydi — sotuv `/grant` orqali qilinadi. Ikkalasini bitta
    joyga qo'shsak, xatoni tuzatish daromadni ham o'zgartirib yuborardi.
    """
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Foydalanuvchi topilmadi")

    if body.name is not None:
        user.name = body.name.strip()
    if body.telegram is not None:
        user.telegram = _clean_telegram(body.telegram)
    if body.credits is not None:
        user.credits = body.credits
    if body.is_active is not None:
        # VIP hisobni bloklab bo'lmaydi — aks holda superadmin o'zini
        # panelidan chiqarib yuborishi va qaytib kira olmasligi mumkin.
        if not body.is_active and is_vip(user.email):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Asosiy hisobni bloklab bo'lmaydi")
        user.is_active = body.is_active

    await db.commit()
    await db.refresh(user)
    return await _to_admin_user(db, user)


@router.delete("/admin/users/{user_id}", summary="Foydalanuvchini o'chirish")
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superadmin),
) -> dict:
    """Foydalanuvchini va unga bog'liq barcha yozuvlarni o'chiradi.

    Qaytarib bo'lmaydi. Ikki himoya bor: VIP hisobni va o'zini o'chirib
    bo'lmaydi — ikkalasi ham paneldan butunlay chiqib qolishga olib keladi.
    """
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Foydalanuvchi topilmadi")
    if is_vip(user.email):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Asosiy hisobni o'chirib bo'lmaydi")
    if user.id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "O'z hisobingizni o'chirib bo'lmaydi")

    email = user.email
    await db.delete(user)  # bog'liq yozuvlar ON DELETE CASCADE bilan ketadi
    await db.commit()
    return {"deleted": True, "email": email}


# --------------------------------------------------------------------------- #
# Tarif so'rovlari
# --------------------------------------------------------------------------- #


@router.post("/upgrade-request", response_model=UpgradeRequestOut, summary="Tarif so'rovi yuborish")
async def create_request(
    body: UpgradeRequestIn,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(optional_user),
) -> UpgradeRequestOut:
    """Foydalanuvchi tarif yoki kredit so'raydi; superadmin panelda ko'radi.

    Kirish talab qilinmaydi: odam boshqa qurilmadan yozishi mumkin. Email
    bo'yicha hisob topilsa bog'lab qo'yamiz, topilmasa ham so'rov saqlanadi —
    superadmin uni qo'lda hal qiladi.
    """
    email = str(body.email).lower().strip()
    linked = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()

    record = UpgradeRequest(
        user_id=(linked.id if linked else (user.id if user else None)),
        email=email,
        telegram=_clean_telegram(body.telegram),
        plan=body.plan.strip()[:32],
        credits=body.credits,
        message=body.message.strip()[:1000],
    )
    db.add(record)

    # Telegram username kelgan bo'lsa hisobga ham yozib qo'yamiz — superadmin
    # keyin uni qidiruvdan topa oladi.
    if linked and record.telegram and not linked.telegram:
        linked.telegram = record.telegram

    await db.commit()
    await db.refresh(record)

    return UpgradeRequestOut(
        id=record.id,
        email=record.email,
        telegram=record.telegram,
        plan=record.plan,
        credits=record.credits,
        message=record.message,
        status=record.status.value,
        created_at=record.created_at.isoformat(),
        handled_at=None,
        user_found=linked is not None,
    )


@router.get("/admin/requests", response_model=list[UpgradeRequestOut], summary="Tarif so'rovlari")
async def list_requests(
    status_filter: str = Query(default="pending", pattern="^(pending|approved|rejected|all)$"),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_superadmin),
) -> list[UpgradeRequestOut]:
    stmt = select(UpgradeRequest)
    if status_filter != "all":
        stmt = stmt.where(UpgradeRequest.status == RequestStatus(status_filter))
    stmt = stmt.order_by(UpgradeRequest.created_at.desc()).limit(limit)
    rows = (await db.execute(stmt)).scalars().all()

    return [
        UpgradeRequestOut(
            id=r.id,
            email=r.email,
            telegram=r.telegram,
            plan=r.plan,
            credits=r.credits,
            message=r.message,
            status=r.status.value,
            created_at=r.created_at.isoformat(),
            handled_at=r.handled_at.isoformat() if r.handled_at else None,
            user_found=r.user_id is not None,
        )
        for r in rows
    ]


@router.post("/admin/requests/{request_id}/resolve", summary="So'rovni yopish")
async def resolve_request(
    request_id: str,
    approve: bool = Query(default=True),
    credits: int = Query(default=0, ge=0, le=100_000, description="Tasdiqlashda beriladigan kredit"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superadmin),
) -> dict:
    """So'rovni tasdiqlaydi yoki rad etadi.

    Tasdiqlashda `credits` berilsa, o'sha zahoti hisobga qo'shiladi va daromad
    daftariga yoziladi — superadmin ikkinchi formani ochib o'tirmasin.
    """
    record = (
        await db.execute(select(UpgradeRequest).where(UpgradeRequest.id == request_id))
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "So'rov topilmadi")
    if record.status is not RequestStatus.PENDING:
        raise HTTPException(status.HTTP_409_CONFLICT, "Bu so'rov allaqachon yopilgan")

    record.status = RequestStatus.APPROVED if approve else RequestStatus.REJECTED
    record.handled_at = utcnow()
    record.handled_by = admin.id

    granted = 0
    if approve and credits > 0:
        target = (
            await db.execute(select(User).where(User.email == record.email))
        ).scalar_one_or_none()
        if target is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"{record.email} bo'yicha hisob topilmadi — avval ro'yxatdan o'tsin",
            )
        target.credits = (target.credits or 0) + credits
        if record.telegram and not target.telegram:
            target.telegram = record.telegram
        db.add(CreditGrant(
            user_id=target.id,
            granted_by=admin.id,
            credits=credits,
            amount_usd=round(credits * PRICE_PER_ANALYSIS_USD, 2),
            note=f"So'rov {record.id}",
        ))
        granted = credits

    await db.commit()
    return {"id": record.id, "status": record.status.value, "credits_granted": granted}


# --------------------------------------------------------------------------- #
# Narx ma'lumoti — interfeys uchun
# --------------------------------------------------------------------------- #


@router.get("/pricing-info", summary="Tarif va aloqa ma'lumoti")
async def pricing_info() -> dict:
    """Narxlar sahifasi shu yerdan oladi — raqamlar ikki joyda yozilmasin."""
    from app.config import get_settings

    return {
        "price_per_analysis_usd": PRICE_PER_ANALYSIS_USD,
        "telegram": get_settings().support_telegram,
        "plans": [
            {
                "id": p.value,
                "monthly_limit": PLAN_QUOTA.get(p),
                "price_usd": PLAN_PRICE_USD.get(p),
            }
            for p in Plan
        ],
    }
