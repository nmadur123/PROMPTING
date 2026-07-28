"""Foydalanuvchining saqlangan hisob-kitoblari.

Tarix, sidebar va profil raqamlari shu endpointlardan keladi — interfeysda
qo'lda yozilgan son qolmasligi uchun.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.estimate import Estimate
from app.models.user import User
from app.routers.auth import current_user
from app.services.quota import get_quota

router = APIRouter(prefix="/api/estimates", tags=["Estimates"])


class EstimateSummary(BaseModel):
    id: str
    title: str
    project_title: str
    monthly_usd: float
    stack: list[str]
    created_at: str


class EstimateDetail(EstimateSummary):
    description: str
    lang: str
    region: str
    monthly_users: int
    confidence: int
    analysis: dict


class EstimateList(BaseModel):
    items: list[EstimateSummary]
    total: int
    quota_used: int
    quota_limit: int | None
    quota_remaining: int | None
    resets_at: str


def _summary(row: Estimate) -> EstimateSummary:
    return EstimateSummary(
        id=row.id,
        title=row.title,
        project_title=row.project_title,
        monthly_usd=row.monthly_usd,
        stack=list(row.stack or []),
        created_at=row.created_at.isoformat(),
    )


@router.get("", response_model=EstimateList, summary="Mening hisob-kitoblarim")
async def list_estimates(
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> EstimateList:
    rows = (
        await db.execute(
            select(Estimate)
            .where(Estimate.user_id == user.id)
            .order_by(Estimate.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()

    total = (
        await db.execute(select(func.count(Estimate.id)).where(Estimate.user_id == user.id))
    ).scalar() or 0

    quota = await get_quota(db, user)
    return EstimateList(
        items=[_summary(r) for r in rows],
        total=total,
        quota_used=quota.used,
        quota_limit=quota.limit,
        quota_remaining=quota.remaining,
        resets_at=quota.resets_at,
    )


@router.get("/{estimate_id}", response_model=EstimateDetail, summary="Bitta hisob-kitob")
async def get_estimate(
    estimate_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> EstimateDetail:
    row = (
        await db.execute(
            # user_id shartsiz qidirish boshqa odamning hisob-kitobini ochib
            # berardi — id taxmin qilinadigan bo'lmasa ham, bu tekshiruv
            # bo'lishi shart.
            select(Estimate).where(Estimate.id == estimate_id, Estimate.user_id == user.id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Topilmadi")

    base = _summary(row)
    return EstimateDetail(
        **base.model_dump(),
        description=row.description,
        lang=row.lang,
        region=row.region,
        monthly_users=row.monthly_users,
        confidence=row.confidence,
        analysis=row.analysis or {},
    )


@router.delete("/{estimate_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_estimate(
    estimate_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    row = (
        await db.execute(
            select(Estimate).where(Estimate.id == estimate_id, Estimate.user_id == user.id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Topilmadi")
    await db.delete(row)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
