"""Asosiy oqim: tahlil → prompt generatsiya → saqlangan natijani ochish."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.blueprint import Blueprint
from app.models.estimate import Estimate
from app.models.user import EventKind, LoginEvent, User, utcnow
from app.routers.auth import current_user, optional_user
from app.services import quota as quota_service
from app.services.quota import get_quota
from app.services.summarize import monthly_total, stack_chips, title_from
from app.schemas.blueprint import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClarifyRequest,
    ClarifyResponse,
    GenerateRequest,
    GenerateResponse,
)
from app.ml.engine import engine
from app.ml.training_data import PROJECT_LABELS
from app.services import analyzer, clarifier, prompt_builder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Blueprint"])


@router.post("/analyze", response_model=AnalyzeResponse, summary="Startup g'oyasini tahlil qilish")
async def analyze_idea(
    body: AnalyzeRequest,
    request: Request,
    user: User | None = Depends(optional_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """ML orqali turni aniqlaydi va stack / server / UI-UX / domen tavsiyalarini qaytaradi.

    Ro'yxatdan o'tish majburiy va kvota server tomonda tekshiriladi — faqat
    interfeysda to'sish hech narsani to'smaydi, `curl` bilan aylanib o'tiladi.
    """
    if user:
        quota = await get_quota(db, user)
        if quota.exhausted:
            raise HTTPException(
                status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "code": "quota_exhausted",
                    "plan": quota.plan,
                    "limit": quota.limit,
                    "used": quota.used,
                    "resets_at": quota.resets_at,
                },
            )

    result = await analyzer.analyze(
        description=body.description,
        monthly_users=body.monthly_users,
        region=body.region,
        lang=body.lang,
        brand_hint=body.brand_hint,
        check_domains=body.check_domains,
    )

    # Yozib bo'lmasa ham javob berilishi kerak: foydalanuvchi hisob-kitobni
    # ko'radi, faqat tarixda ko'rinmaydi. Buning teskarisi — javobni ushlab
    # turish — ancha yomon.
    try:
        db.add(Estimate(
            user_id=user.id,
            description=body.description,
            title=title_from(body.description),
            project_type=result["project_type"],
            project_title=result["project_title"],
            confidence=int(result["confidence"] * 100),
            lang=body.lang,
            region=body.region,
            monthly_users=result["monthly_users"],
            monthly_usd=monthly_total(result),
            stack=stack_chips(result),
            analysis=result,
        ))
        # Oylik limit avval, keyin kredit — yechish mantiqi bitta joyda
        await quota_service.consume(db, user, quota)
        user.last_seen_at = utcnow()
        db.add(LoginEvent(
            user_id=user.id,
            kind=EventKind.ESTIMATE,
            user_agent=(request.headers.get("user-agent") or "")[:200],
        ))
        await db.commit()
    except Exception:  # noqa: BLE001
        await db.rollback()
        logger.exception("Hisob-kitob saqlanmadi")

    return result


@router.post("/clarify", response_model=ClarifyResponse, summary="G'oyaga oid aniqlashtiruvchi savollar")
async def clarify_idea(
    body: ClarifyRequest,
    user: User | None = Depends(optional_user),
) -> dict:
    """G'oyani o'qib, arxitekturaga eng ko'p ta'sir qiladigan 3-6 ta savol qaytaradi.

    Kvota yechilmaydi: bu qadam hali hech narsa yaratmaydi, u faqat keyingi
    generatsiyani aniqroq qiladi. Kvotani shu yerda ham yechish bitta g'oyani
    ikki marta sanash bo'lardi.

    Turni aniqlash uchun to'liq `analyzer.analyze()` chaqirilmaydi — u domen
    bandligini tekshirish uchun tashqi RDAP so'rovlari qiladi va sekin. Bu yerda
    faqat lokal klassifikator kerak.
    """
    prediction = engine.classifier.predict(body.description)
    title = PROJECT_LABELS.get(prediction.label, {}).get(body.lang, prediction.label)

    result = await clarifier.ask(
        description=body.description,
        project_type=prediction.label,
        project_title=title,
        lang=body.lang,
        count=body.count,
    )

    return {
        "project_type": result.project_type,
        "project_title": result.project_title,
        "questions": [q.to_dict() for q in result.questions],
        "generated_by_llm": result.generated_by_llm,
        "warning": result.warning,
    }


@router.post("/generate", response_model=GenerateResponse, summary="Tayyor promptni yaratish")
async def generate_prompt(
    body: GenerateRequest,
    user: User | None = Depends(optional_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Tahlil qiladi va tanlangan model uchun moslangan yakuniy promptni qaytaradi.

    Kirish majburiy. Bu endpoint OpenRouter'ga so'rov yuboradi, ya'ni har chaqiruv
    hisobdan pul yechadi — ochiq qoldirilsa, kalitni istagan odam sarflab
    yuborishi mumkin. `/analyze` allaqachon shu sababdan yopiq edi.

    Kvota SHU YERDA tekshiriladi va yechiladi. Ilgari bu endpoint kvotaga
    umuman tegmasdi — "prompt hamisha tahlildan keyin olinadi" degan farazga
    tayangan edi. Chat sahifasi esa `/analyze` ni umuman chaqirmaydi, ya'ni
    har qanday foydalanuvchi cheksiz prompt yaratardi. Ustiga aynan shu
    endpoint OpenRouter'ga pul to'laydi.

    Ikki marta yechilmasligi uchun `already_charged()` tekshiriladi: landing
    sahifada bitta g'oya uchun avval `/analyze`, keyin `/generate` chaqiriladi.
    """
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Prompt yaratish uchun tizimga kiring.",
        )

    quota = await get_quota(db, user)
    charged = await quota_service.already_charged(db, user, body.description)
    if quota.exhausted and not charged:
        raise HTTPException(
            status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "code": "quota_exhausted",
                "plan": quota.plan,
                "limit": quota.limit,
                "used": quota.used,
                "credits": quota.credits,
                "resets_at": quota.resets_at,
            },
        )

    analysis = await analyzer.analyze(
        description=body.description,
        monthly_users=body.monthly_users,
        region=body.region,
        lang=body.lang,
        brand_hint=body.brand_hint,
        check_domains=body.check_domains,
    )

    ctx = prompt_builder.BuildContext(
        description=body.description,
        project_type=analysis["project_type"],
        project_title=analysis["project_title"],
        confidence=analysis["confidence"],
        signals=analysis["signals"],
        monthly_users=analysis["monthly_users"],
        stack=analysis["stack"],
        anti_patterns=analysis["anti_patterns"],
        requirements=analysis["requirements"],
        server_options=analysis["server_options"],
        scaling=analysis["scaling"],
        uiux=analysis["uiux"],
        domains=analysis["domains"],
        payment=analysis["payment"],
        target_model=body.target_model,
        target_model_name=body.target_model_name or body.target_model,
        lang=body.lang,
        # Javob berilmagan savollar promptga kirmaydi — bo'sh satr faqat
        # shovqin qo'shadi va modelni chalg'itadi.
        clarifications=[
            {"question": a.question.strip(), "answer": a.answer.strip()}
            for a in body.answers
            if a.answer and a.answer.strip()
        ],
    )

    result = await prompt_builder.generate(ctx)

    # Yechish generatsiya muvaffaqiyatli tugagach — model javob bermasa
    # foydalanuvchidan hisob yechilmasin.
    if not charged:
        try:
            db.add(Estimate(
                user_id=user.id,
                description=body.description,
                title=title_from(body.description),
                project_type=analysis["project_type"],
                project_title=analysis["project_title"],
                confidence=int(analysis["confidence"] * 100),
                lang=body.lang,
                region=body.region,
                monthly_users=analysis["monthly_users"],
                monthly_usd=monthly_total(analysis),
                stack=stack_chips(analysis),
                analysis=analysis,
            ))
            await quota_service.consume(db, user, quota)
            user.last_seen_at = utcnow()
            await db.commit()
        except Exception:  # noqa: BLE001 — yozilmasa ham natija berilsin
            await db.rollback()
            logger.exception("Kvota yechilmadi")

    saved_id = None
    if body.save:
        try:
            record = Blueprint(
                description=body.description,
                lang=body.lang,
                region=body.region,
                monthly_users=analysis["monthly_users"],
                project_type=analysis["project_type"],
                confidence=int(analysis["confidence"] * 100),
                target_model=body.target_model,
                analysis=analysis,
                prompt=result.prompt,
                used_llm=1 if result.used_llm else 0,
            )
            db.add(record)
            await db.commit()
            saved_id = record.id
        except Exception:  # noqa: BLE001 — saqlanmasa ham natija berilishi kerak
            await db.rollback()
            logger.exception("Blueprint saqlanmadi")

    return {
        "id": saved_id,
        "prompt": result.prompt,
        "skeleton": result.skeleton,
        "used_llm": result.used_llm,
        "target_model": result.target_model,
        "sources": result.sources,
        "warning": result.warning,
        "analysis": analysis,
    }


@router.get("/blueprints/{blueprint_id}", response_model=GenerateResponse, summary="Saqlangan natija")
async def get_blueprint(blueprint_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    record = (
        await db.execute(select(Blueprint).where(Blueprint.id == blueprint_id))
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Bunday natija topilmadi")

    return {
        "id": record.id,
        "prompt": record.prompt,
        "skeleton": record.prompt,
        "used_llm": bool(record.used_llm),
        "target_model": record.target_model,
        "sources": [],
        "warning": None,
        "analysis": record.analysis,
    }
