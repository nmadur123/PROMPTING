"""Tahlil orkestratori — ML natijasini barcha tavsiyachilarga uzatadi.

Bitta joyda: tur aniqlash → signal ajratish → stack → server → UI/UX →
domen → to'lov. Router faqat shu funksiyani chaqiradi.
"""

from __future__ import annotations

from app.ml.engine import CONFIDENCE_FLOOR, engine, extract_expected_users, extract_signals
from app.ml.training_data import PROJECT_LABELS
from app.services import (
    domain_service,
    mobile_platform,
    payment_service,
    server_calculator,
    stack_recommender,
    uiux_advisor,
)

from app.copy import tr


async def analyze(
    description: str,
    monthly_users: int = 1000,
    region: str = "uz",
    lang: str = "uz",
    brand_hint: str = "",
    check_domains: bool = True,
    clarifications: list[dict] | None = None,
) -> dict:
    """To'liq tahlil qaytaradi (AnalyzeResponse shakli).

    `clarifications` — aniqlashtiruvchi savollarga berilgan javoblar. Ular
    tahlilga ta'sir qiladi: masalan mobil ilovada platforma ("Android",
    "iOS", "ikkalasi") aynan shu javobdan bilinadi va stack shunga qarab
    tanlanadi. Ilgari javoblar faqat promptga qo'shilardi, tahlil esa ularni
    ko'rmasdi — natijada foydalanuvchi "faqat Android" desa ham stackda
    kross-platforma yechimi turardi.
    """
    prediction = engine.classifier.predict(description)
    signals = extract_signals(description)

    # Matnda foydalanuvchi soni aytilgan bo'lsa, u so'rovnomadagi qiymatdan ustun:
    # foydalanuvchi o'z g'oyasini yozganda aniqroq raqam beradi.
    detected = extract_expected_users(description)
    effective_users = detected or monthly_users

    requirements = server_calculator.calculate_requirements(
        prediction.label, effective_users, signals, lang
    )
    server_options = server_calculator.recommend_servers(requirements, region, lang)  # type: ignore[arg-type]
    scaling = server_calculator.scaling_advice(requirements, prediction.label, lang)

    # DIQQAT: faqat JAVOB matni olinadi, savol emas. Platforma savolining
    # o'zida "Android, iOS yoki ikkalasi" degan so'zlar bor — savolni ham
    # qo'shsak, foydalanuvchi nima javob berishidan qat'i nazar ikkala
    # platforma topilib, natija doim "both" bo'lib chiqardi.
    answers_text = "\n".join(
        str(c.get("answer") or "") for c in (clarifications or [])
    )
    platform = mobile_platform.detect(description, answers_text)
    is_mobile = mobile_platform.is_mobile_product(description, prediction.label)

    stack = stack_recommender.recommend_stack(
        prediction.label, signals, effective_users, requirements.db_size_gb, region, lang,
        platform=platform,
        mobile=is_mobile,
    )
    anti = stack_recommender.anti_recommendations(prediction.label, signals, effective_users, lang)
    uiux = uiux_advisor.advise(prediction.label, signals, effective_users, lang)
    payment = payment_service.recommend_payment(prediction.label, signals, region, lang)

    domains: list[dict] = []
    if check_domains:
        found = await domain_service.find_domains(description, prediction.label, brand_hint)
        domains = [d.to_dict() for d in found]

    needs_clarification = prediction.confidence < CONFIDENCE_FLOOR

    return {
        "project_type": prediction.label,
        "project_title": PROJECT_LABELS.get(prediction.label, {}).get(lang, prediction.label),
        "confidence": round(prediction.confidence, 4),
        "alternatives": [
            {
                "label": label,
                "title": PROJECT_LABELS.get(label, {}).get(lang, label),
                "confidence": round(score, 4),
            }
            for label, score in prediction.alternatives
        ],
        "signals": signals,
        "detected_users": detected,
        "monthly_users": effective_users,
        "stack": [c.to_dict() for c in stack],
        "anti_patterns": anti,
        "requirements": requirements.to_dict(),
        "server_options": [
            {
                "tier": o.tier,
                "plan": o.plan,
                "fits": o.fits,
                "headroom_pct": o.headroom_pct,
                "monthly_total_usd": o.monthly_total_usd,
                "addons": o.addons,
                "reasoning": o.reasoning,
            }
            for o in server_options
        ],
        "scaling": scaling,
        "prices_updated": server_calculator.PRICES_UPDATED,
        "uiux": uiux,
        "domains": domains,
        "registrars": domain_service.registrar_advice(region),
        "payment": payment,
        "needs_clarification": needs_clarification,
        "clarify_question": tr("clarify", lang) if needs_clarification else None,
    }
