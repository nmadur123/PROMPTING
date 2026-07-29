"""Ko'p provayderli LLM qatlami — biri ishlamasa ikkinchisiga o'tadi.

Nega kerak: bitta provayderning krediti tugashi butun mahsulotni to'xtatib
qo'yardi. Endi Claude tugasa OpenRouter, OpenRouter tugasa Google (Gemini)
ishlaydi. Tartib sozlamadan olinadi.

Ikkita zanjir bor — `complete(heavy=True)` va `complete()`. Sababi iqtisodiy:
to'liq texnik topshiriq yozish (o'n minglab token, mahsulotning o'zi) va
beshta qisqa savol berish bir xil narxdagi modelga arzimaydi. Og'ir ish
Claude'ga, yengili Gemini/OpenRouterga ketadi.

Barcha provayderlar ishlamasa `LLMError` ko'tariladi va yuqoridagi kod
o'zining zaxirasiga o'tadi (savollar uchun statik ro'yxat, prompt uchun
deterministik skelet) — ya'ni mahsulot baribir javob beradi.

Xabar formati OpenAI uslubida (`[{"role": ..., "content": ...}]`), chunki kod
allaqachon shunga qurilgan. Gemini ham, Anthropic ham boshqa format kutadi —
`_to_gemini()` va `_to_anthropic()` tarjima qiladi.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import get_settings
from app.services import openrouter

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(180.0, connect=15.0)
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Claude Opus 5 da fikrlash (thinking) sukut bo'yicha yoqilgan va u ham
# `max_tokens` ichidan yeydi. Chaqiruvchi 12 000 token so'rasa, shuncha joy
# fikrlashga ketib javob yarmida uzilishi mumkin — shuning uchun so'ralgan
# chegaraga qo'shimcha joy beramiz.
_ANTHROPIC_THINKING_HEADROOM = 12_000
_ANTHROPIC_MAX_TOKENS_CAP = 64_000


class LLMError(RuntimeError):
    """Barcha provayderlar ishlamadi."""


# --------------------------------------------------------------------------- #
# Google Gemini
# --------------------------------------------------------------------------- #


def _to_gemini(messages: list[dict]) -> tuple[Optional[dict], list[dict]]:
    """OpenAI uslubidagi xabarlarni Gemini formatiga o'giradi.

    Gemini `system` rolini bilmaydi — u alohida `systemInstruction` maydonida
    beriladi. `assistant` esa u yerda `model` deb ataladi.
    """
    system_parts: list[str] = []
    contents: list[dict] = []

    for m in messages:
        role = m.get("role")
        text = str(m.get("content") or "")
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        else:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            })

    system = {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    return system, contents


async def _complete_google(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    settings = get_settings()
    key = settings.google_api_key
    if not key:
        raise LLMError("GOOGLE_API_KEY sozlanmagan")

    system, contents = _to_gemini(messages)
    if not contents:
        raise LLMError("Gemini uchun bo'sh so'rov")

    body: dict = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system:
        body["systemInstruction"] = system

    url = f"{_GEMINI_BASE}/models/{settings.google_model}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                url,
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=body,
            )
    except httpx.HTTPError as exc:
        raise LLMError(f"Gemini'ga ulanib bo'lmadi: {exc}") from exc

    if resp.status_code != 200:
        raise LLMError(f"Gemini xatosi {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        # Xavfsizlik filtri butun so'rovni bloklaganda `candidates` bo'lmaydi.
        reason = (data.get("promptFeedback") or {}).get("blockReason", "noma'lum")
        raise LLMError(f"Gemini javob bermadi (sabab: {reason})")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise LLMError(f"Gemini bo'sh javob qaytardi (finishReason: {candidates[0].get('finishReason')})")
    return text


# --------------------------------------------------------------------------- #
# Anthropic (Claude)
# --------------------------------------------------------------------------- #


def _to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI uslubidagi xabarlarni Anthropic formatiga o'giradi.

    Gemini'dagi kabi, `system` alohida maydonga chiqadi — Anthropic uni
    `messages` ichida qabul qilmaydi.
    """
    system_parts: list[str] = []
    contents: list[dict] = []

    for m in messages:
        role = m.get("role")
        text = str(m.get("content") or "")
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        else:
            contents.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": text,
            })

    return "\n\n".join(system_parts), contents


async def _complete_anthropic(messages: list[dict], max_tokens: int) -> str:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY sozlanmagan")

    # Import shu yerda: kalit qo'yilmagan o'rnatishlarda `anthropic` paketi
    # bo'lmasa ham qolgan provayderlar ishlayversin.
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover — faqat to'liqsiz o'rnatishda
        raise LLMError("`anthropic` paketi o'rnatilmagan") from exc

    system, contents = _to_anthropic(messages)
    if not contents:
        raise LLMError("Claude uchun bo'sh so'rov")

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    try:
        # Oqim (stream) shart: katta `max_tokens` da oddiy so'rov HTTP
        # taymautiga urilib uziladi. `get_final_message()` to'liq javobni
        # yig'ib beradi — hodisalarni qo'lda ushlash kerak emas.
        #
        # `temperature` ataylab berilmagan: Opus 5 uni qabul qilmaydi (400).
        async with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=min(max_tokens + _ANTHROPIC_THINKING_HEADROOM, _ANTHROPIC_MAX_TOKENS_CAP),
            system=system or anthropic.NOT_GIVEN,
            messages=contents,
            output_config={"effort": "high"},
        ) as stream:
            message = await stream.get_final_message()
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Claude xatosi {exc.status_code}: {str(exc)[:300]}") from exc
    except anthropic.APIError as exc:
        raise LLMError(f"Claude'ga ulanib bo'lmadi: {exc}") from exc

    # Xavfsizlik tasnifagichi so'rovni rad etsa javob HTTP 200 bo'ladi-yu,
    # `content` bo'sh keladi. `content[0]` ga to'g'ridan-to'g'ri murojaat
    # qilinsa shu yerda IndexError bo'lardi.
    if message.stop_reason == "refusal":
        category = getattr(message.stop_details, "category", None) or "noma'lum"
        raise LLMError(f"Claude so'rovni rad etdi (turkum: {category})")

    text = "".join(b.text for b in message.content if b.type == "text").strip()
    if not text:
        raise LLMError(f"Claude bo'sh javob qaytardi (stop_reason: {message.stop_reason})")
    return text


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #


async def _complete_openrouter(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    min_useful: Optional[int],
) -> str:
    try:
        text = await openrouter.client.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            min_useful=min_useful,
        )
    except openrouter.OpenRouterError as exc:
        raise LLMError(str(exc)) from exc
    if not text or not text.strip():
        raise LLMError("OpenRouter bo'sh javob qaytardi")
    return text.strip()


# --------------------------------------------------------------------------- #
# Umumiy kirish nuqtasi
# --------------------------------------------------------------------------- #


async def complete(
    messages: list[dict],
    max_tokens: int = 1500,
    temperature: float = 0.4,
    min_useful: Optional[int] = None,
    heavy: bool = False,
) -> tuple[str, str]:
    """Birinchi ishlagan provayderning javobini qaytaradi.

    Qaytaradi: `(matn, provayder_nomi)`. Provayder nomi jurnalga va interfeysga
    kerak — foydalanuvchi javobni kim yozganini bilishi mumkin.

    `heavy=True` — asosiy ish (to'liq texnik topshiriq). Kuchliroq va
    qimmatroq zanjir ishlatiladi. Standart `False` — qisqa yordamchi
    so'rovlar arzon modelga ketadi.

    Barcha provayderlar ishlamasa `LLMError` ko'tariladi.
    """
    settings = get_settings()
    raw_order = settings.llm_providers_heavy if heavy else settings.llm_providers
    order = [p.strip().lower() for p in raw_order.split(",") if p.strip()]
    errors: list[str] = []

    for provider in order:
        try:
            if provider in ("anthropic", "claude"):
                return await _complete_anthropic(messages, max_tokens), "anthropic"
            if provider == "openrouter":
                return await _complete_openrouter(messages, max_tokens, temperature, min_useful), provider
            if provider in ("google", "gemini"):
                return await _complete_google(messages, max_tokens, temperature), "google"
            logger.warning("Noma'lum LLM provayderi sozlamada: %s", provider)
        except LLMError as exc:
            # Keyingisiga o'tamiz — sabab jurnalda qoladi.
            logger.warning("%s ishlamadi: %s", provider, str(exc)[:200])
            errors.append(f"{provider}: {str(exc)[:120]}")

    raise LLMError("Barcha provayderlar ishlamadi — " + " | ".join(errors))
