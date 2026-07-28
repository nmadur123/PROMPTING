"""Ko'p provayderli LLM qatlami — biri ishlamasa ikkinchisiga o'tadi.

Nega kerak: bitta provayderning krediti tugashi butun mahsulotni to'xtatib
qo'yardi. Endi OpenRouter tugasa Google (Gemini), Google tugasa OpenRouter
ishlaydi. Tartib sozlamadan olinadi.

Ikkala provayder ham ishlamasa `LLMError` ko'tariladi va yuqoridagi kod
o'zining zaxirasiga o'tadi (savollar uchun statik ro'yxat, prompt uchun
deterministik skelet) — ya'ni mahsulot baribir javob beradi.

Xabar formati OpenAI uslubida (`[{"role": ..., "content": ...}]`), chunki kod
allaqachon shunga qurilgan. Gemini boshqa format kutadi, shuning uchun
`_to_gemini()` tarjima qiladi.
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
) -> tuple[str, str]:
    """Birinchi ishlagan provayderning javobini qaytaradi.

    Qaytaradi: `(matn, provayder_nomi)`. Provayder nomi jurnalga va interfeysga
    kerak — foydalanuvchi javobni kim yozganini bilishi mumkin.

    Barcha provayderlar ishlamasa `LLMError` ko'tariladi.
    """
    settings = get_settings()
    order = [p.strip().lower() for p in settings.llm_providers.split(",") if p.strip()]
    errors: list[str] = []

    for provider in order:
        try:
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
