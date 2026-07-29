"""LLM zanjiri marshrutlash testlari — tashqi tarmoqqa chiqmaydi.

Bu yerda provayderlarning o'zi emas, `complete()` ning *qaysi* provayderni
*qaysi tartibda* chaqirishi tekshiriladi. Sabab: `heavy` bayrog'i noto'g'ri
ulansa, to'liq texnik topshiriq jimgina arzon modelga tushib ketadi va buni
faqat natija sifati pasayganidan sezish mumkin bo'ladi — ya'ni kech.
"""

from __future__ import annotations

import pytest

from app.services import llm

_MSG = [{"role": "user", "content": "salom"}]


@pytest.fixture
def calls(monkeypatch):
    """Har bir provayderni chaqirilganini yozib qo'yadigan qo'g'irchoq bilan almashtiradi.

    Zanjirlar ham shu yerda aniq belgilanadi. Sozlamalar `.env` dan o'qiladi
    va u repo'ga kirmaydi — testlar unga tayansa, boshqa mashinada yoki CI'da
    kod o'zgarmagan holda yiqilardi.
    """
    seen: list[str] = []

    settings = llm.get_settings()
    monkeypatch.setattr(settings, "llm_providers", "tokenmix,openrouter,google", raising=False)
    monkeypatch.setattr(
        settings, "llm_providers_heavy", "tokenmix,anthropic,openrouter,google", raising=False
    )

    async def tokenmix(messages, max_tokens, temperature, heavy):
        seen.append(f"tokenmix:{'heavy' if heavy else 'light'}")
        raise llm.LLMError("qo'g'irchoq: tokenmix o'tkazib yuborildi")

    async def anthropic(messages, max_tokens):
        seen.append("anthropic")
        raise llm.LLMError("qo'g'irchoq: anthropic o'tkazib yuborildi")

    async def openrouter(messages, max_tokens, temperature, min_useful):
        seen.append("openrouter")
        raise llm.LLMError("qo'g'irchoq: openrouter o'tkazib yuborildi")

    async def google(messages, max_tokens, temperature):
        seen.append("google")
        return "javob"

    monkeypatch.setattr(llm, "_complete_tokenmix", tokenmix)
    monkeypatch.setattr(llm, "_complete_anthropic", anthropic)
    monkeypatch.setattr(llm, "_complete_openrouter", openrouter)
    monkeypatch.setattr(llm, "_complete_google", google)
    return seen


async def test_heavy_chain_order(calls):
    text, provider = await llm.complete(_MSG, heavy=True)
    # Hammasi yiqilgach Google javob berdi — zanjir oxirigacha bordi.
    assert calls == ["tokenmix:heavy", "anthropic", "openrouter", "google"]
    assert (text, provider) == ("javob", "google")


async def test_light_chain_skips_anthropic(calls):
    """Qisqa so'rovlar qimmat modelga bormasligi kerak."""
    await llm.complete(_MSG)
    assert "anthropic" not in calls
    assert calls == ["tokenmix:light", "openrouter", "google"]


async def test_tokenmix_gets_heavy_flag_through(calls):
    """`heavy` TokenMix'gacha yetib borsin — u model tanlashda shunga qaraydi.

    Bayroq yo'qolsa to'liq texnik topshiriq jimgina arzon modelga yozilardi.
    """
    await llm.complete(_MSG, heavy=True)
    assert calls[0] == "tokenmix:heavy"
    calls.clear()
    await llm.complete(_MSG, heavy=False)
    assert calls[0] == "tokenmix:light"


async def test_all_providers_failing_raises(calls, monkeypatch):
    async def google(messages, max_tokens, temperature):
        calls.append("google")
        raise llm.LLMError("qo'g'irchoq: google ham yiqildi")

    monkeypatch.setattr(llm, "_complete_google", google)

    with pytest.raises(llm.LLMError) as exc:
        await llm.complete(_MSG, heavy=True)
    # Xabarda har bir provayderning sababi qolsin — aks holda nosozlikni
    # topish uchun jurnalga qarashdan boshqa yo'l bo'lmaydi.
    for provider in ("tokenmix", "anthropic", "openrouter", "google"):
        assert provider in str(exc.value)


async def test_unknown_provider_is_skipped_not_fatal(calls, monkeypatch):
    """Sozlamadagi imlo xatosi butun so'rovni yiqitmasin."""
    settings = llm.get_settings()
    monkeypatch.setattr(settings, "llm_providers_heavy", "antropic,google", raising=False)

    text, provider = await llm.complete(_MSG, heavy=True)
    assert (text, provider) == ("javob", "google")
    assert calls == ["google"]


# ------------------------------------------------------- Anthropic tarjimasi


def test_to_anthropic_splits_system_out():
    """Anthropic `system` ni `messages` ichida qabul qilmaydi — alohida chiqishi shart."""
    system, contents = llm._to_anthropic([
        {"role": "system", "content": "Sen texnik yozuvchisan."},
        {"role": "user", "content": "TT yoz"},
        {"role": "assistant", "content": "Mayli"},
        {"role": "system", "content": "Qisqa yoz."},
    ])
    assert system == "Sen texnik yozuvchisan.\n\nQisqa yoz."
    assert contents == [
        {"role": "user", "content": "TT yoz"},
        {"role": "assistant", "content": "Mayli"},
    ]


def test_to_anthropic_drops_empty_messages():
    # Bo'sh `content` API tomonidan rad etiladi (400).
    system, contents = llm._to_anthropic([
        {"role": "user", "content": ""},
        {"role": "user", "content": None},
        {"role": "user", "content": "bor"},
    ])
    assert system == ""
    assert contents == [{"role": "user", "content": "bor"}]


async def test_anthropic_without_key_raises_not_crashes(monkeypatch):
    """Kalitsiz o'rnatishda zanjir keyingisiga o'tsin, `KeyError` bermasin."""
    settings = llm.get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "", raising=False)

    with pytest.raises(llm.LLMError, match="ANTHROPIC_API_KEY"):
        await llm._complete_anthropic(_MSG, 100)


# ------------------------------------------------------------- TokenMix xatolari


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(payload or "")

    def json(self):
        return self._payload


class _FakeClient:
    """`httpx.AsyncClient` o'rnini bosadi — tarmoqqa chiqmaydi."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        return self._response


def _patch_httpx(monkeypatch, response):
    monkeypatch.setattr(llm.httpx, "AsyncClient", lambda **kw: _FakeClient(response))


async def test_tokenmix_model_not_allowed_gives_actionable_message(monkeypatch):
    """Eng ko'p uchraydigan xato aynan shu — xabar nima qilishni aytsin."""
    settings = llm.get_settings()
    monkeypatch.setattr(settings, "tokenmix_api_key", "sk-tm-test", raising=False)
    _patch_httpx(monkeypatch, _FakeResponse(
        400,
        text='{"error":{"message":"API key is not allowed to access the requested model"}}',
    ))

    with pytest.raises(llm.LLMError) as exc:
        await llm._complete_tokenmix(_MSG, 100, 0.3, heavy=True)
    assert "ruxsat berilmagan" in str(exc.value)
    assert "dashboard" in str(exc.value)


async def test_tokenmix_returns_text(monkeypatch):
    settings = llm.get_settings()
    monkeypatch.setattr(settings, "tokenmix_api_key", "sk-tm-test", raising=False)
    _patch_httpx(monkeypatch, _FakeResponse(
        200, {"choices": [{"message": {"content": "  javob  "}}]},
    ))

    assert await llm._complete_tokenmix(_MSG, 100, 0.3, heavy=True) == "javob"


async def test_tokenmix_empty_choices_raises(monkeypatch):
    # `choices[0]` ga to'g'ridan-to'g'ri murojaat IndexError berardi.
    settings = llm.get_settings()
    monkeypatch.setattr(settings, "tokenmix_api_key", "sk-tm-test", raising=False)
    _patch_httpx(monkeypatch, _FakeResponse(200, {"choices": []}))

    with pytest.raises(llm.LLMError, match="bo'sh javob"):
        await llm._complete_tokenmix(_MSG, 100, 0.3, heavy=True)


async def test_tokenmix_without_key_raises(monkeypatch):
    settings = llm.get_settings()
    monkeypatch.setattr(settings, "tokenmix_api_key", "", raising=False)

    with pytest.raises(llm.LLMError, match="TOKENMIX_API_KEY"):
        await llm._complete_tokenmix(_MSG, 100, 0.3, heavy=True)
