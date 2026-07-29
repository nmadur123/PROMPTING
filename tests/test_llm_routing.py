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
    """Har bir provayderni chaqirilganini yozib qo'yadigan qo'g'irchoq bilan almashtiradi."""
    seen: list[str] = []

    async def anthropic(messages, max_tokens):
        seen.append("anthropic")
        raise llm.LLMError("qo'g'irchoq: anthropic o'tkazib yuborildi")

    async def openrouter(messages, max_tokens, temperature, min_useful):
        seen.append("openrouter")
        raise llm.LLMError("qo'g'irchoq: openrouter o'tkazib yuborildi")

    async def google(messages, max_tokens, temperature):
        seen.append("google")
        return "javob"

    monkeypatch.setattr(llm, "_complete_anthropic", anthropic)
    monkeypatch.setattr(llm, "_complete_openrouter", openrouter)
    monkeypatch.setattr(llm, "_complete_google", google)
    return seen


async def test_heavy_chain_tries_anthropic_first(calls):
    text, provider = await llm.complete(_MSG, heavy=True)
    assert calls[0] == "anthropic"
    # Ikkalasi ham yiqilgach Google javob berdi — zanjir oxirigacha bordi.
    assert calls == ["anthropic", "openrouter", "google"]
    assert (text, provider) == ("javob", "google")


async def test_light_chain_skips_anthropic(calls):
    """Qisqa so'rovlar qimmat modelga bormasligi kerak."""
    await llm.complete(_MSG)
    assert "anthropic" not in calls
    assert calls == ["openrouter", "google"]


async def test_all_providers_failing_raises(calls, monkeypatch):
    async def google(messages, max_tokens, temperature):
        calls.append("google")
        raise llm.LLMError("qo'g'irchoq: google ham yiqildi")

    monkeypatch.setattr(llm, "_complete_google", google)

    with pytest.raises(llm.LLMError) as exc:
        await llm.complete(_MSG, heavy=True)
    # Xabarda har bir provayderning sababi qolsin — aks holda nosozlikni
    # topish uchun jurnalga qarashdan boshqa yo'l bo'lmaydi.
    for provider in ("anthropic", "openrouter", "google"):
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
