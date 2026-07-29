"""LLM'siz yo'l testlari — mahsulotning standart rejimi.

`USE_LLM=false` da tashqi model umuman chaqirilmasligi kerak. Buni "shunchaki
ishladi" deb tekshirib bo'lmaydi: LLM zanjiri yiqilganda ham natija qaytadi,
ya'ni chaqiruv sodir bo'lgani sezilmay qolishi mumkin. Shuning uchun quyida
`llm.complete` ataylab portlaydigan qilib qo'yiladi — chaqirilsa test yiqiladi.
"""

from __future__ import annotations

import pytest

from app.services import clarifier, prompt_builder
from app.services.analyzer import analyze


@pytest.fixture
def no_llm(monkeypatch):
    """`USE_LLM=false` va `llm.complete` ni mina qilib qo'yadi."""
    settings = prompt_builder.get_settings()
    monkeypatch.setattr(settings, "use_llm", False, raising=False)

    async def boom(*args, **kwargs):
        raise AssertionError("LLM chaqirildi — USE_LLM=false bo'lsa chaqirilmasligi kerak")

    monkeypatch.setattr(prompt_builder.llm, "complete", boom)
    monkeypatch.setattr(clarifier.llm, "complete", boom)
    return settings


_IDEA = (
    "Sartaroshxonalar uchun onlayn navbat tizimi. Mijoz bo'sh vaqtni tanlaydi, "
    "karta orqali oldindan to'laydi, usta tasdiqlaydi, SMS eslatma keladi."
)


async def _context(idea: str = _IDEA) -> prompt_builder.BuildContext:
    """Tahlildan `BuildContext` yig'adi — router bilan bir xil maydonlar."""
    a = await analyze(description=idea, monthly_users=1000, check_domains=False)
    return prompt_builder.BuildContext(
        description=idea,
        project_type=a["project_type"],
        project_title=a["project_title"],
        confidence=a["confidence"],
        signals=a["signals"],
        monthly_users=a["monthly_users"],
        stack=a["stack"],
        anti_patterns=a["anti_patterns"],
        requirements=a["requirements"],
        server_options=a["server_options"],
        scaling=a["scaling"],
        uiux=a["uiux"],
        domains=a["domains"],
        payment=a["payment"],
        target_model="anthropic/claude-opus-5",
        target_model_name="Claude Opus 5",
        lang="uz",
    )


# --------------------------------------------------------------- texnik topshiriq


async def test_generate_produces_prompt_without_llm(no_llm):
    ctx = await _context()

    result = await prompt_builder.generate(ctx)

    assert result.used_llm is False
    # Eng muhimi: ogohlantirish BO'LMASLIGI kerak. Ilgari bu yo'l zaxira edi
    # va "AI ishlamayapti" deb belgilanardi — endi u asosiy yo'l.
    assert result.warning is None
    assert result.prompt == result.skeleton
    assert len(result.prompt) > 1000


async def test_generate_output_is_deterministic(no_llm):
    """Bir xil kirishga bir xil chiqish — LLM'siz yo'lning asosiy afzalligi."""
    ctx = await _context()

    first = await prompt_builder.generate(ctx)
    second = await prompt_builder.generate(ctx)
    assert first.prompt == second.prompt


async def test_generate_includes_ml_derived_facts(no_llm):
    """Topshiriqda ML/qoidalar bergan xulosalar bo'lsin, quruq shablon emas."""
    ctx = await _context()

    prompt = (await prompt_builder.generate(ctx)).prompt

    # Foydalanuvchining o'z matni va tasniflagichning xulosasi ichida bo'lsin.
    assert "navbat" in prompt.lower()
    assert ctx.project_title in prompt
    # Retriever topgan manbalar javobda alohida qaytadi.
    assert (await prompt_builder.generate(ctx)).sources


# ------------------------------------------------------------------- savollar


async def test_questions_follow_detected_signals(no_llm):
    result = await clarifier.ask(_IDEA, "booking_service", "Bron", lang="uz", count=5)

    assert result.generated_by_llm is False
    assert result.warning is None
    ids = [q.id for q in result.questions]
    # Matnda to'lov bor — shu savol chiqishi shart.
    assert "payments" in ids
    # Hajmni chegaralaydigan savol har doim oxirida.
    assert ids[-1] == "scope"


async def test_questions_differ_between_ideas(no_llm):
    """Turli g'oyalarga turli savollar — statik ro'yxatning asosiy kamchiligi shu edi."""
    delivery = await clarifier.ask(
        "Taom yetkazish. Xaritada kuryerni real vaqtda kuzatish mumkin.",
        "delivery_logistics", "Yetkazish", lang="uz", count=5,
    )
    blog = await clarifier.ask(
        "Shaxsiy blog. Muallif maqola yozadi, o'quvchilar o'qiydi va izoh qoldiradi.",
        "content_media", "Blog", lang="uz", count=5,
    )

    assert {q.id for q in delivery.questions} != {q.id for q in blog.questions}
    assert "geo" in {q.id for q in delivery.questions}


async def test_question_count_is_respected(no_llm):
    for count in (3, 4, 5, 6):
        result = await clarifier.ask(_IDEA, "booking_service", "Bron", lang="uz", count=count)
        assert len(result.questions) == count, f"count={count}"


@pytest.mark.parametrize("lang", ["uz", "ru", "en"])
async def test_questions_translated(no_llm, lang):
    """Har bir savol so'ralgan tilda bo'lsin — aralash til javobni buzadi."""
    result = await clarifier.ask(_IDEA, "booking_service", "Bron", lang=lang, count=5)
    assert all(q.question.strip() for q in result.questions)
    # Signalga bog'liq savol tarjimasi tushib qolsa o'zbekchaga qaytadi;
    # shuni sezish uchun to'lov savolini aniq solishtiramiz.
    payments = next((q for q in result.questions if q.id == "payments"), None)
    assert payments is not None
    expected = clarifier._SIGNAL_QUESTIONS["payments"][lang][0]
    assert payments.question == expected
