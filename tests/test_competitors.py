"""Raqobat tahlili testlari — tarmoqqa chiqmaydi, lokal dataset ustida.

Bu yerda "to'g'ri raqobatchi topildimi" degan savol tekshiriladi. Uni ko'z
bilan tekshirish yaramaydi: indeks har qanday so'rovga nimadir qaytaradi,
shuning uchun natija hamisha ishonchli KO'RINADI. Aynan shu sababdan
ishlab chiqish davomida ikkita jimgina xato bo'lgan — biri kesh-tilli
(datasetning yarmi umuman topilmasdi), ikkinchisi ballni oshirib yuborish.
"""

from __future__ import annotations

import pytest

from app.services import competitors


@pytest.fixture(scope="module")
def index():
    idx = competitors.get_index()
    if idx is None:
        pytest.skip("raqobat dataseti yo'q — `python -m scripts.fetch_competitors`")
    return idx


def _names(matches) -> set[str]:
    return {m.name for m in matches}


# ------------------------------------------------------------ til ko'prigi


def test_finds_english_competitor_from_uzbek_query(index):
    """Eng muhim test: dataset ikki tilda, so'rov bitta tilda.

    REZVO tavsifi inglizcha ("Online Reservation & CRM Platform for
    Restaurants"), so'rov o'zbekcha. Til ko'prigi buzilsa bu moslik butunlay
    yo'qoladi va buni faqat qo'lda tekshirib sezish mumkin — indeks o'rniga
    boshqa, tasodifiy startuplarni qaytaraveradi.
    """
    matches = index.search(
        "Restoranlar uchun stol band qilish va mijozlar bazasi tizimi", top_k=5
    )
    assert "REZVO" in _names(matches)


def test_finds_uzbek_competitor_from_english_query(index):
    """Teskari yo'nalish ham ishlasin."""
    matches = index.search(
        "restaurant process automation platform for venues", top_k=6
    )
    assert "Alipos" in _names(matches)


def test_both_sources_are_reachable(index):
    """Ikkala manba ham qidiruvda chiqsin — biri butunlay ko'rinmas bo'lmasin."""
    seen: set[str] = set()
    for query in (
        "restoran stol band qilish tizimi",
        "yuk tashish va logistika platformasi",
        "telegram uchun suniy intellekt yordamchisi",
        "onlayn oquv kurslari platformasi",
    ):
        seen.update(m.source for m in index.search(query, top_k=6))
    assert seen == {"uzcombinator", "thepitch"}, f"topilgan manbalar: {seen}"


# ------------------------------------------------------------ til kengaytmasi


def test_concept_tokens_are_bounded():
    """Har bir tushunchaga bitta token (vaznlangan), guruhning hammasi emas.

    Ilgari guruhdagi barcha shakllar qo'shilardi va bir umumiy tushuncha
    o'xshashlikni keskin ko'tarardi: "ot boqish uchun ilova" restoran
    startupiga 0.48 ball olib "bevosita raqobatchi" bo'lib chiqardi.
    """
    text = competitors._expand_terms("restoran uchun band qilish tizimi")
    tokens = [t for t in text.split() if t.startswith("kncpt")]
    # Ikkita tushuncha topilishi kerak (restoran, band qilish), har biri
    # `_CONCEPT_WEIGHT` marta.
    assert len(set(tokens)) >= 2
    assert len(tokens) == len(set(tokens)) * competitors._CONCEPT_WEIGHT


def test_expansion_is_noop_without_known_terms():
    text = "zzz qqq www"
    assert competitors._expand_terms(text) == text


# ------------------------------------------------------------------- dedupe


def test_duplicate_products_are_merged():
    """Bir mahsulot ikki slug bilan kelsa bitta bo'lib qolsin.

    Aks holda to'yinganlik soxta oshadi: asoschi bozorni haqiqatdan
    zichroq deb o'ylaydi.
    """
    rows = [
        {"name": "Hisobchi", "description": "Telegramda ishlaydigan hisobchi", "investment": ""},
        {"name": "Hisobchi Ai", "description": "Telegramda ishlaydigan hisobchi", "investment": "TAKLIF OLGAN"},
        {"name": "Boshqa", "description": "Butunlay boshqa mahsulot tavsifi", "investment": ""},
    ]
    merged = competitors._dedupe(rows)
    assert len(merged) == 2
    # Ma'lumoti to'liqrogi qoladi.
    assert any(r["investment"] == "TAKLIF OLGAN" for r in merged)


# ------------------------------------------------------------------ tahlil


def test_analysis_gives_type_specific_advantages():
    """Ustunlik maslahati loyiha turiga bog'lansin, umumiy bo'lmasin."""
    booking = competitors.analyze("stol band qilish tizimi restoran uchun", "booking_service")
    fintech = competitors.analyze("mikrokredit berish platformasi skoring bilan", "fintech")
    assert booking.advantages and fintech.advantages
    assert booking.advantages[0] != fintech.advantages[0]


def test_signal_adds_advantage():
    with_payments = competitors.analyze("onlayn dokon", "ecommerce", {"payments": True})
    without = competitors.analyze("onlayn dokon", "ecommerce", {})
    assert len(with_payments.advantages) > len(without.advantages)


def test_funded_competitor_changes_strategy_note():
    """Investitsiya olgan raqobatchi bo'lsa strategiya maslahati qo'shilsin."""
    analysis = competitors.analyze(
        "Restoranlarda jarayonlarni avtomatlashtirish platformasi", "saas_dashboard"
    )
    if any(m.funded for m in analysis.matches):
        assert any("Investitsiya olgan" in a for a in analysis.advantages)


def test_missing_dataset_is_not_fatal(monkeypatch, tmp_path):
    """Dataset yo'q bo'lsa tahlil bo'sh qaytsin, istisno ko'tarmasin —
    aks holda butun generatsiya to'xtab qolardi."""
    monkeypatch.setattr(competitors, "_DATASET", tmp_path / "yoq.json")
    competitors.get_index.cache_clear()
    try:
        result = competitors.analyze("qandaydir goya matni bu yerda", "ecommerce")
        assert result.matches == []
        assert result.dataset_size == 0
    finally:
        competitors.get_index.cache_clear()


def test_investment_parsing_distinguishes_funded():
    funded = competitors.CompetitorMatch(
        name="x", description="y", similarity=0.5, source="thepitch",
        investment="$100.000 TAKLIF OLGAN",
    )
    rejected = competitors.CompetitorMatch(
        name="x", description="y", similarity=0.5, source="thepitch",
        investment="TAKLIF OLMAGAN",
    )
    assert funded.funded is True
    # "OLMAGAN" ichida ham "OLGAN" bor — oddiy `in` tekshiruvi bu yerda
    # aldardi.
    assert rejected.funded is False
