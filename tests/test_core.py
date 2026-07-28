"""Yadro mantiq testlari — tashqi tarmoqqa chiqmaydi."""

from __future__ import annotations

import pytest

from app.ml.corpus import load_chunks
from app.ml.engine import engine, extract_expected_users, extract_signals, normalize
from app.services import domain_service, payment_service, server_calculator, stack_recommender, uiux_advisor


# --------------------------------------------------------------------- ML


def test_normalize_strips_apostrophes():
    # Dataset tutuqsiz yozilgan; foydalanuvchi tutuq bilan yozadi.
    assert normalize("Qo'shish") == normalize("Qoshish") == "qoshish"
    assert normalize("O‘zbek  tili") == "ozbek tili"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("sartaroshxona uchun onlayn navbat, mijoz bosh vaqtni tanlaydi usta tasdiqlaydi", "booking_service"),
        ("dasturchilar api kalit oladi va sorovlar soni boyicha tolaydi sdk bor", "devtool_api"),
        ("kuryer buyurtmani oladi xaritada boradi mijoz real vaqtda kuzatadi", "delivery_logistics"),
        ("elektron hamyon pul otkazish tranzaksiya tarixi va limitlar", "fintech"),
    ],
)
def test_classifier_predicts_expected_type(text, expected):
    prediction = engine.classifier.predict(text)
    assert prediction.label == expected, f"{expected} kutilgandi, {prediction.label} chiqdi"
    assert prediction.confidence > 0.2


def test_classifier_returns_alternatives():
    prediction = engine.classifier.predict("onlayn dokon kiyim sotaman savat va tolov")
    assert len(prediction.alternatives) == 2
    # Muqobil variantlar asosiy tanlovdan past bo'lishi shart.
    assert all(score <= prediction.confidence for _, score in prediction.alternatives)


def test_retriever_finds_model_specific_guidance():
    hits = engine.retriever.search(
        "response length and verbosity", top_k=3, family="claude", model="opus-5"
    )
    assert hits, "hech narsa topilmadi"
    # Model-maxsus hujjat bonusi ishlashi kerak.
    assert any(h.chunk.model == "opus-5" for h in hits)


def test_retriever_family_only_excludes_other_families():
    hits = engine.retriever.search(
        "describe the result you need", top_k=6, family="openai", family_only=True
    )
    assert hits
    assert all(h.chunk.family == "openai" for h in hits)


@pytest.mark.parametrize(
    "model_id,expected_family",
    [("anthropic/claude-opus-5", "claude"), ("openai/gpt-5", "openai")],
)
def test_guidance_targets_the_right_doc_family(model_id, expected_family):
    """Claude korpusi kattaroq — kvotasiz GPT uchun ham Claude hujjati chiqardi."""
    from app.services.prompt_builder import BuildContext, retrieve_guidance

    ctx = BuildContext(
        description="onlayn navbat tizimi sartaroshxonalar uchun",
        project_type="booking_service", project_title="Bron", confidence=0.9,
        signals={}, monthly_users=5000, stack=[], anti_patterns=[],
        requirements={}, server_options=[], scaling=[],
        uiux={"render_strategy": {"rule": "", "why": ""}, "tips": [], "performance_budget": {}},
        target_model=model_id,
    )
    hits = retrieve_guidance(ctx)
    assert hits
    families = [h.chunk.family for h in hits]
    assert families.count(expected_family) >= 4, f"{expected_family} kvotasi bajarilmadi: {families}"
    # Manbalar skor bo'yicha kamayish tartibida bo'lsin.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_corpus_chunks_have_sources():
    chunks = load_chunks()
    assert len(chunks) > 50
    assert all(c.source.startswith("https://") for c in chunks)


# ----------------------------------------------------------------- signals


def test_extract_signals_multilingual():
    assert extract_signals("real vaqtda chat kerak")["realtime"] is True
    assert extract_signals("нужна оплата картой")["payments"] is True
    assert extract_signals("we need a map with gps tracking")["geo"] is True
    assert extract_signals("oddiy blog sayti")["realtime"] is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5000 foydalanuvchi kutyapman", 5000),
        ("10 ming foydalanuvchi", 10_000),
        ("2 mln foydalanuvchi boladi", 2_000_000),
        ("ожидаю 3000 пользователей", 3000),
        ("hech qanday raqam yoq", None),
    ],
)
def test_extract_expected_users(text, expected):
    assert extract_expected_users(text) == expected


# --------------------------------------------------------- server sizing


def test_requirements_scale_with_users():
    small = server_calculator.calculate_requirements("saas_dashboard", 1_000, {})
    large = server_calculator.calculate_requirements("saas_dashboard", 100_000, {})
    assert large.peak_rps > small.peak_rps
    assert large.vcpu >= small.vcpu
    assert large.db_size_gb > small.db_size_gb


def test_realtime_signal_increases_ram_and_connections():
    plain = server_calculator.calculate_requirements("social_community", 50_000, {})
    realtime = server_calculator.calculate_requirements("social_community", 50_000, {"realtime": True})
    assert realtime.concurrent_connections > plain.concurrent_connections
    assert realtime.ram_gb >= plain.ram_gb


def test_recommended_plan_actually_fits():
    req = server_calculator.calculate_requirements("marketplace", 200_000, {"media_heavy": True})
    options = server_calculator.recommend_servers(req, "eu")
    assert options
    for option in options:
        plan = option.plan
        assert plan["vcpu"] >= req.vcpu, f"{plan['name']} vCPU talabni qoplamaydi"
        assert plan["ram_gb"] >= req.ram_gb, f"{plan['name']} RAM talabni qoplamaydi"


def test_uz_region_prefers_local_provider_on_equal_price():
    req = server_calculator.calculate_requirements("booking_service", 5_000, {})
    options = server_calculator.recommend_servers(req, "uz")
    assert options[0].plan["provider"] in ("PS.UZ", "UZCLOUD")


def test_huge_scale_falls_back_to_horizontal_advice():
    req = server_calculator.calculate_requirements("game", 20_000_000, {"realtime": True})
    options = server_calculator.recommend_servers(req, "global")
    # Katalogdan oshib ketsa ham javob bo'sh qolmasligi kerak.
    assert options
    assert options[0].reasoning


# ------------------------------------------------------------------ stack


def test_ai_signal_adds_queue_and_ai_layer():
    choices = stack_recommender.recommend_stack("ai_tool", {"ai": True}, 10_000)
    categories = {c.category for c in choices}
    assert "AI qatlami" in categories
    assert "Navbat" in categories


def test_uz_region_recommends_local_payments():
    choices = stack_recommender.recommend_stack("ecommerce", {"payments": True}, 10_000, region="uz")
    payment = next(c for c in choices if c.category == "To'lov")
    assert "Payme" in payment.pick


def test_anti_recommendations_always_present():
    anti = stack_recommender.anti_recommendations("saas_dashboard", {}, 5_000)
    assert len(anti) >= 3
    assert any("Kubernetes" in a for a in anti)


# ------------------------------------------------------------------ UI/UX


def test_uiux_includes_conditional_tips():
    advice = uiux_advisor.advise("social_community", {"realtime": True, "media_heavy": True}, 50_000)
    areas = {tip["area"] for tip in advice["tips"]}
    assert "Realtime" in areas
    assert "Video" in areas
    assert advice["render_strategy"]["rule"]
    assert advice["performance_budget"]["LCP"]


# ----------------------------------------------------------------- domain


def test_slugify_strips_uzbek_suffixes():
    names = domain_service.suggest_names(
        "Samarqanddagi sartaroshxonalar uchun navbat tizimi", "navbat"
    )
    assert "navbat" in names
    # Grammatik shakl domen nomiga tushmasligi kerak.
    assert not any(n.startswith("samarqanddagi") for n in names)


def test_suggest_names_are_valid_labels():
    names = domain_service.suggest_names("onlayn kurs platformasi video dars va test", "")
    assert names
    for name in names:
        assert 3 <= len(name) <= 20
        assert name.replace("-", "").isalnum()


# ---------------------------------------------------------------- payment


def test_unconfigured_provider_falls_back_to_mock():
    # Kalitlarsiz Payme mock'ga tushishi kerak — jimgina "to'landi" demasligi shart.
    provider = payment_service.get_provider("payme")
    assert provider.name == "mock"


def test_mock_webhook_and_intent():
    provider = payment_service.MockProvider()
    intent = provider.create_intent("order-1", 50_000, "http://localhost:3000/done")
    assert intent.amount_uzs == 50_000
    assert intent.checkout_url.startswith("http")


def test_payme_checkout_url_encodes_amount_in_tiyin():
    import base64

    provider = payment_service.PaymeProvider("merchant123", "secret")
    intent = provider.create_intent("order-7", 25_000, "http://x/done")
    encoded = intent.checkout_url.replace(payment_service.PaymeProvider.CHECKOUT_BASE, "")
    decoded = base64.b64decode(encoded).decode()
    assert "a=2500000" in decoded  # Payme tiyinda kutadi
    assert "ac.order_id=order-7" in decoded


def test_payme_webhook_rejects_wrong_key():
    import base64

    provider = payment_service.PaymeProvider("merchant123", "right-key")
    good = base64.b64encode(b"Paycom:right-key").decode()
    bad = base64.b64encode(b"Paycom:wrong-key").decode()
    assert provider.verify_webhook({"authorization": f"Basic {good}"}, {}) is True
    assert provider.verify_webhook({"authorization": f"Basic {bad}"}, {}) is False
    assert provider.verify_webhook({}, {}) is False


def test_payment_recommendation_skipped_when_not_needed():
    assert payment_service.recommend_payment("game", {}, "uz") is None
    assert payment_service.recommend_payment("ecommerce", {}, "uz") is not None
