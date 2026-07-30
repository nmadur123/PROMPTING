"""Jonli web-tadqiqot testlari — tarmoqqa CHIQMAYDI.

Linkup so'rovi kredit sarflaydi, shuning uchun testlarda HTTP qatlami
qo'g'irchoq bilan almashtiriladi. Tekshirilayotgan narsa: so'rov to'g'ri
qurilyaptimi, javob to'g'ri o'qilyaptimi, kesh ishlayaptimi va nosozlik
generatsiyani to'xtatmaydimi.
"""

from __future__ import annotations

import json

import pytest

from app.services import web_research as W


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("JSON emas")
        return self._payload


class _FakeClient:
    def __init__(self, response, calls: list):
        self._response = response
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        self._calls.append({"url": url, **kwargs})
        return self._response


@pytest.fixture
def linkup(monkeypatch, tmp_path):
    """Kalitni yoqadi, keshni vaqtinchalik papkaga oladi, HTTP ni to'sadi."""
    settings = W.get_settings()
    monkeypatch.setattr(settings, "linkup_api_key", "test-key", raising=False)
    monkeypatch.setattr(settings, "linkup_enabled", True, raising=False)
    monkeypatch.setattr(W, "_CACHE_DIR", tmp_path / "cache")

    calls: list = []

    def install(response):
        monkeypatch.setattr(W.httpx, "Client", lambda **kw: _FakeClient(response, calls))
        return calls

    return install, calls


_PAYLOAD = {
    "answer": "O'zbekistonda Stolik va NoAsk kabi tizimlar bor.",
    "sources": [
        {"name": "Stolik", "url": "https://stolik.uz", "snippet": "stol band qilish"},
        {"name": "NoAsk", "url": "https://noask.uz", "snippet": "restoran boshqaruvi"},
    ],
}


# ------------------------------------------------------------------ so'rov


def test_query_is_built_from_project_type_not_raw_text():
    """So'rov turdan qurilsin — xom g'oyada qidiruv uchun keraksiz tafsilot ko'p."""
    query = W.build_query(
        "Sartaroshxona uchun navbat, usta tasdiqlaydi, SMS eslatma keladi",
        "booking_service",
    )
    assert "band qilish" in query
    assert "O'zbekistonda" in query


def test_query_omits_place_for_non_uz_region():
    query = W.build_query("onlayn do'kon ochmoqchiman", "ecommerce", region="global")
    assert "O'zbekistonda" not in query


# ------------------------------------------------------------------- javob


def test_parses_answer_and_sources(linkup):
    install, _ = linkup
    install(_FakeResponse(200, _PAYLOAD))

    result = W.search("test so'rov")
    assert result.ok is True
    assert "Stolik" in result.answer
    assert [s.name for s in result.sources] == ["Stolik", "NoAsk"]
    assert result.sources[0].url == "https://stolik.uz"


def test_sourced_answer_is_flagged_as_model_generated(linkup):
    """Foydalanuvchi javobni kim yozganini bilishi kerak."""
    install, _ = linkup
    install(_FakeResponse(200, _PAYLOAD))

    assert W.search("q", output_type="sourcedAnswer").answer_is_model_generated is True
    W.get_settings()  # kesh kaliti output_type ni ham hisobga oladi
    assert W.search("q", output_type="searchResults").answer_is_model_generated is False


def test_sources_without_url_are_dropped(linkup):
    install, _ = linkup
    install(_FakeResponse(200, {"answer": "", "sources": [{"name": "Havolasiz"}]}))
    assert W.search("q").sources == []


def test_source_count_is_capped(linkup):
    install, _ = linkup
    many = {"sources": [{"name": f"S{i}", "url": f"https://s{i}.uz"} for i in range(30)]}
    install(_FakeResponse(200, many))
    assert len(W.search("q").sources) == W._MAX_SOURCES


# -------------------------------------------------------------------- kesh


def test_second_identical_query_uses_cache(linkup):
    """Kesh bo'lmasa bitta g'oya uchun o'nlab so'rov ketardi."""
    install, calls = linkup
    install(_FakeResponse(200, _PAYLOAD))

    first = W.search("bir xil so'rov")
    second = W.search("bir xil so'rov")

    assert first.from_cache is False
    assert second.from_cache is True
    assert len(calls) == 1, "ikkinchi so'rov tarmoqqa chiqmasligi kerak"


def test_different_output_type_is_cached_separately(linkup):
    install, calls = linkup
    install(_FakeResponse(200, _PAYLOAD))
    W.search("q", output_type="sourcedAnswer")
    W.search("q", output_type="searchResults")
    assert len(calls) == 2


# ---------------------------------------------------------------- nosozlik


@pytest.mark.parametrize(
    "response,fragment",
    [
        (_FakeResponse(429, {}), "limiti tugadi"),
        (_FakeResponse(500, {}, text="server xato"), "500"),
        (_FakeResponse(200, None, text="<html>"), "JSON emas"),
    ],
)
def test_failures_return_not_ok_instead_of_raising(linkup, response, fragment):
    """Nosozlik istisno emas — generatsiya to'xtamasligi kerak."""
    install, _ = linkup
    install(response)
    result = W.search("q")
    assert result.ok is False
    assert fragment in result.error


def test_disabled_without_key(monkeypatch):
    settings = W.get_settings()
    monkeypatch.setattr(settings, "linkup_api_key", "", raising=False)
    assert W.enabled() is False
    assert W.search("q").ok is False


def test_competitor_analysis_works_when_web_disabled(monkeypatch):
    """Linkup o'chirilganda lokal tahlil o'z yo'lida ishlasin."""
    from app.services import competitors

    settings = W.get_settings()
    monkeypatch.setattr(settings, "linkup_enabled", False, raising=False)

    result = competitors.analyze(
        "Restoranlar uchun stol band qilish tizimi", "booking_service", include_web=True
    )
    assert result.web is None
    assert result.advantages  # lokal qism baribir ishlaydi
