"""CORS ruxsatlari.

Bu testlar bir marta sodir bo'lgan uzilishdan keyin yozildi: Railway'da
CORS_ORIGINS `.env.example` dagi localhost ro'yxati bilan qolib ketgan va
production frontend (sys42.vercel.app) butunlay to'silgan edi — brauzerda
"No 'Access-Control-Allow-Origin' header", serverda preflightga 400.

Shuning uchun asosiy tekshiruv: env qanday bo'lishidan qat'i nazar,
production domenlari o'tishi shart.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app

PREFLIGHT = {
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "content-type,authorization",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize(
    "origin",
    [
        "https://sys42.vercel.app",
        "https://sys42.xyz",
        "https://www.sys42.xyz",
        # Vercel preview deploylari — har build yangi domen yasaydi.
        "https://sys42-git-main-isafrdev.vercel.app",
        "https://pr-nu-three.vercel.app",
    ],
)
@pytest.mark.parametrize("path", ["/api/models", "/api/clarify", "/api/generate"])
def test_preflight_ruxsat_etilgan_domenlar(client, origin, path):
    r = client.options(path, headers={"Origin": origin, **PREFLIGHT})

    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == origin
    assert r.headers.get("access-control-allow-credentials") == "true"


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example.com",
        # `*.vercel.app` ochiq emas — faqat shu loyihaning domenlari.
        "https://evil.vercel.app",
        "https://notsys42.vercel.app",
        # Naqsh `fullmatch` bilan tekshiriladi: prefiks bo'lib qo'shilmasin.
        "https://sys42.vercel.app.evil.com",
        "http://sys42.vercel.app",
    ],
)
def test_preflight_begona_domen_toldirilmaydi(client, origin):
    r = client.options("/api/models", headers={"Origin": origin, **PREFLIGHT})

    assert r.headers.get("access-control-allow-origin") is None


def test_env_ni_ochirib_qoyilsa_ham_production_ishlaydi():
    """CORS_ORIGINS bo'sh qolsa ham doimiy domenlar joyida turadi."""
    s = Settings(cors_origins="", cors_origin_regex="")

    assert "https://sys42.vercel.app" in s.cors_list
    assert "https://sys42.xyz" in s.cors_list


def test_env_dagi_royxat_almashtirmaydi_balki_qoshiladi():
    """Eskirgan env qiymati production domenlarini o'chirib yubormasin."""
    s = Settings(cors_origins="http://localhost:3000", cors_origin_regex="")

    assert "http://localhost:3000" in s.cors_list
    assert "https://sys42.vercel.app" in s.cors_list


def test_qoshimcha_naqsh_doimiysiga_qoshiladi():
    import re

    s = Settings(cors_origin_regex=r"https://staging-[a-z]+\.example\.com")
    rx = re.compile(s.cors_regex)

    assert rx.fullmatch("https://sys42.vercel.app")
    assert rx.fullmatch("https://staging-abc.example.com")
    assert not rx.fullmatch("https://evil.com")
