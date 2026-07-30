"""Ilova sozlamalari — hammasi .env dan o'qiladi."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
PDF_DIR = DATA_DIR / "pdf"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

# Loyihaning doimiy frontend domenlari. Bular ATAYLAB env'dan tashqarida va
# CORS_ORIGINS ga qo'shimcha bo'lib qo'shiladi, uni almashtirmaydi.
#
# Sabab tajribadan: Railway'da CORS_ORIGINS `.env.example` dagi localhost
# ro'yxati bilan qolib ketgan edi va production frontend butunlay to'silgandi
# (brauzerda "No 'Access-Control-Allow-Origin' header", serverda esa
# preflightga 400). Doimiy domen kodda tursa, panel sozlamasi eskirsa ham
# mahsulot ishlashda davom etadi.
PRODUCTION_ORIGINS: tuple[str, ...] = (
    "https://sys42.xyz",
    "https://www.sys42.xyz",
    "https://sys42.vercel.app",
)

# Vercel har deploy uchun yangi domen yasaydi:
#   <loyiha>-<hash>-<jamoa>.vercel.app
#   <loyiha>-git-<branch>-<jamoa>.vercel.app
# Loyiha Vercel'da avval `pr`, keyin `sys42` nomi bilan turgan — ikkalasi ham
# qoldirildi, eski preview linklar ham ishlayversin.
#
# Naqsh ataylab tor: butun `*.vercel.app` ga ochib qo'yish begona saytlarga
# ham API ni ochib berardi (allow_credentials=True bilan bu ayniqsa xavfli).
PRODUCTION_ORIGIN_REGEX = r"https://(?:pr|sys42)(?:-[a-z0-9-]+)?\.vercel\.app"

# Ulangan disk yo'li. Railway buni o'zi qo'yadi, boshqa hostingda qo'lda.
VOLUME_ENV_VARS = ("RAILWAY_VOLUME_MOUNT_PATH", "DATA_VOLUME_PATH")


def volume_path() -> str | None:
    """Ulangan disk bo'lsa uning yo'li, bo'lmasa None."""
    for var in VOLUME_ENV_VARS:
        value = (os.getenv(var) or "").strip()
        if value:
            return value
    return None


def _default_database_url() -> str:
    """SQLite faylini deploylardan omon qoladigan joyga qo'yadi.

    Konteyner fayl tizimi HAR DEPLOYDA toza holatdan boshlanadi. Standart
    `./prompting.db` konteyner ichida yotadi, ya'ni har deployda barcha
    foydalanuvchi, kvota va to'lov yozuvi yo'qoladi. Tashqi belgisi shuki,
    kirgan odam keyingi deploydan keyin 401 ola boshlaydi: tokeni butun va
    imzosi to'g'ri, lekin u ko'rsatayotgan hisob bazada yo'q.

    Disk ulangan bo'lsa baza o'sha yerda yotadi va deploydan omon qoladi.
    Postgres ishlatilsa DATABASE_URL baribir shuni ustidan yozadi.
    """
    mount = volume_path()
    if mount:
        # Bu URL, OS yo'li emas — `Path` Windows'da teskari chiziq qo'yardi va
        # hosil bo'lgan URL noto'g'ri bo'lardi. Shuning uchun oldinga chiziq.
        return f"sqlite+aiosqlite:///{mount.rstrip('/')}/prompting.db"
    return "sqlite+aiosqlite:///./prompting.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_writer_model: str = "anthropic/claude-sonnet-5"
    openrouter_app_url: str = "https://sys42.xyz"
    openrouter_app_title: str = "sys42"

    # Sotib olish uchun aloqa. Kodda emas — o'zgarganda deploy kerak bo'lmasin.
    support_telegram: str = "its_mansurov"

    # --- LLM provayderlari ---
    # Tartib muhim: birinchisi ishlamasa keyingisiga o'tiladi. Bittasining
    # krediti tugashi butun mahsulotni to'xtatib qo'ymasligi uchun.
    #
    # Ikkita zanjir bor, chunki ikki xil ish bir xil narxga arzimaydi:
    #   llm_providers       — arzon/tez ish: aniqlashtiruvchi savollar.
    #   llm_providers_heavy — asosiy ish: to'liq texnik topshiriq yozish.
    #
    # Og'ir zanjirda Claude birinchi turadi — TT sifati mahsulotning o'zi.
    # Savollar uchun esa Gemini/OpenRouter yetarli va bir necha barobar arzon.
    # Asosiy tugma: mahsulot LLM'siz ishlaydi.
    #
    # Texnik topshiriq ML yadro va qoidalar asosida yig'iladi — turni tasniflash,
    # signallarni ajratish, stack/server tavsiyasi, playbook, UI-UX va domen —
    # bularning hammasi allaqachon shu yerda, tashqi modelsiz. LLM faqat tayyor
    # matnni qayta yozib chiqardi.
    #
    # `False` (standart) — hech qanday tashqi model chaqirilmaydi: bepul,
    # bir zumda, tarmoqqa bog'liq emas va natija har safar bir xil.
    # `True` — LLM matnni qayta yozadi (zanjir quyida).
    use_llm: bool = False

    llm_providers: str = "moonshot,openrouter,google"
    llm_providers_heavy: str = "moonshot,openrouter,google"

    google_api_key: str = ""
    google_model: str = "gemini-2.5-flash"

    # Anthropic to'g'ridan-to'g'ri — hozir zanjirda emas, Claude TokenMix
    # orqali olinadi. Kalit qo'yilib, zanjirga `anthropic` yozilsa ishlaydi.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"

    # --- Jonli web-tadqiqot (Linkup) ---
    # Lokal portfel dataseti faqat ikkita akselerator ro'yxatidan yig'ilgan,
    # ya'ni bozorning kichik qismi. Linkup internetdan qidiradi va portfelda
    # yo'q raqobatchilarni topadi (sinovda NoAsk, Stolik).
    #
    # `linkup_output_type`:
    #   searchResults — xom natijalar, hech qanday model ishlatilmaydi;
    #   sourcedAnswer — Linkup tomonida model javobni yig'ib beradi.
    # Standart `sourcedAnswer`: bir necha manbani bir joyga yig'adi va
    # foydaliroq. Model umuman kerak bo'lmasa `searchResults` qo'yiladi.
    linkup_api_key: str = ""
    linkup_enabled: bool = True
    linkup_depth: str = "standard"           # standard | deep
    linkup_output_type: str = "sourcedAnswer"

    # TokenMix — standart zanjirdan OLIB TASHLANGAN.
    #
    # Sabab: kalitda ruxsat berilgan yagona model `kimi-k3` va u promo
    # kreditda ishlamaydi. Ikki dona ishlamaydigan provayderni zanjirda
    # ushlab turish faqat kechikish qo'shadi — har so'rov navbat bilan
    # yiqiladi. Kod joyida: balans to'lsa zanjirga `tokenmix` yozilsa
    # yana ishlaydi.
    tokenmix_api_key: str = ""
    tokenmix_base_url: str = "https://api.tokenmix.ai/v1"
    tokenmix_model: str = "kimi-k3"
    tokenmix_model_light: str = "kimi-k3"

    # Moonshot (Kimi) to'g'ridan-to'g'ri.
    # DIQQAT: xalqaro kalit `api.moonshot.ai` da ishlaydi; `.cn` boshqa hisob
    # tizimi va o'sha kalit bilan 401 qaytaradi.
    moonshot_api_key: str = ""
    moonshot_base_url: str = "https://api.moonshot.ai/v1"
    moonshot_model: str = "kimi-k2.6"
    moonshot_model_light: str = "kimi-k2.6"

    database_url: str = Field(default_factory=_default_database_url)

    # Qo'shimcha domenlar (lokal portlar, boshqa front). Doimiy production
    # domenlari `PRODUCTION_ORIGINS` da — bu yerda takrorlash shart emas.
    cors_origins: str = "http://localhost:3000,http://localhost:3100"

    # Qo'shimcha naqsh. `PRODUCTION_ORIGIN_REGEX` ga qo'shiladi, uni
    # almashtirmaydi — pastdagi `cors_regex` ga qarang.
    cors_origin_regex: str = ""

    secret_key: str = "change-me"

    # Firebase ID tokenini tekshirish uchun yetarli — service account kaliti
    # kerak emas, chunki imzo Google'ning ochiq sertifikatlari bilan
    # tekshiriladi.
    firebase_project_id: str = ""

    payment_provider: str = "mock"
    payme_merchant_id: str = ""
    payme_key: str = ""
    click_merchant_id: str = ""
    click_service_id: str = ""
    click_secret_key: str = ""

    @property
    def cors_list(self) -> list[str]:
        """Env'dagi ro'yxat + doimiy production domenlari (tartib saqlanadi)."""
        configured = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return list(dict.fromkeys([*configured, *PRODUCTION_ORIGINS]))

    @property
    def cors_regex(self) -> str:
        """Doimiy naqsh + env'dagi qo'shimcha naqsh, bittaga birlashtirilgan.

        Starlette naqshni `fullmatch` bilan tekshiradi, shuning uchun har bir
        qismni alohida guruhga o'rab alternatsiya qilamiz.
        """
        extra = (self.cors_origin_regex or "").strip()
        if not extra:
            return PRODUCTION_ORIGIN_REGEX
        return f"(?:{PRODUCTION_ORIGIN_REGEX})|(?:{extra})"


@lru_cache
def get_settings() -> Settings:
    return Settings()
