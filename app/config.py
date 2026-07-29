"""Ilova sozlamalari — hammasi .env dan o'qiladi."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
PDF_DIR = DATA_DIR / "pdf"
ARTIFACTS_DIR = DATA_DIR / "artifacts"


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

    llm_providers: str = "tokenmix,openrouter,google"
    llm_providers_heavy: str = "tokenmix,openrouter,google"

    google_api_key: str = ""
    google_model: str = "gemini-2.5-flash"

    # Anthropic to'g'ridan-to'g'ri — hozir zanjirda emas, Claude TokenMix
    # orqali olinadi. Kalit qo'yilib, zanjirga `anthropic` yozilsa ishlaydi.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"

    # TokenMix — bitta kalit ostida ko'p provayder (OpenAI-mos API).
    # Og'ir va yengil ish uchun ikki xil model qo'yish mumkin; hozir
    # ikkalasi ham `kimi-k3`, chunki kalitda ruxsat berilgan yagona model shu.
    tokenmix_api_key: str = ""
    tokenmix_base_url: str = "https://api.tokenmix.ai/v1"
    tokenmix_model: str = "kimi-k3"
    tokenmix_model_light: str = "kimi-k3"

    database_url: str = "sqlite+aiosqlite:///./prompting.db"
    cors_origins: str = "https://sys42.xyz,https://www.sys42.xyz,https://pr-nu-three.vercel.app"

    # Vercel har bir deploy uchun yangi domen yasaydi
    # (`pr-git-main-xxx.vercel.app`, `pr-a1b2c3.vercel.app` ...). Ularni
    # qo'lda CORS_ORIGINS ga qo'shib borish imkonsiz — har preview deploydan
    # keyin backend .env ini yangilash kerak bo'lardi. Shuning uchun shu
    # loyihaning Vercel domenlari naqsh bo'yicha ruxsat etiladi.
    #
    # Naqsh ataylab tor: faqat `pr-...vercel.app`. Butun `*.vercel.app` ga
    # ochib qo'yish begona saytlarga ham API ni ochib berardi.
    cors_origin_regex: str = r"https://pr-[a-z0-9-]*\.vercel\.app"

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
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
