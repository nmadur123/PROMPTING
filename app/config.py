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
    llm_providers: str = "openrouter,google"
    google_api_key: str = ""
    google_model: str = "gemini-2.5-flash"

    database_url: str = "sqlite+aiosqlite:///./prompting.db"
    cors_origins: str = "https://sys42.xyz"
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
