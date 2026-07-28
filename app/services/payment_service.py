"""To'lov qatlami — pluggable adapter.

Bitta interfeys (`PaymentProvider`), uchta amalga oshirish: mock (dev uchun),
Payme va Click. Provayder `.env` dagi `PAYMENT_PROVIDER` bilan tanlanadi, shuning
uchun boshqasiga o'tish uchun kod o'zgartirish shart emas.

DIQQAT: Payme va Click adapterlari to'lov havolasini yasaydi va webhook imzosini
tekshiradi. Merchant kabinetida ro'yxatdan o'tib, kalitlarni `.env` ga
yozmaguningizcha ular `mock` rejimda ishlaydi — bu ataylab: sozlanmagan
provayder jimgina "to'landi" deb qaytarmasligi kerak.
"""

from __future__ import annotations

import base64
import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional

from app.config import get_settings
from app.copy import tr


@dataclass
class PaymentIntent:
    """Yaratilgan to'lov — foydalanuvchi shu havolaga yuboriladi."""

    intent_id: str
    provider: str
    amount_uzs: int
    checkout_url: str
    status: str          # created | paid | failed | mock
    created_at: float

    def to_dict(self) -> dict:
        return asdict(self)


class PaymentError(RuntimeError):
    pass


class PaymentProvider(ABC):
    name: str = "base"

    @abstractmethod
    def create_intent(self, order_id: str, amount_uzs: int, return_url: str) -> PaymentIntent:
        ...

    @abstractmethod
    def verify_webhook(self, headers: dict, body: dict) -> bool:
        ...

    @property
    def configured(self) -> bool:
        return True


class MockProvider(PaymentProvider):
    """Dev rejim: haqiqiy pul harakati yo'q, oqim to'liq sinaladi."""

    name = "mock"

    def create_intent(self, order_id: str, amount_uzs: int, return_url: str) -> PaymentIntent:
        return PaymentIntent(
            intent_id=f"mock_{uuid.uuid4().hex[:12]}",
            provider=self.name,
            amount_uzs=amount_uzs,
            checkout_url=f"{return_url}?mock_payment=success&order_id={order_id}",
            status="mock",
            created_at=time.time(),
        )

    def verify_webhook(self, headers: dict, body: dict) -> bool:
        return True


class PaymeProvider(PaymentProvider):
    """Payme (Paycom) — checkout havolasi base64 parametrlar bilan yasaladi."""

    name = "payme"
    CHECKOUT_BASE = "https://checkout.paycom.uz/"

    def __init__(self, merchant_id: str, key: str) -> None:
        self.merchant_id = merchant_id
        self.key = key

    @property
    def configured(self) -> bool:
        return bool(self.merchant_id and self.key)

    def create_intent(self, order_id: str, amount_uzs: int, return_url: str) -> PaymentIntent:
        if not self.configured:
            raise PaymentError("PAYME_MERCHANT_ID / PAYME_KEY sozlanmagan.")
        # Payme summani tiyinda kutadi.
        params = (
            f"m={self.merchant_id};ac.order_id={order_id};"
            f"a={amount_uzs * 100};c={return_url}"
        )
        encoded = base64.b64encode(params.encode("utf-8")).decode("ascii")
        return PaymentIntent(
            intent_id=f"payme_{order_id}",
            provider=self.name,
            amount_uzs=amount_uzs,
            checkout_url=self.CHECKOUT_BASE + encoded,
            status="created",
            created_at=time.time(),
        )

    def verify_webhook(self, headers: dict, body: dict) -> bool:
        """Payme Basic auth yuboradi: base64("Paycom:<KEY>")."""
        auth = headers.get("authorization") or headers.get("Authorization") or ""
        if not auth.lower().startswith("basic "):
            return False
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return False
        login, _, password = decoded.partition(":")
        return login == "Paycom" and password == self.key


class ClickProvider(PaymentProvider):
    """Click — havola query parametrlari bilan, webhook MD5 imzo bilan."""

    name = "click"
    CHECKOUT_BASE = "https://my.click.uz/services/pay"

    def __init__(self, merchant_id: str, service_id: str, secret_key: str) -> None:
        self.merchant_id = merchant_id
        self.service_id = service_id
        self.secret_key = secret_key

    @property
    def configured(self) -> bool:
        return bool(self.merchant_id and self.service_id and self.secret_key)

    def create_intent(self, order_id: str, amount_uzs: int, return_url: str) -> PaymentIntent:
        if not self.configured:
            raise PaymentError("CLICK_MERCHANT_ID / CLICK_SERVICE_ID / CLICK_SECRET_KEY sozlanmagan.")
        url = (
            f"{self.CHECKOUT_BASE}?service_id={self.service_id}"
            f"&merchant_id={self.merchant_id}&amount={amount_uzs}"
            f"&transaction_param={order_id}&return_url={return_url}"
        )
        return PaymentIntent(
            intent_id=f"click_{order_id}",
            provider=self.name,
            amount_uzs=amount_uzs,
            checkout_url=url,
            status="created",
            created_at=time.time(),
        )

    def verify_webhook(self, headers: dict, body: dict) -> bool:
        """Click `sign_string` ni MD5 bilan tekshiradi."""
        received = body.get("sign_string")
        if not received:
            return False
        raw = (
            f"{body.get('click_trans_id','')}{body.get('service_id','')}"
            f"{self.secret_key}{body.get('merchant_trans_id','')}"
            f"{body.get('amount','')}{body.get('action','')}"
            f"{body.get('sign_time','')}"
        )
        expected = hashlib.md5(raw.encode("utf-8")).hexdigest()  # noqa: S324 — Click shuni talab qiladi
        return expected == received


def get_provider(name: Optional[str] = None) -> PaymentProvider:
    """`.env` dagi sozlamaga qarab provayderni qaytaradi.

    Provayder tanlangan, lekin kalitlari yo'q bo'lsa — `mock` ga tushamiz va
    bu holat `configured=False` orqali API javobida ko'rinadi.
    """
    settings = get_settings()
    choice = (name or settings.payment_provider or "mock").lower()

    if choice == "payme":
        provider = PaymeProvider(settings.payme_merchant_id, settings.payme_key)
    elif choice == "click":
        provider = ClickProvider(
            settings.click_merchant_id, settings.click_service_id, settings.click_secret_key
        )
    else:
        return MockProvider()

    return provider if provider.configured else MockProvider()


# --------------------------------------------------------------------------- #
# Loyihaga to'lov tizimi tavsiyasi
# --------------------------------------------------------------------------- #


def recommend_payment(
    project_type: str, signals: dict, region: str = "uz", lang: str = "uz"
) -> Optional[dict]:
    """Qurilayotgan startup uchun qaysi to'lov tizimi mos kelishi."""
    needs_payment = signals.get("payments") or project_type in (
        "ecommerce", "marketplace", "fintech", "booking_service", "edtech",
        "content_media", "saas_dashboard", "delivery_logistics",
    )
    if not needs_payment:
        return None

    if project_type == "marketplace":
        prefix = "pay.marketplace"
    elif project_type == "fintech":
        prefix = "pay.fintech"
    elif region == "uz":
        prefix = "pay.local"
    else:
        prefix = "pay.intl"

    # Mahalliy to'lov uchun "nega" matni stack tavsiyasi bilan bir xil, shuning
    # uchun takrorlanmaydi — bitta kalitdan ikkala joyda foydalaniladi.
    why_key = "why.pay.uz" if prefix == "pay.local" else (
        "why.pay.intl" if prefix == "pay.intl" else f"{prefix}.why"
    )

    return {
        "provider": tr(f"{prefix}.provider", lang),
        "why": tr(why_key, lang),
        "integration": tr(f"{prefix}.integration", lang),
        "caveat": tr(f"{prefix}.caveat", lang),
    }
