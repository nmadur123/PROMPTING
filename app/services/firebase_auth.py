"""Firebase ID tokenini tekshirish.

`firebase-admin` ishlatilmaydi — u service account kaliti talab qiladi, bu esa
yana bitta sir saqlash demak. Firebase ID tokeni oddiy RS256 JWT, va Google
uni tekshirish uchun ochiq sertifikatlarni e'lon qiladi. Loyiha ID si yetarli.

DIQQAT: token client'dan keladi, shuning uchun undagi email'ga ISHONIB
BO'LMAYDI — imzo tekshirilmasa, har kim istalgan email bilan kira oladi.
Quyidagi tekshiruvlarning hammasi majburiy: imzo, `iss`, `aud`, `exp`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx
from jose import jwt
from jose.exceptions import JWTError

from app.config import get_settings

logger = logging.getLogger(__name__)

# Google shu yerda ID tokenlarni imzolagan ochiq kalitlarni e'lon qiladi.
CERT_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
ISSUER = "https://securetoken.google.com/{project_id}"

_certs: dict[str, str] = {}
_certs_expire_at: float = 0.0


class FirebaseAuthError(Exception):
    """Token yaroqsiz yoki tekshirib bo'lmadi."""


async def _get_certs(force: bool = False) -> dict[str, str]:
    """Google sertifikatlarini oladi va `Cache-Control` bo'yicha keshlaydi."""
    global _certs, _certs_expire_at
    if not force and _certs and time.time() < _certs_expire_at:
        return _certs

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(CERT_URL)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        if _certs:
            # Tarmoq uzilganda eski kalitlar bilan davom etamiz: ular hali ham
            # yaroqli, va kirishni butunlay to'xtatgandan yaxshiroq.
            logger.warning("Firebase sertifikatlari yangilanmadi, eskisi ishlatilyapti: %s", exc)
            return _certs
        raise FirebaseAuthError(f"Sertifikatlarni olib bo'lmadi: {exc}") from exc

    max_age = 3600
    cache_control = resp.headers.get("cache-control", "")
    for part in cache_control.split(","):
        part = part.strip()
        if part.startswith("max-age="):
            try:
                max_age = int(part.split("=", 1)[1])
            except ValueError:
                pass

    _certs = data
    _certs_expire_at = time.time() + max_age
    return _certs


async def verify_id_token(id_token: str) -> dict[str, Any]:
    """Tokenni tekshirib, payload qaytaradi. Yaroqsiz bo'lsa xato tashlaydi."""
    settings = get_settings()
    project_id = settings.firebase_project_id
    if not project_id:
        raise FirebaseAuthError("FIREBASE_PROJECT_ID sozlanmagan")

    try:
        header = jwt.get_unverified_header(id_token)
    except JWTError as exc:
        raise FirebaseAuthError("Token o'qilmadi") from exc

    kid = header.get("kid")
    if not kid:
        raise FirebaseAuthError("Tokenda kalit identifikatori (kid) yo'q")

    certs = await _get_certs()
    if kid not in certs:
        # Google kalitlarni davriy almashtiradi — bir marta majburan yangilaymiz.
        certs = await _get_certs(force=True)
    cert = certs.get(kid)
    if not cert:
        raise FirebaseAuthError("Tokenni imzolagan kalit topilmadi")

    try:
        payload = jwt.decode(
            id_token,
            cert,
            algorithms=["RS256"],
            audience=project_id,
            issuer=ISSUER.format(project_id=project_id),
            options={"verify_at_hash": False},
        )
    except JWTError as exc:
        raise FirebaseAuthError(f"Token tekshiruvdan o'tmadi: {exc}") from exc

    # `sub` — Firebase foydalanuvchi identifikatori. Bo'sh bo'lsa token soxta.
    if not payload.get("sub"):
        raise FirebaseAuthError("Tokenda foydalanuvchi identifikatori yo'q")

    return payload


def extract_profile(payload: dict[str, Any]) -> tuple[str, str, str, bool]:
    """Payloaddan (uid, email, ism, email tasdiqlanganmi) ni ajratadi."""
    uid = str(payload["sub"])
    email = str(payload.get("email") or "").lower().strip()
    name = str(payload.get("name") or "").strip()

    if not name and email:
        name = email.split("@")[0]

    # Qaysi provayder orqali kirgani — Google emailni o'zi tasdiqlaydi,
    # GitHub esa har doim emas.
    verified = bool(payload.get("email_verified"))
    return uid, email, name, verified


def provider_of(payload: dict[str, Any]) -> str:
    firebase = payload.get("firebase") or {}
    provider = firebase.get("sign_in_provider")
    return str(provider) if provider else "firebase"
