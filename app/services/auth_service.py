"""Parol xeshlash va JWT.

`bcrypt` to'g'ridan-to'g'ri chaqiriladi: passlib 1.7 bcrypt 4.x da versiyani
o'qiyolmay har chaqiruvda traceback yozadi, holbuki bu yerda passlib beradigan
qo'shimcha imkoniyatlarning hech biri kerak emas.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from app.config import get_settings

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TTL = timedelta(days=7)

# bcrypt 72 baytdan uzun parolni jimgina kesib tashlaydi — bu holda uzun
# parolning oxiri hech qanday himoya bermaydi. Shuning uchun oldindan rad
# etamiz, kesib emas.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LEN = 8


class AuthError(Exception):
    """Autentifikatsiya bilan bog'liq kutilgan xato."""


def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise AuthError("password_too_long")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # Buzilgan yoki boshqa algoritmda yozilgan xesh — kirish rad etiladi.
        logger.warning("Yaroqsiz parol xeshi uchradi")
        return False


def create_access_token(user_id: str, role: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + ACCESS_TTL).timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Token yaroqli bo'lsa payload, aks holda None qaytaradi."""
    try:
        return jwt.decode(token, get_settings().secret_key, algorithms=[ALGORITHM])
    except JWTError:
        return None
