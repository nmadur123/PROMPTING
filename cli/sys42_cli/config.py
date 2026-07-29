"""CLI sozlamalari va sessiya — `~/.sys42/config.json`.

Token diskda oddiy matnda yotadi. Bu `gh`, `docker`, `npm` va boshqalar
qiladigan ishning o'zi, lekin fayl huquqi 0600 ga qo'yiladi — bir mashinada
bir necha foydalanuvchi bo'lsa, birov ikkinchisining tokenini o'qiy olmasin.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

# Production backend. `SYS42_API_URL` bilan ustidan yozish mumkin — lokal
# ishlab chiqishda `http://localhost:8010` ga qaratib qo'yiladi.
DEFAULT_API_URL = "https://prompting-production.up.railway.app"

CONFIG_DIR = Path(os.environ.get("SYS42_HOME") or (Path.home() / ".sys42"))
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class Session:
    api_url: str = DEFAULT_API_URL
    token: str = ""
    email: str = ""

    @property
    def logged_in(self) -> bool:
        return bool(self.token)


def load() -> Session:
    """Saqlangan sessiyani o'qiydi. Fayl buzilgan bo'lsa — bo'sh sessiya.

    Buzilgan JSON uchun ataylab xato ko'tarilmaydi: foydalanuvchi `sys42 login`
    qilib chiqa olishi kerak, qo'lda fayl tozalash talab qilinmasin.
    """
    env_url = os.environ.get("SYS42_API_URL", "").strip().rstrip("/")

    if not CONFIG_PATH.exists():
        return Session(api_url=env_url or DEFAULT_API_URL)

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return Session(api_url=env_url or DEFAULT_API_URL)

    session = Session(
        api_url=str(raw.get("api_url") or DEFAULT_API_URL).rstrip("/"),
        token=str(raw.get("token") or ""),
        email=str(raw.get("email") or ""),
    )
    # Muhit o'zgaruvchisi saqlangan qiymatdan ustun — bitta buyruq uchun
    # boshqa serverga qarash kerak bo'lganda qulay.
    if env_url:
        session.api_url = env_url
    return session


def save(session: Session) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(session), indent=2), encoding="utf-8")
    try:
        CONFIG_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        # Windows'da bu ko'pincha ma'nosiz — sozlamani saqlashdan voz
        # kechishga arzimaydi.
        pass


def clear() -> None:
    CONFIG_PATH.unlink(missing_ok=True)
