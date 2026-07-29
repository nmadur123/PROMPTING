"""sys42 backendiga HTTP mijoz.

Har bir metod tayyor `dict` qaytaradi yoki `ApiError` ko'taradi. Server
xatolarini shu yerda o'qiladigan matnga aylantiramiz — buyruqlar fayli
`resp.status_code` bilan shug'ullanmasin.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from .config import Session

# Texnik topshiriq yozish eng kuchli modelga boradi va bu bir necha daqiqa
# davom etishi mumkin. httpx ning standart 5 soniyasi bu yerda so'rovni
# doim uzib qo'yardi.
_TIMEOUT = httpx.Timeout(600.0, connect=15.0)


class ApiError(RuntimeError):
    """Server yoki tarmoq xatosi — matni foydalanuvchiga ko'rsatish uchun tayyor."""

    def __init__(self, message: str, *, status: Optional[int] = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class Api:
    def __init__(self, session: Session):
        self.session = session

    # ---------------------------------------------------------------- ichki

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.session.token:
            headers["Authorization"] = f"Bearer {self.session.token}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.session.api_url}{path}"
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.request(method, url, headers=self._headers(), **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(
                "Server javob bermadi (vaqt tugadi). Yana urinib ko'ring."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Serverga ulanib bo'lmadi: {self.session.api_url}\n  {exc}") from exc

        if resp.status_code == 204:
            return None
        if resp.is_success:
            return resp.json()

        raise self._to_error(resp)

    @staticmethod
    def _to_error(resp: httpx.Response) -> ApiError:
        try:
            body = resp.json()
        except ValueError:
            body = None

        detail = body.get("detail") if isinstance(body, dict) else None

        if resp.status_code == 401:
            return ApiError("Sessiya tugagan. `sys42 login` bilan qayta kiring.", status=401)

        if resp.status_code == 402 and isinstance(detail, dict):
            used, limit = detail.get("used", "?"), detail.get("limit", "?")
            return ApiError(
                f"Limit tugadi ({used}/{limit}, tarif: {detail.get('plan', '?')}).\n"
                "  Tarifni sys42.xyz sahifasidan oshirishingiz mumkin.",
                status=402,
                payload=detail,
            )

        if isinstance(detail, str):
            return ApiError(detail, status=resp.status_code, payload=body)
        if isinstance(detail, list) and detail:
            # FastAPI validatsiya xatosi: [{"loc": [...], "msg": ...}, ...]
            first = detail[0]
            field = ".".join(str(p) for p in first.get("loc", [])[1:]) or "so'rov"
            return ApiError(f"{field}: {first.get('msg', 'noto‘g‘ri qiymat')}", status=422)

        return ApiError(f"Server xatosi {resp.status_code}", status=resp.status_code, payload=body)

    # ---------------------------------------------------------------- ommaviy

    def health(self) -> dict:
        return self._request("GET", "/api/health")

    def login(self, email: str, password: str) -> dict:
        return self._request("POST", "/api/auth/login", json={"email": email, "password": password})

    def me(self) -> dict:
        return self._request("GET", "/api/auth/me")

    def models(self, only_popular: bool = True) -> list[dict]:
        # Endpoint `{count, families, models: [...]}` qaytaradi — chaqiruvchiga
        # faqat ro'yxatning o'zi kerak.
        data = self._request("GET", "/api/models", params={"only_popular": only_popular})
        return (data or {}).get("models", [])

    def clarify(self, description: str, lang: str = "uz", count: int = 5) -> dict:
        return self._request(
            "POST", "/api/clarify",
            json={"description": description, "lang": lang, "count": count},
        )

    def generate(
        self,
        *,
        description: str,
        target_model: str,
        answers: list[dict],
        monthly_users: int = 1000,
        region: str = "uz",
        lang: str = "uz",
        save: bool = True,
    ) -> dict:
        return self._request(
            "POST", "/api/generate",
            json={
                "description": description,
                "monthly_users": monthly_users,
                "region": region,
                "lang": lang,
                "target_model": target_model,
                "answers": answers,
                "save": save,
                # Domen tekshiruvi tashqi so'rovlar qo'shadi va terminalda
                # ko'rsatilmaydi — CLI da o'chirilgan.
                "check_domains": False,
            },
        )
