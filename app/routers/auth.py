"""Ro'yxatdan o'tish, kirish, chiqish va joriy foydalanuvchi."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import EventKind, LoginEvent, Plan, Role, User, is_vip, utcnow
from app.services.firebase_auth import (
    FirebaseAuthError,
    extract_profile,
    provider_of,
    verify_id_token,
)
from app.services.quota import Quota, get_quota
from app.services.auth_service import (
    AuthError,
    MIN_PASSWORD_LEN,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_LEN, max_length=128)
    name: str = Field(default="", max_length=120)
    lang: str = Field(default="uz", pattern="^(uz|ru|en)$")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class FirebaseRequest(BaseModel):
    id_token: str = Field(min_length=20)
    lang: str = Field(default="uz", pattern="^(uz|ru|en)$")


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str
    plan: str
    provider: str
    lang: str
    estimates_used: int
    created_at: str
    # Kvota `/me` da beriladi, chunki interfeys uni har sahifada ko'rsatadi;
    # alohida chaqiruv qilish ortiqcha aylanma bo'lardi.
    quota_used: int = 0
    quota_limit: int | None = None
    quota_remaining: int | None = None
    resets_at: str = ""
    # Oylik limitdan tashqari sotib olingan analizlar
    credits: int = 0
    # Oylik qoldiq + kredit. None — cheksiz (VIP yoki Enterprise)
    total_remaining: int | None = None
    telegram: str = ""


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
    # Kirish so'ralgan, lekin hisob shu payt yaratilgan bo'lsa — interfeys
    # buni aytishi kerak, aks holda odam eski hisobiga kirdim deb o'ylaydi.
    created: bool = False


def _to_out(user: User, quota: Optional[Quota] = None) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role.value,
        plan=user.plan.value,
        provider=user.provider,
        lang=user.lang,
        estimates_used=user.estimates_used,
        created_at=user.created_at.isoformat(),
        quota_used=quota.used if quota else 0,
        quota_limit=quota.limit if quota else None,
        quota_remaining=quota.remaining if quota else None,
        resets_at=quota.resets_at if quota else "",
        credits=user.credits or 0,
        total_remaining=quota.total_remaining if quota else None,
        telegram=user.telegram or "",
    )


async def _record(db: AsyncSession, user_id: str, kind: EventKind, request: Request) -> None:
    db.add(LoginEvent(
        user_id=user_id,
        kind=kind,
        user_agent=(request.headers.get("user-agent") or "")[:200],
    ))


async def _create_user(
    db: AsyncSession,
    *,
    email: str,
    name: str,
    password_hash: str = "",
    lang: str = "uz",
    provider: str = "password",
    firebase_uid: Optional[str] = None,
) -> User:
    """Yangi foydalanuvchi.

    Superadmin — FAQAT bitta belgilangan email (`VIP_EMAIL`). Ilgari bu yerda
    "birinchi ro'yxatdan o'tgan odam ham superadmin" qoidasi bor edi: bazani
    tozalab, birinchi bo'lib kirgan istalgan odam panelga to'liq kirish
    huquqini olardi. Endi bunday emas.
    """
    vip = is_vip(email)
    user = User(
        email=email,
        name=name,
        password_hash=password_hash,
        role=Role.SUPERADMIN if vip else Role.USER,
        plan=Plan.ENTERPRISE if vip else Plan.SKETCH,
        lang=lang,
        provider=provider,
        firebase_uid=firebase_uid,
        last_seen_at=utcnow(),
    )
    db.add(user)
    await db.flush()
    return user


# --------------------------------------------------------------------------- #
# Joriy foydalanuvchi
# --------------------------------------------------------------------------- #


async def current_user(
    authorization: Annotated[Optional[str], Header()] = None,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Bearer tokendan foydalanuvchini oladi. Token yaroqsiz bo'lsa 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token yo'q")
    payload = decode_token(authorization[7:].strip())
    if not payload or not payload.get("sub"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token yaroqsiz yoki muddati tugagan")

    user = (await db.execute(select(User).where(User.id == payload["sub"]))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Foydalanuvchi topilmadi")
    return user


async def require_superadmin(user: User = Depends(current_user)) -> User:
    if user.role is not Role.SUPERADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Faqat superadmin uchun")
    return user


async def optional_user(
    authorization: Annotated[Optional[str], Header()] = None,
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Token bo'lsa foydalanuvchi, bo'lmasa None — xato tashlamaydi."""
    if not authorization:
        return None
    try:
        return await current_user(authorization, db)
    except HTTPException:
        return None


# --------------------------------------------------------------------------- #
# Endpointlar
# --------------------------------------------------------------------------- #


@router.post("/register", response_model=TokenOut)
async def register(
    body: RegisterRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> TokenOut:
    """Ro'yxatdan o'tish. Hisob allaqachon bo'lsa va parol to'g'ri kelsa — kirgizadi.

    Shuning uchun status kodi qat'iy emas: yangi hisob ochilsa 201, mavjudiga
    kirilsa 200. Kirishni 201 deb qaytarish "yangi hisob ochildi" degan ma'noni
    beradi va bu yolg'on bo'lardi.
    """
    email = body.email.lower().strip()

    existing = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if existing is not None:
        # Hisob bor ekan — parol to'g'ri bo'lsa, kirgizib yuboramiz. Qaytib
        # kelgan odam ko'pincha qaysi formada ekanini eslamaydi, va uni "bu email
        # band" deb qaytarish hech kimni himoya qilmaydi.
        #
        # Parol MAJBURIY tekshiriladi. Emailni bilgan har kimga token berish —
        # bu ro'yxatdan o'tish formasini hisob o'g'irlash vositasiga aylantiradi.
        if existing.password_hash and verify_password(body.password, existing.password_hash):
            existing.last_seen_at = utcnow()
            await _record(db, existing.id, EventKind.LOGIN, request)
            await db.commit()
            return TokenOut(
                access_token=create_access_token(existing.id, existing.role.value),
                user=_to_out(existing, await get_quota(db, existing)),
                created=False,
            )

        if not existing.password_hash:
            # Google/GitHub orqali ochilgan hisobda parol yo'q. "Parol noto'g'ri"
            # deyish yolg'on bo'lardi — odam parol o'ylab topishga urinadi.
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Bu hisob {existing.provider} orqali ochilgan — o'sha tugma bilan kiring",
            )

        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Bu email allaqachon ro'yxatdan o'tgan, lekin parol mos kelmadi",
        )

    try:
        password_hash = hash_password(body.password)
    except AuthError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Parol juda uzun") from None

    user = await _create_user(
        db, email=email, name=body.name.strip(), password_hash=password_hash,
        lang=body.lang, provider="password",
    )
    await _record(db, user.id, EventKind.REGISTER, request)
    await db.commit()

    response.status_code = status.HTTP_201_CREATED
    return TokenOut(
        access_token=create_access_token(user.id, user.role.value),
        user=_to_out(user, await get_quota(db, user)),
        created=True,
    )


@router.post("/login", response_model=TokenOut)
async def login(body: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)) -> TokenOut:
    """Kirish. Email bazada bo'lmasa — avtomatik ro'yxatdan o'tkazadi.

    Bu ataylab shunday: bir bosqichli kirish konversiyani oshiradi. Narxi bor —
    emailda xato qilgan odam eski hisobiga emas, yangi bo'sh hisobga tushadi va
    ma'lumotlarim yo'qoldi deb o'ylaydi. Shuning uchun javobda `created` bayrog'i
    bor va interfeys buni aniq aytadi.
    """
    email = body.email.lower().strip()
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()

    if user is None:
        # Parol talabi bu yerda ham amal qiladi — kirish formasi orqali zaif
        # parolli hisob yaratib bo'lmaydi.
        if len(body.password) < MIN_PASSWORD_LEN:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                f"Bunday email yo'q. Yangi hisob ochish uchun parol kamida {MIN_PASSWORD_LEN} belgidan iborat bo'lsin",
            )
        try:
            password_hash = hash_password(body.password)
        except AuthError:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Parol juda uzun") from None

        user = await _create_user(db, email=email, name=email.split("@")[0], password_hash=password_hash)
        await _record(db, user.id, EventKind.REGISTER, request)
        await db.commit()
        return TokenOut(
            access_token=create_access_token(user.id, user.role.value),
            user=_to_out(user, await get_quota(db, user)),
            created=True,
        )

    # Google yoki GitHub orqali ochilgan hisobda parol yo'q. "Parol noto'g'ri"
    # deyish bu yerda yolg'on bo'lardi — odam parol o'ylab topishga urinadi.
    if not user.password_hash:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Bu hisob {user.provider} orqali ochilgan — o'sha tugma bilan kiring",
        )

    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Email yoki parol noto'g'ri")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Hisob o'chirilgan")

    user.last_seen_at = utcnow()
    await _record(db, user.id, EventKind.LOGIN, request)
    await db.commit()

    return TokenOut(
        access_token=create_access_token(user.id, user.role.value),
        user=_to_out(user, await get_quota(db, user)),
    )


@router.post("/firebase", response_model=TokenOut, summary="Google / GitHub orqali kirish")
async def firebase_login(
    body: FirebaseRequest, request: Request, db: AsyncSession = Depends(get_db)
) -> TokenOut:
    """Firebase ID tokenini tekshirib, o'z sessiyamizni beradi.

    Tokendagi email'ga to'g'ridan-to'g'ri ishonilmaydi: imzo Google'ning ochiq
    sertifikatlari bilan tekshiriladi. Aks holda har kim istalgan email bilan
    kira olardi.
    """
    try:
        payload = await verify_id_token(body.id_token)
    except FirebaseAuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from None

    uid, email, name, _verified = extract_profile(payload)
    provider = provider_of(payload)

    if not email:
        # GitHub'da email yashirin bo'lishi mumkin. Bunda hisobni bog'lash uchun
        # boshqa yo'l kerak, shuning uchun aniq aytamiz.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Bu hisobda ochiq email yo'q. Provayder sozlamalarida emailni ochiq qiling",
        )

    # Avval firebase_uid bo'yicha: odam emailini o'zgartirsa ham hisob o'ziniki
    # bo'lib qoladi.
    user = (await db.execute(select(User).where(User.firebase_uid == uid))).scalar_one_or_none()
    created = False

    if user is None:
        by_email = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if by_email is not None:
            # Shu email bilan parol orqali ochilgan hisob bor — uni bog'laymiz,
            # ikkinchi hisob yaratmaymiz. Email Google/GitHub tomonidan
            # tasdiqlangani uchun bu xavfsiz.
            by_email.firebase_uid = uid
            if not by_email.name:
                by_email.name = name
            user = by_email
        else:
            user = await _create_user(
                db, email=email, name=name, lang=body.lang,
                provider=provider, firebase_uid=uid,
            )
            created = True

    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Hisob o'chirilgan")

    user.last_seen_at = utcnow()
    await _record(db, user.id, EventKind.REGISTER if created else EventKind.LOGIN, request)
    await db.commit()

    return TokenOut(
        access_token=create_access_token(user.id, user.role.value),
        user=_to_out(user, await get_quota(db, user)),
        created=created,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def logout(
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Tokenni server tomonda bekor qilmaydi — u qisqa muddatli va client uni
    o'chiradi. Bu yerda faqat hodisa yoziladi, statistika uchun."""
    await _record(db, user.id, EventKind.LOGOUT, request)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)) -> UserOut:
    return _to_out(user, await get_quota(db, user))
