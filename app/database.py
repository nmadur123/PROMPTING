"""Async SQLAlchemy sozlamasi."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


def _add_missing_columns(conn) -> None:
    """Mavjud jadvallarga yangi ustunlarni qo'shadi.

    `create_all` faqat YANGI jadval yaratadi — mavjud jadvalga ustun qo'shmaydi.
    Alembic bu loyiha uchun ortiqcha, lekin ustun qo'shilganda bazani o'chirib
    yuborish ham yaramaydi: unda foydalanuvchilar va butun tarix yo'qoladi.
    Shuning uchun yetishmayotgan ustunlar shu yerda qo'shiladi.

    Idempotent: ustun allaqachon bo'lsa o'tkazib yuboriladi.
    """
    import logging

    from sqlalchemy import inspect, text

    log = logging.getLogger(__name__)
    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names())

    # jadval -> [(ustun nomi, SQL turi va standarti)]
    wanted: dict[str, list[tuple[str, str]]] = {
        "users": [
            ("credits", "INTEGER NOT NULL DEFAULT 0"),
            ("telegram", "VARCHAR(64) NOT NULL DEFAULT ''"),
        ],
    }

    for table, columns in wanted.items():
        if table not in existing_tables:
            continue  # create_all uni to'liq holda yaratadi
        have = {c["name"] for c in inspector.get_columns(table)}
        for name, ddl in columns:
            if name in have:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
            log.info("Ustun qo'shildi: %s.%s", table, name)


async def init_db() -> None:
    """Jadvallarni yaratadi. Kichik loyihada alembic o'rniga shu yetarli."""
    # Modellar import qilinmasa, `create_all` ularni ko'rmaydi.
    from app.models import billing, blueprint, estimate, user  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
