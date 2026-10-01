from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from convolve.config import get_settings
from convolve.db.models import Base


class Database:
    def __init__(self, url: str, echo: bool = False) -> None:
        self._url = url
        connect_args: dict = {}
        if url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        self._engine: AsyncEngine = create_async_engine(
            url, echo=echo, pool_pre_ping=True, connect_args=connect_args
        )
        self._factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def url(self) -> str:
        return self._url

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._factory() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    async def create_all(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def dispose(self) -> None:
        await self._engine.dispose()


@lru_cache(maxsize=1)
def get_database() -> Database:
    return Database(get_settings().database_url)


def reset_database_cache() -> None:
    get_database.cache_clear()
