from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from convolve.config import Settings, get_settings
from convolve.db.session import get_database
from convolve.logging_setup import configure_logging
from convolve.proxy.deps import build_container
from convolve.proxy.routes import router
from convolve.version import __version__

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_output=settings.env == "production")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_dirs()
        db = get_database()
        container = build_container(settings, db)
        app.state.container = container
        try:
            yield
        finally:
            await container.aclose()
            await db.dispose()

    app = FastAPI(
        title="Convolve",
        description=(
            "Agentic abstention and sequential uncertainty middleware. "
            "Intercepts LLM agent action proposals and decides whether to "
            "continue or abstain based on the full interaction trajectory."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.include_router(router)

    if STATIC_DIR.exists():
        app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def app_factory() -> FastAPI:  # pragma: no cover
    return create_app()
