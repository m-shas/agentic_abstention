from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from convolve.config import Settings, reset_settings_cache
from convolve.db.session import Database, reset_database_cache
from convolve.proxy.app import create_app


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every test runs with its own database, state store, and artefact dirs."""
    db_path = tmp_path / "test.db"
    traj_dir = tmp_path / "trajectories"
    art_dir = tmp_path / "artifacts"

    monkeypatch.setenv("CONVOLVE_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("CONVOLVE_REDIS_URL", "")
    monkeypatch.setenv("CONVOLVE_TRAJECTORY_DIR", str(traj_dir))
    monkeypatch.setenv("CONVOLVE_EVALUATOR_MODEL_PATH", str(art_dir / "evaluator.joblib"))
    monkeypatch.setenv("CONVOLVE_DISTILLER_CHECKPOINT", str(art_dir / "distiller.pt"))
    monkeypatch.setenv("CONVOLVE_DEFAULT_THRESHOLD", "0.70")
    monkeypatch.setenv("CONVOLVE_MAX_STEPS", "8")
    monkeypatch.setenv("CONVOLVE_DEFAULT_BUDGET", "0.50")
    monkeypatch.setenv("CONVOLVE_COST_KNEE", "0.50")

    reset_settings_cache()
    reset_database_cache()
    yield
    reset_settings_cache()
    reset_database_cache()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest_asyncio.fixture
async def db(settings: Settings):
    database = Database(settings.database_url)
    await database.create_all()
    try:
        yield database
    finally:
        await database.drop_all()
        await database.dispose()


@pytest.fixture
def app(settings: Settings):
    """FastAPI app with an in-memory state store and a fresh SQLite database."""
    application = create_app(settings)
    with TestClient(application) as client:
        # TestClient runs lifespan, so the container is on app.state.
        # Ensure tables exist for the test.
        import anyio

        async def _create():
            db = application.state.container.db
            await db.create_all()

        anyio.run(_create)
        yield application
        _ = client


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


os.environ.setdefault("PYTEST_RUNNING", "1")
