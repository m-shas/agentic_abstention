from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. All fields are overridable via CONVOLVE_* env vars."""

    model_config = SettingsConfigDict(
        env_prefix="CONVOLVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    database_url: str = "sqlite+aiosqlite:///./convolve.db"
    redis_url: str = ""
    state_ttl_seconds: int = 3600

    default_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_steps: int = Field(default=12, ge=1, le=200)
    default_budget: float = Field(default=0.50, ge=0.0)
    cost_knee: float = Field(default=0.50, ge=0.0, lt=1.0)

    evaluator_model_path: Path = Path("artifacts/evaluator.joblib")
    distiller_checkpoint: Path = Path("artifacts/distiller.pt")
    trajectory_dir: Path = Path("data/trajectories")
    playbook_top_k: int = Field(default=2, ge=0, le=10)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def use_redis(self) -> bool:
        return bool(self.redis_url.strip())

    def ensure_dirs(self) -> None:
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.distiller_checkpoint.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests to pick up env changes."""
    get_settings.cache_clear()
