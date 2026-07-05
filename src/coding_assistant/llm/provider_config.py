from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

OPENAI_API_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"

PROVIDER_ENV_KEYS = (OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV, OPENROUTER_API_KEY_ENV)


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str


def _env_value(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value if value else None


def worker_provider_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    provider_env = os.environ if env is None else env
    return {key: value for key in PROVIDER_ENV_KEYS if (value := _env_value(provider_env, key)) is not None}


def require_provider_key(env: Mapping[str, str] | None = None) -> None:
    provider_env = os.environ if env is None else env
    if (
        _env_value(provider_env, OPENAI_API_KEY_ENV) is None
        and _env_value(provider_env, OPENROUTER_API_KEY_ENV) is None
    ):
        raise ValueError(f"{OPENAI_API_KEY_ENV} or {OPENROUTER_API_KEY_ENV} must be set to start the manager.")


def resolve_provider_config(env: Mapping[str, str] | None = None) -> ProviderConfig:
    provider_env = os.environ if env is None else env
    api_key = _env_value(provider_env, OPENAI_API_KEY_ENV) or _env_value(provider_env, OPENROUTER_API_KEY_ENV)
    if api_key is None:
        raise KeyError(OPENAI_API_KEY_ENV)

    base_url = _env_value(provider_env, OPENAI_BASE_URL_ENV)
    if base_url is not None:
        return ProviderConfig(base_url=base_url, api_key=api_key)
    if _env_value(provider_env, OPENAI_API_KEY_ENV) is not None:
        return ProviderConfig(base_url=OPENAI_API_BASE_URL, api_key=api_key)
    return ProviderConfig(base_url=OPENROUTER_API_BASE_URL, api_key=api_key)
