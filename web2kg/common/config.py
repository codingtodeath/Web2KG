import os
from dataclasses import dataclass, field

from dotenv import dotenv_values


@dataclass
class ModelConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)


def model_config(kind="chat", env_file=".env"):
    env = {**dotenv_values(env_file), **os.environ}
    defaults = {
        "chat": ("CHATANYWHERE", "deepseek-v4.1-flash"),
        "embed": ("SILICONFLOW", "BAAI/bge-m3"),
        "vision": ("CHATANYWHERE", "deepseek-v4.1-flash"),
    }
    provider, model = defaults[kind]
    provider = env.get(f"WEB2KG_{kind.upper()}_PROVIDER", provider).upper()
    model = env.get(f"WEB2KG_{kind.upper()}_MODEL", model)
    key = env.get(f"{provider}_API_KEY")
    base = env.get(f"{provider}_BASE_URL")
    if not key or not base:
        raise ValueError(f"Missing {provider}_API_KEY or {provider}_BASE_URL in {env_file}")
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return ModelConfig(provider, model, base, key)
