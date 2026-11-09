"""
Centralized app configuration.

Everything environment-specific (where the model lives, which origins
may call the API, etc.) is read here and nowhere else, so the rest of
the codebase never hardcodes a path or a URL.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Where the fine-tuned transformer (tokenizer + model + labels.json)
    # was exported to. See backend/README.md for how to produce this folder
    # from the training notebook.
    MODEL_DIR: str = "Ahmed-Tarek0/complaint-classifier"

    # Must match the MAX_LEN_TR used when the model was fine-tuned.
    MAX_LENGTH: int = 150

    # How many ranked categories to return alongside the top prediction.
    TOP_K: int = 3

    # Origins allowed to call this API from a browser (the local frontend,
    # by default). Tighten this before deploying publicly.
    CORS_ORIGINS: list[str] = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
