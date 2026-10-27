"""Settings from environment variables (and .env), read and validated once at startup."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    index_name: str = "youtube-transcripts"
    cloud: str = "aws"
    region: str = "us-east-1"
    embed_model: str = "nomic-embed-text"
    embed_dimension: int = 768  # must match embed_model's output size
    chat_model: str = "qwen3:14b"
    window_seconds: int = 60  # chunk size, in seconds of speech
    overlap_snippets: int = 2  # caption lines repeated from the previous chunk
    k: int = 4  # chunks per question
    # relevance (0..1) below which a chunk is ignored. Measured: on-topic questions ~0.88,
    # unrelated ones ~0.73-0.77 (with nomic prefixes).
    # shortcut: tuned on one video, re-check with an eval set once there are more videos
    score_threshold: float = 0.8
    languages: tuple[str, ...] = ("en",)


def load_settings() -> Settings:
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ConfigError("PINECONE_API_KEY is not set (put it in .env or the environment)")

    env = os.getenv
    return Settings(
        pinecone_api_key=api_key,
        index_name=env("YTRAG_INDEX", Settings.index_name),
        cloud=env("YTRAG_CLOUD", Settings.cloud),
        region=env("YTRAG_REGION", Settings.region),
        embed_model=env("YTRAG_EMBED_MODEL", Settings.embed_model),
        embed_dimension=int(env("YTRAG_EMBED_DIMENSION", Settings.embed_dimension)),
        chat_model=env("YTRAG_CHAT_MODEL", Settings.chat_model),
        window_seconds=int(env("YTRAG_WINDOW_SECONDS", Settings.window_seconds)),
        overlap_snippets=int(env("YTRAG_OVERLAP_SNIPPETS", Settings.overlap_snippets)),
        k=int(env("YTRAG_K", Settings.k)),
        score_threshold=float(env("YTRAG_SCORE_THRESHOLD", Settings.score_threshold)),
        languages=tuple(env("YTRAG_LANGUAGES", ",".join(Settings.languages)).split(",")),
    )
