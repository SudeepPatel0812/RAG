"""Video id parsing, transcript fetching and time-window chunking."""

import logging
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi, YouTubeTranscriptApiException

log = logging.getLogger(__name__)

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Snippet:
    text: str
    start: float
    duration: float


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int  # seconds from the beginning of the video


def parse_video_id(url_or_id: str) -> str:
    """Accept a bare id or any common YouTube URL form; return the 11-char video id."""
    value = url_or_id.strip()
    if VIDEO_ID.match(value):
        return value

    url = urlparse(value if "://" in value else "https://" + value)
    host = url.netloc.lower().removeprefix("www.").removeprefix("m.")
    candidate = None
    if host == "youtu.be":
        candidate = url.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if url.path == "/watch":
            candidate = parse_qs(url.query).get("v", [None])[0]
        else:
            parts = url.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                candidate = parts[1]

    if candidate and VIDEO_ID.match(candidate):
        return candidate
    raise ValueError(f"Not a YouTube video URL or id: {url_or_id!r}")


def fetch_snippets(video_id: str, languages: tuple[str, ...]) -> list[Snippet]:
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    except YouTubeTranscriptApiException as e:  # disabled, not found, video unavailable, blocked...
        raise TranscriptUnavailable(f"No transcript for {video_id}: {type(e).__name__}") from e
    snippets = [Snippet(s.text, s.start, s.duration) for s in fetched if s.text.strip()]
    log.info("fetched %d caption lines for %s", len(snippets), video_id)
    return snippets


def chunk_snippets(snippets: list[Snippet], window_seconds: int, overlap_snippets: int) -> list[Chunk]:
    """Group caption lines into ~window_seconds chunks; each chunk repeats the last
    `overlap_snippets` lines of the previous one so a sentence on the boundary isn't cut."""
    if window_seconds <= 0 or overlap_snippets < 0:
        raise ValueError("window_seconds must be > 0 and overlap_snippets >= 0")

    chunks: list[Chunk] = []
    current: list[Snippet] = []

    def flush() -> None:
        chunks.append(Chunk(len(chunks), " ".join(s.text for s in current), int(current[0].start)))

    for snippet in snippets:
        current.append(snippet)
        if snippet.start + snippet.duration - current[0].start >= window_seconds:
            flush()
            # keep overlap, but always drop at least one line so every chunk makes progress
            current = current[-overlap_snippets:] if 0 < overlap_snippets < len(current) else []

    if len(current) > (overlap_snippets if chunks else 0):  # leftover that isn't only overlap
        flush()
    return chunks


def timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)  # int(): Pinecone returns numbers as floats
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def video_url(video_id: str, seconds: float = 0) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"
