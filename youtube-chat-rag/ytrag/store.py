"""Pinecone storage: one namespace per video, idempotent ingestion, no orphan chunks."""

import logging

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Settings
from .embeddings import make_embeddings
from .transcript import Chunk, video_url

log = logging.getLogger(__name__)


def get_index(settings: Settings):
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not pc.has_index(settings.index_name):
        log.info("creating index %s", settings.index_name)
        pc.create_index(  # waits until the index is ready
            name=settings.index_name,
            dimension=settings.embed_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )
    return pc.Index(settings.index_name)


def vector_store(settings: Settings, index, video_id: str) -> PineconeVectorStore:
    # namespace = video_id: every read/write is scoped to one video, no filter to forget
    return PineconeVectorStore(index=index, embedding=make_embeddings(settings.embed_model), namespace=video_id)


def chunk_id(video_id: str, chunk: Chunk) -> str:
    return f"{video_id}#{chunk.index}"


def existing_ids(index, video_id: str) -> set[str]:
    return {vid for page in index.list(namespace=video_id) for vid in page}


def to_documents(video_id: str, chunks: list[Chunk]) -> list[Document]:
    return [
        Document(
            page_content=c.text,
            metadata={"video_id": video_id, "chunk": c.index, "start": c.start, "url": video_url(video_id, c.start)},
        )
        for c in chunks
    ]


def upsert_video(settings: Settings, index, video_id: str, chunks: list[Chunk]) -> int:
    """Upsert this video's chunks, then delete ids from an older chunking that are no longer produced.
    Upsert-then-delete means the video is never empty mid-ingest."""
    old_ids = existing_ids(index, video_id)
    ids = [chunk_id(video_id, c) for c in chunks]

    vector_store(settings, index, video_id).add_documents(to_documents(video_id, chunks), ids=ids)

    stale = sorted(old_ids - set(ids))
    if stale:
        index.delete(ids=stale, namespace=video_id)
        log.info("deleted %d stale chunks from an older ingest", len(stale))
    return len(ids)
