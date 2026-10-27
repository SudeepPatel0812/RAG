"""nomic-embed-text with the task prefixes it was trained on.

OllamaEmbeddings sends text as-is. nomic-embed-text expects "search_document: " on indexed text
and "search_query: " on questions; without them retrieval quality drops. This adds them in one place.
"""

from langchain_ollama import OllamaEmbeddings


class NomicEmbeddings(OllamaEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await super().aembed_documents([f"search_document: {t}" for t in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return await super().aembed_query(f"search_query: {text}")


def make_embeddings(model: str) -> OllamaEmbeddings:
    # only nomic models use these prefixes; other models get plain OllamaEmbeddings
    return NomicEmbeddings(model=model) if "nomic" in model else OllamaEmbeddings(model=model)
