"""
EMBEDDINGS FILTER + COMPRESSOR PIPELINE: cheap contextual compression, NO LLM.

The problem it solves:
    LLMChainExtractor (contextual_compression_retriever.py) makes 1 LLM call PER
    document: great quality, but slow and expensive at k=10 or k=20.
    Often you only need to throw away the obvious junk, and that can be done with
    embeddings alone.

The idea in simple terms:
    A quick bouncer at the door: "is this close enough to the question? no -> out".
    No reading, just a similarity check.

The flow (pipeline of compressors, applied in order):

    question -> base retriever -> k documents
       |
       v
    1. RecursiveCharacterTextSplitter  -> cut each document into sentence-sized pieces
       |
       v
    2. EmbeddingsRedundantFilter       -> drop pieces that are near-duplicates of each other
       |
       v
    3. EmbeddingsFilter(threshold)     -> embed the question and each piece, keep the
                                          top k pieces (k=20 by default), then only those
                                          with similarity > threshold
       |
       v
    list[Document] -> only the relevant PIECES (like the extractor, but without an LLM)

Functions used:
    EmbeddingsFilter(embeddings, similarity_threshold=0.6)
        threshold = RAW cosine similarity (-1..1), NOT the (raw + 1) / 2 relevance score
        used by as_retriever(search_type="similarity_score_threshold").
        So 0.6 here ~ 0.8 there. Tune it on real questions.
        k (default 20) is applied first and also caps the count; k=None -> threshold only
    EmbeddingsRedundantFilter(embeddings) -> removes near-identical pieces
    DocumentCompressorPipeline(transformers=[splitter, redundant_filter, relevant_filter])
        -> runs them in order; it plugs into the same base_compressor slot
    ContextualCompressionRetriever(base_retriever, base_compressor) -> same wrapper as before

When to use:
    + many retrieved docs, need a cheap first cut (milliseconds, no LLM cost)
    + as stage 1 before an LLM compressor or a reranker (cheap filter -> expensive step)
    - only as smart as the embedding similarity: no real "understanding"
    - the threshold is fiddly: too high drops good pieces, too low keeps junk
"""

import os

from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(
    index_name="contextual-compression",  # already filled by contextual_compression_retriever.py
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

pipeline = DocumentCompressorPipeline(
    transformers=[
        RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0, separators=[". "], keep_separator="end"),
        EmbeddingsRedundantFilter(embeddings=embeddings),
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.6),
    ]
)

ccr = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=pipeline)

question = "How many vacation days do I get?"

print("== normal retriever ==")
for i, doc in enumerate(retriever.invoke(question), start=1):
    print(i, doc.metadata["source"], len(doc.page_content), "chars")

print("\n== embeddings filter pipeline (no LLM) ==")
for i, doc in enumerate(ccr.invoke(question), start=1):
    print(i, doc.metadata["source"], len(doc.page_content), "chars |", doc.page_content)
