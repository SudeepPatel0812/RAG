"""
ENSEMBLE RETRIEVER (HYBRID SEARCH): keyword search + vector search, results merged.

The problem it solves:
    vector search understands meaning but misses exact terms (IDs, codes, names);
    BM25 finds exact terms but misses synonyms. Each fails where the other works.

The idea in simple terms:
    Ask two librarians: one who searches by MEANING and one who searches by exact
    WORDS. A book both of them recommend goes to the top of the pile.

The flow (what ensemble.invoke(question) does):

    question
       |------------------------------|
       v                              v
    BM25 retriever (keyword)       vector retriever (Pinecone, meaning)
       |  ranked list A               |  ranked list B
       |------------------------------|
                      v
    Reciprocal Rank Fusion (RRF): every doc gets  weight / (60 + rank)  from each list
    it appears in; scores are added -> docs found by BOTH lists rise to the top
                      v
    list[Document] (merged, no duplicates, re-ordered)

Why RRF uses the RANK, not the score:
    BM25 scores (e.g. 7.3) and cosine scores (e.g. 0.73) are on different scales and
    can't be added. Positions (1st, 2nd, 3rd) can always be compared.

Functions used:
    BM25Retriever.from_documents(docs, k=3)          -> keyword retriever (in memory)
    PineconeVectorStore(index_name=...).as_retriever -> reconnects to the "multi-query"
                                                        index (indexed by multi_query_retriever.py)
    EnsembleRetriever(retrievers=[a, b], weights=[0.5, 0.5])
        weights -> how much each list counts (e.g. [0.3, 0.7] trusts vectors more)

When to use:
    + the production default for many RAG systems: robust to both synonyms and exact terms
    + no LLM call: only 2 searches, so it's fast and cheap
    - two indexes to keep in sync (here BM25 is rebuilt in memory every run)
    - weights need tuning on real questions
    Pinecone, Weaviate, Qdrant and Elasticsearch can also do hybrid search natively.
"""

import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

from multi_query_data import documents

load_dotenv()

docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in documents]


def tokenize(text):
    return re.findall(r"\w+", text.lower())  # see bm25_retriever.py: default split() keeps punctuation


bm25_retriever = BM25Retriever.from_documents(docs, k=3, preprocess_func=tokenize)

vector_store = PineconeVectorStore(
    index_name="multi-query",  # already filled by multi_query_retriever.py
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

question = "My laptop broke, can I get a loaner from the P2 Service Desk?"

for name, retriever in [("BM25", bm25_retriever), ("vector", vector_retriever), ("ensemble", ensemble)]:
    print(f"== {name} ==")
    for i, doc in enumerate(retriever.invoke(question), start=1):
        print(i, doc.metadata["topic"], "|", doc.page_content[:70])
    print()
