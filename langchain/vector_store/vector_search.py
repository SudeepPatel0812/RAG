"""
QUERYING: search an existing vector store, run on every question.

Why a separate script:
    Indexing (vector_store.py) is a batch job; search is the online path.
    Each script starts with empty memory, so here we only RECONNECT to the
    index by name. The data lives in Pinecone, not in Python.

What similarity_search_with_score does:
    1. embed the question with the SAME model used for indexing
       (different model = different vector space = meaningless scores)
    2. send that vector to Pinecone
    3. Pinecone scores and sorts with an ANN index (fast, approximate),
       instead of our Python cosine loop over every vector (exact, O(N))
    4. returns top k as (Document, score), text rebuilt from metadata["text"]

Functions used:
    PineconeVectorStore(index_name, embedding, pinecone_api_key=None)
        -> connects by name; reads PINECONE_API_KEY from env if no key is passed
    similarity_search(query, k)            -> list[Document]
    similarity_search_with_score(query, k) -> list[(Document, score)], higher = closer
    filter={"category": "HR"}              -> metadata filter, like SQL WHERE
    as_retriever(search_kwargs={"k": 2})   -> same search behind LangChain's
                                              Retriever interface (used by chains)

Remember: search ALWAYS returns k results, even for unrelated questions.
Use the score (threshold) to decide whether a result is actually relevant.
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone  # the client (admin); not from langchain_pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(
    index_name="rag-learning",
    embedding=embeddings,
)

# ---------- 1. VECTOR STORE: search directly, you get scores ----------
print("== vector_store.similarity_search_with_score ==")
result = vector_store.similarity_search_with_score("How do I reset my password?", k=2)

for doc, score in result:
    print(round(score, 3), doc.metadata["category"], "|", doc.page_content[:80])


"""
RETRIEVER: "give me a question, I give you the relevant Documents".

Vector store vs retriever:
    vector store -> a STORAGE object: add_documents, delete, similarity_search(query, k),
                    with_score, filters... many methods, you choose each call.
    retriever    -> a SEARCH-ONLY object with ONE method: invoke(query) -> list[Document].
                    How it searches (k, threshold, mmr, filter) is fixed when you create it.

Why the retriever exists:
    1. One interface for everything: Pinecone, FAISS, BM25 keyword search, a web
       search, or your own class. The chain only calls .invoke(question), so you
       can swap the search without touching the rest (like a Go interface / DI).
    2. It is a LangChain Runnable, so it plugs into chains:
       {"context": retriever, "question": ...} | prompt | llm
    3. Search config lives in one place, not in every call.

as_retriever(search_type=..., search_kwargs=...):
    "similarity"                 -> top k by score (default, k=4)
    "similarity_score_threshold" -> top k, but drop results below score_threshold
                                    (fixes "search ALWAYS returns k results")
    "mmr"                        -> Max Marginal Relevance: fetch fetch_k candidates,
                                    then pick k that are relevant AND different from
                                    each other (less duplicate context)
    search_kwargs={"filter": {...}} -> metadata filter, works with all three

Scores: the threshold uses RELEVANCE scores (0..1), not the raw cosine from
similarity_search_with_score. For cosine, LangChain converts: (raw + 1) / 2.
    password question: raw 0.730 -> relevance 0.865 (IT Support, correct)
    weather in Paris : raw 0.442 -> relevance 0.721 (Engineering, irrelevant)
    So score_threshold=0.8 keeps the first and drops the second. Tune it on real questions.

The retriever returns Documents only, no scores: it's what the chain needs (text for the prompt).
"""

# ---------- 2. RETRIEVER: same search, one method, config fixed up front ----------
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

print("\n== retriever (similarity, k=2) ==")
for doc in retriever.invoke("How do I reset my password?"):
    print(doc.metadata["category"], "|", doc.page_content[:80])

# threshold: an unrelated question returns NOTHING instead of 2 junk documents
threshold_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.8},
)

print("\n== retriever (score threshold 0.8) ==")
for question in ["How do I reset my password?", "What is the weather in Paris?"]:
    docs = threshold_retriever.invoke(question)
    print(question, "->", [doc.metadata["category"] for doc in docs])

# filter: only search HR documents
hr_retriever = vector_store.as_retriever(search_kwargs={"k": 2, "filter": {"category": "HR"}})

print("\n== retriever (filter category=HR) ==")
for doc in hr_retriever.invoke("benefits"):
    print(doc.metadata["subcategory"], "|", doc.page_content[:80])

# mmr: 2 results that are relevant but not near-duplicates of each other
mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5})

print("\n== retriever (mmr) ==")
for doc in mmr_retriever.invoke("benefits"):
    print(doc.metadata["category"], "|", doc.page_content[:80])
