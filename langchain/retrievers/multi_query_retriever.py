"""
MULTI-QUERY RETRIEVER: ask the same question several ways, search with each, merge.

The problem it solves:
    A vector search compares ONE question vector with the document vectors.
    If the user's words differ from the documents' words, some relevant
    documents score low and are missed.
        question: "How do I get my money back after a work trip?"
        documents say: "out-of-pocket ... reimbursed", "per diem", "payroll run"
    One phrasing = one point in vector space = one "angle" on the answer.

The idea in simple terms:
    Like asking 3 colleagues to rephrase your question before searching the
    library. Each rephrasing lands near different books; together they find more.

The flow (what mqr.invoke(question) does):

    question
       |
       v
    1. LLM (qwen3) + prompt: "generate 3 different versions of the question,
       separated by newlines"                      -> 1 LLM call
       |
       v
    2. split the reply into lines -> ["How are travel expenses reimbursed?",
                                      "What is the per diem for business trips?",
                                      "How do I claim out-of-pocket costs?"]
       (logged as "Generated queries" at INFO level)
       |
       v
    3. run the NORMAL retriever once per query     -> 3 searches, k docs each
       (each query is embedded, then searched in Pinecone)
       |
       v
    4. unique union: put all results in one list, drop duplicates
       (a doc found by 2 queries appears once)     -> about N x k documents
       (N = lines the LLM wrote, usually 3)
       |
       v
    list[Document]  (no scores, no re-ranking: order = first query's results first)

Functions used:
    MultiQueryRetriever.from_llm(retriever, llm)
        retriever -> the normal retriever it wraps (it reuses its k, filter, etc.)
        llm       -> writes the alternative questions
        prompt=   -> optional custom prompt (default asks for 3 versions)
        include_original=True -> also search with the user's original question
    mqr.invoke(question) -> list[Document], same interface as any retriever
    logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)
        -> prints the generated queries (the only way to see what the LLM wrote)
    ChatOllama(model, reasoning=False) -> qwen3 skips its <think> text, which
        would otherwise end up as extra "questions"

It is still a retriever: it wraps another retriever and has the same .invoke().
A chain can't tell the difference -> swap it in without changing anything else.

Trade-offs:
    + higher recall: finds docs with different wording
    + no change to the index or embeddings
    - an LLM call before every search: slower (seconds) and costs tokens
    - 3x the searches and embeddings
    - more documents -> longer prompt; some may be noise (distractors)
    - only as good as the rephrasings: a bad LLM rewrite searches the wrong thing

Related techniques:
    HyDE          -> LLM writes a fake ANSWER and you search with that
    RAG-Fusion    -> multi-query + rank results by how many queries found them (RRF)
    Query rewriting/decomposition -> split a complex question into sub-questions
"""

import logging
import time
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from multi_query_data import documents

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "multi-query"

if not pc.has_index(index_name):  # like CREATE TABLE IF NOT EXISTS
    pc.create_index(
        name=index_name,
        dimension=768,  # nomic-embed-text gives 768 numbers per text
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index
)

docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in documents]
ids = [d["id"] for d in documents]

vector_store.add_documents(documents=docs, ids=ids)
time.sleep(10)  # wait until Pinecone can search the new vectors


question = "How do I get my money back after a work trip?"

# ---------- BASELINE: one question, one search ----------
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

print("== normal retriever ==")
for doc in retriever.invoke(question):
    print(doc.metadata["topic"], "|", doc.page_content[:70])

# ---------- MULTI-QUERY: LLM writes versions, each is searched, results merged ----------
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)  # prints the generated questions

mqr = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=ChatOllama(model="qwen3:14b", reasoning=False),  # reasoning=False: no thinking text in the output
)

print("\n== multi-query retriever ==")
for i, doc in enumerate(mqr.invoke(question), start=1):
    print(i, doc.metadata["topic"], "|", doc.page_content[:70])

