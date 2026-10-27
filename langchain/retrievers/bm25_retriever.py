"""
BM25 RETRIEVER: classic KEYWORD search (no embeddings, no vector database).

The idea in simple terms:
    Like Ctrl+F with a smart score. A document scores high when it contains the
    question's words, especially RARE words, and loses a bit for being long.
    "ExpenseHub" appears in 1 document -> a very strong signal.
    "the" appears everywhere -> almost worthless.

How BM25 scores (no need to memorise the formula):
    for each word in the question:
        TF  (term frequency)    -> how often the word is in this document (with diminishing returns)
        IDF (inverse doc freq)  -> how rare the word is across all documents (rare = important)
        length normalisation    -> long documents don't win just by having more words
    score = sum over words of IDF x adjusted TF

The flow:
    from_documents(docs) -> tokenise every document (split into words) -> word statistics, in memory
    invoke(question)     -> tokenise the question -> score every document -> top k

Vector search vs BM25:
    vector (semantic)  "money back" finds "reimbursed"     -> understands MEANING, synonyms
                       but can miss exact codes: "P2", "ExpenseHub", "BitLocker", error IDs
    BM25 (keyword)     "ExpenseHub" finds "ExpenseHub" exactly
                       but "money back" does NOT find "reimbursed" (no shared words)
    They fail in opposite places -> combine them (see ensemble_retriever.py).

Functions used:
    BM25Retriever.from_documents(docs, k=3, preprocess_func=tokenize)
        -> builds the keyword index in memory (needs rank-bm25)
    re.findall(r"\w+", text.lower()) -> "How do I use ExpenseHub?" -> ["how", "do", "i", "use", "expensehub"]
    retriever.invoke(question)              -> list[Document], same interface as any retriever

When to use:
    + exact terms: product names, IDs, error codes, people's names, acronyms
    + no embedding model, instant to build, free, explainable ("it matched these words")
    - no synonyms or meaning, and "laptop" != "laptops" (no stemming here)
    - the index lives in memory and is rebuilt every run (fine for small/medium data)

GOTCHA (seen in this file): the default tokeniser is a plain text.split(), so
    "ExpenseHub?" (with the ?) never matches "ExpenseHub", and "The" != "the".
    Without a tokeniser, "How do I use ExpenseHub?" ranked the coffee machine first.
    Fix: preprocess_func=tokenize -> lowercase + keep only words.
"""

import re

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from multi_query_data import documents

docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in documents]


def tokenize(text):
    return re.findall(r"\w+", text.lower())  # lowercase words only, punctuation dropped


bm25_retriever = BM25Retriever.from_documents(docs, k=3, preprocess_func=tokenize)

# 1. exact keyword: BM25 is great at this
# 2. synonyms: BM25 struggles ("money back" vs "reimbursed")
for question in ["How do I use ExpenseHub?", "How do I get my money back after a work trip?"]:
    print(f"== BM25: {question} ==")
    for i, doc in enumerate(bm25_retriever.invoke(question), start=1):
        print(i, doc.metadata["topic"], "|", doc.page_content[:70])
    print()
