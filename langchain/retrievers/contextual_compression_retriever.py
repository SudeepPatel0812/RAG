"""
CONTEXTUAL COMPRESSION RETRIEVER: search as normal, then clean up the results.

The problem it solves:
    A retriever returns WHOLE documents. If a 5-sentence document has 1 relevant
    sentence, the other 4 are noise in the prompt (more tokens, can confuse the LLM).
    Some results are near misses (similar words, wrong meaning) and shouldn't be sent at all.
        question: "How many vacation days do I get?"
        handbook_001: office hours, visitors, 25 days leave, cafeteria, parking
        customer_001: "25 days of free premium support" (trap)

The idea in simple terms:
    The librarian brings 3 books; an assistant reads each one and gives you only the
    relevant paragraph, or throws the book away if nothing in it answers your question.

The flow (what ccr.invoke(question) does):

    question
       |
       v
    1. base_retriever.invoke(question)              -> k whole documents (normal search)
       |
       v
    2. base_compressor.compress_documents(docs, question)
       LLMChainExtractor: for EACH document, ask the LLM
       "extract the parts of this context relevant to the question"
                                                   -> k LLM calls (1 per document)
       |
       v
    3. drop documents where the LLM found nothing ("NO_OUTPUT")
       |
       v
    list[Document] -> shorter, fewer, same metadata

Why a compressor and not llm= (unlike MultiQueryRetriever.from_llm)?
    Multi-query's LLM has ONE job (rewrite the question), so it takes llm directly.
    "Compress the results" can be done MANY ways, not all with an LLM, so the
    retriever has a slot (base_compressor) and you plug a strategy in:
        LLMChainExtractor          -> keep only relevant sentences       (LLM)
        LLMChainFilter             -> keep/drop whole documents          (LLM)
        EmbeddingsFilter           -> drop docs below a similarity score (no LLM)
        CrossEncoderReranker       -> re-score and re-order              (small model)
        DocumentCompressorPipeline -> several of the above in a row
    Backend analogy: strategy pattern / dependency injection.

Multi-query vs compression:
    multi-query: LLM BEFORE search, 1 call     -> improves the QUESTION -> recall (find more)
    compression: LLM AFTER search, 1 call/doc  -> improves the RESULTS  -> precision (less noise)

Functions used:
    LLMChainExtractor.from_llm(llm)               -> the compressor (uses the LLM)
    ContextualCompressionRetriever(base_retriever=..., base_compressor=...)
        (field names are base_*: passing retriever=/llm= gives "Field required")
    ccr.invoke(question)                          -> list[Document], same as any retriever

Trade-offs:
    + much less noise and fewer tokens in the final prompt
    + drops irrelevant near misses
    - k LLM calls per question: slow and expensive at large k
    - the extractor can cut away useful context, or paraphrase it
    - use a cheap EmbeddingsFilter first, then the LLM compressor on what's left
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaEmbeddings, ChatOllama

from compression_data import documents

question = "How many vacation days do I get?"

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "contextual-compression"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

embedding = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(embedding=embedding, index=index)
docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in documents]
ids = [d["id"] for d in documents] 

vector_store.add_documents(documents=docs, ids=ids)
time.sleep(10)  

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

llm = ChatOllama(model="qwen3:14b", reasoning=False)
compressor = LLMChainExtractor.from_llm(llm)

ccr = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor, 
)

print("== normal retriever ==")
for i, doc in enumerate(retriever.invoke(question), start=1):
    print(i, doc.metadata["source"], len(doc.page_content), "chars |", doc.page_content)

print("\n== contextual compression retriever ==")
for i, doc in enumerate(ccr.invoke(question), start=1):
    print(i, doc.metadata["source"], len(doc.page_content), "chars |", doc.page_content)
