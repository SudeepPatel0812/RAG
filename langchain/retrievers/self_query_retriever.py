"""
SELF-QUERY RETRIEVER: the LLM turns a question into a SEARCH TEXT + a METADATA FILTER.

The problem it solves:
    "What are the HR rules about working from home?" contains two things:
        what to search for  -> "working from home"            (meaning -> vector search)
        a condition         -> only documents where category = HR  (exact -> metadata filter)
    A plain vector search ignores the condition. You could hard-code filter={...},
    but users ask in free text, so you don't know the filter in advance.

The idea in simple terms:
    Like a smart search box that turns "cheap red shoes under $50" into
    search "shoes" WHERE colour = "red" AND price < 50. The LLM writes the filter.

The flow (what retriever.invoke(question) does):

    question
       |
       v
    1. LLM + a description of your metadata fields (AttributeInfo) writes a structured query:
         {"query": "working from home", "filter": eq("category", "HR")}        -> 1 LLM call
       |
       v
    2. a translator converts that filter into the database's own syntax
         Pinecone: {"category": {"$eq": "HR"}}                (it's like generating SQL WHERE)
       |
       v
    3. vector search with the query text AND the filter
       |
       v
    list[Document] (only documents that pass the filter)

Functions used:
    AttributeInfo(name, description, type)  -> describes each metadata field to the LLM
                                               (good descriptions and allowed values matter!)
    SelfQueryRetriever.from_llm(llm, vectorstore, document_contents, metadata_field_info)
        document_contents -> one sentence: what the documents are about
        needs the `lark` package (parses the LLM's filter expression)
        works only with vector stores that have a translator (Pinecone, Chroma, Qdrant, pgvector...)
        structured_query_translator=PineconeTranslator() -> the "SQL dialect" for the filter;
                                    normally picked automatically from the vector store
    verbose=True + logging at INFO on "langchain_classic.retrievers.self_query.base"
        -> prints the generated query and filter (both are needed)

When to use:
    + users mention filterable facts: dates, categories, authors, prices, versions
    + documents have clean, consistent metadata
    - an LLM call per question (slower); the LLM can write a wrong or too-strict filter
      -> zero results (log the generated query when debugging!)
    - useless if your metadata is messy or missing
"""

import logging
import os

from dotenv import load_dotenv
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.self_query.base").setLevel(logging.INFO)  # prints the generated query + filter

vector_store = PineconeVectorStore(
    index_name="multi-query",  # already filled by multi_query_retriever.py
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)

metadata_field_info = [
    AttributeInfo(
        name="category",
        description="The department that owns the document. One of: Finance, IT Support, HR, Customer Support, Facilities",
        type="string",
    ),
    AttributeInfo(
        name="topic",
        description="The subject. One of: expenses, travel, hardware, backup, remote work, leave, refunds, office",
        type="string",
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOllama(model="qwen3:14b", reasoning=False, temperature=0),
    vectorstore=vector_store,
    document_contents="Internal company policies for employees",
    metadata_field_info=metadata_field_info,
    search_kwargs={"k": 3},
    verbose=True,  # needed (with the logging line) to print "Generated Query"
    # shortcut: passed explicitly because the automatic lookup crashes with this
    # langchain-community version (ImportError: DatabricksVectorSearch); drop it once fixed upstream
    structured_query_translator=PineconeTranslator(),
)

for question in [
    "What are the HR rules about working from home?",
    "Show me IT Support documents about hardware repairs",
]:
    print(f"\n== {question} ==")
    for i, doc in enumerate(retriever.invoke(question), start=1):
        print(i, doc.metadata["category"], "/", doc.metadata["topic"], "|", doc.page_content[:70])
