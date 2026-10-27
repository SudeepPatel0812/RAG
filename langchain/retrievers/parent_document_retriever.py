"""
PARENT DOCUMENT RETRIEVER: search with SMALL chunks, return the BIG parent document.

The problem it solves (the chunk size trade-off from chunking):
    small chunks -> precise embeddings (one idea each) -> good SEARCH, but too
                    little context for the LLM ("25 days" of what?)
    big chunks   -> lots of context for the LLM, but the embedding averages many
                    topics -> blurry SEARCH
    You want small for searching and big for answering. This retriever does both.

The idea in simple terms:
    Search the book's index cards (small), then hand over the whole chapter (big)
    the matching card came from.

The flow:

    INDEXING  add_documents(parents)
        parent (whole handbook entry) --child_splitter--> child chunks (~120 chars)
        child chunks  -> embedded -> vector store   (each child stores its parent's id)
        parent        -> docstore (key-value: parent_id -> full Document)

    QUERYING  invoke(question)
        question -> vector search over CHILDREN -> best child chunks
                 -> read their parent ids -> look the parents up in the docstore
                 -> return unique PARENT documents

Functions used:
    RecursiveCharacterTextSplitter(chunk_size=120)  -> child_splitter (small pieces to search)
    InMemoryVectorStore(embedding)                  -> holds child vectors (local, no Pinecone)
    InMemoryStore()                                 -> docstore: parent id -> parent Document
                                                       (production: Redis, a database, S3...)
    ParentDocumentRetriever(vectorstore, docstore, child_splitter, parent_splitter=None)
        parent_splitter -> optional: if parents are huge (whole books), split them into
                           medium "parents" too, so you return a section, not the book
    retriever.add_documents(docs)  -> splits, embeds children, stores parents
    vector_store.similarity_search -> shows the small child chunk that actually matched

When to use:
    + long documents where the answer needs surrounding context
    + better search precision without losing context
    - two stores to keep in sync (vector store + docstore)
    - returns big documents -> more tokens in the prompt
      (combine with contextual compression to trim them again)
"""

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from compression_data import documents

docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in documents]

vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model="nomic-embed-text"))
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0),
    search_kwargs={"k": 2},
)

retriever.add_documents(docs)  # ids are generated; parents go to docstore, children to vector store

print(len(docs), "parents ->", len(vector_store.store), "child chunks in the vector store\n")

question = "How many vacation days do I get?"

print("== what the search actually matched (small CHILD chunks) ==")
for doc in vector_store.similarity_search(question, k=2):
    print(len(doc.page_content), "chars |", doc.page_content)

print("\n== what the retriever returns (big PARENT documents) ==")
for doc in retriever.invoke(question):
    print(len(doc.page_content), "chars |", doc.metadata["source"], "|", doc.page_content[:120], "...")
