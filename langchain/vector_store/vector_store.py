import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec  # the client; langchain_pinecone.Pinecone is an old vector store alias
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()  # loads .env from the current folder into env vars (returns True/False, not the key)

pinecone_api_key = os.getenv("PINECONE_API_KEY")

document = [
  {
    "id": "doc_001",
    "text": "To reset your corporate password, navigate to the Identity Portal, click 'Forgot Password', and verify your identity via the multi-factor authentication (MFA) mobile app prompt. Passwords must be at least 16 characters long and contain three character classes.",
    "metadata": {
      "category": "IT Support",
      "subcategory": "Security",
      "author": "SecOps Team",
      "updated_date": "2026-03-15",
      "version": 1.2
    }
  },
  {
    "id": "doc_002",
    "text": "The company's remote work policy allows full-time employees to work from anywhere within the country of their hiring entity. Employees receive a $500 home office stipend upon onboarding to purchase ergonomic equipment, monitors, or peripherals.",
    "metadata": {
      "category": "HR",
      "subcategory": "Benefits",
      "author": "People Ops",
      "updated_date": "2026-01-10",
      "version": 2.0
    }
  },
  {
    "id": "doc_003",
    "text": "Error 404 on the API gateway typically indicates a routing misconfiguration in the ingress controller. Ensure that the service name matches the deployment label exactly and that the target port matches the container port configuration.",
    "metadata": {
      "category": "Engineering",
      "subcategory": "DevOps",
      "author": "Platform Team",
      "updated_date": "2026-05-22",
      "version": 1.0
    }
  },
  {
    "id": "doc_004",
    "text": "Annual health insurance open enrollment runs from November 1st through November 30th. Changes made during this window become active on January 1st of the following calendar year. Vision and dental plans require active re-selection.",
    "metadata": {
      "category": "HR",
      "subcategory": "Benefits",
      "author": "People Ops",
      "updated_date": "2025-10-01",
      "version": 1.1
    }
  },
  {
    "id": "doc_005",
    "text": "Our database utilizes automated daily snapshots taken at 02:00 UTC. Snapshots are retained for 30 days in a geo-replicated object store. To request a point-in-time recovery (PITR), open an emergency ticket with the Database Administration team.",
    "metadata": {
      "category": "Engineering",
      "subcategory": "Database",
      "author": "DBA Team",
      "updated_date": "2026-04-18",
      "version": 3.4
    }
  }
]

# ---------- INDEXING ----------
pc = Pinecone(api_key=pinecone_api_key)

index_name = "rag-learning"
if not pc.has_index(index_name):  # like CREATE TABLE IF NOT EXISTS
    pc.create_index(
        name=index_name,
        dimension=768,  # nomic-embed-text gives 768 numbers per text
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)  # open the index by name

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in document]
ids = [d["id"] for d in document]  # fixed ids -> re-running overwrites instead of duplicating

vector_store.add_documents(documents=docs, ids=ids)  # embed + upsert

# ---------- QUERYING ----------
result = vector_store.similarity_search_with_score("How do I reset my password?", k=1)

for doc, score in result:
    print(round(score, 3), doc.metadata["category"], "|", doc.page_content[:80])
