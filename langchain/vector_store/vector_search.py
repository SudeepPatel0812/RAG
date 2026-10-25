import os

from dotenv import load_dotenv
from pinecone import Pinecone 
from langchain_pinecone import PineconeVectorStore, Pinecone
from langchain_ollama import OllamaEmbeddings

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(
    index_name="rag-learning",
    embedding=embeddings,
)

result = vector_store.similarity_search_with_score("How do I reset my password?", k=1)

for doc, score in result:
    print(round(score, 3), doc.metadata["category"], "|", doc.page_content[:80])
