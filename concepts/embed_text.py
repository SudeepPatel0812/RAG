import ollama
from cosine import cosine

search_documents = ollama.embed(model="nomic-embed-text",  input=[
    "search_documents: Refunds are acceptable within 30 days.",
    "search_documents: Our office is located in Berlin.",
    "search_documents: Shipping takes 3-5 business days.",
    "search_query: How do I get my money back?"], )
search_query = search_documents.embeddings[3]

print("Refunds: ", cosine(search_query, search_documents.embeddings[0]))
print("Berlin: ", cosine(search_query, search_documents.embeddings[1]))
print("Shipping: ", cosine(search_query, search_documents.embeddings[2]))
