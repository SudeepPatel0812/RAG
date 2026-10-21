import ollama
from cosine import cosine

texts = []

scores = []

question = "What is ShopEasy refund policy?"

file_path = "internals/document/store_policy.txt"

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read().strip().split("\n\n")
    content = list(filter(None, content))

    for idx, value in enumerate(content, start = 1):
       texts.append("search_document: " + value)
    #    scores.append(cosine(question, value))

texts.append("search_query: " + question)

result = ollama.embed(model="nomic-embed-text", input=texts)

question_vec = result.embeddings[-1]
chunk_vecs = result.embeddings[:-1]

for chunk, vec in zip (content, chunk_vecs):
    score = cosine(question_vec, vec)
    scores.append((score, chunk))

# print(len(scores))
# print(scores[0])

# print(len(result.embeddings))

# print(scores)

top = sorted(scores, reverse=True)[:2]

for score, chunk in top:
    context = "\n\n".join(chunk for score, chunk in top)

prompt = f"""You are a helpful support assistant. Answer using ONLY the context below.
If the answer isn't in the context, say "I don't know."

question: {question}"""

# print(prompt)

response = ollama.chat(
    model="qwen3:14b",
    messages=[{"role": "user", "content": question}],
    think=False,
)

print(response.message.content)