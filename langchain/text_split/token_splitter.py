from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("concepts/document/store_policy.txt", encoding="utf-8") as file:
    text = file.read()

# models have limits in TOKENS, not characters, so measure chunks with the
# embedding model's own tokenizer (nomic-embed-text); downloads it once from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=50,  # tokens, not characters
    chunk_overlap=10,
)

for chunk in text_splitter.split_text(text):
    print(len(tokenizer.encode(chunk, add_special_tokens=False)), "tokens,", len(chunk), "chars")
    print(repr(chunk))
    print()
