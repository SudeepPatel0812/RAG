"""
Token-based splitting: measure chunk_size in TOKENS, not characters.

What it does:
    Same recursive algorithm, but length = number of tokens from a real
    tokenizer instead of len(text).

Why tokens:
    Models have limits in tokens (an embedding model's max input, an LLM's
    context window). Characters per token varies (~5 here: 223 chars = 46
    tokens), so a character limit can still overflow a model. Use the
    tokenizer of the model you embed with.

Functions used:
    AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
        -> downloads that model's tokenizer from Hugging Face (cached after once)
    RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size, chunk_overlap)
    tokenizer.encode(text, add_special_tokens=False) -> token ids; len() = token count

Alternatives: TokenTextSplitter / from_tiktoken_encoder (OpenAI tokenizers, needs tiktoken).
"""

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
