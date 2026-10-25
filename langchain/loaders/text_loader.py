"""
TextLoader: load a plain .txt file as ONE Document.

What it does:
    Reads the whole file -> Document(page_content=<file text>,
    metadata={"source": <path>}).

Why a Document and not a string:
    Every loader (text, PDF, web) returns the same shape, so splitters, vector
    stores and retrievers only need to understand Document. metadata travels
    with the text all the way to the answer, so you can cite the source.

Functions used:
    TextLoader(path, encoding="utf-8") -> encoding avoids Windows decode errors
    loader.lazy_load() -> generator, one Document at a time
    loader.load()      -> list of all Documents

Plain Python equivalent:
    {"page_content": open(path, encoding="utf-8").read(), "metadata": {"source": path}}
"""

from langchain_community.document_loaders import TextLoader

loader = TextLoader("concepts/document/store_policy.txt", encoding="utf-8")

# a .txt file becomes ONE Document (the whole file)
for document in loader.lazy_load():
    print(document.metadata)
    print(document.page_content[:200])
