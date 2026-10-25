"""
Splitting Documents (not strings) with overlap + start index.

What it does:
    Loader -> Documents -> split_documents -> smaller Documents. Each chunk
    copies the parent's metadata (source) and, with add_start_index=True,
    gets "start_index" = the character position where it starts in the original.

Overlap (chunk_overlap=50):
    When a paragraph has to be cut, the next chunk starts up to 50 chars
    earlier, so a fact on the boundary appears whole in at least one chunk.
    From the output: chunk 3 ends "...original payment method within 5-7
    business days" and chunk 4 starts with that same text.
    Cost: duplicated text -> more embeddings and more tokens.

Functions used:
    TextLoader(...).load()     -> [Document]
    split_documents(documents) -> [Document] with metadata kept
    split_text(text)           -> [str], metadata lost

Use split_documents in a real pipeline: metadata is what lets you cite sources.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("concepts/document/store_policy.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,  # each chunk repeats up to 50 chars from the previous one
    add_start_index=True,  # stores where the chunk starts in the original text
)

# split_documents (not split_text) keeps each chunk's metadata, e.g. the source file
chunks = text_splitter.split_documents(documents)

print(len(chunks), "chunks")
for chunk in chunks:
    print(chunk.metadata)
    print(repr(chunk.page_content))
    print()
