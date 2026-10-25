"""
MarkdownHeaderTextSplitter: split by headings, keep the heading path as metadata.

What it does:
    Cuts the text at every heading in headers_to_split_on. Each chunk is the
    text under one heading, and metadata records all parent headings:
    "Express shipping arrives the next day." ->
        {"h1": "ShopEasy Policy", "h2": "Shipping", "h3": "Express"}

Why it helps RAG:
    The chunk alone doesn't say what "the next day" is about; its metadata
    does. You can prepend the headings before embedding, or filter by section.

Functions used:
    MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ...])
        tuple = (markdown symbol, metadata key name)
    split_text(markdown) -> list[Document] (Documents, not strings)

It ignores size: a long section stays one chunk. In practice, run a
RecursiveCharacterTextSplitter on its output afterwards.
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown = """# ShopEasy Policy

## Refunds
Refunds are accepted within 30 days of delivery.
Money is returned in 5-7 business days.

## Shipping
Standard shipping takes 3-5 business days.

### Express
Express shipping arrives the next day.
"""

# split on headings, and record which headings each chunk sat under
text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
)

for chunk in text_splitter.split_text(markdown):
    print(chunk.metadata)
    print(repr(chunk.page_content))
    print()
