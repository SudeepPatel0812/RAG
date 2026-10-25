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
