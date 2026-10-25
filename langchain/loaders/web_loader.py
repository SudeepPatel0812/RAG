"""
WebBaseLoader: load a web page as a Document.

What it does:
    Downloads the URL (HTTP GET), parses the HTML with BeautifulSoup and keeps
    only the text. metadata = {"source", "title", "language"}.

How it works:
    HTML -> BeautifulSoup -> get_text() -> Document. Menus, footers etc. are
    kept too, so web text is noisy; filter with bs_kwargs if needed.

Functions used:
    WebBaseLoader(url or [urls]) -> one Document per URL
    loader.lazy_load()           -> fetches and yields each page

Needs network access. Set the USER_AGENT env var to identify your scraper.
"""

from langchain_community.document_loaders import WebBaseLoader

# downloads the page and strips HTML tags with BeautifulSoup
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

for document in loader.lazy_load():
    print(document.metadata)
    print(document.page_content[:300])
