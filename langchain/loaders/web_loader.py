from langchain_community.document_loaders import WebBaseLoader

# downloads the page and strips HTML tags with BeautifulSoup
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

for document in loader.lazy_load():
    print(document.metadata)
    print(document.page_content[:300])
