from langchain_community.document_loaders import TextLoader

loader = TextLoader("concepts/document/store_policy.txt", encoding="utf-8")

# a .txt file becomes ONE Document (the whole file)
for document in loader.lazy_load():
    print(document.metadata)
    print(document.page_content[:200])
