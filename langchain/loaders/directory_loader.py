from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# load every PDF in a folder, using PyPDFLoader for each file
loader = DirectoryLoader("docs", glob="*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()
print(len(documents), "documents")
for document in documents:
    print(document.metadata["source"], "page", document.metadata["page"])
