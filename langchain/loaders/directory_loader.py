"""
DirectoryLoader: load every matching file in a folder.

What it does:
    Finds files with `glob`, runs `loader_cls` on each one, and combines all
    their Documents into one list.

Functions used:
    DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
        glob       -> which files ("**/*.pdf" would include subfolders)
        loader_cls -> which loader handles each file
    loader.load() -> 3 PDFs x their pages = 39 Documents here

Plain Python equivalent:
    [doc for pdf in Path("docs").glob("*.pdf") for doc in PyPDFLoader(str(pdf)).load()]
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# load every PDF in a folder, using PyPDFLoader for each file
loader = DirectoryLoader("docs", glob="*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()
print(len(documents), "documents")
for document in documents:
    print(document.metadata["source"], "page", document.metadata["page"])
