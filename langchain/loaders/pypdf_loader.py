from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# PyPDF only extracts the text layer: fast, but no tables/layout (Docling does those)
for pdf in Path("docs").glob("*.pdf"):
    loader = PyPDFLoader(str(pdf))

    # one Document per PAGE, with the page number in metadata
    for document in loader.lazy_load():
        print(document.metadata["source"], "page", document.metadata["page"])
        print(document.page_content[:200])
