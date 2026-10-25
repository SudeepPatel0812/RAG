"""
PyPDFLoader: load a PDF, one Document per PAGE.

What it does:
    Uses the pypdf library to read the PDF's text layer page by page.
    metadata = {"source": <file>, "page": <0-based page number>, ...}

How it works:
    A PDF stores text as positioned characters; pypdf extracts them in order.
    No OCR (scanned images give empty text) and no table understanding.

Functions used:
    Path("docs").glob("*.pdf") -> iterate over the PDFs
    PyPDFLoader(path)          -> one loader per file
    document.metadata["page"]  -> 0-based, so page 0 is the first page

Why one Document per page matters: the page number becomes a citation
("see page 4"), and chunks never mix text from different files.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# PyPDF only extracts the text layer: fast, but no tables/layout (Docling does those)
for pdf in Path("docs").glob("*.pdf"):
    loader = PyPDFLoader(str(pdf))

    # one Document per PAGE, with the page number in metadata
    for document in loader.lazy_load():
        print(document.metadata["source"], "page", document.metadata["page"])
        print(document.page_content[:200])
