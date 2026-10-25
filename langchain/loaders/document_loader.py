"""
DoclingLoader (langchain_docling): load PDFs with layout understanding.

What it does:
    Converts each file into Document(page_content, metadata). Docling runs
    layout models (via torch), so it understands headings, tables, reading
    order and multi-column pages, not just the raw text layer.

How it works:
    PDF -> layout model finds blocks (title, paragraph, table) -> text in
    reading order -> Documents. By default it already chunks by document
    structure (export_type=DOC_CHUNKS); export_type=MARKDOWN gives one
    Markdown Document per file instead.

Functions used:
    Path("docs").glob("*.pdf")     -> every PDF in the folder (a generator)
    DoclingLoader(file_path=[...]) -> accepts one path or a list of paths
    loader.lazy_load()             -> yields Documents one at a time (low memory)
    loader.load()                  -> returns them all as a list

Trade-off vs PyPDFLoader: much better on tables/complex layouts, but slower and
heavy (torch + models). Use PyPDF for clean text PDFs, Docling for messy ones.
"""

from pathlib import Path
from langchain_docling import DoclingLoader

pdf_files = list(Path("docs").glob("*.pdf"))

loader = DoclingLoader(file_path=[str(pdf) for pdf in pdf_files])

for document in loader.lazy_load():
    print(document)


