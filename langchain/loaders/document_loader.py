from pathlib import Path
from langchain_docling import DoclingLoader

pdf_files = list(Path("docs").glob("*.pdf"))

loader = DoclingLoader(file_path=[str(pdf) for pdf in pdf_files])

for document in loader.lazy_load():
    print(document)


