from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("concepts/document/store_policy.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,  # each chunk repeats up to 50 chars from the previous one
    add_start_index=True,  # stores where the chunk starts in the original text
)

# split_documents (not split_text) keeps each chunk's metadata, e.g. the source file
chunks = text_splitter.split_documents(documents)

print(len(chunks), "chunks")
for chunk in chunks:
    print(chunk.metadata)
    print(repr(chunk.page_content))
    print()
