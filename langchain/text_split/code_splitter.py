from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

with open("concepts/search.py", encoding="utf-8") as file:
    code = file.read()

# same recursive idea, but the separators are Python ones: "\nclass ", "\ndef ", then lines
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))

chunks = text_splitter.split_text(code)
print(len(chunks), "chunks")
for chunk in chunks:
    print("-----")
    print(chunk)
