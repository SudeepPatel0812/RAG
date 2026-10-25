"""
Code splitter: RecursiveCharacterTextSplitter with language-aware separators.

What it does:
    Same recursive algorithm, but the separator list comes from the language.
    Python: ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""],
    so it tries to keep whole classes, then whole functions, together.

Why: splitting code on blank lines can cut a function in half; half a
function embeds badly and is useless as context.

Functions used:
    RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, ...)
    RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
        -> the separator list it will use
    Language also has JS, GO, JAVA, MARKDOWN, HTML, ...

Note: search.py has no def/class, so here it falls back to "\n\n" and "\n".
"""

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
