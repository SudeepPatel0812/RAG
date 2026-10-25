"""
RecursiveCharacterTextSplitter: the default splitter for most RAG.

What it does:
    Tries separators from biggest to smallest: ["\n\n", "\n", " ", ""].
    Splits on "\n\n" (paragraphs); any piece still > chunk_size is split
    again on "\n" (lines), then " " (words), then "" (characters), and the
    small pieces are merged back up to chunk_size.

Example (chunk_size=25):
    a 600-char paragraph -> too big -> no "\n" inside -> split on " "
    -> words merged into ~25-char chunks, never cutting a word in half.

Why it's the default: it keeps meaning together (paragraph > line > word)
as much as the size limit allows.

Parameters: chunk_size, chunk_overlap, separators (to change the list).
Functions: split_text(text) -> list[str], split_documents(docs) -> list[Document]
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque dolor mi, pellentesque quis porttitor non, molestie nec mauris. Nunc eget elit nunc. Morbi ultricies posuere turpis, eget convallis purus laoreet ac. Maecenas sed pellentesque lacus. Sed blandit libero vel erat commodo, id tincidunt ligula faucibus. Quisque suscipit blandit orci, et ultrices enim euismod vel. Vestibulum et accumsan nisi. Sed auctor quam a neque hendrerit iaculis. Donec vitae luctus erat. Praesent ultricies dapibus tellus ut rhoncus. Donec lobortis tempus orci. Aliquam accumsan ac ex pulvinar aliquet. Aenean imperdiet feugiat purus, vitae blandit velit tempus nec.

Morbi non sem augue. Vestibulum egestas, lacus viverra faucibus cursus, lectus nulla ultrices mi, nec suscipit dui leo sit amet neque. Duis et lectus sed libero fringilla vulputate. Phasellus quis venenatis elit, eget accumsan ipsum. Nam dui turpis, volutpat ut dignissim ac, cursus in risus. Proin ipsum magna, pulvinar nec ipsum vel, sagittis ornare libero. Aliquam vestibulum bibendum velit, eget pellentesque lectus elementum eget. Proin et ante quis eros placerat eleifend. Integer vehicula cursus euismod. Donec sollicitudin eget tellus a aliquam. Duis malesuada vestibulum lorem sit amet fringilla. Aenean dapibus magna dignissim, congue neque id, aliquet sapien. Donec ac purus sit amet lacus malesuada bibendum vitae ac risus.

Duis hendrerit nulla sed vestibulum aliquet. Nullam tempus erat sed pharetra sodales. Phasellus congue molestie leo, non lobortis diam dignissim vel. Curabitur lorem leo, suscipit at dapibus id, accumsan sit amet dolor. Vestibulum luctus eros ac nisl pellentesque pretium. Duis porta massa eu ligula blandit dignissim. Vestibulum id mauris vitae nisi venenatis tempus id ut eros.

Morbi consequat erat vitae erat placerat blandit. Sed elementum mauris in nisl finibus euismod. Maecenas ipsum metus, finibus eget convallis facilisis, commodo et purus. Donec in viverra nunc, sit amet dictum felis. Curabitur id massa a erat vehicula egestas eu sed felis. Morbi et malesuada risus, et faucibus justo. Curabitur rutrum, lorem at ornare tincidunt, augue lacus porta nisi, sit amet imperdiet augue libero sit amet nibh. Ut facilisis blandit varius.

Integer condimentum pharetra lorem, vel elementum tortor. Donec ut leo mi. Donec ornare, ligula vitae porttitor fringilla, turpis odio mollis felis, dictum volutpat leo justo quis orci. Nullam aliquet risus augue, a eleifend justo porta eget. Interdum et malesuada fames ac ante ipsum primis in faucibus. Mauris et risus non mauris placerat eleifend ac eget metus. Phasellus ornare diam est. Etiam eu lobortis erat. Nulla diam dolor, bibendum quis placerat sit amet, vulputate non justo. Nam id ante ac elit rutrum scelerisque. Fusce convallis gravida lacus eu aliquet. Duis id odio sodales, luctus ligula vitae, porttitor orci. Praesent ultricies sapien et tellus euismod, vitae euismod arcu condimentum."""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=0,
)

result = text_splitter.split_text(text)

print(result)
print(len(result))