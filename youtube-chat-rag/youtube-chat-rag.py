"""
YOUTUBE CHAT RAG: ask questions about a YouTube video, answers cite the moment it was said.

The flow:

    INDEXING (once per video)
    video_id
       |  1. LOAD: YouTubeTranscriptApi().fetch(video_id, languages=["en"])
       v     -> 613 snippets: FetchedTranscriptSnippet(text, start, duration)
    snippets (tiny: 7-40 chars, one caption line each)
       |  2. CHUNK: group snippets into ~60-second windows (+2 snippets overlap)
       v     -> 25 chunks of ~800-1170 chars, each with its start time
    Documents(page_content=text, metadata={video_id, chunk, start, url})
       |  3. EMBED + STORE: add_documents(docs, ids=["<video_id>#0", ...])
       v     -> 25 rows in Pinecone index "youtube-transcripts"

    QUERYING (every question)
    question
       |  4. RETRIEVE: MMR search, k=3, filter={"video_id": video_id}
       v     -> 3 relevant AND different chunks, only from this video
       |  5. COMPRESS: LLMChainExtractor keeps only the relevant sentences (1 LLM call per chunk)
       v
    short passages + timestamp + link (https://www.youtube.com/watch?v=...&t=275s)

Step by step:
    1. LOAD
       YouTubeTranscriptApi() -> create an OBJECT first (fetch is a method, it needs self)
       fetch(video_id, languages=["en"]) -> FetchedTranscript; each snippet is an OBJECT:
       snippet.text / snippet.start (not snippet["text"])
       A bare `except:` would hide real errors as "no captions": use `except Exception as e`.

    2. CHUNK: why not ...
       ... one row for the whole transcript? 23,000 chars -> the embedding averages 22 minutes
           into one blurry point, and nomic-embed-text only reads ~2048 tokens (~8000 chars),
           so most of it would never be embedded.
       ... every 4 snippets?  ~150 chars each -> 154 half-sentences with no context.
       ... RecursiveCharacterTextSplitter? the joined transcript has no "\n", so the default
           separators cut on spaces (mid-sentence). With separators=[". ", "? ", "! ", " "]
           and keep_separator="end" -> 30 chunks ending on full sentences. Works, but
           " ".join(...) THROWS THE TIMESTAMPS AWAY.
       ... time windows (used): snippets already carry `start`, so group ~60 seconds of speech.
           Same size as 1000-char chunks, and every chunk knows WHEN it was said:
           a timestamp is to a video what a page number is to a PDF -> citations + deep links.
       Don't forget the LAST window after the loop.

    3. EMBED + STORE
       metadata must be a DICT: {"video_id", "chunk", "start", "url"}
       stable ids "<video_id>#<i>" -> re-running overwrites (25 rows stay 25, no bloat)
       loop over the CHUNKS: `for chunk in transcript` would loop over 23,000 CHARACTERS
       -> 23,000 one-letter rows
       Changing the chunking (30 -> 25 chunks) leaves old ids #25-#29 behind as orphans:
       delete the video's old rows first when re-chunking.

    4. RETRIEVE
       search_type="mmr", lambda_mult=0.5 -> balance relevance vs variety (talks repeat themselves)
       filter={"video_id": video_id} -> WHERE video_id = ...; one index can hold many videos,
       and a question about video A never gets chunks from video B.
       The filter lives in the retriever's search_kwargs, so everything that wraps the
       retriever (compression, chains) inherits it.

    5. COMPRESS
       ContextualCompressionRetriever(base_retriever=retriever, base_compressor=LLMChainExtractor)
       -> each 1000-char chunk becomes the 1-2 sentences that answer the question.
       reasoning=True makes qwen3 think for EACH of the k chunks: slower; False is fine here.

Next: put the compressed passages + question into a prompt and answer with ChatOllama,
citing the timestamps ("she says this at 4:35").
"""

from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# Get the transcript of the video
# Document Loader

video_id = "JX9xNgN0vmk"

ytt_api = YouTubeTranscriptApi()  # create an object first; Python then fills `self` for you

transcript = ""

try:
    transcript_list = ytt_api.fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
except Exception as e:
    print("Could not fetch captions:", type(e).__name__, e)

if not transcript:
    raise SystemExit("No transcript, nothing to index.")

# Chunking

# the transcript is one long line (no "\n"), so the default separators would cut on spaces,
# often mid-sentence; try sentence endings first, then spaces as a fallback
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[". ", "? ", "! ", " "],
    keep_separator="end",  # keep the "." at the end of the sentence, not at the start of the next chunk
)
split_text = text_splitter.split_text(text=transcript)
print(len(split_text), "chunks")
for chunk in split_text[:3]:  # dry run: check where chunks start and end
    print("---", repr(chunk[:60]), "...", repr(chunk[-60:]))

# Time-based chunking (alternative to the splitter above)
# " ".join(...) throws the timestamps away. Here we group the snippets into ~60-second
# windows instead, so every chunk remembers WHEN it was said (like a page number for a PDF).
# Each snippet looks like: FetchedTranscriptSnippet(text="...", start=0.0, duration=5.2)

window_seconds = 60
overlap_snippets = 2  # repeat the last 2 snippets in the next chunk, like chunk_overlap

time_chunks = []  # list of {"text": ..., "start": ...}
current = []      # snippets in the window being built

for snippet in transcript_list:
    current.append(snippet)
    if snippet.start + snippet.duration - current[0].start >= window_seconds:
        time_chunks.append({"text": " ".join(s.text for s in current), "start": current[0].start})
        current = current[-overlap_snippets:]  # start the next window with the overlap

if len(current) > overlap_snippets:  # the last window, unless it's only the overlap we already saved
    time_chunks.append({"text": " ".join(s.text for s in current), "start": current[0].start})

print("\n", len(time_chunks), "time chunks")
for chunk in time_chunks[:3]:
    minutes, seconds = divmod(int(chunk["start"]), 60)
    print(f"--- {minutes}:{seconds:02d} |", repr(chunk["text"][:60]), "...")

# Creating Vectors and storing them into vector store

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "youtube-transcripts"

if not pinecone.has_index(index_name):
    pinecone.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pinecone.Index(index_name)

embedding = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(embedding=embedding, index=index)

docs = []
ids = []

# Sentence-splitter version (no timestamps):
# for i, chunk in enumerate(split_text):  # loop over CHUNKS; looping over `transcript` gives single characters
#     docs.append(Document(page_content=chunk, metadata={"video_id": video_id, "chunk": i}))
#     ids.append(f"{video_id}#{i}")  # stable ids: re-running overwrites instead of adding duplicates

# Time-based version: the timestamp goes into metadata, so answers can cite "at 3:20" + a link
for i, chunk in enumerate(time_chunks):
    start = int(chunk["start"])
    docs.append(Document(
        page_content=chunk["text"],
        metadata={
            "video_id": video_id,
            "chunk": i,
            "start": start,  # seconds from the beginning of the video
            "url": f"https://www.youtube.com/watch?v={video_id}&t={start}s",  # opens the video at that moment
        },
    ))
    ids.append(f"{video_id}#{i}")  # stable ids: re-running overwrites instead of adding duplicates

# vector_store.add_documents(documents=docs, ids=ids)
print("stored", len(docs), "chunks")

# Retrievers

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5,
        "filter": {"video_id": video_id},  # only search THIS video's chunks (like SQL WHERE video_id = ...)
    },
    )

llm = ChatOllama(model="qwen3:14b", reasoning=True)
compressor = LLMChainExtractor.from_llm(llm=llm)

contextual_compresson_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

question = "Does speaker talk about why Ai won't replace developers ?"
print("\n== contextual compression retriever ==")
for i, doc in enumerate(contextual_compresson_retriever.invoke(question), start=1):
    # Pinecone returns every number as a float (54 -> 54.0), and {:02d} only accepts ints
    minutes, seconds = divmod(int(doc.metadata["start"]), 60)
    print(i, f"{minutes}:{seconds:02d}", doc.metadata["url"], "|", doc.page_content)
