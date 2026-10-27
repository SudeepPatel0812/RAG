# RAG Revision Notes

# RAG Pipeline

![alt text](image.png)

Read this before an interview. Everything here was built by hand in `internals/` (plain Python + Ollama, no framework).

---

## 1. RAG in one line

**Retrieval-Augmented Generation** = before asking the LLM, look up the relevant pieces of *your* documents and paste them into the prompt.

- An LLM only knows its training data, not your private docs. Without RAG it fills gaps with plausible guesses (**hallucination**).
- With RAG it answers from your text (**grounding**). Open-book exam vs closed-book.

## 2. The pipeline

```
INDEXING (once)                         QUERYING (every question)
document                                question
  │ chunk                                 │ embed (same model)
  ▼                                       ▼
chunks ──embed──► vectors ──store──► cosine(question, each chunk) → sort → top-k
                                          │
                                          ▼
                                prompt = rules + context (top-k) + question ──► LLM ──► answer
```

### Say it out loud (interview version)

> "I split the document into paragraph chunks, because embedding a whole document dilutes specific facts. I embed each chunk and the user's question with the same embedding model, using its query/document prefixes. I rank chunks by cosine similarity to the question and take the top k. I put those chunks into a prompt as context, with an instruction to answer only from the context and say 'I don't know' otherwise. The LLM then generates an answer grounded in my documents. That reduces hallucination, but doesn't eliminate it."

---

## 3. Core concepts

### Embeddings
- A model turns text into a list of numbers (a **vector**). `nomic-embed-text` → **768** numbers.
- **Similar meaning → nearby vectors.** "How do I get my money back?" lands near "Refunds are accepted within 30 days", with zero shared words. That's **semantic search** (vs keyword search).
- A single number means nothing on its own. Only **comparing** vectors tells you anything.
- How the model does it: text → tokens → token IDs → neural network layers (context-aware) → average into one vector.
- Why it works: **training** on millions of pairs, pulling related texts closer and pushing unrelated ones apart.
- **Deterministic:** same text → same vector, so embeddings can be **cached**: embed once, reuse forever.
- **Prefixes** (nomic): `search_query: ...` for questions, `search_document: ...` for docs, because the model was trained that way. They go *inside the string*; variable names mean nothing to the model.

### Cosine similarity
```
cosine(a, b) = dot(a, b) / (length(a) × length(b))
dot(a, b)    = a[0]*b[0] + a[1]*b[1] + ...
length(a)    = sqrt(a[0]² + a[1]² + ...)
```
- 1.0 = same direction (same meaning), 0 = unrelated.
- Hand-checked: `[1,0]·[1,0] = 1`, `[1,0]·[0,1] = 0`, `[1,0]·[1,1] ≈ 0.707`.
- Real scores are rarely near 1 (the winner was 0.767). **The ranking within one run matters, not the absolute number.**
- Scores from *different setups* (e.g. with vs without prefixes) **can't be compared**: everything shifts.

### Chunking
- **Why:** an embedding averages all tokens, so one fact buried in 20 pages gets diluted ("one drop of juice in a bucket of water").
- **Too big** → diluted meaning, wasted LLM tokens. **Too small** → lost context ("30 days" of *what*?).
- Started with paragraphs: `split("\n\n")`, not `"\n"` (paragraphs can wrap over several lines). `.strip()` + `filter(None, ...)` drops empty/whitespace chunks.
- Keep chunk numbers/source → lets you **cite sources** and debug bad answers.

### Retrieval (top-k)
- Score every chunk against the question, keep `(score, chunk)` tuples, `sorted(..., reverse=True)[:k]`.
- **Retrieval always returns something**, even when nothing is relevant (the CEO question still got 2 chunks).
- **Choosing k:** more context vs more noise and cost. Irrelevant chunks can confuse the LLM. Add a **score threshold** to skip junk. Real answer: *measure it*.

### Generation (prompt)
```
You are a helpful support assistant. Answer using ONLY the context below.
If the answer isn't in the context, say "I don't know."

Context:
<top-k chunks joined with blank lines>

Question: <question>
```
- "ONLY the context" → grounding. "Say I don't know" → safe exit (it worked for "Who is the CEO?").
- The score is for *ranking*; send only the chunk **text** to the LLM.

---

## 4. Experiments I ran (and what they proved)

| Experiment | Result | Lesson |
|---|---|---|
| "money back" vs refunds/Berlin/shipping | refunds won (0.571 vs 0.373/0.354) | semantic match without shared words |
| Wrong index for query (used a doc as the query) | plausible-looking but meaningless scores | bugs in RAG are often *silent* |
| With vs without prefixes | all scores rose, gaps barely moved | one example proves nothing → need an **eval set** |
| Search the store policy for a refund question | refunds 0.767, Welcome 0.581 (#2 via "customer support") | retrieval on a real document works; big gap = confident |
| "Who is the CEO?" with RAG | "I don't know" | the prompt rule plus irrelevant context → safe refusal |
| Refund policy, **no** context | long generic answer: 30 days (lucky), 5–10 days (wrong, real = 5–7), invented rules | that's a hallucination: plausible, confident, partly wrong |

---

## 5. Likely interview questions

**What is RAG and why use it instead of fine-tuning?**
Retrieve relevant docs at query time and put them in the prompt. Cheaper than fine-tuning, docs can be updated instantly (no retraining), you can cite sources, and it reduces hallucination on private/fresh data. Fine-tuning teaches *style/behaviour*; RAG supplies *facts*.

**What is an embedding?**
A fixed-length vector representing meaning; similar texts → nearby vectors. Produced by a model trained to pull related texts together.

**Why cosine similarity?**
It measures direction (meaning), not length, so long and short texts compare fairly. Dot product = cosine when vectors are normalised.

**Why chunk? How do you pick chunk size?**
Whole-document embeddings dilute facts and waste context. Size is a trade-off (precision vs context). Start with ~paragraphs and add **overlap** so facts on a boundary aren't cut. Tune with an eval set.

**How do you choose k?**
Trade-off between recall (enough context) and noise/cost. Use a score threshold, and measure with recall@k on test questions.

**How do you know your RAG is good?**
Build an eval set: questions + the correct chunk. Measure **hit rate / recall@k** for retrieval, and answer correctness/faithfulness for generation. Change one variable at a time.

**How do you reduce hallucinations?**
Good retrieval, an "answer only from context / say I don't know" prompt, score thresholds, asking the model to **cite the chunk**, and evaluating with *unanswerable* questions. Reduces, never eliminates.

**Why must queries and docs use the same embedding model?**
Different models produce different vector spaces. Comparing across them is meaningless (like comparing GPS coordinates with postcodes).

**What breaks at scale?**
See the next section.

---

## 6. From my version to production (scaling)

| My version | Limit | Scalable fix |
|---|---|---|
| Re-embeds every chunk on every run | slow, wasteful | embed once, **store vectors** (Chroma, pgvector, Qdrant) |
| Python loop of cosine over every chunk | O(N) per query, too slow at 1M chunks | **ANN vector index** (HNSW, IVF): approximate, fast |
| `split("\n\n")` | messy PDFs, huge paragraphs | size-based chunking + **overlap**, structure-aware splitting |
| Always top-2 | sends junk when nothing matches | **score threshold**, re-ranking (cross-encoder) |
| Vectors only | misses exact terms (IDs, error codes) | **hybrid search**: vector + keyword (BM25) |
| One embed call per run | big corpora | **batch** embedding, async, cache by content hash |
| "Looks right to me" | can't tell if a change helps | **eval set**, recall@k, regression tests |
| No sources | user can't verify | store metadata (file, chunk id), **cite sources** |

Backend analogies: vector store = an index by meaning; embedding cache = memoising a pure function; ANN index = trading exactness for speed, like a cache.

---

## 7. LangChain mapping (what each class replaces)

| I wrote | LangChain |
|---|---|
| `open(...).read().split("\n\n")` | `TextLoader` + `TextSplitter` (e.g. `RecursiveCharacterTextSplitter`) |
| `ollama.embed(...)` | `OllamaEmbeddings` |
| list of vectors + `cosine` + `sorted()[:k]` | `VectorStore.similarity_search(query, k)` / `Retriever` |
| `f"""...{context}...{question}"""` | `PromptTemplate` / `ChatPromptTemplate` |
| `ollama.chat(...)` | `ChatOllama` |
| the whole script | a chain (LCEL: `retriever | prompt | llm`) |

Framework trade-off: faster to build and easy to swap components, but hides the internals. When retrieval is wrong you still need to know what's underneath.

---

## 8. Python gotchas I hit

- `No module named 'ollama'` → venv not active / package installed into a different Python. Check `python -c "import sys; print(sys.executable)"`.
- `NameError` → a variable used before it's created (`total = total + ...` with no `total = 0`).
- `ZeroDivisionError` → a vector length came out 0 (wrong formula). Debug by printing the values right before the crash.
- `TypeError: unexpected keyword argument` → that parameter doesn't exist; check with `help(fn)`.
- `ValidationError ... input_type=list` → passed a list where a string was expected. **Read the last line of a traceback, then the first line in *your* file.**
- **Text vs vectors:** `cosine` only takes vectors from `result.embeddings`, never strings.
- Don't name your variables after API fields (`content` vs the message's `"content"` key).
- `zip(a, b)` pairs two lists (stops silently at the shorter). `enumerate(xs, start=1)` gives (position, item). `xs[-1]` = last, `xs[:-1]` = all but the last.
- File paths are relative to **where you run the command**, not where the script lives.

---

## 9. LangChain loaders and splitters (code: `langchain/`, theory in each file's docstring)

**Loaders** turn any source into the same shape: `Document(page_content, metadata)`.

| File | Class | One Document per | Note |
|---|---|---|---|
| `loaders/text_loader.py` | `TextLoader` | file | simplest |
| `loaders/pypdf_loader.py` | `PyPDFLoader` | PDF page (`metadata["page"]`, 0-based) | text layer only, no tables/OCR |
| `loaders/document_loader.py` | `DoclingLoader` | structure-aware chunk | layout models: tables, headings; slow + heavy |
| `loaders/directory_loader.py` | `DirectoryLoader` | whatever `loader_cls` gives | glob a folder (3 PDFs → 39 pages) |
| `loaders/web_loader.py` | `WebBaseLoader` | URL | HTML → BeautifulSoup text; noisy |

**Splitters** are all one algorithm, "try the biggest separator first, merge pieces up to `chunk_size`". Only two things change: **which separators** and **how length is measured**.

| File | Separators | Length | Use when |
|---|---|---|---|
| `text_split/length_splitter.py` (`CharacterTextSplitter`) | one (`"\n\n"`) | chars | simple; big paragraphs stay big |
| `text_split/recursive_splitter.py` | `"\n\n"` → `"\n"` → `" "` → `""` | chars | the default |
| `text_split/document_splitter.py` | recursive + `chunk_overlap` + `add_start_index` | chars | real pipelines (`split_documents` keeps metadata) |
| `text_split/markdown_splitter.py` | `#`, `##`, `###` | none (by section) | structured docs; headings go into metadata |
| `text_split/code_splitter.py` | `\nclass `, `\ndef `, ... | chars | source code |
| `text_split/token_splitter.py` | recursive | **tokens** (model's tokenizer) | production: models limit tokens, not chars |

- `split_text` → `list[str]` (metadata lost). `split_documents` → `list[Document]` (metadata kept → citations).
- **Overlap** repeats the end of the previous chunk, so a fact on a boundary appears whole in one chunk. Cost: more embeddings and tokens.
- Not covered yet: **semantic chunking** (split where the embedding meaning changes; `langchain-experimental`).

---

## 10. Vector stores (code: `langchain/vector_store/`, theory in each file's docstring)

**What it does:** stores `(id, vector, metadata)` and answers "which stored vectors are closest to this one?" quickly. It replaces my list of vectors + `cosine` loop + `sorted()[:k]`.

```
INDEXING (vector_store.py, once)                 QUERYING (vector_search.py, every question)
Documents ─embed─► (id, 768 values, metadata)    question ─embed (same model)─► vector
           ─upsert─► Pinecone index                ─► Pinecone ANN search ─► top-k (Document, score)
```

| Concept | Meaning | Backend analogy |
|---|---|---|
| index | a named collection with a fixed `dimension` + `metric` | a table with a schema |
| `create_index(dimension=768, metric="cosine")` | dimension MUST equal the embedding model's output | `CREATE TABLE` |
| record | `id` + `values` (the vector) + `metadata` | a row |
| upsert with ids | same id overwrites → no duplicates | `INSERT ... ON CONFLICT DO UPDATE` |
| metadata `filter` | `{"category": "HR"}` | `WHERE` |
| ANN index (HNSW etc.) | approximate nearest neighbours, fast at millions of vectors | an index trading exactness for speed |
| eventual consistency | new vectors aren't searchable instantly | read-replica lag |

- `pinecone.Pinecone` = **client** (admin: create/delete/describe indexes). `PineconeVectorStore` = **LangChain** (add/search). Never import `Pinecone` from `langchain_pinecone`: it's an old alias of the vector store.
- Pinecone has no "document" field: LangChain puts `page_content` in `metadata["text"]`. The console shows metadata; the 768 values are there (`index.fetch(ids=[...])`).
- Search **always** returns k results. Use the score to judge relevance.
- Index and query with the **same embedding model**.
- `load_dotenv()` takes no argument; `os.getenv("KEY")` reads the value.

### Vector store vs vector database

People mix these up. The difference is **what else it does besides similarity search**.

| | Vector store | Vector database |
|---|---|---|
| What | anything that can save vectors and do nearest-neighbour search | a full database system built around vectors |
| Examples | FAISS (a library), LangChain `InMemoryVectorStore`, my Python list + cosine | Pinecone, Weaviate, Qdrant, Milvus, Chroma (server mode), Postgres + pgvector |
| Where it lives | often in-process memory or a file | a separate server/service with an API |
| Persistence | maybe (save/load a file) | yes, durable storage |
| CRUD by id, upsert, delete | limited / rebuild | yes |
| Metadata filtering | basic or none | yes, combined with vector search |
| Scale | one machine's RAM | sharding, replication, millions-billions of vectors |
| Concurrency, auth, backups, monitoring | no | yes |
| Analogy | an in-memory hash map / SQLite file | PostgreSQL / a managed DB like RDS |

- In LangChain, **`VectorStore` is the interface name for both**: `PineconeVectorStore` is a *vector store wrapper* around Pinecone, which is a *vector database*.
- Every vector database contains a vector store (the index + search). Not every vector store is a database.
- **When to use which:** prototypes, notebooks, small fixed data → FAISS / in-memory. Many users, data that changes, filters, millions of vectors → a vector database. Already run Postgres → pgvector (one less system to operate).

**Interview line:** "A vector store is the capability: store embeddings and find nearest neighbours. A vector database adds the database parts around it: persistence, CRUD, metadata filtering, scaling, replication and access control. FAISS is a vector store; Pinecone is a vector database; LangChain wraps both behind the same VectorStore interface."

### Retriever (code: second half of `vector_search.py`)

| | Vector store | Retriever |
|---|---|---|
| Role | storage + search | search only |
| Methods | `add_documents`, `delete`, `similarity_search(query, k)`, `_with_score`, ... | **one**: `invoke(query) -> list[Document]` |
| Search config | passed on every call | fixed once in `as_retriever(search_type, search_kwargs)` |
| Scores | available (`_with_score`) | not returned (Documents only, which is what the prompt needs) |
| Plugs into chains | no | yes, it's a Runnable: `{"context": retriever, ...} \| prompt \| llm` |
| Backends | one database | anything: vector store, BM25, web search, my own class |

- Why: **one interface** (like a Go interface / dependency injection). The chain calls `.invoke(question)` and doesn't care what's behind it, so search can be swapped without touching the chain.
- `search_type`: `"similarity"` (top k), `"similarity_score_threshold"` (drop weak results, fixes "always returns k"), `"mmr"` (relevant **and** diverse: fetch `fetch_k`, keep `k` that aren't near-duplicates). `search_kwargs={"filter": {...}}` works with all.
- The threshold uses **relevance** scores 0..1, not raw cosine: `(raw + 1) / 2`. Measured: password question raw 0.73 → 0.865; "weather in Paris" raw 0.44 → 0.72. `score_threshold=0.8` kept the first and returned `[]` for the second.

#### In simple terms: the library

- **Vector store = the library.** It holds all the books (documents). At the front desk you can **add** a book, **remove** one, or **search**, giving the instructions every time ("2 books about passwords", optionally "and show how well each matches").
- **Retriever = a helper with fixed instructions.** Tell it once: "always bring me the 2 best books, only if they're actually relevant". After that you just ask the question. It can only do **one thing**: bring books.

```python
vector_store.similarity_search("reset password", k=2)              # instructions on every call

retriever = vector_store.as_retriever(search_kwargs={"k": 2})     # instructions once
retriever.invoke("reset password")                                # then just ask
```

Both return the **same** documents. Why have the helper: the rest of the app (prompt + LLM) only says "bring me books for this question" and doesn't care whether they come from Pinecone, FAISS, Google or my own list. Change the library → replace the helper, nothing else changes.

Backend analogy: vector store = the **database**; retriever = a **repository function** `getRelevantDocs(question)` that the rest of the code calls instead of talking to the database.

| | What it is | What you do with it |
|---|---|---|
| **Vector store** | the library | add, delete, search with any settings |
| **Retriever** | a helper with fixed instructions | ask a question → get documents |

---

## 11. Multi-query retriever (code + flow docstring: `langchain/retrievers/multi_query_retriever.py`, data: `multi_query_data.py`)

**Problem:** one question = one vector = one "angle". If the user's words differ from the documents' words ("money back" vs "reimbursed", "per diem"), relevant docs score low and are missed.

**In simple terms:** ask 3 colleagues to rephrase your question, search with each version, and combine what they find.

```
question
  └─► LLM: "write 3 different versions of this question"     (1 LLM call)
        └─► ["How are travel expenses reimbursed?", "What is the per diem...?", "How do I claim out-of-pocket costs?"]
              └─► normal retriever on EACH version             (3 searches × k docs)
                    └─► unique union (drop duplicates)         (≈ N × k docs, N = lines the LLM wrote, usually 3)
                          └─► list[Document]
```

```python
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=ChatOllama(model="qwen3:14b", reasoning=False))
mqr.invoke(question)   # same interface as any retriever
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)  # see the generated queries
```

| Plus | Minus |
|---|---|
| higher **recall**: finds docs with different wording | an **LLM call before every search** (slower, costs tokens) |
| no change to the index or embeddings | 3× searches and embeddings |
| still a retriever → drop-in replacement in a chain | more docs → longer prompt, some noise |
| | only as good as the LLM's rephrasings |

- No scores, no re-ranking: just the merged unique list. `include_original=True` also searches with the user's own question.
- Related: **HyDE** (LLM writes a fake *answer*, search with it), **RAG-Fusion** (multi-query + rank by how many queries found each doc, RRF), **query decomposition** (split a complex question into sub-questions).
- **Interview line:** "Multi-query improves recall when users phrase things differently from the documents. It trades one extra LLM call and N× searches for finding more relevant chunks; I'd measure recall@k on test questions to see if it's worth the latency."

---

## 12. Contextual compression retriever (code + flow docstring: `langchain/retrievers/contextual_compression_retriever.py`, data: `compression_data.py`)

**Problem:** a retriever returns **whole** documents. If a 5-sentence doc has 1 relevant sentence, the other 4 are noise in the prompt (more tokens, can confuse the LLM). Some returned docs are near misses that aren't relevant at all.

**Idea:** search as normal, **then clean up the results**: cut each doc down to the relevant part, or drop irrelevant docs.

```
ContextualCompressionRetriever
 ├── base_retriever  → finds documents (normal retriever)
 └── base_compressor → cleans them     (you choose the strategy)
```

```python
compressor = LLMChainExtractor.from_llm(llm)
ccr = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
ccr.invoke(question)   # still a retriever: same .invoke()
```

### Why `LLMChainExtractor` and not just `llm=` (unlike multi-query)?

- **Multi-query:** the LLM has only **one** possible job (rewrite the question), so it takes `llm` directly: `MultiQueryRetriever.from_llm(retriever, llm)`.
- **Compression:** "clean up the results" can be done **many ways, and not all use an LLM**. So the retriever has a **slot** for a compressor; the LLM belongs to the compressor you plug in.

| Compressor | What it does | Uses an LLM? |
|---|---|---|
| `LLMChainExtractor` | keeps only the relevant **sentences** of each doc | ✅ |
| `LLMChainFilter` | keeps or drops **whole docs** (yes/no per doc) | ✅ |
| `EmbeddingsFilter` | drops docs below a similarity threshold | ❌ embeddings only |
| Cross-encoder / reranker | re-scores and re-orders docs | ❌ small ranking model |
| `DocumentCompressorPipeline` | chains several of the above | depends |

Backend analogy: **strategy pattern / dependency injection**. The service takes any `Compressor`, like a service that takes any `Cache` (Redis, in-memory, none) instead of hard-coding one.

### When the LLM runs

```
Multi-query:  question → LLM (rewrite) → search → results
                         BEFORE search, 1 call                 → better RECALL (find more)

Compression:  question → search → results → LLM on each doc → shorter results
                                            AFTER search, 1 call PER DOC → better PRECISION (less noise)
```

- Cost: `k=3` → 3 LLM calls; `k=10` → 10 calls. That's why the cheap `EmbeddingsFilter` exists: use it first to drop obvious junk, then an LLM compressor on what's left (a pipeline).
- They combine: multi-query to find more, then compression to clean it up.

### Errors I hit
- `ValidationError ... base_compressor Field required / base_retriever Field required` → passed `retriever=`/`llm=`. The real field names are `base_retriever` and `base_compressor`. Pydantic lists every **missing** field by name; read those names.

---

## 13. All retrievers: differences and when to use which (code: `langchain/retrievers/`)

Every retriever has the **same interface**: `retriever.invoke(question) -> list[Document]`. So they can be swapped, wrapped and combined, and a chain never knows which one it got.

### Files

| File | Retriever | Data |
|---|---|---|
| `multi_query_data.py` / `compression_data.py` | practice documents + test questions (data only, safe to import) | — |
| `vector_store/vector_search.py` | plain vector retriever (`as_retriever`: similarity / threshold / filter / mmr) | Pinecone `rag-learning` |
| `multi_query_retriever.py` | MultiQueryRetriever | Pinecone `multi-query` |
| `contextual_compression_retriever.py` | ContextualCompressionRetriever + LLMChainExtractor | Pinecone `contextual-compression` |
| `embeddings_filter_retriever.py` | compression with EmbeddingsFilter pipeline (no LLM) | Pinecone `contextual-compression` |
| `bm25_retriever.py` | BM25Retriever (keyword) | in memory |
| `ensemble_retriever.py` | EnsembleRetriever (BM25 + vector = hybrid) | in memory + Pinecone `multi-query` |
| `parent_document_retriever.py` | ParentDocumentRetriever (search small, return big) | InMemoryVectorStore + InMemoryStore |
| `self_query_retriever.py` | SelfQueryRetriever (LLM writes the metadata filter) | Pinecone `multi-query` |

Run the `*_retriever.py` that **index** first (multi_query, contextual_compression); the others reconnect to those indexes.

### What each one changes

Think of retrieval as 3 stages; each advanced retriever improves one of them:

```
   QUESTION ───────────► SEARCH ───────────► RESULTS
   multi-query           BM25 / vector       contextual compression (LLM)
   self-query (filter)   ensemble (hybrid)   embeddings filter (no LLM)
                         parent document     reranker (cross-encoder)
```

| Retriever | Simple idea | Fixes | LLM calls | Cost |
|---|---|---|---|---|
| Vector (similarity) | find by **meaning** | baseline | 0 | 1 embedding + 1 search |
| Vector + threshold | drop weak matches | "always returns k" | 0 | same |
| MMR | relevant **and** different | near-duplicate results | 0 | fetch_k then pick k |
| BM25 | find by **exact words** | codes, IDs, names, acronyms | 0 | in memory, very cheap |
| Ensemble (hybrid) | meaning + words, merged by rank (RRF) | each one's blind spot | 0 | 2 searches |
| Multi-query | ask 3 ways, merge | user words ≠ document words (**recall**) | 1, before search | + 3× searches |
| Self-query | LLM writes search text + **filter** | conditions in the question ("HR docs about...") | 1, before search | wrong filter → 0 results |
| Parent document | search small chunks, return the parent | chunk size trade-off (precision vs context) | 0 | 2 stores to sync |
| Compression: LLMChainExtractor | keep only relevant sentences | noise in the prompt (**precision**) | k, after search | slow at large k |
| Compression: LLMChainFilter | keep/drop whole docs | near-miss documents | k, after search | slow at large k |
| Compression: EmbeddingsFilter | drop pieces below a similarity score (**raw** cosine -1..1, not the (raw+1)/2 relevance score) | noise, cheaply | 0 | needs threshold tuning; `k` (default 20) also caps it |

### When to use which

| Situation | Use |
|---|---|
| Starting out, small data | plain vector retriever, `k` 3–5 |
| Unrelated questions return junk | `similarity_score_threshold` |
| Results are near-duplicates (many similar chunks) | MMR |
| Users search for codes, IDs, product names, error messages | BM25, or better **ensemble** (hybrid) |
| A good default for production | **ensemble** (hybrid) + a reranker or compression |
| Users phrase things differently from the docs | multi-query |
| Questions contain conditions (department, date, author, version) and metadata is clean | self-query |
| Long documents; the answer needs surrounding context | parent document |
| Retrieved docs are long and mostly irrelevant | contextual compression (EmbeddingsFilter first, LLM extractor if needed) |
| Latency/cost matters | avoid per-question LLM calls: hybrid + EmbeddingsFilter, no multi-query |

**They combine:** e.g. multi-query (find more) → ensemble (meaning + words) → compression (trim). Every extra stage costs latency, so add one only when an **eval set** (recall@k on test questions) shows it helps.

### Things I learned running them

- **BM25 tokenisation:** the default is `text.split()`, so `"ExpenseHub?"` ≠ `"ExpenseHub"`. Without a tokeniser, the coffee-machine doc ranked first. Fix: `preprocess_func=lambda t: re.findall(r"\w+", t.lower())`.
- **Ensemble uses rank, not score** (RRF: `weight / (60 + rank)`), because BM25 scores (~7) and cosine (~0.7) aren't comparable.
- **Parent document:** 8 parents → 27 child chunks; the search matched a 114-char child ("...25 days of paid annual leave...") but returned the 392-char handbook entry.
- **EmbeddingsFilter pipeline (no LLM):** 3 docs (347 + 392 + 232 chars) → 1 sentence of 88 chars, and it dropped the `customer_001` trap.
- Data files must contain **only data**: importing a file runs all of it (it would re-index Pinecone and call the LLM).
- **Multi-query (measured):** the LLM wrote "reclaim expenses incurred during a business trip", "request a reimbursement for travel-related costs", ... → found exp_001 **plus** 2 travel docs the normal search missed, but also let the customer-refund trap in (recall up, precision down).
- **Contextual compression (measured):** 3 docs (347 + 392 + 232 chars) → 1 sentence (88 chars); the LLM dropped the "25 days of free premium support" trap.
- **Self-query (measured):** "What are the HR rules about working from home?" → `query='working from home'` + filter `category = HR AND topic = remote work` → only HR / remote-work docs. To see the generated query you need **both** `verbose=True` and the INFO logging line.
- **Self-query gotcha:** `ImportError: DatabricksVectorSearch` comes from LangChain's automatic translator lookup (langchain-community is being sunset). Fix: pass `structured_query_translator=PineconeTranslator()`.

---

## 14. Big documents: chunk → many rows → search the chunks

**One big document = many rows.** Chunk it, and each chunk becomes its own record (id + vector + metadata). Search runs over chunks, never the whole document.

Why not one row per document: one vector can't hold 200 pages of meaning (the embedding averages everything → facts get diluted), and the LLM only needs the relevant part (tokens, context window).

| id | values (768 numbers) | metadata |
|---|---|---|
| `handbook#0` | [0.02, -0.11, ...] | `{source: "employee_handbook.pdf", page: 1, chunk: 0, text: "Welcome to..."}` |
| `handbook#457` | [...] | `{source: "employee_handbook.pdf", page: 112, chunk: 457, text: "25 days of paid annual leave..."}` |

**Metadata ties chunks back to the document**: source + page → citations ("page 112"), filters, finding neighbours.

```
INDEXING  big PDF ─ PyPDFLoader ─► pages ─ RecursiveCharacterTextSplitter ─► 800 chunks ─ add_documents(ids) ─► 800 rows
SEARCH    question ─ embed ─► search all chunk vectors ─► top k chunks (with page) ─► prompt ─► LLM
```

```python
pages = PyPDFLoader("employee_handbook.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(pages)  # keeps source + page
vector_store.add_documents(chunks, ids=[f"handbook#{i}" for i in range(len(chunks))])
```

| Problem with big docs | Fix |
|---|---|
| chunk too small to make sense | overlap, or parent document retriever |
| fact split across two chunks | `chunk_overlap` |
| near-duplicate chunks | MMR |
| need one doc out of thousands | metadata filter `{"source": ...}` |
| **Pinecone metadata limit: 40 KB per record** (the text lives in metadata) | keep chunks small; store big parent text elsewhere (DB, S3, docstore) and keep only its id in Pinecone |
| re-indexing an updated doc | stable chunk ids; delete the doc's old chunks first (orphans) |

Backend analogy: Elasticsearch full-text search. You index passages, not one giant field, and each hit points back to the book + page.

### Choosing a retriever: measure, don't guess

1. **Baseline:** plain vector retriever, `k=4`.
2. **Eval set:** 10–20 test questions + which chunk/page holds each answer. Measure **recall@k** (is the right chunk in the top k?).
3. **Look at failures, match to a fix:**

| What you see | Cause | Add |
|---|---|---|
| right chunk found, but the answer lacks context | chunks too small | parent document |
| codes / names / IDs missed | vectors weak at exact terms | ensemble (hybrid BM25 + vector) |
| "money back" vs "reimbursement" missed | wording mismatch | multi-query |
| answer comes from the wrong document | no scoping | metadata filter / self-query |
| right chunks, but noisy / long prompts | chunks too big or too many | compression (EmbeddingsFilter first, LLM extractor if needed) |
| top k are near-copies | repeated text | MMR |
| unrelated questions return junk | always returns k | score threshold |

4. **Add one at a time, re-measure.** Keep it if recall@k improves. Multi-query, self-query and the LLM compressors add an LLM call per question.

Typical setup for big documents:
```
chunks (~500–1000 chars, ~10–15% overlap)
  → hybrid search (BM25 + vector)          robust default
  → parent document (return the section)   big docs need context
  → (optional) compression / reranker      if prompts get noisy
```

Backend analogy: performance tuning. Don't add caching + sharding on day one; **measure, find the bottleneck, fix that**. The eval set is the profiler.
