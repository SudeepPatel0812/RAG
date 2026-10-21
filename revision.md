# RAG Revision Notes

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
