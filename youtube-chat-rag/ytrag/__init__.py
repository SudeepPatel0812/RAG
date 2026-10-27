"""
ytrag: production version of youtube-chat-rag.py (same R -> A -> G, built to be run and maintained).

Usage (from the youtube-chat-rag folder, Ollama running, PINECONE_API_KEY in .env):
    python -m ytrag ingest "https://www.youtube.com/watch?v=JX9xNgN0vmk"
    python -m ytrag ask JX9xNgN0vmk "Does the speaker talk about why AI won't replace developers?"
    python -m unittest discover -s tests          # tests, no Ollama/Pinecone needed

Modules (each does one job; the learning script did all of them top to bottom):
    config.py      settings from env vars, validated once at startup
    transcript.py  video id parsing, fetching, time-window chunking (pure, unit-tested)
    embeddings.py  nomic-embed-text WITH its search_document:/search_query: prefixes
    store.py       Pinecone: one namespace per video, idempotent re-ingest, stale-chunk cleanup
    chain.py       LCEL chain: retrieve -> (no docs? "I don't know") -> prompt -> llm, + sources
    __main__.py    CLI: ingest / ask, logging, clean error messages, exit codes

What changed vs the learning script, and why:
    learning script                           production
    -------------------------------------     ----------------------------------------------------
    everything at module level, runs on       functions + a CLI; importing a module has no side
    import (re-indexes, calls the LLM)        effects; the same code is reusable from an API later
    hard-coded video id, models, k            config.py reads env vars with defaults (12-factor)
    print + bare except                       logging + specific exceptions -> clear message, exit 1
    one video per index, metadata filter      Pinecone NAMESPACE per video: isolation by design
                                              (Pinecone's multitenancy pattern), and no filter to forget
    re-chunking leaves orphan ids             upsert new chunks, then delete ids that are no longer
                                              produced (no empty window, no orphans)
    re-runs re-embed everything               `ingest` skips a video that's already indexed
                                              (unless --force)
    OllamaEmbeddings without prefixes         nomic prefixes added (the model was trained with them)
    always k chunks -> LLM                    score threshold; no relevant chunk -> answer
                                              "I don't know" WITHOUT calling the LLM (cheaper, safer)
    sources = everything retrieved            sources = the timestamps the answer actually cites
    print the whole answer at the end         streaming tokens (chain.stream) like a chat UI
    no tests                                  unit tests for the pure logic (parsing, chunking,
                                              formatting, citations, prefixes)

Not included on purpose (add when needed): an HTTP API (wrap chain.py with FastAPI), async
ingestion queue, caching answers, evaluation set / recall@k, auth, multi-language transcripts.
"""
