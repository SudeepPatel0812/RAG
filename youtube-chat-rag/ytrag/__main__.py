"""CLI: python -m ytrag ingest <url-or-id> | python -m ytrag ask <url-or-id> "question" """

import argparse
import logging
import sys

from .chain import build_chain, cited_sources
from .config import ConfigError, load_settings
from .store import existing_ids, get_index, upsert_video
from .transcript import TranscriptUnavailable, chunk_snippets, fetch_snippets, parse_video_id, timestamp, video_url

log = logging.getLogger("ytrag")


def ingest(args, settings) -> None:
    video_id = parse_video_id(args.video)
    index = get_index(settings)
    if existing_ids(index, video_id) and not args.force:
        print(f"{video_id} is already indexed (use --force to re-ingest)")
        return

    snippets = fetch_snippets(video_id, settings.languages)
    chunks = chunk_snippets(snippets, settings.window_seconds, settings.overlap_snippets)
    if not chunks:
        raise TranscriptUnavailable(f"Transcript for {video_id} is empty")
    count = upsert_video(settings, index, video_id, chunks)
    print(f"indexed {count} chunks for {video_id}")


def ask(args, settings) -> None:
    video_id = parse_video_id(args.video)
    index = get_index(settings)
    if not existing_ids(index, video_id):
        raise TranscriptUnavailable(f"{video_id} is not indexed yet: run `python -m ytrag ingest {video_id}` first")

    chain = build_chain(settings, index, video_id)
    result = {"answer": ""}
    if args.no_stream:
        result = chain.invoke(args.question)
        print(result["answer"])
    else:
        for part in chain.stream(args.question):  # dict pieces: question, docs, then answer tokens
            if "answer" in part:
                print(part["answer"], end="", flush=True)
                result["answer"] += part["answer"]
            if "docs" in part:
                result["docs"] = part["docs"]
        print()

    sources = cited_sources(result["answer"], result.get("docs", []))
    if sources:
        print("\nsources:")
        for doc in sources:
            print(f"  [{timestamp(doc.metadata['start'])}] {video_url(video_id, doc.metadata['start'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ytrag", description="Chat with YouTube videos (RAG)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="fetch, chunk, embed and store a video's transcript")
    p_ingest.add_argument("video", help="YouTube URL or video id")
    p_ingest.add_argument("--force", action="store_true", help="re-ingest even if already indexed")
    p_ingest.set_defaults(func=ingest)

    p_ask = sub.add_parser("ask", help="ask a question about an ingested video")
    p_ask.add_argument("video", help="YouTube URL or video id")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-stream", action="store_true", help="print the answer at the end instead of streaming")
    p_ask.set_defaults(func=ask)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:  # "no relevant docs" is a normal outcome here (we answer "I don't know")
        logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)

    try:
        args.func(args, load_settings())
    except (ConfigError, TranscriptUnavailable, ValueError) as e:  # expected problems: clean message
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception:  # unexpected: keep the traceback for debugging
        log.exception("unexpected error")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
