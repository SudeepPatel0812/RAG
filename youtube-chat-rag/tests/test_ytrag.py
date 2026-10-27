"""Unit tests for the pure logic. No Ollama or Pinecone needed:
    cd youtube-chat-rag; python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import ytrag` works from anywhere

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from ytrag.chain import NO_ANSWER, cited_sources, format_docs
from ytrag.embeddings import NomicEmbeddings
from ytrag.store import chunk_id, to_documents
from ytrag.transcript import Chunk, Snippet, chunk_snippets, parse_video_id, timestamp


class ParseVideoIdTest(unittest.TestCase):
    def test_accepts_ids_and_url_forms(self):
        for value in [
            "JX9xNgN0vmk",
            "https://www.youtube.com/watch?v=JX9xNgN0vmk",
            "https://www.youtube.com/watch?v=JX9xNgN0vmk&t=54s&list=abc",
            "youtube.com/watch?v=JX9xNgN0vmk",
            "https://m.youtube.com/watch?v=JX9xNgN0vmk",
            "https://youtu.be/JX9xNgN0vmk?t=10",
            "https://www.youtube.com/shorts/JX9xNgN0vmk",
            "https://www.youtube.com/embed/JX9xNgN0vmk",
        ]:
            self.assertEqual(parse_video_id(value), "JX9xNgN0vmk", value)

    def test_rejects_garbage(self):
        for value in ["", "hello", "https://example.com/watch?v=JX9xNgN0vmk", "https://youtube.com/watch?v=short"]:
            with self.assertRaises(ValueError, msg=value):
                parse_video_id(value)


def snippets(n, seconds_each=5.0):
    return [Snippet(f"line{i}", i * seconds_each, seconds_each) for i in range(n)]


class ChunkSnippetsTest(unittest.TestCase):
    def test_windows_overlap_and_last_chunk(self):
        chunks = chunk_snippets(snippets(30), window_seconds=60, overlap_snippets=2)  # 150 s of speech
        self.assertEqual([c.start for c in chunks], [0, 50, 100])
        self.assertTrue(chunks[0].text.startswith("line0 ") and chunks[0].text.endswith("line11"))
        self.assertTrue(chunks[1].text.startswith("line10 line11 "))  # 2 lines of overlap
        self.assertTrue(chunks[-1].text.endswith("line29"))  # the leftover after the loop isn't lost
        self.assertEqual([c.index for c in chunks], [0, 1, 2])

    def test_no_chunk_made_only_of_overlap(self):
        chunks = chunk_snippets(snippets(12), window_seconds=60, overlap_snippets=2)  # exactly one window
        self.assertEqual(len(chunks), 1)

    def test_short_video_and_empty(self):
        self.assertEqual(len(chunk_snippets(snippets(3), 60, 2)), 1)
        self.assertEqual(chunk_snippets([], 60, 2), [])

    def test_overlap_larger_than_window_still_progresses(self):
        chunks = chunk_snippets(snippets(20, seconds_each=30.0), window_seconds=60, overlap_snippets=5)
        self.assertTrue(chunks[-1].text.endswith("line19"))
        self.assertLess(len(chunks), 40)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            chunk_snippets(snippets(3), 0, 2)


class FormattingTest(unittest.TestCase):
    def test_timestamp(self):
        self.assertEqual(timestamp(54.0), "0:54")  # Pinecone returns floats
        self.assertEqual(timestamp(650), "10:50")
        self.assertEqual(timestamp(3725), "1:02:05")

    def test_format_docs_and_cited_sources(self):
        docs = [Document(page_content="a", metadata={"start": 54.0}), Document(page_content="b", metadata={"start": 650.0})]
        self.assertEqual(format_docs(docs), "[0:54] a\n\n[10:50] b")
        self.assertEqual(cited_sources("She says so [10:50].", docs), [docs[1]])
        self.assertEqual(cited_sources("No citation here.", docs), docs)  # fall back to everything retrieved
        self.assertEqual(cited_sources(NO_ANSWER, docs), [])

    def test_documents_and_ids(self):
        chunk = Chunk(3, "text", 54)
        doc = to_documents("JX9xNgN0vmk", [chunk])[0]
        self.assertEqual(doc.metadata["url"], "https://www.youtube.com/watch?v=JX9xNgN0vmk&t=54s")
        self.assertEqual(chunk_id("JX9xNgN0vmk", chunk), "JX9xNgN0vmk#3")


class NomicPrefixTest(unittest.TestCase):
    def test_prefixes_are_added(self):
        with patch.object(OllamaEmbeddings, "embed_documents", lambda self, texts: texts), \
             patch.object(OllamaEmbeddings, "embed_query", lambda self, text: text):
            emb = NomicEmbeddings(model="nomic-embed-text")
            self.assertEqual(emb.embed_documents(["x"]), ["search_document: x"])
            self.assertEqual(emb.embed_query("q"), "search_query: q")


if __name__ == "__main__":
    unittest.main()
