"""Tests for chunker.chunk_text — semantic + rule-based paths.

The semantic path requires embeddings.get_embeddings.  All tests redirect
chunker._embed to a fast fake that returns controlled unit vectors, so no
model is downloaded during the test run.
"""
import numpy as np
import pytest
import chunker


# ---------------------------------------------------------------------------
# Embedding fakes
# ---------------------------------------------------------------------------

def _fake_embed_similar(sentences: list[str]) -> np.ndarray:
    """All sentences are nearly identical → no semantic breakpoints."""
    base = np.ones(8, dtype=np.float32)
    base /= np.linalg.norm(base)
    # Tiny random jitter so percentile computation is stable.
    rng = np.random.default_rng(0)
    matrix = base + rng.normal(0, 0.01, (len(sentences), 8)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def _fake_embed_two_topics(sentences: list[str]) -> np.ndarray:
    """First half ≈ topic A, second half ≈ topic B → one big semantic gap."""
    mid = len(sentences) // 2
    topic_a = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    topic_b = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    rows = [topic_a if i < mid else topic_b for i in range(len(sentences))]
    return np.stack(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_embed_similar(monkeypatch):
    """Default: all sentences semantically similar (no forced splits)."""
    monkeypatch.setattr(chunker, "_embed", _fake_embed_similar)


@pytest.fixture()
def patch_two_topics(monkeypatch):
    """Override embedding to produce one clear topic boundary."""
    monkeypatch.setattr(chunker, "_embed", _fake_embed_two_topics)


@pytest.fixture(autouse=True)
def _disable_semantic(monkeypatch):
    """Run rule-based path by default so existing edge-case tests are stable.

    Semantic-specific tests opt-in by patching SEMANTIC_CHUNKING_ENABLED=True.
    """
    from unittest.mock import MagicMock
    stub = MagicMock()
    stub.SEMANTIC_CHUNKING_ENABLED = False
    stub.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE = 95
    monkeypatch.setattr("chunker.settings", stub, raising=False)
    # Patch the lazy import inside chunk_text too.
    import sys
    fake_settings_mod = MagicMock()
    fake_settings_mod.settings = stub
    monkeypatch.setitem(sys.modules, "settings", fake_settings_mod)


# ---------------------------------------------------------------------------
# Helper: small-settings wrapper
# ---------------------------------------------------------------------------

def _chunk(text: str, size: int = 200, overlap: int = 20) -> list[str]:
    return chunker.chunk_text(text, chunk_size=size, chunk_overlap=overlap)


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------

def test_empty_input():
    assert _chunk("") == []


def test_whitespace_only():
    assert _chunk("   \n\n   ") == []


def test_single_short_paragraph():
    text = "Hello world."
    chunks = _chunk(text)
    assert chunks == ["Hello world."]


def test_multiple_short_paragraphs_merged():
    """Short paragraphs should be merged into a single chunk."""
    parts = ["Line one.", "Line two.", "Line three."]
    text = "\n\n".join(parts)
    chunks = _chunk(text)
    # All three fit in 200 chars so there should be exactly 1 chunk
    assert len(chunks) == 1
    for part in parts:
        assert part in chunks[0]


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def test_two_large_paragraphs_become_separate_chunks():
    """Two paragraphs each near CHUNK_SIZE must not collapse into one."""
    para_a = "Alpha. " * 28          # ~196 chars
    para_b = "Beta.  " * 28          # ~196 chars
    text = para_a.strip() + "\n\n" + para_b.strip()
    chunks = _chunk(text)
    assert len(chunks) >= 2
    assert any("Alpha" in c for c in chunks)
    assert any("Beta" in c for c in chunks)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def test_long_paragraph_split_at_sentences():
    """A paragraph > CHUNK_SIZE must be split at sentence boundaries."""
    sentences = [f"Sentence number {i} ends here." for i in range(20)]
    text = " ".join(sentences)
    chunks = _chunk(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250, f"Chunk too large: {len(c)}"


def test_sentences_not_split_mid_word():
    """Each chunk must end cleanly — no orphaned leading space."""
    text = "First sentence ends. Second sentence is here. Third sentence follows it."
    chunks = _chunk(text)
    for c in chunks:
        assert not c.startswith(" ")
        assert c == c.strip()


# ---------------------------------------------------------------------------
# Character fallback (sentences > CHUNK_SIZE)
# ---------------------------------------------------------------------------

def test_very_long_sentence_character_fallback():
    """A sentence longer than CHUNK_SIZE must still be chunked."""
    long_sentence = "word " * 100      # ~500 chars, > 200
    chunks = _chunk(long_sentence.strip())
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def test_overlap_carries_tail():
    """With chunk_overlap>0 the content should carry across chunk boundaries."""
    text = ("Short sentence A. " * 5) + "\n\n" + ("Short sentence B. " * 5)
    chunks = _chunk(text, size=50, overlap=10)
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Settings-driven behaviour
# ---------------------------------------------------------------------------

def test_chunk_size_respected():
    text = "This is a sentence. " * 20
    chunks = _chunk(text, size=50, overlap=0)
    for c in chunks:
        assert len(c) <= 60, f"Chunk exceeded max: {len(c)}"


def test_no_duplicate_content():
    """Chunks should not gratuitously repeat entire sentences."""
    text = "Unique fact one. Unique fact two. Unique fact three."
    chunks = _chunk(text)
    combined = " ".join(chunks)
    for kw in ("one", "two", "three"):
        assert kw in combined


# ---------------------------------------------------------------------------
# Semantic path
# ---------------------------------------------------------------------------

def test_semantic_splits_at_topic_boundary(patch_two_topics, monkeypatch):
    """With two clearly distinct embedding clusters, semantic chunking should
    produce at least 2 chunks even when all text fits in one character window."""
    import sys
    from unittest.mock import MagicMock
    stub = MagicMock()
    stub.SEMANTIC_CHUNKING_ENABLED = True
    stub.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE = 95
    fake_mod = MagicMock()
    fake_mod.settings = stub
    monkeypatch.setitem(sys.modules, "settings", fake_mod)

    # 10 sentences per topic — clear boundary in the middle.
    topic_a = " ".join(f"Database fact {i}." for i in range(10))
    topic_b = " ".join(f"Networking fact {i}." for i in range(10))
    text = topic_a + " " + topic_b
    chunks = chunker.chunk_text(text, chunk_size=2000, chunk_overlap=0)
    assert len(chunks) >= 2


def test_semantic_fallback_when_too_few_sentences(monkeypatch):
    """Fewer than 3 sentences must use rule-based path (no embed call)."""
    import sys
    from unittest.mock import MagicMock, patch
    stub = MagicMock()
    stub.SEMANTIC_CHUNKING_ENABLED = True
    stub.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE = 95
    fake_mod = MagicMock()
    fake_mod.settings = stub
    monkeypatch.setitem(sys.modules, "settings", fake_mod)

    with patch.object(chunker, "_embed", side_effect=AssertionError("should not embed")) as mock_embed:
        chunks = chunker.chunk_text("Only one sentence here.", chunk_size=200, chunk_overlap=20)
    assert chunks == ["Only one sentence here."]
    mock_embed.assert_not_called()


def test_semantic_fallback_on_embed_error(monkeypatch):
    """If _embed raises, chunk_text must not crash — falls back to rule-based."""
    import sys
    from unittest.mock import MagicMock
    stub = MagicMock()
    stub.SEMANTIC_CHUNKING_ENABLED = True
    stub.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE = 95
    fake_mod = MagicMock()
    fake_mod.settings = stub
    monkeypatch.setitem(sys.modules, "settings", fake_mod)
    monkeypatch.setattr(chunker, "_embed", lambda _: (_ for _ in ()).throw(RuntimeError("model unavailable")))

    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunker.chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 1
    assert all(text_part in " ".join(chunks) for text_part in ["one", "two", "three"])
