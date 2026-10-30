"""RAG Chat CLI — init, ingest, and dev server.

Usage:
    python cli.py init              Create knowledge/ dir and .env from template
    python cli.py ingest [DIR]      Index knowledge base into pgvector
    python cli.py dev               Start uvicorn with hot-reload
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rag-cli")


def cmd_init(args):
    """Scaffold project: create knowledge/ dir, copy .env.example → .env."""
    root = Path(__file__).resolve().parent.parent

    # Create knowledge dir
    kb = root / "knowledge"
    if not kb.exists():
        kb.mkdir()
        logger.info("[+] Created knowledge/ directory")
    else:
        logger.info("[=] knowledge/ already exists")

    # Copy .env.example → .env (if not exists)
    env_example = root / ".env.example"
    env_file = root / ".env"
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        logger.info("[+] Created .env from .env.example — add your API key!")
    elif env_file.exists():
        logger.info("[=] .env already exists")
    else:
        logger.warning("[!] No .env.example found")

    # Check for example KB
    example = kb / "example.txt"
    if not example.exists() and not any(kb.glob("*.txt")):
        logger.info("[!] No knowledge base files found.  Add .txt/.md files to knowledge/")
    else:
        logger.info(f"[=] Knowledge base: {len(list(kb.glob('*')))} file(s)")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Edit .env with your LLM_API_KEY")
    logger.info("  2. Add knowledge base files to knowledge/")
    logger.info("  3. Start PostgreSQL: docker compose up postgres -d")
    logger.info("  4. Run: python cli.py dev")


def cmd_ingest(args):
    """Read knowledge base files, chunk, embed, store in pgvector."""
    from settings import settings
    from chunker import chunk_text

    kb_dir = Path(args.dir or settings.KNOWLEDGE_DIR)
    if not kb_dir.exists():
        logger.error(f"Directory not found: {kb_dir}")
        sys.exit(1)

    files = sorted(
        p for p in kb_dir.iterdir()
        if p.suffix in (".txt", ".md") and p.is_file()
    )
    if not files:
        logger.error(f"No .txt or .md files in {kb_dir}")
        sys.exit(1)

    logger.info(f"Indexing {len(files)} file(s) from {kb_dir}/")

    # Init DB
    import query_db
    if not query_db.init_db():
        logger.error("Database connection failed.  Is PostgreSQL running?")
        sys.exit(1)

    # Clear existing chunks
    query_db.clear_document_chunks()
    logger.info("Cleared existing document index")

    # Chunk and store (uses the same semantic chunker as main.py)
    total_chunks = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if chunks:
            query_db.store_document_chunks(chunks, source=path.name)
            total_chunks += len(chunks)
            logger.info(f"  {path.name}: {len(chunks)} chunks")

    logger.info(f"Done — {total_chunks} total chunks indexed in pgvector")


def cmd_dev(args):
    """Start uvicorn development server with hot-reload."""
    import subprocess

    from settings import settings

    host = args.host or settings.HOST
    port = args.port or settings.PORT

    backend_dir = Path(__file__).resolve().parent
    logger.info(f"Starting dev server at http://{host}:{port}")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", host,
            "--port", str(port),
            "--reload",
        ],
        cwd=str(backend_dir),
        check=False,
    )


# ---------------------------------------------------------------------------
#  Memory inspector commands
# ---------------------------------------------------------------------------

def _ensure_db():
    """Initialize DB, exit if unavailable."""
    import query_db
    if not query_db.init_db():
        logger.error("Database connection failed.  Is PostgreSQL running?")
        sys.exit(1)
    return query_db


def cmd_memory(args):
    """Route memory sub-commands: inspect | query."""
    sub = getattr(args, "memory_command", None)
    if sub == "inspect":
        _cmd_memory_inspect(args)
    elif sub == "query":
        _cmd_memory_query(args)
    else:
        logger.error("Usage: python cli.py memory {inspect|query}")
        sys.exit(1)


def _cmd_memory_inspect(args):
    """Print full cognitive state: threads, insights, concepts."""
    query_db = _ensure_db()
    cid = getattr(args, "conversation", None)
    insights_only = getattr(args, "insights_only", False)

    # Determine which conversations to inspect
    if cid:
        conv = query_db.get_conversation(cid)
        if not conv:
            logger.error(f"Conversation {cid} not found")
            sys.exit(1)
        conversations = [conv]
    else:
        conversations = query_db.list_conversations(limit=20)

    if not conversations:
        logger.info("No conversations found.")
        return

    bar = "\u2550" * 60
    print(f"\n\u2550\u2550\u2550 Memory State {bar}\n")

    for conv in conversations:
        conv_id = conv["id"]
        title = conv.get("title", "Untitled")
        print(f"  Conversation: {conv_id[:12]}...  \"{title}\"")

        threads = query_db.get_threads(conv_id)
        if threads:
            print(f"  Threads: {len(threads)} active\n")
            for t in threads:
                label = t.get("label") or "(unlabeled)"
                mc = t.get("message_count", 0)
                summary = t.get("summary") or "(no summary yet)"
                tid = t["id"]
                # Thread box
                header = f" Thread: \"{label}\" " + "\u2500" * max(1, 48 - len(label)) + f" {mc} msgs "
                print(f"  \u250C\u2500{header}\u2500\u2510")
                # Summary (wrap at ~60 chars)
                _print_wrapped(f"Summary: {summary}", indent=5, width=58)
                # Insights for this thread
                insights = query_db.get_insights_for_thread(tid, limit=10)
                if insights:
                    print(f"  \u2502  Insights:")
                    for ins in insights:
                        itype = ins["insight_type"]
                        itext = ins["insight_text"]
                        conf = ins.get("confidence_score", 0)
                        # Truncate long text
                        if len(itext) > 55:
                            itext = itext[:52] + "..."
                        print(f"  \u2502    [{itype:<15}] {itext} ({conf:.2f})")
                print(f"  \u2514{'─' * 60}\u2518\n")
        elif not insights_only:
            print("  Threads: 0\n")

        if insights_only:
            all_insights = query_db.get_insights(conv_id, limit=30)
            if all_insights:
                print("  All Insights:")
                for ins in all_insights:
                    itype = ins["insight_type"]
                    itext = ins["insight_text"]
                    conf = ins.get("confidence_score", 0)
                    tid_short = (ins.get("thread_id") or "")[:8]
                    if len(itext) > 55:
                        itext = itext[:52] + "..."
                    print(f"    [{itype:<15}] {itext} ({conf:.2f})  thread:{tid_short}")
                print()

        # Concept links
        concepts = query_db.get_concepts_for_conversation(conv_id)
        if concepts:
            print(f"  Concept Links: {len(concepts)}")
            for c in concepts[:15]:
                concept = c["concept"]
                src = c.get("source_type", "")
                print(f"    \"{concept}\" \u2190\u2192 {src}")
            if len(concepts) > 15:
                print(f"    ... and {len(concepts) - 15} more")
            print()

        print(f"{'─' * 65}\n")

    print(f"{'═' * 65}\n")


def _cmd_memory_query(args):
    """Cross-thread semantic search over insights and concepts."""
    q = args.query_text
    if not q:
        logger.error("Usage: python cli.py memory query \"your search text\"")
        sys.exit(1)

    query_db = _ensure_db()
    from embeddings import get_query_embedding

    embedding = get_query_embedding(q)
    k = getattr(args, "k", 10)

    print(f"\n─── Cross-Thread Search {'─' * 42}\n")
    print(f"  Query: \"{q}\"\n")

    # Insights
    insight_type = getattr(args, "type", None)
    insights = query_db.search_similar_insights(embedding, k=k, insight_type=insight_type)
    if insights:
        print("  Matching Insights:")
        for i, ins in enumerate(insights, 1):
            itype = ins["insight_type"]
            itext = ins["insight_text"]
            conf = ins.get("confidence_score", 0)
            sim = ins.get("similarity", 0)
            tid = (ins.get("thread_id") or "—")[:8]
            if len(itext) > 60:
                itext = itext[:57] + "..."
            print(f"    {i}. [{itype}] \"{itext}\"")
            print(f"       Thread: {tid} │ Confidence: {conf:.2f} │ Sim: {sim:.2f}")
        print()
    else:
        print("  No matching insights found.\n")

    # Concepts
    concepts = query_db.search_similar_concepts(embedding, k=k)
    if concepts:
        print("  Matching Concepts:")
        for i, c in enumerate(concepts, 1):
            concept = c["concept"]
            sim = c.get("similarity", 0)
            src = c.get("source_type", "")
            print(f"    {i}. \"{concept}\" ←→ {src} ({sim:.2f})")
        print()
    else:
        print("  No matching concepts found.\n")

    print(f"{'─' * 65}\n")


def _print_wrapped(text: str, indent: int = 4, width: int = 60):
    """Print text wrapped to *width* with leading indent inside a box."""
    prefix = "  │" + " " * (indent - 3)
    words = text.split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            print(f"{prefix}{line}")
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        print(f"{prefix}{line}")


def main():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Stateful AI reference architecture — CLI tools",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init
    sub.add_parser("init", help="Initialize project structure")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index knowledge base into pgvector")
    p_ingest.add_argument("dir", nargs="?", help="Knowledge base directory (default: knowledge/)")

    # dev
    p_dev = sub.add_parser("dev", help="Start development server")
    p_dev.add_argument("--host", help="Bind host")
    p_dev.add_argument("--port", type=int, help="Bind port")

    # memory — with sub-sub-commands
    p_mem = sub.add_parser("memory", help="Inspect cognitive state or query across threads")
    mem_sub = p_mem.add_subparsers(dest="memory_command", help="Memory commands")

    p_inspect = mem_sub.add_parser("inspect", help="Print threads, insights, concepts")
    p_inspect.add_argument("--conversation", "-c", help="Limit to one conversation ID")
    p_inspect.add_argument("--insights-only", action="store_true", help="Show only insights")

    p_query = mem_sub.add_parser("query", help="Cross-thread semantic search")
    p_query.add_argument("query_text", help="Natural-language search query")
    p_query.add_argument("--type", choices=["decision", "conclusion", "hypothesis", "open_question", "observation"],
                         help="Filter by insight type")
    p_query.add_argument("-k", type=int, default=10, help="Max results (default: 10)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "dev":
        cmd_dev(args)
    elif args.command == "memory":
        cmd_memory(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
