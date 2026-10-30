"""Tests for query_db.py — connection pool and Q&A retrieval scoping (no live DB)."""

import threading
from unittest.mock import MagicMock, patch

import pytest

import query_db


@pytest.fixture
def fresh_pool():
    saved = query_db._pool
    query_db._pool = None
    yield
    query_db._pool = saved


class TestConnectionPool:
    def test_pool_is_thread_safe_and_created_once(self, fresh_pool):
        created = MagicMock()
        created.return_value.closed = False
        with patch.object(query_db.pool, "ThreadedConnectionPool", created):
            threads = [threading.Thread(target=query_db._get_pool) for _ in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert created.call_count == 1


class TestRetrieveSimilarQueries:
    def test_filters_by_user_id(self):
        conn = MagicMock()
        cur = conn.cursor.return_value
        cur.fetchall.return_value = []
        with patch.object(query_db, "get_connection", return_value=conn), \
             patch.object(query_db, "put_connection"):
            query_db.retrieve_similar_queries([0.1, 0.2], k=4, user_id="alice")

        sql, params = cur.execute.call_args[0]
        assert "WHERE user_id = %s" in sql
        assert "alice" in params
