"""Tests for query_db.delete_last_exchange — used by /chat/regenerate (no live DB)."""

from unittest.mock import MagicMock, patch

import query_db


def _run(fetchone, fetchall=()):
    conn = MagicMock()
    cur = conn.cursor.return_value
    cur.fetchone.return_value = fetchone
    cur.fetchall.return_value = list(fetchall)
    with patch.object(query_db, "get_connection", return_value=conn), \
         patch.object(query_db, "put_connection"):
        result = query_db.delete_last_exchange("conv-1")
    return result, cur, conn


def test_deletes_last_user_message_and_everything_after():
    result, cur, conn = _run(fetchone=(42, "what is rag?"), fetchall=[(42,), (43,)])

    assert result == 2
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    params = [c.args[1] for c in cur.execute.call_args_list]
    assert "id >= %s" in sqls[1] and params[1] == ("conv-1", 42)
    # Only the Q&A row for that exact user message is removed
    assert "user_queries" in sqls[2] and params[2] == ("conv-1", "what is rag?")
    assert params[3] == (2, "conv-1")
    conn.commit.assert_called_once()


def test_no_user_message_is_a_noop():
    result, cur, conn = _run(fetchone=None)

    assert result == 0
    assert cur.execute.call_count == 1
    conn.commit.assert_not_called()
