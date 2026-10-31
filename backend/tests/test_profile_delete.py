"""Tests for owner-scoped profile deletion (no live DB)."""

from unittest.mock import MagicMock, patch

import query_db


def _delete(rowcount):
    conn = MagicMock()
    cur = conn.cursor.return_value
    cur.rowcount = rowcount
    with patch.object(query_db, "get_connection", return_value=conn), \
         patch.object(query_db, "put_connection"):
        ok = query_db.delete_profile_entry(5, user_id="alice")
    return ok, cur


def test_delete_is_scoped_to_owner():
    ok, cur = _delete(rowcount=1)
    sql, params = cur.execute.call_args.args
    assert ok is True
    assert "id = %s AND user_id = %s" in sql
    assert params == (5, "alice")


def test_other_users_entry_is_not_deleted():
    ok, _ = _delete(rowcount=0)
    assert ok is False
