"""Tests for profile entry updates by ID (no live DB)."""

from unittest.mock import MagicMock, patch

import psycopg2

import query_db


def _update(fetchone=(7,), execute_error=None):
    conn = MagicMock()
    cur = conn.cursor.return_value
    cur.fetchone.return_value = fetchone
    if execute_error:
        cur.execute.side_effect = execute_error
    with patch.object(query_db, "get_connection", return_value=conn), \
         patch.object(query_db, "put_connection"):
        status = query_db.update_profile_entry_by_id(7, " city ", "Paris", user_id="alice")
    return status, cur


def test_updates_the_given_entry_for_that_user():
    status, cur = _update()
    sql, params = cur.execute.call_args.args
    assert status == "updated"
    assert "WHERE id = %s AND user_id = %s" in sql
    assert params == ("city", "Paris", "general", 7, "alice")


def test_missing_or_other_users_entry_is_not_found():
    status, _ = _update(fetchone=None)
    assert status == "not_found"


def test_renaming_onto_existing_key_is_a_conflict():
    status, _ = _update(execute_error=psycopg2.errors.UniqueViolation())
    assert status == "conflict"
