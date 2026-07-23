"""Behavioral tests for the immutable MUSE context DAG shadow ledger."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB, context_hash


def _db(tmp_path: Path) -> SessionDB:
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", "test")
    return db


def test_context_dag_schema_is_additive_and_immutable(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tables = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    triggers = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }

    assert {
        "context_nodes",
        "context_edges",
        "context_active_heads",
        "context_node_projections",
    } <= tables
    assert {"context_nodes_no_update", "context_edges_no_update"} <= triggers


def test_context_payload_round_trips_replay_fields_without_runtime_markers(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    messages = [
        {
            "role": "user",
            "content": "clean input",
            "api_content": "provider-visible input",
            "_db_persisted": True,
        },
        {
            "role": "assistant",
            "content": "working",
            "reasoning": "private reasoning",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
    ]

    node_id = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:one",
        kind="turn",
        input_context_hash=None,
        output_messages=messages,
        payload_messages=messages,
    )
    node = db.get_context_node(node_id, decode_payload=True)

    assert node["payload"] == [
        {
            "role": "user",
            "content": "clean input",
            "api_content": "provider-visible input",
        },
        {
            "role": "assistant",
            "content": "working",
            "reasoning": "private reasoning",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
    ]
    assert len(node["output_context_hash"]) == 64


def test_context_hash_ignores_persistence_only_message_metadata() -> None:
    model_messages = [{"role": "user", "content": "same prompt"}]
    persisted_messages = [
        {
            "role": "user",
            "content": "same prompt",
            "timestamp": 123.5,
            "_db_persisted": True,
        }
    ]

    assert context_hash(model_messages) == context_hash(persisted_messages)


def test_context_dag_payload_uses_bounded_multimodal_projection(tmp_path: Path) -> None:
    db = _db(tmp_path)
    data_url = "data:image/png;base64," + ("A" * 100_000)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect this"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
            "_runtime_only": data_url,
        }
    ]

    node_id = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:multimodal",
        kind="turn",
        input_context_hash=None,
        output_messages=messages,
        payload_messages=messages,
    )
    payload = db.get_context_node(node_id, decode_payload=True)["payload"]

    assert payload == [{"role": "user", "content": "inspect this\n[screenshot]"}]
    assert data_url not in repr(payload)
    assert messages[0]["content"][1]["image_url"]["url"] == data_url


def test_context_hash_matches_durable_multimodal_projection() -> None:
    live = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ]
    durable = [{"role": "user", "content": "inspect this\n[screenshot]"}]

    assert context_hash(live) == context_hash(durable)


def test_context_event_is_idempotent_but_rejects_conflicting_replay(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    kwargs = {
        "session_id": "session-a",
        "conversation_id": "session-a",
        "event_key": "turn:same",
        "kind": "turn",
        "input_context_hash": None,
        "output_messages": [{"role": "user", "content": "one"}],
        "payload_messages": [{"role": "user", "content": "one"}],
    }

    first = db.append_context_node(**kwargs)
    second = db.append_context_node(**kwargs)

    assert first == second
    assert len(db.list_context_nodes("session-a")) == 1
    with pytest.raises(ValueError, match="event key already exists"):
        db.append_context_node(
            **{
                **kwargs,
                "output_messages": [{"role": "user", "content": "different"}],
            }
        )


def test_context_event_rejects_same_output_with_different_payload(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    output = [{"role": "user", "content": "complete"}]
    kwargs = {
        "session_id": "session-a",
        "conversation_id": "session-a",
        "event_key": "turn:payload-conflict",
        "kind": "turn",
        "storage_mode": "checkpoint",
        "input_context_hash": None,
        "output_messages": output,
    }
    db.append_context_node(**kwargs, payload_messages=output)

    with pytest.raises(ValueError, match="event key already exists"):
        db.append_context_node(
            **kwargs,
            payload_messages=[{"role": "user", "content": "truncated"}],
        )


def test_context_nodes_and_edges_reject_updates(tmp_path: Path) -> None:
    db = _db(tmp_path)
    first = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:first",
        kind="turn",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "first"}],
        payload_messages=[{"role": "user", "content": "first"}],
    )
    second = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:second",
        kind="turn",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "second"}],
        payload_messages=[{"role": "user", "content": "second"}],
        parent_node_ids=[first],
    )

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db._conn.execute(
            "UPDATE context_nodes SET kind = 'compression' WHERE id = ?", (second,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db._conn.execute(
            "UPDATE context_edges SET ordinal = 2 WHERE child_node_id = ?",
            (second,),
        )


def test_context_edges_cross_session_boundaries_and_cascade(tmp_path: Path) -> None:
    db = _db(tmp_path)
    parent = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:parent",
        kind="turn",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "before"}],
        payload_messages=[{"role": "user", "content": "before"}],
    )
    db.create_session("session-b", "test", parent_session_id="session-a")
    child = db.append_context_node(
        session_id="session-b",
        conversation_id="session-a",
        event_key="compression:one",
        kind="compression",
        input_context_hash=None,
        output_messages=[{"role": "assistant", "content": "summary"}],
        payload_messages=[{"role": "assistant", "content": "summary"}],
        parent_node_ids=[parent],
    )

    assert db.get_context_parents(child) == [parent]
    db.delete_session("session-b")
    assert db.get_context_node(child) is None
    assert db.get_context_node(parent) is not None
    assert db.get_context_parents(child) == []


def test_surviving_compression_child_keeps_context_identity_after_root_delete(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    parent = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:deletion-parent",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "before"}],
        payload_messages=[{"role": "user", "content": "before"}],
    )
    db.end_session("session-a", "compression")
    db.create_session("session-b", "test", parent_session_id="session-a")
    checkpoint_messages = [{"role": "user", "content": "summary"}]
    db.append_context_node(
        session_id="session-b",
        conversation_id="session-a",
        event_key="compression:deletion-child",
        kind="compression",
        storage_mode="checkpoint",
        input_context_hash=context_hash(
            [{"role": "user", "content": "before"}]
        ),
        output_messages=checkpoint_messages,
        payload_messages=checkpoint_messages,
        parent_node_ids=[parent],
        metadata={"source_span_complete": True},
    )

    db.delete_session("session-a")

    assert db.get_context_conversation_id("session-b") == "session-a"
    assert db.replay_context_messages("session-a") == checkpoint_messages


def test_original_history_bypasses_compression_summary_nodes(tmp_path: Path) -> None:
    db = _db(tmp_path)
    first = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:first-history",
        kind="turn",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "first"}],
        payload_messages=[{"role": "user", "content": "first"}],
    )
    summary = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="compression:history",
        kind="compression",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=[{"role": "assistant", "content": "summary"}],
        payload_messages=[{"role": "assistant", "content": "summary"}],
        parent_node_ids=[first],
    )
    second = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:second-history",
        kind="turn",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "second"}],
        payload_messages=[{"role": "user", "content": "second"}],
        parent_node_ids=[summary],
    )

    assert db.get_context_parents(second) == [summary]
    assert db.get_context_active_chain_node_ids(second) == [first, summary, second]
    assert db.get_context_history_node_ids("session-a") == [first, second]
    relations = {
        tuple(row)
        for row in db._conn.execute(
            """SELECT child_node_id, parent_node_id, relation
               FROM context_edges WHERE relation LIKE 'history_%'"""
        ).fetchall()
    }
    assert (second, first, "history_prev") in relations
    assert (first, second, "history_next") in relations


def test_active_tip_tracks_appends_and_can_be_repointed(tmp_path: Path) -> None:
    db = _db(tmp_path)
    first = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:active-first",
        kind="turn",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "first"}],
        payload_messages=[{"role": "user", "content": "first"}],
        storage_mode="checkpoint",
    )
    second = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:active-second",
        kind="turn",
        input_context_hash=None,
        output_messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
        payload_messages=[{"role": "assistant", "content": "second"}],
        parent_node_ids=[first],
    )

    assert db.get_context_active_tip_node_id("session-a") == second
    db.set_context_active_tip("session-a", first)
    assert db.get_context_active_tip_node_id("session-a") == first


def test_stale_parent_append_does_not_steal_active_tip(tmp_path: Path) -> None:
    db = _db(tmp_path)
    first_messages = [{"role": "user", "content": "first"}]
    first = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:cas-first",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=first_messages,
        payload_messages=first_messages,
    )
    active_messages = first_messages + [
        {"role": "assistant", "content": "active"}
    ]
    active = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:cas-active",
        kind="turn",
        input_context_hash=context_hash(first_messages),
        output_messages=active_messages,
        payload_messages=active_messages[1:],
        parent_node_ids=[first],
    )
    stale_messages = first_messages + [
        {"role": "assistant", "content": "stale"}
    ]
    db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:cas-stale",
        kind="turn",
        input_context_hash=context_hash(first_messages),
        output_messages=stale_messages,
        payload_messages=stale_messages[1:],
        parent_node_ids=[first],
    )

    assert db.get_context_active_tip_node_id("session-a") == active
    assert db.replay_context_messages("session-a") == active_messages


def test_delegate_session_owns_independent_context_conversation(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.create_session(
        "delegate-a",
        "delegate",
        parent_session_id="session-a",
        model_config={"_delegate_from": "session-a"},
    )

    assert db.get_context_conversation_id("session-a") == "session-a"
    assert db.get_context_conversation_id("delegate-a") == "delegate-a"


def test_context_replay_reconstructs_checkpoint_and_deltas(tmp_path: Path) -> None:
    db = _db(tmp_path)
    checkpoint_messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first"},
    ]
    complete_messages = checkpoint_messages + [
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next"},
    ]
    checkpoint = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="compression:replay-checkpoint",
        kind="compression",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=checkpoint_messages,
        payload_messages=checkpoint_messages,
    )
    db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:replay-delta",
        kind="turn",
        storage_mode="delta",
        input_context_hash=context_hash(checkpoint_messages),
        output_messages=complete_messages,
        payload_messages=complete_messages[len(checkpoint_messages):],
        parent_node_ids=[checkpoint],
    )

    assert db.replay_context_messages("session-a") == complete_messages


def test_context_replay_falls_back_when_tip_hash_does_not_match(tmp_path: Path) -> None:
    db = _db(tmp_path)
    canonical = [{"role": "user", "content": "canonical"}]
    db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:mismatched-payload",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=canonical,
        payload_messages=[{"role": "user", "content": "corrupt"}],
    )

    assert db.get_context_messages_or_fallback("session-a", canonical) is canonical


def test_context_replay_rejects_unexplained_checkpoint_input_drift(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    first_messages = [{"role": "user", "content": "first"}]
    first = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:checkpoint-parent",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=first_messages,
        payload_messages=first_messages,
    )
    replacement = [{"role": "user", "content": "replacement"}]
    db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="compression:bad-input",
        kind="compression",
        storage_mode="checkpoint",
        input_context_hash="0" * 64,
        output_messages=replacement,
        payload_messages=replacement,
        parent_node_ids=[first],
    )

    assert db.replay_context_messages("session-a") is None


def test_level_one_projection_is_mutable_without_changing_original_node(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    original = [{"role": "tool", "content": "full output"}]
    node_id = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:projection-source",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=original,
        payload_messages=original,
    )
    projected = [{"role": "tool", "content": "[terminal] output pruned"}]

    db.upsert_context_projection(
        node_id=node_id,
        level=1,
        projected_messages=projected,
        metadata={"strategy": "deterministic_tool_pruning"},
    )

    assert db.get_context_projection(node_id, level=1)["payload"] == projected
    assert db.get_context_node(node_id, decode_payload=True)["payload"] == original


def test_level_two_summary_records_replaced_contiguous_span(tmp_path: Path) -> None:
    db = _db(tmp_path)
    first = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:span-first",
        kind="turn",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "one"}],
        payload_messages=[{"role": "user", "content": "one"}],
        storage_mode="checkpoint",
    )
    second_messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    second = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:span-second",
        kind="turn",
        input_context_hash=context_hash(second_messages[:1]),
        output_messages=second_messages,
        payload_messages=second_messages[1:],
        parent_node_ids=[first],
    )
    summary_messages = [{"role": "user", "content": "summary"}]
    summary = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="compression:span-summary",
        kind="compression",
        storage_mode="checkpoint",
        input_context_hash=context_hash(second_messages),
        output_messages=summary_messages,
        payload_messages=summary_messages,
        parent_node_ids=[second],
        summary_source_node_ids=[first, second],
        metadata={"compression_level": 2},
    )

    sources = db.get_context_summary_source_node_ids(summary)
    assert sources == [first, second]
    assert db.get_context_active_tip_node_id("session-a") == summary
    assert db.replay_context_messages("session-a") == summary_messages


def test_runtime_compression_marks_incomplete_source_and_skips_projection(
    tmp_path: Path,
) -> None:
    from agent.conversation_compression import _record_context_dag_compression

    db = _db(tmp_path)
    parent_messages = [{"role": "user", "content": "finalized"}]
    parent = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:runtime-parent",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=parent_messages,
        payload_messages=parent_messages,
    )
    before_messages = parent_messages + [
        {"role": "assistant", "content": "in-progress"}
    ]
    compressed_messages = [{"role": "user", "content": "summary"}]
    agent = SimpleNamespace(
        _session_db=db,
        session_id="session-a",
        context_compressor=SimpleNamespace(
            _last_compression_level=2,
            _last_level1_projection=[
                {"role": "user", "content": "pruned in-progress"}
            ],
        ),
    )

    _record_context_dag_compression(
        agent,
        before_messages=before_messages,
        compressed_messages=compressed_messages,
        boundary_parent_session_id="session-a",
        attempt_id="runtime-incomplete",
        split_status="in_place_committed",
        in_place=True,
    )

    tip = db.get_context_active_tip_node_id("session-a")
    node = db.get_context_node(tip, decode_payload=True)
    assert node["metadata"]["source_span_complete"] is False
    assert db.get_context_summary_source_node_ids(tip) == []
    assert db.get_context_projection(parent, level=1) is None
    assert db.replay_context_messages("session-a") == compressed_messages


def test_runtime_compression_persists_complete_projection_and_source_span(
    tmp_path: Path,
) -> None:
    from agent.conversation_compression import _record_context_dag_compression

    db = _db(tmp_path)
    before_messages = [{"role": "tool", "content": "full output"}]
    parent = db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:runtime-complete-parent",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=before_messages,
        payload_messages=before_messages,
    )
    level_one = [{"role": "tool", "content": "[terminal] output pruned"}]
    compressed_messages = [{"role": "user", "content": "summary"}]
    agent = SimpleNamespace(
        _session_db=db,
        session_id="session-a",
        context_compressor=SimpleNamespace(
            _last_compression_level=2,
            _last_level1_projection=level_one,
        ),
    )

    _record_context_dag_compression(
        agent,
        before_messages=before_messages,
        compressed_messages=compressed_messages,
        boundary_parent_session_id="session-a",
        attempt_id="runtime-complete",
        split_status="in_place_committed",
        in_place=True,
    )

    tip = db.get_context_active_tip_node_id("session-a")
    node = db.get_context_node(tip, decode_payload=True)
    assert node["metadata"]["source_span_complete"] is True
    assert db.get_context_summary_source_node_ids(tip) == [parent]
    assert db.get_context_projection(parent, level=1)["payload"] == level_one


def test_shadow_dag_does_not_change_resume_projection(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.append_message("session-a", "user", "question", api_content="question + context")
    db.append_message("session-a", "assistant", "answer")
    before = db.get_messages_as_conversation("session-a")

    db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:shadow-only",
        kind="turn",
        input_context_hash=None,
        output_messages=before,
        payload_messages=before,
    )

    assert db.get_messages_as_conversation("session-a") == before


def test_live_resume_keeps_canonical_projection_when_dag_hash_matches(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    db.append_message("session-a", "user", "canonical")
    replay = [{"role": "user", "content": "canonical"}]
    db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:authoritative-replay",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=replay,
        payload_messages=replay,
    )

    restored = db.get_messages_as_conversation(
        "session-a", repair_alternation=True
    )

    assert restored[0]["content"] == "canonical"
    assert "timestamp" in restored[0]


def test_live_resume_keeps_canonical_rows_when_dag_replay_is_invalid(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    db.append_message("session-a", "user", "canonical")
    db.append_context_node(
        session_id="session-a",
        conversation_id="session-a",
        event_key="turn:invalid-authoritative-replay",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=[{"role": "user", "content": "canonical"}],
        payload_messages=[{"role": "user", "content": "different"}],
    )

    restored = db.get_messages_as_conversation(
        "session-a", repair_alternation=True
    )

    assert restored[0]["content"] == "canonical"
    assert "timestamp" in restored[0]
