"""Regression tests for iteration-limit exit normalization (#61631)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_finalizer import finalize_turn


class _LimitAgent:
    def __init__(
        self,
        *,
        max_iterations=60,
        budget_remaining=0,
        completion_explainer=False,
    ):
        self.max_iterations = max_iterations
        self.iteration_budget = SimpleNamespace(
            remaining=budget_remaining, used=max_iterations, max_total=max_iterations
        )
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._handle_max_iterations_called = False
        self._completion_explainer = completion_explainer

    def _handle_max_iterations(self, messages, api_call_count):
        self._handle_max_iterations_called = True
        return "summary from extra call"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return self._completion_explainer

    def _format_turn_completion_explanation(self, _reason):
        return "iteration-limit explanation"

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _finalize(
    agent,
    *,
    final_response,
    exit_reason,
    api_call_count=60,
    pending_verification_response=None,
    messages=None,
    current_turn_user_idx=None,
):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=messages or [{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response=pending_verification_response,
        current_turn_user_idx=current_turn_user_idx,
    )


def test_completed_skill_turn_triggers_evidence_linked_review(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(max_iterations=60, budget_remaining=10)
    agent.valid_tool_names = ["skill_manage"]
    reviews = []
    agent._spawn_background_review = lambda **kwargs: reviews.append(kwargs)
    messages = [
        {
            "role": "user",
            "content": (
                '[IMPORTANT: The user has invoked the "deployment-patterns" '
                "skill, indicating they want you to follow its instructions. "
                "The full skill content is loaded below.]"
            ),
        },
        {"role": "assistant", "content": "Completed with verification."},
    ]

    result = _finalize(
        agent,
        final_response="Completed with verification.",
        exit_reason="text_response(stop)",
        api_call_count=1,
        messages=messages,
        current_turn_user_idx=0,
    )

    assert result["completed"] is True
    assert len(reviews) == 1
    assert reviews[0]["review_skills"] is True
    assert reviews[0]["skill_evidence"] == ["deployment-patterns"]


def test_completed_turn_appends_immutable_context_node(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-test", "test")
    agent = _LimitAgent(max_iterations=60, budget_remaining=10)
    agent._session_db = db
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "completed"},
    ]

    _finalize(
        agent,
        final_response="completed",
        exit_reason="text_response(stop)",
        api_call_count=1,
        messages=messages,
        current_turn_user_idx=0,
    )

    nodes = db.list_context_nodes("sess-test")
    assert len(nodes) == 1
    assert nodes[0]["kind"] == "turn"
    assert nodes[0]["storage_mode"] == "checkpoint"
    node = db.get_context_node(nodes[0]["id"], decode_payload=True)
    assert node["payload"] == messages
    assert node["metadata"]["turn_exit_reason"] == "text_response(stop)"


def test_first_context_node_checkpoints_preexisting_history(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-test", "test")
    agent = _LimitAgent(max_iterations=60, budget_remaining=10)
    agent._session_db = db
    messages = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "history"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "completed"},
    ]

    _finalize(
        agent,
        final_response="completed",
        exit_reason="text_response(stop)",
        api_call_count=1,
        messages=messages,
        current_turn_user_idx=2,
    )

    node = db.get_context_node(
        db.get_context_active_tip_node_id("sess-test"), decode_payload=True
    )
    assert node["storage_mode"] == "checkpoint"
    assert node["payload"] == messages


def test_canonical_rewrite_forces_next_turn_checkpoint(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-test", "test")
    old_messages = [{"role": "user", "content": "old branch"}]
    old_tip = db.append_context_node(
        session_id="sess-test",
        conversation_id="sess-test",
        event_key="turn:old-branch",
        kind="turn",
        storage_mode="checkpoint",
        input_context_hash=None,
        output_messages=old_messages,
        payload_messages=old_messages,
    )
    agent = _LimitAgent(max_iterations=60, budget_remaining=10)
    agent._session_db = db
    rewritten_messages = [
        {"role": "user", "content": "rewritten task"},
        {"role": "assistant", "content": "new answer"},
    ]

    _finalize(
        agent,
        final_response="new answer",
        exit_reason="text_response(stop)",
        api_call_count=1,
        messages=rewritten_messages,
        current_turn_user_idx=0,
    )

    new_tip = db.get_context_active_tip_node_id("sess-test")
    assert new_tip != old_tip
    node = db.get_context_node(new_tip, decode_payload=True)
    assert node["storage_mode"] == "checkpoint"
    assert node["payload"] == rewritten_messages
    assert db.replay_context_messages("sess-test") == rewritten_messages


def test_context_dag_failure_does_not_lose_final_response(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-test", "test")
    monkeypatch.setattr(
        db,
        "append_context_node",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("dag unavailable")),
    )
    agent = _LimitAgent(max_iterations=60, budget_remaining=10)
    agent._session_db = db

    result = _finalize(
        agent,
        final_response="completed",
        exit_reason="text_response(stop)",
        api_call_count=1,
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "completed"},
        ],
        current_turn_user_idx=0,
    )

    assert result["final_response"] == "completed"
    assert "cleanup_errors" not in result
    assert db.list_context_nodes("sess-test") == []


def test_canonical_persistence_failure_prevents_context_node(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-test", "test")
    agent = _LimitAgent(max_iterations=60, budget_remaining=10)
    agent._session_db = db
    agent._persist_session = lambda *_a, **_kw: (_ for _ in ()).throw(
        RuntimeError("canonical write failed")
    )

    result = _finalize(
        agent,
        final_response="completed",
        exit_reason="text_response(stop)",
        api_call_count=1,
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "completed"},
        ],
        current_turn_user_idx=0,
    )

    assert result["final_response"] == "completed"
    assert result["cleanup_errors"] == [
        "persist_session: canonical write failed"
    ]
    assert db.list_context_nodes("sess-test") == []


def test_pending_verify_response_is_preserved_for_cron_delivery(monkeypatch):
    """A held-back verification response survives last-turn exhaustion."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "complete cron report body"

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response=report,
    )

    assert result["final_response"] == report
    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    assert agent._handle_max_iterations_called is False


def test_pending_pre_verify_response_is_preserved_on_budget_exhaustion(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "budget exhausted but complete"

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="budget_exhausted",
        pending_verification_response=report,
    )

    assert result["final_response"] == report
    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    assert agent._handle_max_iterations_called is False


def test_empty_pending_verification_response_uses_summary_fallback(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="",
    )

    assert result["final_response"] == "summary from extra call"
    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    assert agent._handle_max_iterations_called is True


def test_short_generated_summary_keeps_abnormal_turn_explainer(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(completion_explainer=True)
    agent._handle_max_iterations = lambda *_args: "The"

    result = _finalize(agent, final_response=None, exit_reason="unknown")

    assert result["final_response"] == "The\n\niteration-limit explanation"


def test_short_preserved_verification_response_is_not_rewritten(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(completion_explainer=True)

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="The",
    )

    assert result["final_response"] == "The"


def test_text_response_exit_not_rewritten_at_iteration_limit(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(budget_remaining=5)
    exit_reason = "text_response(finish_reason=stop)"

    result = _finalize(
        agent,
        final_response="normal answer",
        exit_reason=exit_reason,
        api_call_count=59,
    )

    assert result["turn_exit_reason"] == exit_reason
    assert agent._handle_max_iterations_called is False


@pytest.mark.parametrize(
    "exit_reason",
    [
        "error_near_max_iterations(boom)",
        "guardrail_halt",
        "partial_stream_recovery",
        "fallback_prior_turn_content",
        "empty_response_exhausted",
    ],
)
def test_unrelated_non_success_response_is_not_reclassified(monkeypatch, exit_reason):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response="diagnostic or partial content",
        exit_reason=exit_reason,
    )

    assert result["turn_exit_reason"] == exit_reason
    assert result["completed"] is False
    assert agent._handle_max_iterations_called is False


@pytest.mark.parametrize(
    ("exit_reason", "interrupted", "failed"),
    [
        ("interrupted_by_user", True, False),
        ("all_retries_exhausted_no_response", False, False),
        ("provider_failure", False, True),
    ],
)
def test_pending_response_does_not_mask_later_terminal_exit(
    monkeypatch, exit_reason, interrupted, failed
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=interrupted,
        failed=failed,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response="stale premature report",
    )

    assert result["final_response"] is None
    assert result["turn_exit_reason"] == exit_reason
    assert result["completed"] is False
    assert agent._handle_max_iterations_called is False


def test_pending_response_records_kanban_timeout(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db._record_task_failure", record)
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="composed report",
    )

    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    record.assert_called_once_with(
        conn,
        "task-123",
        error=(
            "Iteration budget exhausted (60/60) — task could not complete "
            "within the allowed iterations"
        ),
        outcome="timed_out",
        release_claim=True,
        end_run=True,
        event_payload_extra={"budget_used": 60, "budget_max": 60},
    )


def test_published_pending_candidate_is_not_duplicated_by_finalizer(monkeypatch):
    """When budget exhaustion preserves a verification candidate that is
    already the tail assistant message, the finalizer must NOT append a
    duplicate. The content-comparison guard prevents this. (#65919 §7)
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "the composed report"

    result = finalize_turn(
        agent,
        final_response=report,
        api_call_count=60,
        interrupted=False,
        failed=False,
        # The candidate is already in messages as the tail assistant.
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": report},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
        _pending_verification_response=report,
    )

    # The tail assistant already matches final_response — no duplicate appended.
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant"]
    # Persisted messages should also have no duplicate.
    assert agent.persisted_messages is not None
    persisted_roles = [m["role"] for m in agent.persisted_messages]
    assert persisted_roles == ["user", "assistant"]


def test_terminal_verification_failure_is_persisted_as_one_correction(monkeypatch):
    """When verification fails terminally (nudge present but budget exhausted),
    the finalizer drops the synthetic nudge and the assistant candidate
    persists as a single correction. No duplicate assistant appended. (#65919 §7)
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "terminal failure correction"

    result = finalize_turn(
        agent,
        final_response=report,
        api_call_count=60,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": report},
            # Synthetic nudge — should be dropped by _drop_verification_continuation_scaffolding.
            {"role": "user", "content": "[System: run tests]", "_verification_stop_synthetic": True},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
        _pending_verification_response=report,
    )

    # The nudge is dropped; the assistant candidate is the tail and matches
    # final_response, so no duplicate is appended.
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant"]
    # The nudge is gone from persisted messages too.
    assert agent.persisted_messages is not None
    persisted_contents = [m.get("content") for m in agent.persisted_messages]
    assert "[System: run tests]" not in persisted_contents
    assert report in persisted_contents
