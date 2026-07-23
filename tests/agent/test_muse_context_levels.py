"""Behavioral coverage for MUSE's two-level context compression policy."""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=8000):
        return ContextCompressor(
            model="test-model",
            threshold_percent=0.75,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def test_level_one_pruning_avoids_level_two_summary_when_it_restores_budget() -> None:
    compressor = _compressor()
    messages = [
        {"role": "user", "content": f"question {idx}"}
        if idx % 2 == 0
        else {"role": "assistant", "content": "x" * 800}
        for idx in range(8)
    ]
    projected = [dict(message) for message in messages]
    projected[1]["content"] = "[pruned]"

    with patch.object(
        compressor, "_prune_old_tool_results", return_value=(projected, 1)
    ), patch.object(
        compressor,
        "_generate_summary",
        side_effect=AssertionError("Level 2 should not run"),
    ):
        result = compressor.compress(messages, current_tokens=6100)

    assert result == projected
    assert compressor._last_compression_level == 1
    assert compressor._last_level1_projection == projected
    assert compressor._last_compression_made_progress is True
