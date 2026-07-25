"""Regression tests for the OAuth User-Agent header in anthropic_adapter.py.

Two DIFFERENT Anthropic endpoints impose distinct User-Agent requirements:

- Inference (``/v1/messages`` via build_anthropic_client) matches the current
  Claude Agent SDK fingerprint: ``claude-cli/... (external, sdk-cli)``.
- OAuth token endpoint (``/v1/oauth/token`` login exchange + refresh):
  Anthropic now RATE-LIMITS (HTTP 429) any UA whose prefix is ``claude-code/``
  (or ``Mozilla/``). Verified empirically against platform.claude.com:
  ``claude-code/2.1.200`` -> 429; ``axios/*`` / ``node`` -> 400 (reached code
  validation). The token endpoint must therefore use a non-``claude-code/`` UA
  (we send ``axios/*``, matching the real Claude Code CLI's exchange client).
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest


class TestOAuthUserAgentPrefix:
    """Inference uses the SDK CLI UA; the OAuth token endpoint must not."""

    def test_build_anthropic_client_oauth_ua(self):
        """OAuth inference must match the current Claude Agent SDK UA."""
        from agent.anthropic_adapter import build_anthropic_client

        mock_sdk = MagicMock()
        with patch("agent.anthropic_adapter._get_anthropic_sdk", return_value=mock_sdk):
            build_anthropic_client("sk-ant-oauth-abc123", "https://api.anthropic.com")

        # Inspect the kwargs passed to Anthropic()
        call_kwargs = mock_sdk.Anthropic.call_args[1]
        headers = call_kwargs.get("default_headers", {})
        ua = headers.get("user-agent", "") or headers.get("User-Agent", "")

        assert "claude-cli/" in ua, f"Expected claude-cli/ in UA, got: {ua}"
        assert "(external, sdk-cli)" in ua

    def test_no_legacy_claude_code_cli_ua_in_source(self):
        """Source must not send the obsolete claude-code/(external, cli) UA."""
        import inspect
        import agent.anthropic_adapter as mod

        source = inspect.getsource(mod)
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if (
                "claude-code/" in stripped
                and "(external, cli)" in stripped
                and ("User-Agent" in stripped or "user-agent" in stripped)
            ):
                pytest.fail(
                    f"Line {i}: legacy OAuth User-Agent still present: {stripped}"
                )

    def test_token_exchange_ua_not_throttled(self):
        """run_hermes_oauth_login_pure must NOT send a throttled token-endpoint UA.

        Anthropic 429s both ``claude-cli/`` and ``claude-code/`` UAs at the
        token endpoint. The login exchange must use the shared
        ``_OAUTH_TOKEN_USER_AGENT`` constant (a non-claude-code UA).
        """
        import inspect
        import agent.anthropic_adapter as mod

        try:
            source = inspect.getsource(mod.run_hermes_oauth_login_pure)
        except AttributeError:
            pytest.skip("run_hermes_oauth_login_pure not found")

        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.strip()
            if ("User-Agent" in stripped or "user-agent" in stripped) and (
                "claude-cli/" in stripped or "claude-code/" in stripped
            ):
                pytest.fail(
                    f"Line {i}: throttled UA in token-exchange header: {stripped}"
                )
        assert "_OAUTH_TOKEN_USER_AGENT" in source, (
            "run_hermes_oauth_login_pure should send the shared "
            "_OAUTH_TOKEN_USER_AGENT (non-claude-code) on the token endpoint"
        )
        assert not mod._OAUTH_TOKEN_USER_AGENT.startswith(("claude-code/", "claude-cli/")), (
            f"_OAUTH_TOKEN_USER_AGENT must not be a throttled prefix: "
            f"{mod._OAUTH_TOKEN_USER_AGENT!r}"
        )

    def test_token_refresh_ua_not_throttled(self):
        """refresh_anthropic_oauth_pure must NOT send a throttled token-endpoint UA."""
        import inspect
        import agent.anthropic_adapter as mod

        func = getattr(mod, "refresh_anthropic_oauth_pure", None)
        if func is None or not callable(func):
            pytest.skip("refresh_anthropic_oauth_pure not found")
        source = inspect.getsource(func)

        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.strip()
            if ("User-Agent" in stripped or "user-agent" in stripped) and (
                "claude-cli/" in stripped or "claude-code/" in stripped
            ):
                pytest.fail(
                    f"Line {i}: throttled UA in refresh header: {stripped}"
                )
        assert "_OAUTH_TOKEN_USER_AGENT" in source, (
            "refresh_anthropic_oauth_pure should send the shared "
            "_OAUTH_TOKEN_USER_AGENT (non-claude-code) on the token endpoint"
        )
