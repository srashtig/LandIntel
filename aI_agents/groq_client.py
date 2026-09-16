"""Shared Groq client plumbing for the AI layer (:mod:`aI_agents.qa`,
:mod:`aI_agents.qa_agent`) — lazily constructs (and caches) the Groq client,
and wraps ``chat.completions.create`` with one retry on a transient
rate-limit error.

Extracted from the original notebook-era ``decision_app/chat.py`` (kept only
in the frozen pre-reorg snapshot): that file's structured preference
parsing (``parse_preference_overrides``) and custom-facts extraction
(``parse_custom_facts``), and its dependency on the old
``decision_engine``-style factor catalog, are notebook-only and not part of
this app's actual grounded-Q&A path — only the client/retry core below is.
"""

from __future__ import annotations

import time

from data_analysis_pipeline import config

_client = None
_client_checked = False

# gpt-oss models on Groq emit hidden "reasoning" tokens before the visible
# reply; left at the default effort these can be verbose enough to eat a
# small max-token budget (empty visible content, finish_reason "length")
# and to burn through Groq's free-tier tokens-per-minute limit within a
# couple of calls. "low" keeps calls fast, cheap, and reliable for these
# grounded/closed-vocabulary tasks — confirmed via a real call (~6 reasoning
# tokens vs. ~170-300 at the default effort).
_REASONING_EFFORT = "low"

# One retry on a transient 429 (tokens-per-minute) — the free/on-demand
# Groq tier used here has an 8000 TPM cap, easily hit by two chat calls in
# quick succession (confirmed during verification of this module).
_RATE_LIMIT_RETRY_SECONDS = 10.0


class ChatUnavailable(RuntimeError):
    """Raised when no Groq API key is configured, the ``groq`` package is
    not installed, or the call fails after a retry (e.g. a persistent rate
    limit). Callers should catch this specifically and show a clear message
    rather than crash."""


def _get_client():
    """Lazily construct (and cache) the Groq client, or ``None`` if no API
    key is configured anywhere (``GROQ_API_KEY`` env/``.env``, or the
    legacy key file — see :mod:`data_analysis_pipeline.config`) or the
    ``groq`` package is not installed."""

    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True

    if not config.GROQ_API_KEY:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def reset_client() -> None:
    """Force the next :func:`_get_client` call to re-check
    ``config.GROQ_API_KEY`` and construct a fresh client — call this after
    changing the key at runtime (e.g. via the sidebar), since
    ``_get_client`` otherwise caches its first result (including a cached
    ``None``, from before a key existed) for the life of the process."""

    global _client, _client_checked
    _client = None
    _client_checked = False


def _require_client():
    client = _get_client()
    if client is None:
        raise ChatUnavailable(
            "No Groq client available: either GROQ_API_KEY is not configured "
            "(set it in .env, see .env.example) or the 'groq' "
            "package is not installed (pip install groq)."
        )
    return client


def _create_completion(client, **kwargs):
    """``client.chat.completions.create`` with one retry on a transient
    rate-limit error or a malformed-output error, surfaced as
    :class:`ChatUnavailable` if it persists.

    The malformed-output case (Groq's ``json_validate_failed`` /
    ``tool_use_failed`` error codes) is a real, observed model quirk: in
    ``response_format={"type": "json_object"}`` mode (used throughout this
    app — there is no ``tools=`` param registered anywhere), the model can
    still occasionally emit output shaped like a native tool call instead
    of the requested plain JSON object, which Groq rejects outright rather
    than returning it as content. A retry (same prompt, same temperature)
    was confirmed to get a valid response back in practice, since this is
    model sampling variance, not a persistent problem with the request
    itself.

    Args:
        client: A Groq client, from :func:`_get_client`.
        **kwargs: Forwarded to ``chat.completions.create``.

    Returns:
        The Groq completion response.

    Raises:
        ChatUnavailable: If the call still fails after one retry.
    """

    from groq import BadRequestError, RateLimitError

    _MALFORMED_OUTPUT_CODES = {"json_validate_failed", "tool_use_failed"}

    def _is_malformed_output(error: BadRequestError) -> bool:
        body = error.body if isinstance(error.body, dict) else {}
        return body.get("error", {}).get("code") in _MALFORMED_OUTPUT_CODES

    try:
        return client.chat.completions.create(**kwargs)
    except RateLimitError as error:
        time.sleep(_RATE_LIMIT_RETRY_SECONDS)
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as retry_error:
            raise ChatUnavailable(
                f"Groq rate limit exceeded even after retrying: {retry_error}"
            ) from retry_error
    except BadRequestError as error:
        if not _is_malformed_output(error):
            raise
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as retry_error:
            if not _is_malformed_output(retry_error):
                raise
            raise ChatUnavailable(
                f"Groq returned malformed output even after retrying: {retry_error}"
            ) from retry_error
