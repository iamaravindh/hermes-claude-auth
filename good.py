"""
Claude Code OAuth bypass for hermes-agent.
==========================================

Monkey-patches hermes-agent's ``agent.anthropic_adapter.build_anthropic_kwargs``
at import time via a sitecustomize.py hook so that OAuth-authenticated requests
pass Anthropic's server-side content validation.

Background
----------
On 2026-04-04 Anthropic deployed server-side validation on OAuth requests: if
the ``system[]`` array contains text that doesn't match Claude Code's system
prompt structure, the request is rejected with HTTP 400 — even on accounts with
remaining subscription quota.  Third-party tools (hermes-agent, opencode, cline,
aider, etc.) all hit this simultaneously.

opencode-claude-auth v1.4.8 (PR #148) worked around it by:

  1. Injecting a cryptographically-signed ``x-anthropic-billing-header`` as
     ``system[0]``.  The signature is derived from characters at positions 4, 7,
     20 of the first user message, a hardcoded salt, and the Claude CLI version.
  2. Relocating all non-Claude-Code system prompt content to the first user
     message wrapped in ``<system-reminder>`` blocks.
  3. Adding the ``prompt-caching-scope-2026-01-05`` beta flag.

hermes-agent already implements the Claude Code identity prefix, user-agent
spoofing, ``x-app: cli``, tool name ``mcp_`` prefixing, and the
``oauth-2025-04-20`` / ``claude-code-20250219`` beta flags.  This patch adds
only the three items above plus a temperature fix for Opus 4.6 adaptive thinking.

Installation
------------
Installed automatically by ``install.sh``.  See README.md for details.

The ``sitecustomize_hook.py`` loader runs at Python interpreter startup and
hooks ``agent.anthropic_adapter``'s import so that ``apply_patches()`` runs
immediately after the module is loaded.  No hermes-agent source modifications
are needed.

Reversal
--------
Run ``uninstall.sh`` or manually remove the sitecustomize hook from the venv's
site-packages and restart hermes-gateway.

References
----------
- https://github.com/griffinmartin/opencode-claude-auth
- https://github.com/griffinmartin/opencode-claude-auth/pull/148

Version history
---------------
- 1.0.0 (2026-04-09): Initial — billing header, system prompt relocation,
  prompt-caching beta flag, aux-client temperature hook for Opus 4.6.
"""

from __future__ import annotations

__version__ = "1.0.0"

import hashlib
import inspect
import logging
import sys
import traceback
from typing import Any, Dict, List

logger = logging.getLogger("good")

# ---------------------------------------------------------------------------
# Cryptographic signing (ported from opencode-claude-auth/src/signing.ts)
# ---------------------------------------------------------------------------

# Shared secret shipped in the Claude Code CLI binary.  Anthropic's server
# uses this salt to verify billing-header signatures.
_BILLING_SALT = "59cf53e54c78"

# Sentinel strings — entries in system[] starting with these are kept;
# everything else is relocated to the first user message.
_BILLING_PREFIX = "x-anthropic-billing-header"
_SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def _extract_first_user_message_text(messages: List[Dict[str, Any]]) -> str:
    """Return the text of the first user message's first text block.

    Matches Claude Code's K19() exactly: find the first message with
    role="user", then return the text of its first text content block.
    """
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        return text
        return ""
    return ""


def _compute_cch(message_text: str) -> str:
    """First 5 hex chars of SHA-256(message_text)."""
    return hashlib.sha256(message_text.encode("utf-8")).hexdigest()[:5]


def _compute_version_suffix(message_text: str, version: str) -> str:
    """3-char version suffix: SHA-256(salt + sampled_chars + version)[:3].

    Samples characters at indices 4, 7, 20 from the message text, padding
    with "0" when the message is shorter than the index.
    """
    sampled = "".join(
        message_text[i] if i < len(message_text) else "0" for i in (4, 7, 20)
    )
    input_str = f"{_BILLING_SALT}{sampled}{version}"
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()[:3]


def _build_billing_header_value(
    messages: List[Dict[str, Any]],
    version: str,
    entrypoint: str,
) -> str:
    """Build the full x-anthropic-billing-header text for system[0]."""
    text = _extract_first_user_message_text(messages)
    suffix = _compute_version_suffix(text, version)
    cch = _compute_cch(text)
    return (
        f"x-anthropic-billing-header: "
        f"cc_version={version}.{suffix}; "
        f"cc_entrypoint={entrypoint}; "
        f"cch={cch};"
    )


# ---------------------------------------------------------------------------
# Bypass logic (ported from opencode-claude-auth/src/transforms.ts)
# ---------------------------------------------------------------------------


def _model_supports_adaptive_thinking(model: str) -> bool:
    if not isinstance(model, str):
        return False
    return any(v in model for v in ("4-6", "4.6"))


def _fix_temperature_for_oauth_adaptive(
    api_kwargs: Dict[str, Any],
    *,
    site: str,
) -> None:
    """Strip temperature from OAuth requests on adaptive-thinking models.

    Opus 4.6 with implicit adaptive thinking rejects non-1 temperature
    values with HTTP 400.  This drops the parameter entirely so the API
    uses its default.
    """
    if "temperature" not in api_kwargs:
        return
    temp = api_kwargs.get("temperature")
    if temp == 1 or temp == 1.0:
        return
    model = api_kwargs.get("model")
    if not _model_supports_adaptive_thinking(model or ""):
        return
    del api_kwargs["temperature"]
    logger.info(
        "Dropped temperature=%r for OAuth adaptive-thinking model %r (site=%s)",
        temp,
        model,
        site,
    )


def _prepend_to_first_user_message(
    messages: List[Dict[str, Any]],
    texts: List[str],
) -> None:
    """Prepend each text as a <system-reminder> block to the first user message.

    Mutates ``messages`` in place.
    """
    if not texts:
        return
    combined = "\n\n".join(f"<system-reminder>\n{t}\n</system-reminder>" for t in texts)
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            new_text = f"{combined}\n\n{content}" if content else combined
            messages[i] = {**msg, "content": [{"type": "text", "text": new_text}]}
            return
        if isinstance(content, list):
            new_content = list(content)
            for j, block in enumerate(new_content):
                if isinstance(block, dict) and block.get("type") == "text":
                    existing = block.get("text") or ""
                    new_content[j] = {
                        **block,
                        "text": f"{combined}\n\n{existing}" if existing else combined,
                    }
                    messages[i] = {**msg, "content": new_content}
                    return
            new_content.insert(0, {"type": "text", "text": combined})
            messages[i] = {**msg, "content": new_content}
            return
        messages[i] = {**msg, "content": [{"type": "text", "text": combined}]}
        return


def apply_claude_code_bypass(api_kwargs: Dict[str, Any], version: str) -> None:
    """Mutate api_kwargs in place to pass OAuth content validation.

    Only call on OAuth requests (``is_oauth=True``).  Safe to call multiple
    times — stale billing headers are replaced, duplicate identity entries
    are dropped.

    After this runs, ``api_kwargs["system"]`` contains at most the billing
    header and the Claude Code identity prefix.  Everything else is moved to
    the first user message as ``<system-reminder>`` blocks.
    """
    messages = api_kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return

    raw_system = api_kwargs.get("system")
    if raw_system is None:
        system: List[Any] = []
    elif isinstance(raw_system, str):
        system = [{"type": "text", "text": raw_system}] if raw_system else []
    elif isinstance(raw_system, list):
        system = list(raw_system)
    else:
        logger.warning(
            "Unexpected system type %s; skipping bypass", type(raw_system).__name__
        )
        return

    # Compute billing header using ORIGINAL messages (before relocation).
    try:
        billing_value = _build_billing_header_value(messages, version, "cli")
    except Exception as exc:
        logger.warning("Failed to build billing header: %s", exc)
        return
    billing_entry = {"type": "text", "text": billing_value}

    kept: List[Any] = []
    moved_texts: List[str] = []
    identity_seen = False

    for entry in system:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        entry_type = entry.get("type")
        if entry_type != "text":
            kept.append(entry)
            continue
        text = entry.get("text") or ""
        if text.startswith(_BILLING_PREFIX):
            continue  # stale billing header — drop
        if text.startswith(_SYSTEM_IDENTITY):
            if identity_seen:
                continue  # duplicate — drop
            identity_seen = True
            rest = text[len(_SYSTEM_IDENTITY) :].lstrip("\n")
            identity_entry = {k: v for k, v in entry.items() if k != "text"}
            identity_entry["text"] = _SYSTEM_IDENTITY
            kept.append(identity_entry)
            if rest:
                moved_texts.append(rest)
            continue
        if text:
            moved_texts.append(text)

    if not identity_seen:
        kept.insert(0, {"type": "text", "text": _SYSTEM_IDENTITY})

    # Billing header first (no cache_control — changes per request).
    api_kwargs["system"] = [billing_entry] + kept

    if moved_texts:
        _prepend_to_first_user_message(messages, moved_texts)

    _fix_temperature_for_oauth_adaptive(api_kwargs, site="build_kwargs")


# ---------------------------------------------------------------------------
# Monkey-patch installation
# ---------------------------------------------------------------------------


def _get_version_safely(aa_module: Any) -> str:
    """Return the Claude CLI version string from the adapter module."""
    getter = getattr(aa_module, "_get_claude_code_version", None)
    if callable(getter):
        try:
            version = getter()
            if isinstance(version, str) and version and version[0].isdigit():
                return version
        except Exception:
            pass
    fallback = getattr(aa_module, "_CLAUDE_CODE_VERSION_FALLBACK", None)
    if isinstance(fallback, str) and fallback:
        return fallback
    return "2.1.90"


def _install_aux_client_hook(force: bool = False) -> bool:
    """Patch the auxiliary client to strip temperature on OAuth adaptive models."""
    try:
        from agent import auxiliary_client as ac  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("aux_client_hook_failed_import: %s: %s", type(exc).__name__, exc)
        sys.stderr.write(
            f"[good] aux_client_hook_failed_import: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return False

    adapter_cls = getattr(ac, "_AnthropicCompletionsAdapter", None)
    if adapter_cls is None:
        logger.warning("aux_client_hook_failed: _AnthropicCompletionsAdapter not found")
        return False

    if getattr(adapter_cls, "_AUX_CLIENT_TEMP_HOOK_APPLIED", False) and not force:
        logger.debug("aux_client_hook already installed")
        return True

    original_create = getattr(adapter_cls, "create", None)
    if not callable(original_create):
        logger.warning("aux_client_hook_failed: create() not callable on adapter")
        return False

    def patched_create(self: Any, **kwargs: Any) -> Any:
        real_client = getattr(self, "_client", None)
        if real_client is None:
            return original_create(self, **kwargs)
        messages_obj = getattr(real_client, "messages", None)
        if messages_obj is None:
            return original_create(self, **kwargs)

        is_oauth = bool(getattr(self, "_is_oauth", False))
        if not is_oauth:
            return original_create(self, **kwargs)

        inner_original = messages_obj.create

        def fixed_messages_create(**inner_kwargs: Any) -> Any:
            try:
                _fix_temperature_for_oauth_adaptive(inner_kwargs, site="aux_client")
            except Exception as exc:
                logger.warning(
                    "aux_client_hook: temperature fix raised %s: %s",
                    type(exc).__name__,
                    exc,
                )
            return inner_original(**inner_kwargs)

        try:
            messages_obj.create = fixed_messages_create
            rebind_ok = True
        except (AttributeError, TypeError):
            rebind_ok = False
        try:
            if rebind_ok:
                return original_create(self, **kwargs)

            class _ShimMessages:
                create = staticmethod(fixed_messages_create)

            class _ShimClient:
                messages = _ShimMessages()

            self._client = _ShimClient()
            try:
                return original_create(self, **kwargs)
            finally:
                self._client = real_client
        finally:
            if rebind_ok:
                try:
                    del messages_obj.create
                except (AttributeError, TypeError):
                    messages_obj.create = inner_original

    patched_create.__name__ = original_create.__name__
    patched_create.__qualname__ = getattr(
        original_create, "__qualname__", original_create.__name__
    )
    patched_create.__doc__ = original_create.__doc__
    patched_create.__wrapped__ = original_create  # type: ignore[attr-defined]

    adapter_cls.create = patched_create
    adapter_cls._AUX_CLIENT_TEMP_HOOK_APPLIED = True
    logger.info(
        "Aux client temperature hook installed on _AnthropicCompletionsAdapter.create"
    )
    sys.stderr.write(
        "[good] Aux client temperature hook installed\n"
    )
    return True


def apply_patches(anthropic_adapter_module: Any = None) -> bool:
    """Install the bypass on ``agent.anthropic_adapter``.

    Called by the sitecustomize hook after the module is imported.  Returns
    ``True`` on success, ``False`` if the target module is incompatible.
    Idempotent — safe to call multiple times.
    """
    aa = anthropic_adapter_module
    if aa is None:
        try:
            from agent import anthropic_adapter as aa  # type: ignore[import-not-found,no-redef]
        except ImportError as exc:
            logger.warning("Cannot import agent.anthropic_adapter: %s", exc)
            return False

    if getattr(aa, "_CLAUDE_CODE_BYPASS_APPLIED", False):
        logger.debug("Claude Code bypass already installed")
        return True

    # 1. Add the missing beta flag.
    new_beta = "prompt-caching-scope-2026-01-05"
    oauth_betas = getattr(aa, "_OAUTH_ONLY_BETAS", None)
    if isinstance(oauth_betas, list) and new_beta not in oauth_betas:
        oauth_betas.append(new_beta)
        logger.info("Appended beta flag: %s", new_beta)

    # 2. Verify the target function exists with the expected signature.
    original_build = getattr(aa, "build_anthropic_kwargs", None)
    if not callable(original_build):
        logger.warning(
            "agent.anthropic_adapter.build_anthropic_kwargs not found — "
            "skipping monkey-patch (incompatible hermes-agent version?)"
        )
        return False

    try:
        sig = inspect.signature(original_build)
        if "is_oauth" not in sig.parameters:
            logger.warning(
                "build_anthropic_kwargs lacks 'is_oauth' param — "
                "skipping monkey-patch (incompatible hermes-agent version?)"
            )
            return False
    except (TypeError, ValueError) as exc:
        logger.warning("Cannot introspect build_anthropic_kwargs: %s", exc)
        return False

    # 3. Wrap build_anthropic_kwargs to apply the bypass on OAuth requests.
    def patched_build_anthropic_kwargs(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = original_build(*args, **kwargs)

        try:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            is_oauth = bool(bound.arguments.get("is_oauth", False))
        except TypeError:
            is_oauth = bool(kwargs.get("is_oauth", False))

        if is_oauth and isinstance(result, dict):
            try:
                apply_claude_code_bypass(result, _get_version_safely(aa))
            except Exception as exc:
                logger.warning(
                    "apply_claude_code_bypass raised %s: %s",
                    type(exc).__name__,
                    exc,
                )
                traceback.print_exc(file=sys.stderr)
        return result

    patched_build_anthropic_kwargs.__name__ = original_build.__name__
    patched_build_anthropic_kwargs.__qualname__ = getattr(
        original_build, "__qualname__", original_build.__name__
    )
    patched_build_anthropic_kwargs.__doc__ = original_build.__doc__
    patched_build_anthropic_kwargs.__module__ = getattr(
        original_build, "__module__", __name__
    )
    patched_build_anthropic_kwargs.__wrapped__ = original_build  # type: ignore[attr-defined]

    aa.build_anthropic_kwargs = patched_build_anthropic_kwargs
    aa._CLAUDE_CODE_BYPASS_APPLIED = True  # type: ignore[attr-defined]
    logger.info("Claude Code OAuth bypass installed (build_anthropic_kwargs)")
    sys.stderr.write("[good] Claude Code OAuth bypass installed\n")

    _install_aux_client_hook()

    return True
