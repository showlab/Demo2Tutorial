from __future__ import annotations

from typing import Any


def chat_completion_create(client: Any, **kwargs: Any) -> Any:
    """Call Chat Completions with model-family compatible parameters."""
    normalized = _normalize_chat_kwargs(kwargs)
    try:
        return client.chat.completions.create(**normalized)
    except Exception as exc:
        retry = _normalize_from_error(normalized, str(exc))
        if retry == normalized:
            raise
        return client.chat.completions.create(**retry)


def _normalize_chat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    model = str(out.get("model") or "")
    if _uses_completion_token_param(model) and "max_tokens" in out:
        out["max_completion_tokens"] = out.pop("max_tokens")
    if _uses_default_temperature_only(model):
        out.pop("temperature", None)
    if _is_gpt5_reasoning_model(model) and "reasoning_effort" not in out:
        out["reasoning_effort"] = "none" if model.startswith("gpt-5.1") else "minimal"
    return out


def _normalize_from_error(kwargs: dict[str, Any], message: str) -> dict[str, Any]:
    out = dict(kwargs)
    if (
        "max_tokens" in message
        and "max_completion_tokens" in message
        and "max_tokens" in out
    ):
        out["max_completion_tokens"] = out.pop("max_tokens")
    if "temperature" in message and "Unsupported value" in message:
        out.pop("temperature", None)
    return out


def _uses_completion_token_param(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith("gpt-5") or normalized.startswith(("o1", "o3", "o4"))


def _uses_default_temperature_only(model: str) -> bool:
    return model.lower().startswith("gpt-5")


def _is_gpt5_reasoning_model(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith("gpt-5") and "chat" not in normalized
