"""
title: Context Manager
author: Joshua Dixon
author_url: https://github.com/Joshua-Dixon-AI
version: 0.1.0
license: MIT
description: >
    Deterministic prompt-context compaction for Open WebUI. Preserves system
    instructions and recent turns, compacts stale history into a short summary,
    and drops raw older messages before the request reaches the model.
repository: https://github.com/Joshua-Dixon-AI/openwebui
"""

import copy
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type"):
                    parts.append(f"[{item.get('type')}]")
                else:
                    parts.append(json.dumps(item, default=str, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, default=str, ensure_ascii=False)
    return str(content)


def _message_text(message: dict) -> str:
    parts = []
    content = _content_to_text(message.get("content"))
    if content:
        parts.append(content)
    for key in ("tool_calls", "name"):
        if message.get(key):
            parts.append(json.dumps(message.get(key), default=str, ensure_ascii=False))
    return "\n".join(parts)


def _estimate_message_tokens(message: dict) -> int:
    return 4 + _approx_tokens(message.get("role", "")) + _approx_tokens(_message_text(message))


def _estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(_estimate_message_tokens(m) for m in messages)


def _first_line(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _usage_bar(tokens: int, budget: int, width: int = 20) -> str:
    if budget <= 0:
        percent = 100
    else:
        percent = round((tokens / budget) * 100)
    filled = min(width, max(0, round((min(percent, 100) / 100) * width)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {percent}%"


def _head_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars < 120:
        return text[:max_chars]
    head = int(max_chars * 0.65)
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        text[:head]
        + f"\n\n[... {omitted} characters removed by Context Manager ...]\n\n"
        + text[-tail:]
    )


def _trim_content(content: Any, max_chars: int) -> Any:
    if isinstance(content, str):
        return _head_tail(content, max_chars)
    if isinstance(content, list):
        remaining = max_chars
        trimmed = []
        for item in content:
            item_copy = copy.deepcopy(item)
            if isinstance(item_copy, dict) and isinstance(item_copy.get("text"), str):
                text = item_copy["text"]
                item_copy["text"] = _head_tail(text, max(0, remaining))
                remaining -= len(item_copy["text"])
            elif isinstance(item_copy, str):
                item_copy = _head_tail(item_copy, max(0, remaining))
                remaining -= len(item_copy)
            trimmed.append(item_copy)
        return trimmed
    if content is None:
        return content
    text = _content_to_text(content)
    return _head_tail(text, max_chars)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Filter priority. Lower values run earlier.")
        enabled: bool = Field(default=True, description="Enable context pruning.")
        prompt_token_budget: int = Field(
            default=24000,
            description="Approximate maximum input tokens to send to the model.",
        )
        always_keep_recent_messages: int = Field(
            default=12,
            description="Number of newest non-tool chat messages to preserve before dropping older history.",
        )
        always_keep_system_messages: bool = Field(
            default=True,
            description="Always preserve system/developer messages.",
        )
        max_old_message_chars: int = Field(
            default=6000,
            description="Trim older individual messages to this many characters before dropping history.",
        )
        prefer_drop_tool_messages: bool = Field(
            default=True,
            description="Drop stale tool/function messages before ordinary chat history.",
        )
        pinned_regex: str = Field(
            default="",
            description="Optional regex. Matching messages are preserved when possible.",
        )
        add_pruning_notice: bool = Field(
            default=True,
            description="Insert a compact system summary when messages are compacted or dropped.",
        )
        compaction_detail_messages: int = Field(
            default=8,
            description="Maximum number of compacted older messages to describe in the retained summary.",
        )
        compaction_snippet_chars: int = Field(
            default=220,
            description="Maximum characters per compacted message snippet.",
        )
        show_usage_status: bool = Field(
            default=True,
            description="Show a visible context usage status before each model call.",
        )
        debug_events: bool = Field(default=False, description="Show pruning status events in chat.")

    def __init__(self):
        self.valves = self.Valves()

    async def _emit(self, emitter, description: str, done: bool = False):
        if emitter and self.valves.debug_events:
            await emitter({"type": "status", "data": {"description": description, "done": done}})

    async def _emit_usage(self, emitter, tokens: int, budget: int, label: str = "Context"):
        if emitter and self.valves.show_usage_status:
            await emitter({
                "type": "status",
                "data": {
                    "description": f"{label}: {_usage_bar(tokens, budget)} (~{tokens}/{budget} tokens)",
                    "done": True,
                },
            })

    def _pinned_indexes(self, messages: list[dict]) -> set[int]:
        pattern = (self.valves.pinned_regex or "").strip()
        if not pattern:
            return set()
        try:
            rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error:
            return set()
        return {i for i, message in enumerate(messages) if rx.search(_message_text(message))}

    @staticmethod
    def _active_tool_indexes(messages: list[dict]) -> set[int]:
        """Preserve tool results only when they are the active tail of the prompt."""
        tool_roles = {"tool", "function"}
        keep: set[int] = set()
        i = len(messages) - 1
        while i >= 0 and messages[i].get("role") in tool_roles:
            keep.add(i)
            i -= 1
        if keep and i >= 0 and messages[i].get("role") == "assistant":
            keep.add(i)
        return keep

    def _compaction_message(
        self,
        original_count: int,
        dropped: list[dict],
        trimmed_count: int,
        original_tokens: int,
        final_tokens_without_summary: int,
        detail_limit_override: Optional[int] = None,
    ) -> dict:
        role_counts: dict[str, int] = {}
        for message in dropped:
            role = message.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        role_summary = ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items()))
        if not role_summary:
            role_summary = "none"
        if detail_limit_override is None:
            detail_limit = max(0, int(self.valves.compaction_detail_messages or 0))
        else:
            detail_limit = max(0, int(detail_limit_override))
        snippet_chars = max(40, int(self.valves.compaction_snippet_chars or 0))
        details = []
        for message in dropped[:detail_limit]:
            role = message.get("role", "unknown")
            snippet = _first_line(_message_text(message), snippet_chars)
            if snippet:
                details.append(f"- {role}: {snippet}")
        remaining = len(dropped) - len(details)
        if remaining > 0 and detail_limit > 0:
            details.append(f"- ... {remaining} additional older message(s) compacted.")

        if detail_limit == 0:
            content_parts = [
                "Context Manager compacted older chat history before model invocation.",
                f"Original messages: {original_count}; raw older messages removed: {len(dropped)} ({role_summary}); older messages trimmed: {trimmed_count}.",
                "Preserved: system/developer instructions, recent turns, pinned matches, and active trailing tool results.",
                "Treat this as lossy context.",
            ]
        else:
            content_parts = [
                "Context Manager compacted older chat history before model invocation.",
                f"Original messages: {original_count}.",
                f"Original estimated prompt: ~{original_tokens} tokens.",
                f"Compacted prompt before this summary: ~{final_tokens_without_summary} tokens.",
                f"Raw older messages removed: {len(dropped)} ({role_summary}).",
                f"Oversized older messages trimmed before removal: {trimmed_count}.",
                "Preserved: system/developer instructions, recent conversational turns, pinned matches, and active trailing tool results.",
            ]
        if details:
            content_parts.append("Compacted older-message trace:")
            content_parts.extend(details)
        if detail_limit > 0:
            content_parts.append(
                "Treat this summary as lossy context. Ask the user before relying on details that may have been removed."
            )
        content = "\n".join(content_parts)
        return {"role": "system", "content": content}

    def _insert_notice(self, messages: list[dict], notice: dict) -> list[dict]:
        insert_at = 0
        for i, message in enumerate(messages):
            if message.get("role") in {"system", "developer"}:
                insert_at = i + 1
        return messages[:insert_at] + [notice] + messages[insert_at:]

    async def inlet(self, body: dict, __event_emitter__=None, __user__: Optional[dict] = None) -> dict:
        """
        Prune chat history before the model sees it.

        This filter is intentionally deterministic. It does not summarize with
        another model call, so it adds no hidden LLM cost and does not create a
        second place where sensitive chat content is sent.
        """
        if not self.valves.enabled:
            return body

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body

        budget = max(1000, int(self.valves.prompt_token_budget or 0))
        original_messages = copy.deepcopy(messages)
        working = [copy.deepcopy(m) for m in messages if isinstance(m, dict)]
        if len(working) != len(messages):
            body["messages"] = working

        original_tokens = _estimate_messages_tokens(working)
        if original_tokens <= budget:
            await self._emit_usage(__event_emitter__, original_tokens, budget)
            await self._emit(
                __event_emitter__,
                f"Context Manager: prompt fits budget (~{original_tokens}/{budget} tokens).",
                done=True,
            )
            return body

        total = len(working)
        recent_n = max(1, int(self.valves.always_keep_recent_messages or 1))
        recent_chat_indexes = [
            i for i, message in enumerate(working)
            if message.get("role") not in {"tool", "function"}
        ][-recent_n:]
        keep_indexes = set(recent_chat_indexes)
        keep_indexes.update(self._active_tool_indexes(working))
        if self.valves.always_keep_system_messages:
            keep_indexes.update(
                i for i, message in enumerate(working)
                if message.get("role") in {"system", "developer"}
            )
        keep_indexes.update(self._pinned_indexes(working))

        trimmed_count = 0
        max_old_chars = int(self.valves.max_old_message_chars or 0)
        if max_old_chars > 0:
            for i, message in enumerate(working):
                if i in keep_indexes:
                    continue
                text = _message_text(message)
                if len(text) > max_old_chars:
                    message["content"] = _trim_content(message.get("content"), max_old_chars)
                    trimmed_count += 1

        dropped: list[dict] = []
        if _estimate_messages_tokens(working) > budget:
            candidate_indexes = [i for i in range(total) if i not in keep_indexes]
            if self.valves.prefer_drop_tool_messages:
                candidate_indexes.sort(
                    key=lambda i: (
                        0 if working[i].get("role") in {"tool", "function"} else 1,
                        i,
                    )
                )
            else:
                candidate_indexes.sort()

            drop_indexes: set[int] = set()
            for i in candidate_indexes:
                if _estimate_messages_tokens([m for j, m in enumerate(working) if j not in drop_indexes]) <= budget:
                    break
                drop_indexes.add(i)
                dropped.append(working[i])

            working = [m for i, m in enumerate(working) if i not in drop_indexes]

        final_tokens_without_summary = _estimate_messages_tokens(working)
        if self.valves.add_pruning_notice and (dropped or trimmed_count):
            max_details = max(0, int(self.valves.compaction_detail_messages or 0))
            for detail_limit in range(max_details, -1, -1):
                notice = self._compaction_message(
                    len(original_messages),
                    dropped,
                    trimmed_count,
                    original_tokens,
                    final_tokens_without_summary,
                    detail_limit_override=detail_limit,
                )
                candidate = self._insert_notice(working, notice)
                if _estimate_messages_tokens(candidate) <= budget or detail_limit == 0:
                    working = candidate
                    break

        final_tokens = _estimate_messages_tokens(working)
        body["messages"] = working
        await self._emit_usage(__event_emitter__, final_tokens, budget, label="Context after compaction")
        await self._emit(
            __event_emitter__,
            (
                "Context Manager: reduced prompt from "
                f"~{original_tokens} to ~{final_tokens} tokens; "
                f"dropped {len(dropped)} message(s), trimmed {trimmed_count}."
            ),
            done=True,
        )
        return body
