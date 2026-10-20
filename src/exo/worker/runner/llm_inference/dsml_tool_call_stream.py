"""Stream DSML tool calls while they are generated.

DeepSeek writes a tool call as DSML markup; the JSON arguments clients expect
are only known once the block ends, so a long tool call (a file written by a
`write` call) used to reach the client after minutes of silence. This builds,
from the markup generated so far, the part of each call's JSON arguments that
can no longer change: a prefix of what `parse_dsml_output` returns for the
finished block.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from exo.worker.engines.mlx.vendor.dsml_encoding import DSML_TOKEN

_INVOKE_START = re.compile(rf"<{re.escape(DSML_TOKEN)}invoke\s+name=\"([^\"]+)\">")
_INVOKE_END = f"</{DSML_TOKEN}invoke>"
_PARAM_START = re.compile(
    rf"<{re.escape(DSML_TOKEN)}parameter\s+name=\"([^\"]+)\"\s+string=\"(true|false)\">"
)
_PARAM_END = f"</{DSML_TOKEN}parameter>"


def _without_marker_prefix(text: str, marker: str) -> str:
    """``text`` without a suffix that could be the start of ``marker``."""
    for keep in range(max(0, len(text) - len(marker) + 1), len(text) + 1):
        if marker.startswith(text[keep:]):
            return text[:keep]
    return text


def _json_value(raw: str) -> str:
    # As parse_dsml_output: a non-string parameter is JSON, or else a string.
    try:
        value: Any = json.loads(raw)  # pyright: ignore[reportAny]
    except (json.JSONDecodeError, ValueError):
        value = raw
    return json.dumps(value)


def stable_tool_calls(text: str) -> list[tuple[str, str]]:
    """(name, arguments prefix) of every tool call started in ``text``.

    Each prefix only grows as more text arrives, and for a finished block
    equals the arguments parse_dsml_output returns (when every parameter
    name is used once).
    """
    calls: list[tuple[str, str]] = []
    pos = 0
    while (invoke := _INVOKE_START.search(text, pos)) is not None:
        body_end = text.find(_INVOKE_END, invoke.end())
        body = text[invoke.end() : body_end if body_end >= 0 else len(text)]
        parts: list[str] = []
        partial: str | None = None
        body_pos = 0
        while (param := _PARAM_START.search(body, body_pos)) is not None:
            key = json.dumps(param.group(1))
            is_string = param.group(2) == "true"
            value_end = body.find(_PARAM_END, param.end())
            if value_end < 0:
                if is_string:
                    value = _without_marker_prefix(body[param.end() :], _PARAM_END)
                    partial = f"{key}: {json.dumps(value)[:-1]}"
                break
            value = body[param.end() : value_end]
            parts.append(
                f"{key}: {json.dumps(value) if is_string else _json_value(value)}"
            )
            body_pos = value_end + len(_PARAM_END)
        arguments = "{" + ", ".join(parts)
        if partial is not None:
            arguments += (", " if parts else "") + partial
        elif body_end >= 0:
            arguments += "}"
        calls.append((invoke.group(1), arguments))
        if body_end < 0:
            break
        pos = body_end + len(_INVOKE_END)
    return calls


@dataclass
class StreamedToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class DsmlToolCallStream:
    """Turns the growing text of a DSML tool-call block into deltas."""

    calls: list[StreamedToolCall] = field(default_factory=list)

    def update(self, text: str) -> list[tuple[int, StreamedToolCall, bool, str]]:
        """(index, call, is_new, arguments delta) for what ``text`` added."""
        deltas: list[tuple[int, StreamedToolCall, bool, str]] = []
        for index, (name, arguments) in enumerate(stable_tool_calls(text)):
            if index == len(self.calls):
                call = StreamedToolCall(id=str(uuid4()), name=name, arguments="")
                self.calls.append(call)
                is_new = True
            else:
                call = self.calls[index]
                is_new = False
            # Malformed markup can shrink what parses (a parameter that an
            # early </invoke> cuts off); never take back what was sent.
            if not arguments.startswith(call.arguments):
                continue
            added = arguments[len(call.arguments) :]
            if added or is_new:
                call.arguments = arguments
                deltas.append((index, call, is_new, added))
        return deltas
