"""Tool calls streamed while generated add up to the arguments of the
complete call, piece by piece."""

import json
from collections.abc import Generator

from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallDeltaResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.vendor.dsml_encoding import DSML_TOKEN, parse_dsml_output
from exo.worker.runner.llm_inference.dsml_tool_call_stream import (
    DsmlToolCallStream,
    stable_tool_calls,
)
from exo.worker.runner.llm_inference.model_output_parsers import parse_deepseek_v4

_START = f"<{DSML_TOKEN}tool_calls>"
_END = f"</{DSML_TOKEN}tool_calls>"


def _invoke(name: str, *params: tuple[str, str, bool]) -> str:
    body = "".join(
        f'<{DSML_TOKEN}parameter name="{key}" string="{"true" if is_string else "false"}">'
        f"{value}</{DSML_TOKEN}parameter>\n"
        for key, value, is_string in params
    )
    return f'<{DSML_TOKEN}invoke name="{name}">\n{body}</{DSML_TOKEN}invoke>\n'


_BLOCK = (
    _START
    + "\n"
    + _invoke(
        "write",
        ("path", "/workspace/slides/s5.py", True),
        ("content", 'print("年末商戦 </p>")\n\tx = "\\u3042" < 3\n', True),
    )
    + _invoke("exec", ("command", "ls -la", True), ("timeout", "30", False))
    + _invoke("getTime")
    + _END
)


def _streamed(text: str, step: int) -> list[tuple[str, str, str]]:
    """(id, name, arguments) of each call, from deltas fed ``step`` chars at
    a time."""
    stream = DsmlToolCallStream()
    calls: dict[int, list[str]] = {}
    for end in range(step, len(text) + step, step):
        for index, call, is_new, added in stream.update(text[:end]):
            if is_new:
                calls[index] = [call.id, call.name, ""]
            calls[index][2] += added
    return [(c[0], c[1], c[2]) for _, c in sorted(calls.items())]


def test_streamed_arguments_add_up_to_the_parsed_ones() -> None:
    parsed = parse_dsml_output(_BLOCK)
    assert parsed is not None
    for step in (1, 2, 7, 50):
        streamed = _streamed(_BLOCK, step)
        assert [(name, arguments) for _, name, arguments in streamed] == [
            (item.name, item.arguments) for item in parsed
        ]


def test_a_value_is_streamed_before_its_parameter_closes() -> None:
    text = (
        _START
        + f'<{DSML_TOKEN}invoke name="write">'
        + f'<{DSML_TOKEN}parameter name="path" string="true">a.py</{DSML_TOKEN}parameter>'
        + f'<{DSML_TOKEN}parameter name="content" string="true">line 1\nline 2</'
    )
    # The trailing "</" may start the closing tag, so it is held back.
    assert stable_tool_calls(text) == [
        ("write", '{"path": "a.py", "content": "line 1\\nline 2')
    ]


def test_a_non_string_value_waits_for_its_closing_tag() -> None:
    text = (
        _START
        + f'<{DSML_TOKEN}invoke name="exec">'
        + f'<{DSML_TOKEN}parameter name="timeout" string="false">3'
    )
    assert stable_tool_calls(text) == [("exec", "{")]


def test_parser_streams_pieces_and_keeps_their_ids() -> None:
    def source() -> Generator[GenerationResponse | None]:
        for i in range(0, len(_BLOCK), 5):
            yield GenerationResponse(text=_BLOCK[i : i + 5], token=0, usage=None)
        yield GenerationResponse(text="", token=0, usage=None, finish_reason="stop")

    deltas: list[ToolCallDeltaResponse] = []
    final: list[ToolCallResponse] = []
    for response in parse_deepseek_v4(source()):
        if isinstance(response, ToolCallDeltaResponse):
            deltas.append(response)
        elif isinstance(response, ToolCallResponse):
            final.append(response)

    assert len(final) == 1
    tool_calls = final[0].tool_calls
    assert [d.name for d in deltas if d.name is not None] == [
        "write",
        "exec",
        "getTime",
    ]
    for index, item in enumerate(tool_calls):
        pieces = [d for d in deltas if d.index == index]
        assert {d.id for d in pieces} == {item.id}
        assert "".join(d.arguments for d in pieces) == item.arguments
    assert json.loads(tool_calls[1].arguments) == {"command": "ls -la", "timeout": 30}
