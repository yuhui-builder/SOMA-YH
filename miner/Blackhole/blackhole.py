"""Blackhole — CoT compression miner for SOMA SN114.

Two entry points:
  - R2 CLI: `python3 blackhole.py assemble` (stdin/stdout JSON, stateful)
  - R1 fn:  `main(task, compression_ratio) -> str` (for local eval)
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


logger = logging.getLogger(__name__)


_CHARS_PER_TOKEN_SAFETY = 1.4
_MAX_TOOL_RESULT_CHARS = 1500
_MAX_LATEST_EDIT_CHARS = 4000

_SECTION_BUDGET = {
    "task": 0.18,
    "state": 0.42,
    "reasoning": 0.18,
    "trace": 0.22,
}

_RE_USER_MESSAGE = re.compile(
    r'<message\s+role="user"\s*>(.*?)</message>', re.DOTALL | re.IGNORECASE
)
_RE_USER_TEXT_INNER = re.compile(
    r'<text>(.*?)</text>', re.DOTALL | re.IGNORECASE
)
_RE_THINKING = re.compile(
    r'<thinking>(.*?)</thinking>', re.DOTALL | re.IGNORECASE
)
_RE_TOOL_CALL = re.compile(
    r'<tool_call\s+name="([^"]+)"\s+id="([^"]+)"\s*>(.*?)</tool_call>',
    re.DOTALL | re.IGNORECASE,
)
_RE_TOOL_RESULT = re.compile(
    r'<tool_result\s+tool="([^"]+)"\s+tool_call_id="([^"]+)"\s*>(.*?)</tool_result>',
    re.DOTALL | re.IGNORECASE,
)
_RE_FINAL_TEXT = re.compile(
    r'<message\s+role="assistant"\s*>.*?<text>(.*?)</text>\s*</message>\s*$',
    re.DOTALL | re.IGNORECASE,
)

_RE_FILE_PATH = re.compile(
    r'(?:["\'`])?'
    r'(/?(?:[A-Za-z_][A-Za-z0-9_./-]*/)+[A-Za-z_][A-Za-z0-9_.-]*\.(?:py|js|ts|tsx|jsx|java|c|cpp|h|go|rs|rb|php|cs|json|yaml|yml|toml|cfg|ini|md|txt|html|css))'
    r'(?:["\'`])?'
)

_RE_PY_SYMBOL = re.compile(
    r'\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)'
)

_RE_ERROR_INDICATOR = re.compile(
    r'^(?:Traceback|.*?Error|.*?Exception|FAIL(?:URE|ED)?:?|FAILED|'
    r'AssertionError|ImportError|ModuleNotFoundError|SyntaxError|TypeError|'
    r'ValueError|AttributeError|KeyError|IndexError|RuntimeError|fatal:)\b',
    re.IGNORECASE | re.MULTILINE,
)


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode_ordinary(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if _ENCODER is None:
        return text[: max_tokens * 4]
    try:
        ids = _ENCODER.encode_ordinary(text)
        if len(ids) <= max_tokens:
            return text
        return _ENCODER.decode(ids[:max_tokens])
    except Exception:
        return text[: max_tokens * 4]


@dataclass
class _ToolCall:
    name: str
    call_id: str
    body: str
    position: int


@dataclass
class _ToolResult:
    tool: str
    call_id: str
    body: str
    position: int


@dataclass
class _ParsedTranscript:
    user_message: str
    thinkings: list[tuple[int, str]] = field(default_factory=list)
    tool_calls: list[_ToolCall] = field(default_factory=list)
    tool_results: list[_ToolResult] = field(default_factory=list)
    final_assistant_text: str = ""


def _parse(task: str) -> _ParsedTranscript:
    out = _ParsedTranscript(user_message="")

    m = _RE_USER_MESSAGE.search(task)
    if m:
        inner = m.group(1)
        t = _RE_USER_TEXT_INNER.search(inner)
        out.user_message = (t.group(1) if t else inner).strip()

    for m in _RE_THINKING.finditer(task):
        out.thinkings.append((m.start(), m.group(1).strip()))

    for m in _RE_TOOL_CALL.finditer(task):
        out.tool_calls.append(
            _ToolCall(
                name=m.group(1),
                call_id=m.group(2),
                body=m.group(3).strip(),
                position=m.start(),
            )
        )

    for m in _RE_TOOL_RESULT.finditer(task):
        out.tool_results.append(
            _ToolResult(
                tool=m.group(1),
                call_id=m.group(2),
                body=m.group(3).strip(),
                position=m.start(),
            )
        )

    m = _RE_FINAL_TEXT.search(task)
    if m:
        out.final_assistant_text = m.group(1).strip()

    return out


def _pair_calls_to_results(
    calls: list[_ToolCall], results: list[_ToolResult]
) -> list[tuple[_ToolCall, _ToolResult | None]]:
    by_id: dict[str, _ToolResult] = {r.call_id: r for r in results}
    return [(c, by_id.get(c.call_id)) for c in calls]


def _is_error_result(result: _ToolResult | None) -> bool:
    if result is None or not result.body:
        return False
    return bool(_RE_ERROR_INDICATOR.search(result.body[:1000]))


def _extract_files(parsed: _ParsedTranscript) -> list[str]:
    seen: dict[str, int] = {}
    sources: Iterable[str] = (
        [t for _, t in parsed.thinkings]
        + [c.body for c in parsed.tool_calls]
        + [r.body for r in parsed.tool_results]
    )
    for src in sources:
        for m in _RE_FILE_PATH.finditer(src):
            path = m.group(1)
            short = path
            if "/.openclaw/workspace/" in short:
                short = short.split("/.openclaw/workspace/", 1)[1]
            if short not in seen:
                seen[short] = len(seen)
    return list(seen.keys())


def _extract_symbols(parsed: _ParsedTranscript) -> list[str]:
    seen: dict[str, int] = {}
    for src in (
        [t for _, t in parsed.thinkings]
        + [c.body for c in parsed.tool_calls]
        + [r.body for r in parsed.tool_results]
    ):
        for m in _RE_PY_SYMBOL.finditer(src):
            name = m.group(1)
            if name not in seen and not name.startswith("_"):
                seen[name] = len(seen)
    return list(seen.keys())


def _is_test_path(body: str) -> bool:
    if not body:
        return False
    head = body[:500]
    return bool(re.search(r'(?:^|[/"\'])(test_[^/"\']+|tests?[/"\']|_test\.py)', head))


def _find_canonical_edits(
    pairs: list[tuple[_ToolCall, _ToolResult | None]],
    max_count: int = 3,
) -> list[tuple[_ToolCall, _ToolResult | None]]:
    successful = [
        (c, r) for (c, r) in pairs
        if c.name == "edit" and not _is_error_result(r)
    ]
    if not successful:
        return []
    source_edits = [(c, r) for (c, r) in successful if not _is_test_path(c.body)]
    test_edits = [(c, r) for (c, r) in successful if _is_test_path(c.body)]
    out = source_edits[-max(1, max_count - 1):]
    if test_edits and len(out) < max_count:
        out.append(test_edits[-1])
    return out


def _find_latest_successful_edit(
    pairs: list[tuple[_ToolCall, _ToolResult | None]]
) -> tuple[_ToolCall, _ToolResult | None] | None:
    edits = _find_canonical_edits(pairs, max_count=1)
    return edits[0] if edits else None


def _find_latest_error(
    pairs: list[tuple[_ToolCall, _ToolResult | None]]
) -> _ToolResult | None:
    for c, r in reversed(pairs):
        if _is_error_result(r):
            return r
    return None


def _find_failed_approaches(
    pairs: list[tuple[_ToolCall, _ToolResult | None]],
    max_count: int = 4,
) -> list[tuple[_ToolCall, _ToolResult]]:
    out: list[tuple[_ToolCall, _ToolResult]] = []
    for c, r in pairs:
        if c.name in ("edit", "exec") and _is_error_result(r):
            assert r is not None
            out.append((c, r))
    return out[-max_count:]


def _select_recent_successful_trace(
    pairs: list[tuple[_ToolCall, _ToolResult | None]],
    max_count: int = 3,
) -> list[tuple[_ToolCall, _ToolResult | None]]:
    out = [(c, r) for (c, r) in pairs if not _is_error_result(r)]
    return out[-max_count:]


def _short_body(body: str, limit_chars: int) -> str:
    body = body.strip()
    if len(body) <= limit_chars:
        return body
    head = body[: max(0, limit_chars // 2 - 30)]
    tail = body[-(limit_chars - len(head) - 6):]
    return f"{head}\n…\n{tail}"


def _render_state(
    files: list[str],
    symbols: list[str],
    canonical_edits: list[tuple[_ToolCall, _ToolResult | None]],
    latest_error: _ToolResult | None,
    failed_paths: list[tuple[_ToolCall, _ToolResult]],
    budget_tokens: int,
) -> str:
    parts: list[str] = ["<state>"]

    if files:
        files_text = "\n".join(f"  - {p}" for p in files[:12])
        parts.append(f"<files>\n{files_text}\n</files>")

    if symbols:
        symbols_text = ", ".join(symbols[:15])
        parts.append(f"<symbols>{symbols_text}</symbols>")

    if canonical_edits:
        per_edit_chars = max(800, _MAX_LATEST_EDIT_CHARS // max(1, len(canonical_edits)))
        edit_blocks: list[str] = []
        for edit_call, edit_result in canonical_edits:
            edit_body = _short_body(edit_call.body, per_edit_chars)
            result_text = ""
            if edit_result is not None:
                result_text = _short_body(edit_result.body, 300)
            block = f"  <edit>\n  call: {edit_body}"
            if result_text:
                block += f"\n  result: {result_text}"
            block += "\n  </edit>"
            edit_blocks.append(block)
        parts.append(
            "<canonical-edits>\n" + "\n".join(edit_blocks) + "\n</canonical-edits>"
        )

    if latest_error is not None:
        err_text = _short_body(latest_error.body, _MAX_TOOL_RESULT_CHARS)
        parts.append(f"<latest-error>\n{err_text}\n</latest-error>")

    if failed_paths:
        bullets = []
        for c, r in failed_paths:
            head = _short_body(c.body, 200)
            err = _short_body(r.body, 200)
            bullets.append(f"  - {c.name}: {head}\n    -> {err}")
        parts.append(
            "<failed-paths>\n" + "\n".join(bullets) + "\n</failed-paths>"
        )

    parts.append("</state>")
    section = "\n".join(parts)

    if _count_tokens(section) <= budget_tokens:
        return section

    fallback_orders: list[list[str]] = [
        ["files", "symbols", "canonical-edits", "latest-error", "failed-paths"],
        ["files", "symbols", "canonical-edits", "latest-error"],
        ["files", "symbols", "canonical-edits"],
        ["files", "canonical-edits"],
        ["canonical-edits"],
    ]
    pieces_by_key = {
        "files": next((p for p in parts if p.startswith("<files>")), None),
        "symbols": next((p for p in parts if p.startswith("<symbols>")), None),
        "canonical-edits": next((p for p in parts if p.startswith("<canonical-edits>")), None),
        "latest-error": next((p for p in parts if p.startswith("<latest-error>")), None),
        "failed-paths": next((p for p in parts if p.startswith("<failed-paths>")), None),
    }
    for order in fallback_orders:
        keep = [pieces_by_key[k] for k in order if pieces_by_key.get(k)]
        candidate = "<state>\n" + "\n".join(keep) + "\n</state>"
        if _count_tokens(candidate) <= budget_tokens:
            return candidate

    if canonical_edits:
        edit_body = _short_body(canonical_edits[0][0].body, max(200, budget_tokens * 2))
        truncated = (
            f"<state>\n<canonical-edits>\n  <edit>\n  call: {edit_body}\n  </edit>\n</canonical-edits>\n</state>"
        )
        return _truncate_to_tokens(truncated, budget_tokens)
    return _truncate_to_tokens("<state></state>", budget_tokens)


def _render_reasoning(parsed: _ParsedTranscript, budget_tokens: int) -> str:
    if not parsed.thinkings and not parsed.final_assistant_text:
        return ""
    parts: list[str] = ["<key-reasoning>"]
    if parsed.thinkings:
        first = parsed.thinkings[0][1]
        first_text = _short_body(first, 800)
        parts.append(f"<opening>{first_text}</opening>")
    closing_source = (
        parsed.final_assistant_text or
        (parsed.thinkings[-1][1] if len(parsed.thinkings) > 1 else "")
    )
    if closing_source:
        parts.append(f"<closing>{_short_body(closing_source, 1200)}</closing>")
    parts.append("</key-reasoning>")
    block = "\n".join(parts)
    if _count_tokens(block) <= budget_tokens:
        return block
    if parsed.thinkings:
        first_text = _short_body(parsed.thinkings[0][1], 400)
        return _truncate_to_tokens(
            f"<key-reasoning>\n<opening>{first_text}</opening>\n</key-reasoning>",
            budget_tokens,
        )
    return ""


def _render_trace(
    trace: list[tuple[_ToolCall, _ToolResult | None]],
    budget_tokens: int,
) -> str:
    if not trace:
        return ""
    blocks: list[str] = ["<trace>"]
    per_entry_budget = max(80, budget_tokens // max(1, len(trace) + 1))
    per_entry_chars = per_entry_budget * 3
    for c, r in trace:
        head = _short_body(c.body, max(80, per_entry_chars // 2))
        result_text = ""
        if r is not None:
            result_text = _short_body(r.body, max(80, per_entry_chars // 2))
        block = f"  <step name=\"{c.name}\">\n    {head}"
        if result_text:
            block += f"\n    -> {result_text}"
        block += "\n  </step>"
        blocks.append(block)
    blocks.append("</trace>")
    rendered = "\n".join(blocks)
    if _count_tokens(rendered) <= budget_tokens:
        return rendered
    while len(trace) > 1 and _count_tokens(rendered) > budget_tokens:
        trace = trace[1:]
        return _render_trace(trace, budget_tokens)
    return _truncate_to_tokens(rendered, budget_tokens)


def _assemble(
    parsed: _ParsedTranscript,
    target_tokens: int,
) -> str:
    if target_tokens <= 0:
        return ""

    pairs = _pair_calls_to_results(parsed.tool_calls, parsed.tool_results)
    files = _extract_files(parsed)
    symbols = _extract_symbols(parsed)
    canonical_edits = _find_canonical_edits(pairs, max_count=3)
    latest_error = _find_latest_error(pairs)
    failed_paths = _find_failed_approaches(pairs)
    trace = _select_recent_successful_trace(pairs)

    task_budget = max(50, int(target_tokens * _SECTION_BUDGET["task"]))
    state_budget = max(80, int(target_tokens * _SECTION_BUDGET["state"]))
    reasoning_budget = max(40, int(target_tokens * _SECTION_BUDGET["reasoning"]))
    trace_budget = max(40, int(target_tokens * _SECTION_BUDGET["trace"]))

    task_text = parsed.user_message or "(no user message recovered)"
    task_block = f"<task>\n{task_text}\n</task>"
    if _count_tokens(task_block) > task_budget:
        truncated_task = _truncate_to_tokens(task_text, max(40, task_budget - 8))
        task_block = f"<task>\n{truncated_task}\n</task>"

    state_block = _render_state(
        files, symbols, canonical_edits, latest_error, failed_paths, state_budget
    )
    reasoning_block = _render_reasoning(parsed, reasoning_budget)
    trace_block = _render_trace(trace, trace_budget)

    full_parts = [task_block, state_block]
    if reasoning_block:
        full_parts.append(reasoning_block)
    if trace_block:
        full_parts.append(trace_block)
    combined = "\n\n".join(full_parts)

    if _count_tokens(combined) <= target_tokens:
        return combined

    for drop in ("trace", "reasoning"):
        if drop == "trace" and trace_block:
            full_parts.remove(trace_block)
            trace_block = ""
            combined = "\n\n".join(p for p in full_parts if p)
            if _count_tokens(combined) <= target_tokens:
                return combined
        elif drop == "reasoning" and reasoning_block:
            if reasoning_block in full_parts:
                full_parts.remove(reasoning_block)
            reasoning_block = ""
            combined = "\n\n".join(p for p in full_parts if p)
            if _count_tokens(combined) <= target_tokens:
                return combined
    return _truncate_to_tokens(combined, target_tokens)


def _lexical_fallback(task: str, target_tokens: int) -> str:
    lines = [ln for ln in task.splitlines() if ln.strip()]
    if not lines:
        return _truncate_to_tokens(task, target_tokens)

    def score(ln: str) -> float:
        s = 1.0
        if _RE_FILE_PATH.search(ln):
            s *= 1.5
        if _RE_ERROR_INDICATOR.search(ln):
            s *= 1.3
        if re.search(r'\b(?:def|class|import|from)\b', ln):
            s *= 1.2
        if len(ln) > 400:
            s *= 0.4
        if re.match(r'^[-=*_]{5,}$', ln.strip()):
            s *= 0.1
        return s

    scored = sorted(
        ((score(ln), i, ln) for i, ln in enumerate(lines)),
        key=lambda t: (-t[0], t[1]),
    )
    kept: list[tuple[int, str]] = []
    used_tokens = 0
    for _, idx, ln in scored:
        cost = _count_tokens(ln) + 1
        if used_tokens + cost > target_tokens:
            continue
        kept.append((idx, ln))
        used_tokens += cost
    kept.sort(key=lambda t: t[0])
    return "\n".join(ln for _, ln in kept)


def main(task: str, compression_ratio: float | None = None) -> str:
    if not task:
        return ""

    if compression_ratio is None:
        compression_ratio = 0.2
    compression_ratio = max(0.001, min(1.0, float(compression_ratio)))

    original_tokens = _count_tokens(task)
    target_tokens = max(1, int(original_tokens * compression_ratio))

    if compression_ratio >= 1.0 or original_tokens <= target_tokens:
        return task

    try:
        parsed = _parse(task)
        if not parsed.user_message and not parsed.tool_calls:
            return _lexical_fallback(task, target_tokens)
        out = _assemble(parsed, target_tokens)
        if not out.strip():
            return _lexical_fallback(task, target_tokens)
        if _count_tokens(out) > target_tokens:
            out = _truncate_to_tokens(out, target_tokens)
        return out
    except Exception as exc:
        logger.warning("blackhole: pipeline failed, falling back to lexical: %s", exc)
        try:
            return _lexical_fallback(task, target_tokens)
        except Exception:
            return _truncate_to_tokens(task, target_tokens)


def _writeback(task: str, out: str) -> None:
    out_dir = os.environ.get("MINER_OUTPUT_DIR")
    tag = os.environ.get("MINER_OUTPUT_TAG", "task")
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{tag}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
    except Exception:
        pass


_inner_main = main


def main(task: str, compression_ratio: float | None = None) -> str:  # noqa: F811
    out = _inner_main(task, compression_ratio)
    _writeback(task, out)
    return out


import copy
import hashlib
import json as _json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

EVENT_NAMES = frozenset({"assemble"})
STATE_VERSION = 1
STATE_DIR_NAME = "state"

KEEP_TOOL_RESULT_COUNT = 6
MAX_TOOL_RESULT_CHARS = 2000


def _r2_normalize_role(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("_", "").replace("-", "")


def _r2_extract_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_r2_extract_text(item) for item in value)
    if isinstance(value, dict):
        for k in ("text", "content"):
            if isinstance(value.get(k), str):
                return value[k]
        return "\n".join(_r2_extract_text(v) for v in value.values())
    return str(value)


def _r2_estimate_tokens(messages: list) -> int:
    total_chars = 0
    for m in messages:
        if isinstance(m, dict):
            total_chars += len(_r2_extract_text(m.get("content")))
    return max(1, math.ceil(total_chars / 4)) if messages else 0


def _r2_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _r2_safe_file_part(value, fallback: str = "session") -> str:
    raw = value.strip() if isinstance(value, str) and value.strip() else fallback
    norm = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in raw)
    norm = norm.strip("-")
    return (norm or fallback)[:120]


def _r2_canonical_json(value) -> str:
    return _json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), default=str)


def _r2_fingerprint(messages: list) -> str:
    return hashlib.sha256(_r2_canonical_json(messages).encode("utf-8")).hexdigest()


def _r2_get_params(payload: dict) -> dict:
    params = payload.get("params")
    return params if isinstance(params, dict) else payload


def _r2_plugin_dir(payload: dict) -> Path:
    v = payload.get("pluginDir")
    if isinstance(v, str) and v.strip():
        return Path(v.strip())
    return Path(__file__).resolve().parent


def _r2_session_identity(payload: dict) -> tuple[str, str | None, str | None]:
    params = _r2_get_params(payload)
    sid = params.get("sessionId") if isinstance(params.get("sessionId"), str) else None
    skey = params.get("sessionKey") if isinstance(params.get("sessionKey"), str) else None
    return _r2_safe_file_part(sid or skey or "session"), sid, skey


def _r2_state_path(payload: dict) -> Path:
    return _r2_plugin_dir(payload) / "logs" / STATE_DIR_NAME / f"{_r2_session_identity(payload)[0]}.json"


def _r2_load_state(state_path: Path) -> dict | None:
    try:
        s = _json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(s, dict) or s.get("version") != STATE_VERSION:
        return None
    if not isinstance(s.get("messages"), list):
        return None
    if not isinstance(s.get("sourceMessageCount"), int):
        return None
    if not isinstance(s.get("sourceFingerprint"), str):
        return None
    return s


def _r2_save_state(state_path: Path, payload: dict,
                   raw_messages: list, current_messages: list) -> None:
    _, sid, skey = _r2_session_identity(payload)
    state = {
        "version": STATE_VERSION,
        "updatedAt": _r2_utc_now(),
        "sessionId": sid,
        "sessionKey": skey,
        "sourceMessageCount": len(raw_messages),
        "sourceFingerprint": _r2_fingerprint(raw_messages),
        "messageCount": len(current_messages),
        "messages": current_messages,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(f"{state_path.suffix}.tmp")
    tmp.write_text(_json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(state_path)


def _r2_resolve_stateful(payload: dict, raw: list) -> tuple[list, dict, Path]:
    state_path = _r2_state_path(payload)
    state = _r2_load_state(state_path)
    meta = {
        "statePath": str(state_path),
        "stateLoaded": False,
        "stateResetReason": None,
        "rawInputMessageCount": len(raw),
        "previousSourceMessageCount": None,
        "previousStateMessageCount": None,
        "newMessageCount": len(raw),
        "workingMessageCount": len(raw),
    }
    if state is None:
        return raw, meta, state_path
    src_count = state.get("sourceMessageCount")
    if not isinstance(src_count, int) or src_count < 0:
        meta["stateResetReason"] = "invalid_source_count"
        return raw, meta, state_path
    meta["previousSourceMessageCount"] = src_count
    meta["previousStateMessageCount"] = len(state.get("messages", []))
    if src_count > len(raw):
        meta["stateResetReason"] = "source_shorter_than_state"
        return raw, meta, state_path
    if _r2_fingerprint(raw[:src_count]) != state.get("sourceFingerprint"):
        meta["stateResetReason"] = "source_prefix_changed"
        return raw, meta, state_path
    new_tail = raw[src_count:]
    working = [*state["messages"], *new_tail]
    meta.update({
        "stateLoaded": True,
        "newMessageCount": len(new_tail),
        "workingMessageCount": len(working),
    })
    return working, meta, state_path


def _r2_tool_result_ids(m) -> set:
    if not isinstance(m, dict) or _r2_normalize_role(m.get("role")) != "toolresult":
        return set()
    out = set()
    for k in ("toolCallId", "toolUseId", "id"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            out.add(v.strip())
    return out


def _r2_tool_call_ids(m) -> set:
    if not isinstance(m, dict) or _r2_normalize_role(m.get("role")) != "assistant":
        return set()
    out = set()
    content = m.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                v = block.get("id")
                if isinstance(v, str) and v.strip():
                    out.add(v.strip())
    for k in ("toolCalls", "tool_calls"):
        tcs = m.get(k)
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict):
                    v = tc.get("id")
                    if isinstance(v, str) and v.strip():
                        out.add(v.strip())
    return out


def _r2_tool_call_blocks(m) -> list:
    if not isinstance(m, dict):
        return []
    out = []
    content = m.get("content")
    if isinstance(content, list):
        out.extend(b for b in content
                   if isinstance(b, dict) and b.get("type") == "toolCall")
    for k in ("toolCalls", "tool_calls"):
        tcs = m.get(k)
        if isinstance(tcs, list):
            out.extend(b for b in tcs if isinstance(b, dict))
    return out


def _r2_block_is_error(text: str) -> bool:
    if not text:
        return False
    head = text[:1000]
    return bool(_RE_ERROR_INDICATOR.search(head))


def _r2_is_edit_call(block: dict) -> bool:
    if not isinstance(block, dict):
        return False
    name = block.get("name") or block.get("function") or ""
    if isinstance(name, str) and name.lower() == "edit":
        return True
    args = block.get("input") or block.get("args") or {}
    if isinstance(args, dict) and "edits" in args and "path" in args:
        return True
    return False


def _r2_call_path_hint(block: dict) -> str:
    if not isinstance(block, dict):
        return ""
    args = block.get("input") or block.get("args") or {}
    if isinstance(args, dict):
        p = args.get("path") or args.get("file") or ""
        if isinstance(p, str):
            return p
    return ""


def _r2_is_test_path(path: str) -> bool:
    if not path:
        return False
    return bool(re.search(r"(?:^|[/\\])(?:test_[^/\\]+|tests?[/\\]|_test\.py$)", path))


def _r2_sanitize_content(content):
    if not isinstance(content, list):
        return content, False
    new_blocks = []
    changed = False
    for b in content:
        if isinstance(b, dict) and b.get("type") == "thinking":
            changed = True
            continue
        new_blocks.append(b)
    return new_blocks, changed


def _r2_truncate_text_block(block, max_chars: int) -> tuple[dict, bool]:
    if not isinstance(block, dict):
        return block, False
    text = block.get("text")
    if not isinstance(text, str) or len(text) <= max_chars:
        return block, False
    new = copy.deepcopy(block)
    new["text"] = _short_body(text, max_chars)
    return new, True


def _r2_trim_tool_result(message: dict) -> tuple[dict, bool]:
    if _r2_normalize_role(message.get("role")) != "toolresult":
        return message, False
    content = message.get("content")
    if isinstance(content, str):
        if len(content) <= MAX_TOOL_RESULT_CHARS:
            return message, False
        new = copy.deepcopy(message)
        new["content"] = _short_body(content, MAX_TOOL_RESULT_CHARS)
        return new, True
    if isinstance(content, list):
        new_blocks = []
        changed = False
        for b in content:
            nb, ch = _r2_truncate_text_block(b, MAX_TOOL_RESULT_CHARS)
            if ch:
                changed = True
            new_blocks.append(nb)
        if not changed:
            return message, False
        new = copy.deepcopy(message)
        new["content"] = new_blocks
        return new, True
    return message, False


def _r2_is_failed_placeholder(m: dict) -> bool:
    if not isinstance(m, dict) or _r2_normalize_role(m.get("role")) != "assistant":
        return False
    if not isinstance(m.get("errorMessage"), str):
        return False
    c = m.get("content")
    return c in (None, "") or c == []


def _r2_has_runtime_content(content) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return bool(content)
    if isinstance(content, dict):
        return bool(content)
    return content is not None


def _r2_first_user_index(messages: list) -> int | None:
    for i, m in enumerate(messages):
        if isinstance(m, dict) and _r2_normalize_role(m.get("role")) == "user":
            return i
    return None


def _r2_sanitize_messages(messages: list) -> tuple[list, dict]:
    out = []
    removed_msgs = 0
    removed_thinking = 0
    trimmed_results = 0
    changed = False
    for m in messages:
        if _r2_is_failed_placeholder(m):
            changed = True
            removed_msgs += 1
            continue
        if not isinstance(m, dict):
            out.append(m)
            continue
        nxt = m
        new_content, content_changed = _r2_sanitize_content(m.get("content"))
        if content_changed:
            nxt = copy.deepcopy(m)
            nxt["content"] = new_content
            changed = True
            removed_thinking += 1
        if _r2_normalize_role(nxt.get("role")) != "toolresult" \
                and not _r2_has_runtime_content(nxt.get("content")):
            changed = True
            removed_msgs += 1
            continue
        nxt, trim_changed = _r2_trim_tool_result(nxt)
        if trim_changed:
            changed = True
            trimmed_results += 1
        out.append(nxt)
    meta = {
        "changed": changed,
        "removedMessageCount": removed_msgs,
        "removedThinkingBlockCount": removed_thinking,
        "trimmedToolResultCount": trimmed_results,
    }
    return (out if changed else messages), meta


def _r2_prune_messages(messages: list) -> tuple[list, dict]:
    first_user = _r2_first_user_index(messages)
    if first_user is None:
        return messages, {"changed": False, "reason": "missing_first_user_message"}

    tool_result_idx = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and _r2_normalize_role(m.get("role")) == "toolresult"
    ]
    if len(tool_result_idx) < KEEP_TOOL_RESULT_COUNT:
        return messages, {
            "changed": False,
            "reason": "fewer_than_keep_count_tool_results",
            "toolResultCount": len(tool_result_idx),
        }

    kept_tr_idx = set(tool_result_idx[-KEEP_TOOL_RESULT_COUNT:])
    kept_tr_ids: set = set()
    for i in kept_tr_idx:
        kept_tr_ids.update(_r2_tool_result_ids(messages[i]))

    kept_call_idx: set = set()
    if kept_tr_ids:
        for i, m in enumerate(messages):
            if _r2_tool_call_ids(m) & kept_tr_ids:
                kept_call_idx.add(i)

    canonical_call_idx: set = set()
    canonical_call_ids: set = set()
    for i, m in enumerate(messages):
        if _r2_normalize_role(m.get("role")) != "assistant":
            continue
        for block in _r2_tool_call_blocks(m):
            if not _r2_is_edit_call(block):
                continue
            path = _r2_call_path_hint(block)
            if _r2_is_test_path(path):
                continue
            cid = block.get("id")
            if not isinstance(cid, str):
                continue
            for j, mr in enumerate(messages):
                if _r2_normalize_role(mr.get("role")) != "toolresult":
                    continue
                if cid in _r2_tool_result_ids(mr):
                    rtext = _r2_extract_text(mr.get("content"))
                    if not _r2_block_is_error(rtext):
                        canonical_call_idx.add(i)
                        canonical_call_ids.add(cid)
                        kept_tr_idx.add(j)
                    break

    keep = {first_user} | kept_tr_idx | kept_call_idx | canonical_call_idx
    pruned = [m for i, m in enumerate(messages) if i in keep]

    changed = len(pruned) != len(messages)
    return (pruned if changed else messages), {
        "changed": changed,
        "reason": "pruned" if changed else "nothing_to_remove",
        "originalMessageCount": len(messages),
        "messageCount": len(pruned),
        "keptToolResultCount": len(kept_tr_idx),
        "keptToolCallMessageCount": len(kept_call_idx),
        "canonicalEditCount": len(canonical_call_idx),
    }


def handle_assemble(payload: dict) -> dict:
    params = _r2_get_params(payload)
    raw_messages = params.get("messages") if isinstance(params.get("messages"), list) else []
    working, state_meta, state_path = _r2_resolve_stateful(payload, raw_messages)
    runtime, sanitize_meta = _r2_sanitize_messages(working)
    pruned, prune_meta = _r2_prune_messages(runtime)

    output_differs = _r2_fingerprint(pruned) != _r2_fingerprint(raw_messages)
    pruned_flag = bool(prune_meta.get("changed"))
    sanitized_flag = bool(sanitize_meta.get("changed"))
    changed = pruned_flag or sanitized_flag or output_differs
    reason = prune_meta.get("reason")
    if pruned_flag:
        reason = "pruned"
    elif sanitized_flag:
        reason = "sanitized"
    elif output_differs and state_meta.get("stateLoaded"):
        reason = "state_reused"

    metadata = {
        **prune_meta,
        **state_meta,
        "changed": changed,
        "reason": reason,
        "originalMessageCount": len(raw_messages),
        "messageCount": len(pruned),
        "sanitized": sanitized_flag,
        "removedMessageCount": sanitize_meta.get("removedMessageCount", 0),
        "removedThinkingBlockCount": sanitize_meta.get("removedThinkingBlockCount", 0),
        "trimmedToolResultCount": sanitize_meta.get("trimmedToolResultCount", 0),
        "pruned": pruned_flag,
    }

    try:
        _r2_save_state(state_path, payload, raw_messages, pruned)
        metadata["stateSaved"] = True
    except Exception as e:
        metadata["stateSaved"] = False
        metadata["stateError"] = str(e)

    return {
        "assembled": True,
        "messages": pruned,
        "estimatedTokens": _r2_estimate_tokens(pruned),
        "baseMiner": metadata,
    }


def _r2_run_event(event_name: str) -> int:
    try:
        raw = sys.stdin.buffer.read()
        payload = _json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Connector payload must be a JSON object")
        if event_name not in EVENT_NAMES:
            raise ValueError(f"Unknown event: {event_name}")
        response = {"ok": True, "result": handle_assemble(payload)}
    except Exception as e:
        response = {"ok": False, "error": str(e), "errorType": e.__class__.__name__}
    sys.stdout.buffer.write(_json.dumps(response, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


def _r2_cli_main(argv: list) -> int:
    return _r2_run_event(argv[0] if argv else "assemble")


if __name__ == "__main__":
    raise SystemExit(_r2_cli_main(sys.argv[1:]))
