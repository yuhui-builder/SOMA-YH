# Blackhole — CoT compression miner (SOMA SN114, R2)

Second-generation miner for the working-agent eval that ships on `upstream/dev`. Different bet than Apollo, designed against the actual R2 mechanism, not R1's QA-style grader.

**Two interfaces supported in one file** — see [§ Interfaces](#interfaces) below.

## Why a second miner

Apollo was tuned when the validator was QA-on-compressed-text (R1). The R2 validator instead loads Blackhole as a `base_miner.py` inside the SOMA OpenClaw plugin. Every time OpenClaw is about to make an LLM call, the plugin invokes `python3 base_miner.py assemble` and asks Blackhole to prune the conversation history. The agent (running `qwen/qwen3-coder`) then proceeds on Blackhole's pruned message list. After the agent finishes, the diff it emits is graded by the SWE-bench Verified test suite. Different signal entirely:

- The **agent**, not an LLM judge, consumes your output.
- You're scored on whether the **tests pass**, not on whether your text looks like a good summary.
- Anything that helps the agent produce a correct patch in one shot is valuable; anything else is filler.

Apollo's "rank events by tier, fill budget by importance" still works, but it preserves chronological narrative when what the agent actually needs is **current state of the world**. Blackhole takes the opposite stance: parse the transcript, extract the agent state, render it as a "case file" for the next agent to read.

## Architecture

Single file, stdlib + `tiktoken` only. No `sentence-transformers`, no embeddings, no model downloads — those aren't reliably available in the R2 sandbox (the `sandbox_image/` directory was deleted on `upstream/dev`).

```
PARSE      raw transcript -> typed events (tolerant regex; not strict XML)
EXTRACT    files / symbols / latest source-file edit / failed approaches
           / latest error / recent successful trace
RENDER     <task> verbatim
           <state> canonical-edits, files, symbols, latest-error, failed-paths
           <key-reasoning> opening hypothesis + closing synthesis
           <trace> most recent successful tool steps
BUDGET     per-section token budgets; tail-trim if over
SAFETY     wrap everything; lexical-line-scoring fallback if parsing
           recovers no structure; hard token-cap at end
```

Output passes the platform's `_is_compressed_enough` gates by construction:

- `compressed_tokens / original_tokens <= ratio` — enforced via final truncation
- `chars_per_token(compressed) / chars_per_token(original) <= 1.8` — Blackhole emits plain ASCII / standard punctuation; no BPE-unfriendly characters

## How it differs from Apollo

| Axis | Apollo | Blackhole |
|---|---|---|
| Selection unit | event with tier T0..T4 | section (task / state / reasoning / trace) |
| Ranker | rules + MiniLM-L3 embedding for T2 | pure rule-based |
| External deps | `sentence-transformers` (may be unavailable on R2) | stdlib + tiktoken only |
| Output order | chronological | reorganized by role |
| Failed-approach handling | demoted | explicitly preserved as `<failed-paths>` |
| Source-vs-test edits | mixed | source-file edits prioritized in `<canonical-edits>` |
| Warm-call latency | 50–150ms | 3–11ms |
| Cold-start | 5–6s (MiniLM load) | <50ms |

## How it aligns with the R2 strategy doc

See `chats/Strategy.md` §3 — the three R2-specific improvements, applied to both interfaces:

1. **State-resumption** — preserve current working state. In R2 mode, this means keeping the canonical (source-file) edit messages even when they fall outside the recent-history window. In R1 mode, this means the `<canonical-edits>` block in the rendered output.
2. **Failure-trace preservation** — in R2 mode, failed edits stay in the window (the last 6 tool_results include errors); in R1 mode, the explicit `<failed-paths>` block. Both reasons by oli 5/22 explicitly allow structural markers.
3. **Per-ratio behavior** — R1 only: section budgets scale with the ratio. R2 has no ratio knob — pruning is implicit and the operating point is "how aggressive should the window + canonical-edit selection be". Tune `KEEP_TOOL_RESULT_COUNT` and `MAX_TOOL_RESULT_CHARS` to shift it.

Anti-injection scope (oli 5/22 — *"no added words, no inserted instructions, no extra content that wasn't already there"*): Blackhole never inserts new content. In R2 mode it only **selects** existing messages, **strips** thinking blocks, and **truncates** oversize tool_result bodies — every retained byte comes from the original input. In R1 mode the output uses structural tag markers (`<task>`, `<state>`, etc.) which oli explicitly allowed; content inside the tags is lifted verbatim.

## Interfaces

Blackhole supports both R1 and R2 shapes in the same file. The platform picks which one runs based on how the script is invoked.

### R2 production (the one that actually counts)

The SOMA-plugin's `index.js` spawns `python3 base_miner.py assemble` and pipes a JSON payload via stdin. Blackhole's `if __name__ == "__main__":` block handles this:

```bash
# what the plugin runs in production:
echo '{"params":{"messages":[...],"sessionId":"abc"},"pluginDir":"..."}' \
    | python3 blackhole.py assemble
# stdout: {"ok": true, "result": {"assembled": true, "messages": [...], ...}}
```

The R2 entry point is the module-level function `handle_assemble(payload: dict) -> dict`. It:
- Reads OpenClaw message objects from `payload["params"]["messages"]` (each has `role`, `content`, `toolCalls`, etc.)
- Maintains per-session state in `logs/state/<sessionId>.json` (across `assemble` calls)
- Returns `{"assembled": True, "messages": [...pruned...], "estimatedTokens": int, "baseMiner": {<metadata>}}`

What R2 pruning does:
1. **Sanitize** — strip `{type: "thinking"}` content blocks; drop empty/failed-placeholder assistant messages; truncate huge tool_result bodies to ≤2KB
2. **Find canonical edits** — assistant messages whose `toolCall.name == "edit"` and whose target path is *not* a test file, paired with successful (non-error) tool_results. These are always kept.
3. **Window** — keep the first user message + the last 6 tool_results + the assistant messages that invoked them
4. **Union** — keep set = first_user ∪ canonical_edits ∪ last_window
5. **Save state** under `logs/state/<sessionId>.json` so the next call can resume from where we left off

Smoke test:

```bash
python -m miner.Blackhole.test_r2_smoke
```

Exercises: thinking-block stripping, tool_result trimming, canonical-edit preservation across windowing, stateful incremental replay, state-file persistence.

### R1 / local-eval compatibility

The R1 function is preserved unchanged so the local task-completion proxy still works:

```python
from miner.Blackhole.blackhole import main
compressed_text = main(transcript_xml, compression_ratio=0.20)
```

Local eval (works against the published CoT-Compression-2 samples):

```bash
python -m miner.run_task_completion_proxy miner.Blackhole.blackhole \
    --task-dir CoT-Compression-2 \
    --challenge-id 1533,1534,1535 \
    --ratios 0.10,0.20,0.30 \
    --prefer-provider openrouter --model "qwen/qwen3-coder" \
    --log miner/sample_results/cmp_blackhole.jsonl
```

This tests the R1 shape with the R2 production model (`qwen/qwen3-coder`). It's a useful directional signal but it does **not** exercise:
- the stateful `assemble`/`assemble`/... loop the real plugin runs
- the OpenClaw agent's actual tool execution
- the SWE-bench harness applying patches and running tests

For a real-fidelity local R2 test, run the public benchmark runner against your fork of `SOMA-plugin` with `blackhole.py` swapped in as `base_miner.py`:

```bash
# in your SOMA-plugin fork: cp /path/to/blackhole.py ./base_miner.py
uv run python -m soma_bench benchmark-solve \
    --agent-name openclaw \
    --benchmark SWE-bench/SWE-bench_Verified \
    --instance-id django__django-11095 \
    --execute \
    --openclaw-plugin-path /path/to/your/SOMA-plugin
```

`miner/.env` needs `OPENROUTER_API_TOKEN`.

## Local eval results vs Apollo

5 challenges (1533–1537), 3 ratios (0.10 / 0.20 / 0.30), via `qwen/qwen3-coder` on OpenRouter (the R2 production model).

See [`comparison_v1.md`](comparison_v1.md) for the full per-task breakdown. Summary:

| Aggregate | Apollo | Blackhole | Δ |
|---|---|---|---|
| Quality (avg) | 0.481 | **0.518** | +0.037 |
| Token savings (avg) | 0.861 | 0.871 | +0.010 |
| Combined score (avg) | 0.448 | **0.485** | +0.037 (+8.3%) |
| Per-task wins | 4 | **7** | (1 tie) |
| Compress time (warm) | 50–150ms | **3–11ms** | ~10x faster |

By compression ratio:

| ratio | Apollo combined | Blackhole combined |
|---|---|---|
| 0.10 | **0.527** | 0.482 |
| 0.20 | 0.460 | **0.525** |
| 0.30 | 0.339 | **0.455** |

Apollo edges Blackhole at the most aggressive ratio (0.10), where its event-tier ranking better preserves the few highest-value tokens. Blackhole wins clearly at 0.20 and 0.30, where there's enough budget for the structured `<state>` block to pay off.

## What to do with this

The Strategy.md asymmetric-bet thesis is: maintain Apollo as a floor, ship Blackhole as the upgraded primary. The pattern above (Blackhole's edge growing with the ratio) suggests Blackhole is the right pick for any submission targeting ratio ≥ 0.20, which is also the ratio range the strategy doc recommends for working-agent evaluation. If you submit, submit Blackhole at ratio 0.20 or 0.25 — that's where its advantage is sharpest.

If you want a safer hedge, the cheap thing is to ensemble them: try Blackhole, fall back to Apollo's output if Blackhole's structured parser finds nothing (i.e., the input isn't an OpenClaw transcript). That's already roughly what the `_lexical_fallback` path does inside Blackhole, but a true Apollo-fallback would need to import Apollo as a library.

## Known limitations

- **Sample size is 5 challenges**; statistical power is limited. The +8.3% combined edge is suggestive, not definitive. Re-run on 20+ challenges before treating this as the final ordering.
- **The local task-completion proxy is not the real eval.** It does not run the OpenClaw agent loop or apply patches to a SWE-bench container. It LLM-judges your candidate patch vs gold. A miner that fails here is almost certainly bad; a miner that wins here is a *candidate*, not a guaranteed production winner.
- **The local R1 comparison results above are NOT R2 production results.** They tested `main(task, ratio) → str` against the gpt/qwen3-coder agent simulation. The R2 production path uses `handle_assemble(payload) → dict` inside a real OpenClaw agent loop. The strategic *direction* (canonical-edits + state, no embeddings) is what carries over; the numbers do not.
- **R2 production threshold is 80% on the 5 simplest screener tasks** (oli 5/25 PM). Local task-completion-proxy scores in the 0.45–0.55 combined range mean *something* but not directly the 80% bar — they're directional.
- **Production validation uses `qwen/qwen3-coder`** but reward-proportion rules and exact scoring function (especially the token-savings weighting) aren't published yet. Treat all combined scores here as relative, not absolute.
- **The smoke test does not run the actual OpenClaw plugin.** It only verifies the CLI shape and the stateful pruning logic. A real-fidelity test requires forking `DendriteHQ/SOMA-plugin`, dropping `blackhole.py` in as `base_miner.py`, and running `python -m soma_bench benchmark-solve` from `DendriteHQ/SOMA-benchmark`.
