# Blackhole vs Apollo — local eval v1

**Setup**
- Eval: `miner/run_task_completion_proxy.py` (R2 working-agent proxy: miner-compresses → LLM-acts-as-agent → produces unified diff → LLM-grades diff vs gold SWE-bench patch)
- Model: `qwen/qwen3-coder` via OpenRouter (the R2 production default per `swebench_default_model` on `upstream/dev`)
- Tasks: CoT-Compression-2, challenge IDs 1533, 1534, 1535, 1536, 1537 (Django bugs, mapped to SWE-bench Verified instances `django__django-11095`, `-11133`, `-11149`, `-11179`, `-11292`)
- Ratios: 0.10, 0.20, 0.30
- Total runs: 30 (15 per miner). Of those, 2 Apollo runs and 1 Blackhole run failed mid-LLM with `'NoneType' object is not subscriptable` — qwen3-coder returned malformed responses, not a miner bug. 12 paired runs available for direct comparison.

**Aggregate (averaged across all valid runs)**

| Metric | Apollo (n=13) | Blackhole (n=14) | Δ (B − A) |
|---|---|---|---|
| Patch quality | 0.481 | **0.518** | +0.037 (+7.7%) |
| Token savings | 0.861 | 0.871 | +0.010 |
| Combined score | 0.448 | **0.485** | +0.037 (+8.3%) |

**Per-ratio**

| ratio | Apollo q | Blackhole q | Apollo combined | Blackhole combined | Winner |
|---|---|---|---|---|---|
| 0.10 | **0.550** | 0.500 | **0.527** | 0.482 | Apollo |
| 0.20 | 0.500 | **0.562** | 0.460 | **0.525** | Blackhole |
| 0.30 | 0.375 | **0.500** | 0.339 | **0.455** | Blackhole |

**Per-task (paired only)**

| cid | ratio | A.qual | B.qual | A.combined | B.combined | Winner |
|---|---|---|---|---|---|---|
| 1533 | 0.10 | 0.50 | 0.50 | 0.479 | 0.484 | Blackhole |
| 1534 | 0.10 | 0.50 | 0.50 | 0.478 | 0.480 | Blackhole |
| 1534 | 0.20 | 0.50 | 0.50 | 0.458 | 0.457 | Apollo (margin 0.001) |
| 1534 | 0.30 | 0.50 | **0.75** | 0.444 | **0.669** | Blackhole (+0.225) |
| 1535 | 0.10 | 0.50 | 0.50 | 0.477 | 0.485 | Blackhole |
| 1535 | 0.30 | 0.50 | 0.50 | 0.444 | 0.450 | Blackhole |
| 1536 | 0.10 | **0.75** | 0.50 | **0.716** | 0.477 | Apollo (+0.239) |
| 1536 | 0.20 | 0.50 | 0.50 | 0.458 | 0.466 | Blackhole |
| 1536 | 0.30 | 0.00 | 0.00 | 0.000 | 0.000 | TIE (both INCORRECT) |
| 1537 | 0.10 | 0.50 | 0.50 | 0.484 | 0.483 | Apollo (margin 0.001) |
| 1537 | 0.20 | 0.50 | 0.50 | 0.468 | 0.471 | Blackhole |
| 1537 | 0.30 | 0.50 | 0.50 | 0.466 | 0.456 | Apollo (margin 0.010) |

**Win counts**: Blackhole 7 / Apollo 4 / Tie 1

**Speed**

| | Apollo (warm) | Blackhole (warm) |
|---|---|---|
| Compression latency | 50–150 ms | 3–11 ms |
| Cold-start | 5–6 s (MiniLM load) | < 50 ms |

Blackhole is ~10x faster warm and has no model-loading cold start. Both are well under the 10s sandbox budget.

**Reading the result**

1. At ratio = 0.10 (aggressive), Apollo wins on average. Its embedding-based tier-ranking surfaces a tighter subset of high-value events when the budget is genuinely tight. The single large win for Apollo (1536@0.10, quality=0.75) is what tips the average — three other 0.10 runs were near-ties.
2. At ratio = 0.20 and 0.30, Blackhole wins clearly. The structured `<task> / <state> / <key-reasoning> / <trace>` rendering uses the extra budget on agent-actionable state (canonical edits, file paths, failed approaches) more effectively than Apollo's chronological event preservation.
3. The headline +8.3% combined-score edge is suggestive at this sample size (n=12 paired), not significant. The pattern matters more than the magnitude: Blackhole's edge grows with the budget, which matches the design hypothesis.

**Recommendation for R2 submission**

The Strategy.md decision tree calibrates the operating point by running through multiple ratios; if you target ratio ≥ 0.20 (the regime where token savings is mild and resolve-rate dominates), Blackhole is the pick. At ratio ≤ 0.15, Apollo's lead is small but real, and either is defensible.

A safer mixed strategy would be: pick **Blackhole at ratio 0.25** as the single submission. This is the regime where Blackhole's structured render most reliably preserves agent-actionable state, and it's also the operating point Strategy.md hypothesizes is closest to the resolve-rate knee.

**Caveats**

- 5 tasks × 3 ratios is small. Statistical confidence is limited.
- The proxy is not the real eval. It does not apply patches and run tests; it LLM-judges candidate vs gold. A pass here is a *necessary* but not *sufficient* signal.
- qwen3-coder returned malformed responses for 3 of 30 runs. This noise will appear in production too; the miner that *handles* this (graceful agent re-prompts) wins long-term, but that's out of scope for a compression algorithm.
- Reward-proportion rules for R2 are not published yet (oli 5/21: "Exact reward proportions will be announced before the competition starts"). Combined-score weighting here is `quality × (1 + savings) / 2` — production may weight differently.
