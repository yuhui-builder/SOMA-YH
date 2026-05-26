# Running eval on Blackhole — what works on your machine today

Three tiers of self-eval, in increasing fidelity to the R2 production validator. Tiers 1 and 2 already work locally; tier 3 needs Docker.

## Quick answer

| Tier | What it tests | Runs locally now? | Setup cost |
|---|---|---|---|
| **1. R2 smoke test** | CLI shape, pruning logic, state persistence | ✅ Yes | Already done |
| **2. R1 task-completion proxy** | Directional quality vs. gpt-judged gold patch | ✅ Yes | Already done |
| **3. Full SOMA-benchmark (real R2)** | End-to-end: miner → OpenClaw agent → SWE-bench tests | ❌ Needs Docker Desktop | ~1 hr install + ~$1–5 per task in OpenRouter credits |

You ran tier 2 earlier (Apollo vs Blackhole on cids 1533–1537). Tier 1 is already wired up. Tier 3 is what you need before submitting with high confidence.

---

## Tier 1 — R2 smoke test (verifies the CLI shape)

```bash
python -m miner.Blackhole.test_r2_smoke
```

What this exercises:
- `python3 blackhole.py assemble` reads JSON from stdin, writes JSON to stdout
- Thinking blocks are stripped
- Long noisy tool_results are trimmed
- Canonical source-file edits survive windowing (even if older than the last 6 tool_results)
- State persists to `logs/state/<sessionId>.json`
- Second invocation does incremental replay (`stateLoaded=True`, `newMessageCount=1`)

What it does *not* test:
- Whether `qwen/qwen3-coder` can actually solve a task with Blackhole's output
- The SWE-bench test harness
- The OpenClaw plugin venv / dependencies in the real sandbox

Cost: free. Runs in <1 second.

## Tier 2 — R1 task-completion proxy (directional quality signal)

This is what you already ran. It tests the R1 interface (`main(task, ratio) → str`), but uses the R2 production model (`qwen/qwen3-coder`) and grades the agent's patch against the gold SWE-bench patch.

```bash
# Setup: miner/.env already has OPENROUTER_API_TOKEN. Just run:
python -m miner.run_task_completion_proxy miner.Blackhole.blackhole \
    --task-dir CoT-Compression-2 \
    --challenge-id 1533,1534,1535,1536,1537 \
    --ratios 0.10,0.20,0.30 \
    --prefer-provider openrouter --model "qwen/qwen3-coder" \
    --log miner/sample_results/blackhole_eval.jsonl
```

What this exercises:
- Compression quality (does the agent produce a correct-shaped patch?)
- Token savings vs original input
- Relative ranking against Apollo

What it does *not* test:
- The actual R2 `handle_assemble(payload) → dict` shape (it tests R1's `main` instead)
- The stateful per-turn assembly loop
- SWE-bench's actual test execution (uses LLM-judge of patch vs gold)
- The OpenClaw agent's tool-using behavior

Cost: ~$0.10–0.30 per challenge × ratio combo on qwen3-coder. The 5-challenge × 3-ratio batch we ran earlier was ~$1–2.

## Tier 3 — Full SOMA-benchmark (the real R2 eval)

This runs the **same code** the production validator runs. Requires Docker because OpenClaw and the SWE-bench eval harness are both containerized.

### Prerequisites

1. **Docker Desktop on Windows**: https://www.docker.com/products/docker-desktop/
   - WSL2 backend recommended
   - Allocate ≥16 GB RAM, ≥50 GB disk (SWE-bench eval images are large)
2. **uv** (Python package manager): `pip install --user uv` (already installed in your env)
3. **OpenRouter API key with credits** — same one you already have in `miner/.env`
4. **HF token (optional)**: makes SWE-bench dataset download faster. Get from https://huggingface.co/settings/tokens

### Setup

```bash
# 1) Clone the public Dendrite repos.
cd $HOME  # or wherever you want them
git clone https://github.com/DendriteHQ/SOMA-plugin.git
git clone https://github.com/DendriteHQ/SOMA-benchmark.git

# 2) Drop Blackhole in as the plugin's base_miner.py.
cp /e/Jobs/Soma/miner/Blackhole/blackhole.py $HOME/SOMA-plugin/base_miner.py

# 3) Set up the benchmark env.
cd $HOME/SOMA-benchmark
uv sync
cp .env.example .env
# Edit .env:
#   LLM_API_KEY=<your OpenRouter key>
#   LLM_BASE_URL=https://openrouter.ai/api/v1
#   LLM_MODEL=qwen/qwen3-coder
#   OPENROUTER_API_KEY=<your OpenRouter key>
#   OPENROUTER_MODEL=qwen/qwen3-coder
#   HF_TOKEN=<optional>

# 4) Pre-pull Docker images (the first run will do it anyway, but pre-pulling shows download progress).
docker pull alpine/openclaw:latest

# 5) Run a single SWE-bench Verified instance through the full pipeline.
uv run python -m soma_bench benchmark-solve \
    --agent-name openclaw \
    --benchmark SWE-bench/SWE-bench_Verified \
    --instance-id django__django-11095 \
    --execute \
    --openclaw-plugin-path $HOME/SOMA-plugin \
    --openclaw-plugin-reinstall-on-run-start \
    --openclaw-command "--timeout 180"

# Output appears in outputs/soma-bench-local/<run-id>/output.jsonl
# Look at the row for: resolved: true/false, total_tokens, agent_steps.
```

Cost per task:
- Disk: ~3–10 GB per SWE-bench Verified instance image (cached locally after first run)
- Time: 3–10 minutes per task wallclock
- API: $1–5 per task on qwen3-coder depending on session length

### What to test

Pick the 5 easiest Django tasks first (lowest agent_steps when run unaided). Those are the same shape as the live screener's "5 simplest tasks" set:

```bash
for id in django__django-11095 django__django-11133 django__django-11149 django__django-11179 django__django-11292; do
    uv run python -m soma_bench benchmark-solve \
        --agent-name openclaw \
        --benchmark SWE-bench/SWE-bench_Verified \
        --instance-id $id \
        --execute \
        --openclaw-plugin-path $HOME/SOMA-plugin \
        --openclaw-plugin-reinstall-on-run-start \
        --openclaw-command "--timeout 180"
done
```

Then read `outputs/soma-bench-local/*/output.jsonl` and count `resolved: true`. The threshold for screener qualification is **4 of 5** (80%). If Blackhole hits 4 or 5, that's a green light to submit. If it hits 0–3, the configurable knobs in `blackhole.py` are:

- `KEEP_TOOL_RESULT_COUNT` (currently 6) — bigger window keeps more recent state, less aggressive compression
- `MAX_TOOL_RESULT_CHARS` (currently 2000) — bigger means less aggressive tool_result trimming

### Disabling the plugin for a baseline comparison

To see how much value Blackhole adds vs raw OpenClaw, run the same instance without the plugin:

```bash
uv run python -m soma_bench benchmark-solve \
    --agent-name openclaw \
    --benchmark SWE-bench/SWE-bench_Verified \
    --instance-id django__django-11095 \
    --execute \
    --openclaw-disable-plugin
```

Compare `total_tokens` and `resolved` between the two runs. The Δ is your value-add.

---

## What I checked while writing this

- `docker` not installed on your machine; needed for Tier 3.
- `uv` installed via pip in your user site-packages (path: `C:\Users\aaa\AppData\Roaming\Python\Python313\Scripts\uv.exe`). May need PATH update.
- `openclaw` on PyPI is a *client SDK for a remote openclaw service*, NOT the local OpenClaw CLI. The real OpenClaw is the `alpine/openclaw:latest` Docker image used by SOMA-benchmark. Confirmed in `SOMA-benchmark/src/soma_bench/benchmark/backends/openclaw.py:35` (`OPENCLAW_GATEWAY_IMAGE = "alpine/openclaw:latest"`).
- `OPENCLAW_WORKSPACE_ERROR` in the same file states: *"OpenClaw backend currently supports only docker workspace execution."* So Tier 3 fundamentally requires Docker; there is no Docker-free path for the full pipeline.

If you install Docker Desktop, I can wrap the Tier 3 commands into a one-shot script in this directory. Just say the word.
