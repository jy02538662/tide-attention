# Tide Attention Evaluation Protocol

## What this package proves today

This repository implements the **minimal complete architecture** of:

```text
Condensate Manifold (yang support)
  + phase interference (input × memory)
  + defect density (yin)
  + D_proxy = coherence / noise
  + 2.5-window sparse/full switching
  + attractor memory deep/shallow retrieve
```

Current benchmarks compare Tide against **controlled baselines** under the same encoder:

| Baseline | Meaning |
|---|---|
| `full` | O(n²) attention + shallow vector retrieve |
| `sparse` | Fixed condensate sparse (no defect switch) |
| `tide` | Defect-gated yang/yin switch + attractor memory |

They do **not** yet prove superiority over commercial systems (Kimi MoBA, Claude, GPT, Gemini long-context).

## What would count as "exceeding market architectures"

All of the following must pass. Missing any one means the claim is invalid.

### A. Capability parity or win on public long-context suites

Run the same frozen backbone (e.g. Qwen2.5-7B / Llama-3.1-8B) with:

1. Native full / FlashAttention
2. Fixed condensate sparse (or MoBA if available)
3. Tide defect-gated controller (this repo)

Suites:

```text
Needle-in-a-Haystack (1k–128k)
RULER
LongBench / LongBench-v2
InfiniteBench (subset)
Multi-needle / variable tracking
```

Pass rule:

```text
Tide accuracy >= best baseline - 0.5pp
AND Tide wins on at least one hard subset (multi-needle conflict / memory interference)
```

### B. Cost / latency advantage at equal accuracy

At matched accuracy:

```text
FLOPs_tide / FLOPs_full  < 0.25 at 32k+
OR latency_tide / latency_full < 0.5 on same GPU
```

Condensate Theorem's 159× numbers require their **commercial Triton kernel**. This repo's FLOPs estimate is algorithmic support size, not a production kernel claim.

### C. Ablations that isolate the theory

Must show each module contributes:

```text
no_phase_interference
no_defect_gate (always sparse)
no_defect_gate (always full)
no_attractor_memory (vector NN only)
xi_c sweep: {1.0, 1.5, 2.0, 2.5, 3.0, 4.0}
```

Pass rule for the 2.5 hypothesis:

```text
A stable threshold near 2.5 after normalization predicts switch quality
better than arbitrary thresholds on held-out tasks.
```

If the best threshold is unstable across tasks, keep `xi_c` as a tunable gate, not a universal constant.

### D. Memory interaction specific claim

For input↔memory understanding (this project's focus), evaluate:

```text
memory conflict accuracy
false memory rejection
attractor recovery under noise
deep-retrieve trigger precision/recall
```

Pass rule:

```text
Tide deep-retrieve triggers more often under conflict than under clear memory
AND conflict accuracy > fixed-sparse
AND average FLOPs < full
```

## Claim boundary language (external)

Safe:

```text
We implement a falsifiable coherence-over-noise controller that switches between
condensate-sparse and full attention using phase-interference defect metrics,
and evaluate it against full/sparse baselines.
```

Unsafe until A–D pass on public models:

```text
We exceed all market long-context architectures.
```

## Reproduction

```bash
cd tide_attention
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/demo.py
python benchmarks/run_benchmark.py --trials 16
python benchmarks/claim_eval.py --trials 12 --seq-len 256
```

`benchmarks/claim_eval.py` emits `benchmarks/claim_results.json` with gates A-D and a strict verdict. Unless real public frozen-LM suites are attached with `--public-lm`, the verdict intentionally remains `NO_MARKET_CLAIM`.

## Latest controlled baseline snapshot (hash embeddings, 2026-07-18)

These are **not** commercial-model results.

| Setting | full / sparse | tide |
|---|---|---|
| Conflict memory top-1 accuracy | 37.5% | **100%** |
| Clear short context (len=64) mode | full / fixed sparse | **yang_sparse** (FLOPs matched to sparse) |
| Clear vs conflict switch | n/a | clear→yang, conflict→yin+deep retrieve |

Optional next step (real LM):

```text
1. Clone condensate-theorem validation scripts for manifold mass checks
2. Hook TideController into a HuggingFace attention forward
3. Run Needle + RULER with fixed seed and publish tables
```
