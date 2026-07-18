# Tide Attention v0.1-preview

**Clear memory stays cheap. Conflicting memory makes the model think harder.**

Tide Attention is a CPU-friendly proof-of-path for defect-gated long-context control. It is not positioned as a Transformer replacement. It demonstrates a different technical path: use defect signals from attention, phase interference, and memory contradictions to decide when a system can stay sparse and when it should switch to full/deep retrieval.

```text
clear memory      -> yang_sparse -> shallow retrieve -> low cost
conflicted memory -> yin_full    -> deep retrieve    -> conflict recovery
```

No NVIDIA GPU is required for the default demo. No model download. No API key.

> License: non-commercial use only. See [LICENSE](LICENSE).

## 10-second demo

```bash
cd tide_attention
pip install -r requirements.txt
py -3 scripts/visual_demo.py
```

What you should see:

```text
Clear memory:
  mode = yang_sparse
  FLOPs = 0.191x full
  answer = PHOENIX

Conflict memory:
  mode = yin_full
  deep_retrieve = True
  answer = PHOENIX
```

If you want the small-model harness later:

```bash
py -3 scripts/qwen_needle_demo.py --dry-run --context-len 512
```

It shows the same story without downloading Qwen:

```text
Needle / clear memory          -> yang_sparse, no deep retrieve
Multi-Needle / conflict memory -> yin_full, deep retrieve
```

For a true Qwen run, start with quick CPU settings:

```bash
py -3 scripts/qwen_needle_demo.py --quick --only clear --context-len 256 --device cpu
py -3 scripts/qwen_needle_demo.py --quick --only conflict --context-len 256 --device cpu
```

## Shareable one-liner

```text
Tide Attention is a defect-gated long-context controller: clear memory stays sparse, conflicting memory triggers full/deep retrieval.
```

## Why it is interesting

Most long-context systems face a trade-off:

| Mode | Strength | Weakness |
|---|---|---|
| full attention | robust | expensive |
| fixed sparse attention | cheap | can miss conflict / false memory |
| Tide Attention | sparse when clear, full/deep when contradicted | experimental controller, not a production kernel |

The core idea is to treat contradictions as **defects**. When memory is coherent, Tide stays on a condensate-sparse support. When memory contains mutually incompatible facts, Tide triggers the yin path: full attention + deeper attractor retrieval.

## Proof-of-path claim

Tide Attention is designed to prove a conditional-computation path:

```text
Do not always think cheaply.
Do not always think expensively.
Think harder only when memory defects appear.
```

This project therefore emphasizes path behavior:

```text
clear memory    -> sparse-like cost profile
conflict memory -> full/deep recovery profile
```

The current results should be read as evidence for this controller path, not as a claim that Tide beats commercial models.

## Controlled claim-gate

```bash
py -3 benchmarks/claim_eval.py --trials 6 --seq-len 256 --lengths 64,128,256
```

Latest local snapshot from the controlled offline harness:

| Gate | Result |
|---|---|
| Conflict accuracy win | pass: Tide 100% vs full/sparse 66.7% |
| Cost gate | pass: 0.191x full FLOPs on clear 256-token setting |
| Ablation gate | pass: conflict triggers deep path, clear stays sparse |
| Threshold gate | pass in local harness; 2.5 is acceptable but not uniquely optimal |

Important boundary:

```text
These are controlled offline results. They do not prove superiority over GPT, Claude, Gemini, or Kimi.
```

## Qwen2.5-0.5B real-model benchmark

The repository includes a small real-LM harness for `Qwen/Qwen2.5-0.5B-Instruct`.

Fast dry run, with no model download:

```bash
py -3 scripts/qwen_needle_demo.py --dry-run --quick --only clear --context-len 256
```

Fast real Qwen run on CPU:

```bash
py -3 scripts/qwen_needle_demo.py --quick --only clear --context-len 256 --device cpu
py -3 scripts/qwen_needle_demo.py --quick --only conflict --context-len 256 --device cpu
```

Latest real Qwen2.5-0.5B-Instruct CPU snapshot:

| Case | Tide mode | Deep retrieve | Answer | FLOPs vs full |
|---|---|---:|---|---:|
| Needle / clear memory | `yang_sparse` | no | `PHOENIX` | 0.191x |
| Multi-Needle / conflict memory | `yin_full` | yes | `PHOENIX` | 1.020x |

## Baseline comparison

The stronger benchmark compares three policies on the same Qwen2.5-0.5B backbone:

| Policy | Behavior |
|---|---|
| `tide` | defect-gated switching: sparse on clear memory, deep/full on conflict |
| `always_sparse` | shallow retrieval and sparse-cost estimate for every case |
| `always_full` | deep retrieval and full-cost estimate for every case |

The benchmark includes a `hard_conflict` case, where the shallow/sparse path only sees stale "current value" memories and the corrected `VALID_EVIDENCE` entry is reachable only through deep retrieval.

Run:

```bash
py -3 benchmarks/qwen_benchmark.py --trials 2 --lengths 512,1024,2048 --cases clear,conflict,hard_conflict --device cpu
```

Latest real Qwen2.5-0.5B CPU snapshot (`2 trials × 3 lengths × 3 cases × 3 policies = 54 generations`):

| Policy | Split | n | Accuracy | Deep retrieve | Mean FLOPs vs full |
|---|---|---:|---:|---:|---:|
| `tide` | clear | 6 | 100% | 0% | 0.056x |
| `tide` | conflict | 6 | 100% | 100% | 1.006x |
| `tide` | hard_conflict | 6 | 100% | 100% | 1.006x |
| `always_sparse` | clear | 6 | 100% | 0% | 0.055x |
| `always_sparse` | conflict | 6 | 100% | 0% | 0.055x |
| `always_sparse` | hard_conflict | 6 | **0%** | 0% | 0.055x |
| `always_full` | clear | 6 | 100% | 100% | 1.002x |
| `always_full` | conflict | 6 | 100% | 100% | 1.006x |
| `always_full` | hard_conflict | 6 | 100% | 100% | 1.006x |

Overall accuracy: `tide` 100%, `always_full` 100%, `always_sparse` 66.7%.

Interpretation:

```text
clear memory   -> Tide stays sparse (as cheap as always_sparse), correct
conflict       -> Tide goes deep (as capable as always_full), correct
hard_conflict  -> always_sparse is misled by stale memory and fails (0%),
                  Tide detects the defect, switches to deep retrieval, recovers
```

The key result is not raw accuracy on a toy task. It is that a fixed sparse policy fails exactly where defect-gated switching is needed, while Tide keeps the low clear-memory cost and still recovers under hard conflict.

FLOPs note: `flops_vs_full` values are analytic estimates for the controller path, not measured Qwen kernel FLOPs. On clear memory at 2048 tokens the Tide path estimate drops to about `0.024x` full, and rises to about `1.006x` only when deep retrieval is triggered.

This is a small-model demonstration, not a commercial-model comparison.

## Architecture

```text
Input chunks ──┐
               ├─ PhaseInterference (input × memory fringe clarity)
Memory bank ───┘
               │
        CondensateManifold = Anchor ∪ Window ∪ TopK
               │
        DefectDetector = off-manifold mass + entropy + fringe defect
               │
        MemoryDiagnostics = margin + agreement + conflict density
               │
        TideController = yang_score / D_effective gate
               │
     ┌─────────┴─────────┐
 clear / coherent    contradicted / noisy
 yang_sparse         yin_full + deep retrieve
```

## What is included

```text
tide_attention/
  condensate.py     # Anchor + Window + TopK sparse support
  phase.py          # input × memory interference metrics
  defect.py         # off-manifold / entropy / fringe defects
  controller.py     # yang/yin switching controller
  memory.py         # attractor memory + conflict diagnostics
  attention.py      # full / sparse / tide attention reference kernels
  long_context.py   # end-to-end input-memory interaction engine

scripts/
  visual_demo.py    # best first demo for GitHub readers
  demo.py           # compact diagnostic demo
  smoke_test.py     # quick correctness checks

benchmarks/
  run_benchmark.py  # controlled full/sparse/tide comparison
  claim_eval.py     # local claim-gate A-D evaluator
  qwen_benchmark.py # Qwen2.5-0.5B real-model benchmark with baselines
```

## Promo assets

To generate a browser-friendly promo page and poster:

```bash
py -3 scripts/make_promo_video.py
```

This writes:

```text
assets/promo/tide_attention_promo.html
assets/promo/tide_attention_poster.svg
assets/promo/storyboard.md
```

Open the HTML file in a browser and screen-record it if you want an MP4 or GIF. The repository intentionally does not commit binary video files.

## Install

```bash
pip install -r requirements.txt
```

Dependencies are intentionally light:

```text
torch
numpy
transformers
tqdm
```

`transformers` is required only for real Qwen runs; the default demos do not download a model.

## Recommended release claim

Safe:

```text
Tide Attention is an experimental proof-of-path for defect-gated long-context control. In CPU-friendly memory-conflict tasks and a Qwen2.5-0.5B harness, it stays sparse on clear memory and switches to full/deep retrieval when contradictions appear.
```

Not safe yet:

```text
Tide Attention beats GPT/Claude/Kimi.
Tide Attention is a production Transformer replacement.
```

To make stronger claims, run public frozen-model suites such as Needle-in-a-Haystack, RULER, LongBench, and Multi-Needle Conflict on the same backbone. See [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md).

## License

This project is released for **non-commercial use only** under the PolyForm Noncommercial License 1.0.0. Commercial use, including use in paid products, hosted services, internal business workflows, or commercial research and development, is not permitted without a separate written license.

See [LICENSE](LICENSE).

## Next step

The next useful milestone is a public benchmark subset:

```text
Needle-in-a-Haystack / Multi-Needle with more seeds and longer contexts
RULER subset if hardware budget allows
Qwen2.5-1.5B as the next real-model backbone
```

The current v0.1-preview intentionally stays CPU-friendly and reproducible.
