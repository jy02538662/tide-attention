# Tide Attention v0.1-preview

**Clear memory stays cheap. Conflicting memory makes the model think harder.**

Tide Attention is a CPU-friendly experimental controller for long-context memory conflicts. It switches between sparse and full/deep paths using defect signals from attention, phase interference, and memory contradictions.

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
python scripts/visual_demo.py
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
python scripts/qwen_needle_demo.py --dry-run --context-len 512
```

It shows the same story without downloading Qwen:

```text
Needle / clear memory          -> yang_sparse, no deep retrieve
Multi-Needle / conflict memory -> yin_full, deep retrieve
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

## Promo assets

To generate a browser-friendly promo page and poster:

```bash
python scripts/make_promo_video.py
```

This writes:

```text
assets/promo/tide_attention_promo.html
assets/promo/tide_attention_poster.svg
assets/promo/storyboard.md
```

Open the HTML file in a browser and screen-record it if you want an MP4 or GIF. The repository intentionally does not commit binary video files.

## Controlled claim-gate

```bash
python benchmarks/claim_eval.py --trials 6 --seq-len 256 --lengths 64,128,256
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
```

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

`transformers` is included for the next real-LM evaluation step, but the default demos do not download a model.

## Recommended release claim

Safe:

```text
Tide Attention is an experimental defect-gated long-context controller. In controlled CPU-friendly memory-conflict tasks, it stays sparse on clear memory and switches to full/deep retrieval when contradictions appear.
```

Not safe yet:

```text
Tide Attention beats GPT/Claude/Kimi.
```

To make stronger claims, run public frozen-model suites such as Needle-in-a-Haystack, RULER, LongBench, and Multi-Needle Conflict on the same backbone. See [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md).

## Optional: Qwen2.5-0.5B Needle demo

The repository includes a small real-LM harness for `Qwen/Qwen2.5-0.5B-Instruct`.

Dry run first, with no model download:

```bash
python scripts/qwen_needle_demo.py --dry-run --context-len 512
```

Real Qwen run, if you have enough RAM/VRAM and network access:

```bash
python scripts/qwen_needle_demo.py --context-len 512 --device cpu
```

If CUDA is available, the script uses it by default:

```bash
python scripts/qwen_needle_demo.py --context-len 1024
```

The demo runs two cases:

```text
Needle / clear memory          -> expected yang_sparse
Multi-Needle / conflict memory -> expected yin_full + deep retrieve
```

It writes `benchmarks/qwen_needle_demo.json`.

## License

This project is released for **non-commercial use only** under the PolyForm Noncommercial License 1.0.0. Commercial use, including use in paid products, hosted services, internal business workflows, or commercial research and development, is not permitted without a separate written license.

See [LICENSE](LICENSE).

## Next step

The next useful milestone is a fuller small real-LM benchmark:

```text
Qwen2.5-0.5B-Instruct + Needle / Multi-Needle Conflict
```

This keeps the project reproducible for people without expensive GPUs while making the result more credible than a synthetic-only benchmark.
