"""CPU-friendly visual demo for Tide Attention v0.1-preview.

Shows the core open-source story in one screen:
  - clear memory -> yang_sparse, shallow retrieve, sparse-level FLOPs
  - conflicting memory -> yin_full, deep retrieve, contradiction-aware recovery
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tide_attention import TideLongContextEngine, XI_C
from tide_attention.attention import full_attention, sparse_attention


DIM = 128
WINDOW = 32
TOPK = 16
ANSWER = "PHOENIX"


def stable_hash(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def tokenize(text: str) -> List[str]:
    return text.lower().replace("_", " ").replace("=", " ").replace(":", " ").split()


def embed(text: str, dim: int = DIM) -> torch.Tensor:
    vec = torch.zeros(dim)
    toks = tokenize(text)
    for tok in toks:
        vec[stable_hash(tok) % dim] += 1.0
        vec[stable_hash("##" + tok) % dim] += 0.7
    joined = " ".join(toks)
    for mark in ("phoenix", "orion", "dragon", "wolves"):
        if mark in joined:
            vec[stable_hash("ANSWER::" + mark) % dim] += 5.0
    if "secret" in toks and "needle" in toks:
        vec[stable_hash("TASK::secret_needle") % dim] += 3.0
    return F.normalize(vec, dim=0)


def make_context(length: int = 256) -> List[str]:
    chunks = []
    for i in range(length - 1):
        if i == int(length * 0.62):
            chunks.append(f"Evidence: the valid SECRET_NEEDLE value is {ANSWER}.")
        else:
            chunks.append(f"Background block {i}: logistics, sensors, schedules, unrelated archive text.")
    return chunks


def run_case(name: str, memories: Iterable[str], *, shallow_k: int, deep_k: int) -> dict:
    engine = TideLongContextEngine(dim=DIM, embed_fn=embed, window=WINDOW, topk=TOPK, xi_c=XI_C)
    engine.ingest_memory(list(memories))
    context = make_context()
    query = "What is the valid SECRET_NEEDLE value?"
    result = engine.run(context, query, shallow_k=shallow_k, deep_k=deep_k)
    top = result.retrieved[0].text if result.retrieved else ""
    seq_len = len(context) + 1
    full_flops = float(seq_len * seq_len)
    sparse_flops = float(seq_len * min(seq_len, 1 + WINDOW + TOPK))
    return {
        "name": name,
        "mode": result.attention.mode,
        "mode_trace": result.mode_trace,
        "used_deep": result.used_deep_retrieve,
        "d_proxy": result.attention.mean_d,
        "defect": result.attention.mean_defect,
        "flops": result.attention.flops_estimate,
        "full_flops": full_flops,
        "sparse_flops": sparse_flops,
        "flops_vs_full": result.attention.flops_estimate / full_flops,
        "answer": top,
        "hit": ANSWER.lower() in top.lower(),
        "notes": result.notes,
        "retrieved": [(h.score, h.text) for h in result.retrieved[:5]],
    }


def print_case(case: dict) -> None:
    print(f"\n=== {case['name']} ===")
    print(f"mode: {case['mode']} | trace: {case['mode_trace']}")
    print(f"deep_retrieve: {case['used_deep']}")
    print(f"D_proxy: {case['d_proxy']:.3f} | defect: {case['defect']:.3f}")
    print(f"FLOPs: {case['flops']:.0f} ({case['flops_vs_full']:.3f}x full)")
    print(f"answer: {case['answer']} {'OK' if case['hit'] else 'MISS'}")
    print(f"notes: {case['notes']}")
    print("top retrieved:")
    for score, text in case["retrieved"]:
        print(f"  [{score:.3f}] {text}")


def baseline_readout() -> None:
    context = make_context()
    qkv = torch.stack([embed(x) for x in context + ["What is the valid SECRET_NEEDLE value?"]], dim=0)
    _, _ = full_attention(qkv, qkv, qkv)
    _, _, mass = sparse_attention(qkv, qkv, qkv, window=WINDOW, topk=TOPK)
    seq_len = qkv.shape[0]
    full_flops = seq_len * seq_len
    sparse_flops = seq_len * min(seq_len, 1 + WINDOW + TOPK)
    print("\nBaselines on same synthetic context:")
    print(f"  full attention FLOPs:   {full_flops}")
    print(f"  fixed sparse FLOPs:     {sparse_flops} ({sparse_flops / full_flops:.3f}x full)")
    print(f"  sparse manifold mass:   {mass:.3f}")


def main() -> None:
    print("Tide Attention v0.1-preview")
    print("Defect-gated sparse/full switching for long-context memory conflicts.")
    print("Runs on CPU. No GPU, no model download, no API key.\n")

    clear = run_case(
        "Clear memory: stay cheap",
        [
            f"The valid SECRET_NEEDLE value is {ANSWER}.",
            "Unrelated memory about weather and meetings.",
        ],
        shallow_k=2,
        deep_k=5,
    )
    conflict = run_case(
        "Conflict memory: think harder",
        [
            "The valid SECRET_NEEDLE value is ORION.",
            "The valid SECRET_NEEDLE value is DRAGON.",
            "The valid SECRET_NEEDLE value is WOLVES.",
            f"The valid SECRET_NEEDLE value is {ANSWER}.",
            "Stale note: old needle values may be false.",
        ],
        shallow_k=2,
        deep_k=5,
    )

    print_case(clear)
    print_case(conflict)
    baseline_readout()

    print("\nTakeaway:")
    print("  clear    -> yang_sparse: sparse-level cost")
    print("  conflict -> yin_full: deep retrieval only when contradictions appear")
    print("\nBoundary:")
    print("  This is a CPU-friendly controlled demo, not a claim of beating GPT/Claude/Kimi.")


if __name__ == "__main__":
    main()
