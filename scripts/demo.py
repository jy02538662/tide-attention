"""Quick demo: yin/yang defect-driven long-context memory interaction."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tide_attention import TideLongContextEngine, XI_C


def _stable_hash(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def embed(text: str, dim: int = 96) -> torch.Tensor:
    vec = torch.zeros(dim)
    toks = text.lower().replace("_", " ").split()
    for tok in toks:
        vec[_stable_hash(tok) % dim] += 1.0
        vec[_stable_hash(tok[::-1]) % dim] += 0.5
    joined = " ".join(toks)
    for mark in ("phoenix", "orion", "dragon"):
        if mark in joined:
            vec[_stable_hash("ANSWER::" + mark) % dim] += 5.0
    if "secret" in toks and ("needle" in toks or "codename" in toks or "project" in toks):
        vec[_stable_hash("TASK::secret_needle") % dim] += 3.0
    return F.normalize(vec, dim=0)


def main() -> None:
    dim = 96
    engine = TideLongContextEngine(
        dim=dim,
        embed_fn=lambda t: embed(t, dim),
        window=32,
        topk=16,
        xi_c=XI_C,
    )
    engine.ingest_memory(
        [
            "User preference: prefer topological explanations.",
            "The secret project codename is PHOENIX.",
            "ARX5 button sequence is pink blue green yellow.",
            "Conflicting rumor: codename is ORION.",
        ]
    )

    context = [
        "Long filler context about warehouse logistics and sensor noise.",
        "More filler discussing unrelated scheduling conflicts.",
        "Another paragraph of distractor tokens for long context pressure.",
        "Operator asks about the secret project codename in the archive.",
    ] * 20  # ~80 chunks

    query = "What is the secret project codename?"
    result = engine.run(context, query, shallow_k=2, deep_k=6)

    print("=== Tide Long-Context Demo ===")
    print(f"xi_c threshold: {XI_C}")
    print(f"mode: {result.attention.mode}")
    print(f"D_proxy: {result.attention.mean_d:.4f}")
    print(f"defect: {result.attention.mean_defect:.4f}")
    print(f"manifold_mass: {result.attention.manifold_mass:.4f}")
    print(f"interference_clarity: {result.interference_clarity:.4f}")
    print(f"mode_trace: {result.mode_trace}")
    print(f"used_deep_retrieve: {result.used_deep_retrieve}")
    print(f"notes: {result.notes}")
    print("retrieved:")
    for h in result.retrieved:
        print(f"  [{h.score:.3f}] {h.key}: {h.text}")


if __name__ == "__main__":
    main()
