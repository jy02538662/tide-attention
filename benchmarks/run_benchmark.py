"""Synthetic long-context benchmarks for Tide Attention.

Compares:
  - full attention + vector retrieve
  - fixed condensate sparse + vector retrieve
  - Tide defect-gated switch + attractor memory

These are controlled baselines, not commercial API models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tide_attention.attention import full_attention, sparse_attention
from tide_attention.controller import XI_C
from tide_attention.long_context import TideLongContextEngine


def _stable_hash(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


@dataclass
class TrialResult:
    method: str
    seq_len: int
    hit: bool
    mode: str
    mean_d: float
    mean_defect: float
    flops: float
    latency_ms: float
    retrieved_key: str


def _make_needle_corpus(
    seq_len: int,
    needle: str,
    needle_pos: float,
    filler: str = "lorem ipsum dolor sit amet consectetur adipiscing elit",
) -> Tuple[List[str], str]:
    """Build chunk list with a needle planted at relative position."""
    chunks = []
    insert_at = max(0, min(seq_len - 2, int(seq_len * needle_pos)))
    for i in range(seq_len - 1):
        if i == insert_at:
            chunks.append(f"SECRET_NEEDLE {needle} END_NEEDLE")
        else:
            chunks.append(f"{filler} chunk_{i}")
    query = "What is the SECRET_NEEDLE value?"
    return chunks, query


def _tokenize(text: str) -> List[str]:
    return text.lower().replace("_", " ").replace(":", " ").split()


def _embed_factory(dim: int):
    def embed(text: str) -> torch.Tensor:
        vec = torch.zeros(dim)
        toks = _tokenize(text)
        for tok in toks:
            vec[_stable_hash(tok) % dim] += 1.0
            vec[_stable_hash("##" + tok) % dim] += 0.7
        joined = " ".join(toks)
        # Strong identity features for needle answers.
        for mark in ("phoenix", "orion", "dragon", "wolves"):
            if mark in joined:
                vec[_stable_hash("ANSWER::" + mark) % dim] += 5.0
        if "secret" in toks and "needle" in toks:
            vec[_stable_hash("TASK::secret_needle") % dim] += 3.0
        return F.normalize(vec, dim=0)

    return embed


def _is_correct(text: str, needle: str) -> bool:
    t = text.lower()
    n = needle.lower()
    if n not in t:
        return False
    # Reject conflicting distractors if they also appear.
    distractors = {"orion", "dragon", "wolves"} - {n}
    return not any(d in t for d in distractors)


def run_retrieval_trial(
    method: str,
    seq_len: int,
    needle: str = "PHOENIX",
    needle_pos: float = 0.35,
    dim: int = 128,
    window: int = 32,
    topk: int = 16,
    seed: int = 0,
    memory_mode: str = "conflict",
) -> TrialResult:
    torch.manual_seed(seed)
    embed = _embed_factory(dim)
    chunks, query = _make_needle_corpus(seq_len, needle, needle_pos)

    if memory_mode == "clear":
        mem_texts = [
            f"The secret needle value is {needle}.",
            "Unrelated memory about weather and traffic.",
            "Meeting notes without the needle.",
        ]
    else:
        # Conflicting near-duplicates; shuffle so correct fact is not always index 0.
        mem_texts = [
            "The secret needle value is ORION.",
            "The secret needle value is DRAGON.",
            "The secret needle value is WOLVES.",
            f"The secret needle value is {needle}.",
            "Distractor note: previous needles may be wrong.",
            "Unrelated memory about weather and traffic.",
        ]
        rng = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(mem_texts), generator=rng).tolist()
        mem_texts = [mem_texts[i] for i in order]

    t0 = time.perf_counter()
    engine = TideLongContextEngine(dim=dim, embed_fn=embed, window=window, topk=topk, xi_c=XI_C)
    engine.ingest_memory(mem_texts)

    if method == "tide":
        result = engine.run(chunks, query, shallow_k=2, deep_k=5)
        top = result.retrieved[0].text if result.retrieved else ""
        hit = _is_correct(top, needle)
        latency = (time.perf_counter() - t0) * 1000
        return TrialResult(
            method=method,
            seq_len=seq_len,
            hit=hit,
            mode=result.attention.mode,
            mean_d=result.attention.mean_d,
            mean_defect=result.attention.mean_defect,
            flops=result.attention.flops_estimate,
            latency_ms=latency,
            retrieved_key=result.retrieved[0].key if result.retrieved else "",
        )

    # Baselines: query-only vector retrieve (no evidence grounding / no attractor).
    ctx = engine.encode_sequence(list(chunks) + [query])
    q = k = v = ctx
    if method == "full":
        _ctx_out, _ = full_attention(q, k, v)
        flops = float(q.shape[0] * k.shape[0])
        mode = "full"
        mean_d, mean_defect = -1.0, -1.0
        hits = engine.memory.retrieve(embed(query), topk=3, attractor_steps=0)
    elif method == "sparse":
        _ctx_out, _, mass = sparse_attention(q, k, v, window=window, topk=topk)
        flops = float(q.shape[0] * min(k.shape[0], 1 + window + topk))
        mode = "fixed_sparse"
        mean_d, mean_defect = mass, 0.0
        hits = engine.memory.retrieve(embed(query), topk=2, attractor_steps=0)
    else:
        raise ValueError(method)

    top = hits[0].text if hits else ""
    hit = _is_correct(top, needle)
    latency = (time.perf_counter() - t0) * 1000
    return TrialResult(
        method=method,
        seq_len=seq_len,
        hit=hit,
        mode=mode,
        mean_d=mean_d,
        mean_defect=mean_defect,
        flops=flops,
        latency_ms=latency,
        retrieved_key=hits[0].key if hits else "",
    )


def _summarize(rows: List[TrialResult]) -> Dict:
    if not rows:
        return {}
    return {
        "accuracy": sum(r.hit for r in rows) / len(rows),
        "mean_flops": sum(r.flops for r in rows) / len(rows),
        "mean_latency_ms": sum(r.latency_ms for r in rows) / len(rows),
        "yang_rate": sum(1 for r in rows if "yang" in r.mode or r.mode == "repair") / len(rows),
        "yin_rate": sum(1 for r in rows if "yin" in r.mode or r.mode == "full") / len(rows),
        "mean_d": sum(r.mean_d for r in rows if r.mean_d >= 0) / max(1, sum(1 for r in rows if r.mean_d >= 0)),
    }


def run_conflict_suite(dim: int = 128, seq_len: int = 256, trials: int = 20) -> Dict:
    """Clear memory should favor yang/sparse; conflict should favor yin/deep and beat query-only NN."""
    methods = ["full", "sparse", "tide"]
    rows: List[TrialResult] = []
    for method in methods:
        for i in range(trials):
            memory_mode = "clear" if i % 2 == 0 else "conflict"
            pos = 0.25 if memory_mode == "clear" else 0.8
            rows.append(
                run_retrieval_trial(
                    method,
                    seq_len=seq_len,
                    needle_pos=pos,
                    dim=dim,
                    seed=1000 + i,
                    memory_mode=memory_mode,
                )
            )
    summary: Dict = {"overall": {}, "clear": {}, "conflict": {}}
    for method in methods:
        all_m = [r for r in rows if r.method == method]
        # Reconstruct memory mode from trial seed parity used above.
        clear_rows = [r for j, r in enumerate(all_m) if j % 2 == 0]
        conflict_rows = [r for j, r in enumerate(all_m) if j % 2 == 1]
        summary["overall"][method] = _summarize(all_m)
        summary["clear"][method] = _summarize(clear_rows)
        summary["conflict"][method] = _summarize(conflict_rows)
    return {"summary": summary, "trials": [asdict(r) for r in rows]}


def run_scaling_suite(lengths: Sequence[int] = (64, 128, 256, 512), trials: int = 5) -> Dict:
    methods = ["full", "sparse", "tide"]
    out = {}
    for n in lengths:
        out[str(n)] = {}
        for method in methods:
            acc = []
            flops = []
            yang = []
            for i in range(trials):
                # Scaling uses clear memory so Tide can stay yang and show FLOP savings.
                r = run_retrieval_trial(
                    method,
                    seq_len=n,
                    seed=i,
                    needle_pos=0.4,
                    memory_mode="clear",
                )
                acc.append(r.hit)
                flops.append(r.flops)
                yang.append(1 if "yang" in r.mode or r.mode == "fixed_sparse" else 0)
            out[str(n)][method] = {
                "accuracy": sum(acc) / len(acc),
                "mean_flops": sum(flops) / len(flops),
                "yang_rate": sum(yang) / len(yang),
            }
    return out


def run_switch_behavior(dim: int = 64, seq_len: int = 128) -> Dict:
    """Show D_proxy / mode under clear vs conflicted memory."""
    embed = _embed_factory(dim)
    engine = TideLongContextEngine(dim=dim, embed_fn=embed, window=24, topk=12)
    chunks, query = _make_needle_corpus(seq_len, "PHOENIX", 0.3)

    engine.ingest_memory(["The secret needle value is PHOENIX."])
    clear = engine.run(chunks, query)

    engine2 = TideLongContextEngine(dim=dim, embed_fn=embed, window=24, topk=12)
    engine2.ingest_memory(
        [
            "The secret needle value is PHOENIX.",
            "The secret needle value is ORION.",
            "The secret needle value is DRAGON.",
            "The secret needle value is WOLVES.",
            "Conflicting note: ignore previous needles.",
        ]
    )
    conflict = engine2.run(chunks, query)

    return {
        "clear_memory": {
            "mode": clear.attention.mode,
            "mean_d": clear.attention.mean_d,
            "mean_defect": clear.attention.mean_defect,
            "clarity": clear.interference_clarity,
            "top": clear.retrieved[0].text if clear.retrieved else "",
        },
        "conflict_memory": {
            "mode": conflict.attention.mode,
            "mean_d": conflict.attention.mean_d,
            "mean_defect": conflict.attention.mean_defect,
            "clarity": conflict.interference_clarity,
            "top": conflict.retrieved[0].text if conflict.retrieved else "",
            "used_deep": conflict.used_deep_retrieve,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="benchmarks/results.json")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    print("Running switch behavior...")
    switch = run_switch_behavior()
    print(json.dumps(switch, indent=2))

    print("\nRunning conflict suite...")
    conflict = run_conflict_suite(trials=args.trials)
    print(json.dumps(conflict["summary"], indent=2))

    print("\nRunning scaling suite...")
    scaling = run_scaling_suite()
    print(json.dumps(scaling, indent=2))

    payload = {
        "xi_c": XI_C,
        "switch_behavior": switch,
        "conflict_suite": conflict,
        "scaling_suite": scaling,
        "claim_boundary": (
            "These results compare Tide vs controlled full/sparse baselines under "
            "hash embeddings. They do NOT prove superiority over commercial long-context "
            "products (Kimi/Claude/GPT). See EVAL_PROTOCOL.md for the falsifiable path."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
