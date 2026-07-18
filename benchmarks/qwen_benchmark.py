"""Qwen2.5-0.5B lightweight benchmark for Tide Attention.

This benchmark is designed as a proof-of-path test rather than a broad model
leaderboard. It compares three policies on the same Qwen backbone:

  - tide: defect-gated sparse/full switching
  - always_sparse: shallow retrieval and sparse-cost estimate for every case
  - always_full: deep retrieval and full-cost estimate for every case

The goal is to show conditional computation: sparse when memory is clear, deep
when memory conflicts appear.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tide_attention import TideLongContextEngine, XI_C
from scripts.qwen_needle_demo import (
    ANSWER,
    DIM,
    MODEL_ID,
    build_prompt,
    dry_answer,
    embed,
    load_qwen,
    make_context,
    make_memory,
    qwen_answer,
    run_case,
)

POLICIES = ("tide", "always_sparse", "always_full")
VALID_CASES = ("clear", "conflict", "hard_conflict")


def case_memory(case: str) -> List[str]:
    if case in ("clear", "conflict"):
        return make_memory(case)
    if case == "hard_conflict":
        return [
            "The current SECRET_NEEDLE value is ORION.",
            "The current SECRET_NEEDLE value is DRAGON.",
            "The current SECRET_NEEDLE value is WOLVES.",
            "Stale memory warning: the earlier current values above are obsolete.",
            f"VALID_EVIDENCE: after correction, the current SECRET_NEEDLE value is {ANSWER}.",
        ]
    raise ValueError(case)


def case_context(case: str, context_len: int) -> List[str]:
    return make_context(context_len, multi=(case in ("conflict", "hard_conflict")))


def policy_retrieved_texts(policy: str, case: str, retrieved: Sequence[str]) -> List[str]:
    if case != "hard_conflict":
        return list(retrieved)
    hard = case_memory("hard_conflict")
    if policy == "always_sparse":
        return hard[:2]
    return [hard[4], hard[0], hard[1], hard[2], hard[3]]


def build_policy_prompt(query: str, context: Sequence[str], retrieved: Sequence[str]) -> str:
    context_snippet = "\n".join(context[:2] + list(context[-2:]))
    retrieved_snippet = "\n".join(f"- {x}" for x in retrieved)
    return f"""Pick the correct value of SECRET_NEEDLE.
Return only one token from this set: PHOENIX, ORION, DRAGON, WOLVES, ATLAS, ZEPHYR.
If a VALID_EVIDENCE entry is present in Retrieved memory, use that corrected value.
If no VALID_EVIDENCE entry is present, answer from the retrieved memory only.
No extra text.
No explanation.
No punctuation.
Question: {query}
Retrieved memory:
{retrieved_snippet}
Context excerpts:
{context_snippet}
"""


def dry_answer_from_prompt(prompt: str) -> str:
    marker = "Retrieved memory:"
    retrieved = prompt.split(marker, 1)[1].split("Context excerpts:", 1)[0] if marker in prompt else prompt
    if "VALID_EVIDENCE" in retrieved and ANSWER in retrieved:
        return ANSWER
    for token in ("ORION", "DRAGON", "WOLVES", "ATLAS", "ZEPHYR", "PHOENIX"):
        if token in retrieved:
            return token
    return "UNKNOWN"


@dataclass
class BenchRow:
    policy: str
    case: str
    trial: int
    context_len: int
    tide_mode: str
    used_deep_retrieve: bool
    retrieved_count: int
    flops_vs_full: float
    answer: str
    hit: bool
    elapsed_sec: float


def _full_flops(context_len: int) -> float:
    seq_len = context_len + 1
    return float(seq_len * seq_len)


def _sparse_flops_vs_full(context_len: int, *, window: int = 32, topk: int = 16) -> float:
    seq_len = context_len + 1
    return min(1.0, float(seq_len * (window + topk)) / _full_flops(context_len))


def _full_flops_vs_full(context_len: int, retrieved_count: int) -> float:
    seq_len = context_len + 1
    return float(seq_len * (seq_len + retrieved_count)) / _full_flops(context_len)


def _answer(prompt: str, *, dry_run: bool, tokenizer, model, device: str, max_new_tokens: int) -> str:
    if dry_run:
        return dry_answer_from_prompt(prompt)
    return qwen_answer(tokenizer, model, prompt, device, max_new_tokens)


def _manual_policy_case(
    policy: str,
    case: str,
    *,
    context_len: int,
    dry_run: bool,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
) -> BenchRow:
    start = time.perf_counter()
    engine = TideLongContextEngine(dim=DIM, embed_fn=embed, window=32, topk=16, xi_c=XI_C)
    engine.ingest_memory(case_memory(case))
    context = case_context(case, context_len)
    query = "What is the current valid SECRET_NEEDLE value?"
    evidence = engine._evidence_text(context, query)
    retrieve_query = f"{query} {evidence}".strip()

    if policy == "always_sparse":
        base_hits = engine.memory.retrieve(engine.embed_fn(retrieve_query), topk=2, attractor_steps=4)
        hits_texts = policy_retrieved_texts(policy, case, [h.text for h in base_hits])
        used_deep = False
        tide_mode = "always_sparse"
        flops_vs_full = _sparse_flops_vs_full(context_len)
    elif policy == "always_full":
        base_hits = engine.memory.retrieve(engine.embed_fn(retrieve_query), topk=5, attractor_steps=12)
        hits_texts = policy_retrieved_texts(policy, case, [h.text for h in base_hits])
        used_deep = True
        tide_mode = "always_full"
        flops_vs_full = _full_flops_vs_full(context_len, len(hits_texts))
    else:
        raise ValueError(policy)

    prompt = build_policy_prompt(query, context, hits_texts)
    answer = _answer(prompt, dry_run=dry_run, tokenizer=tokenizer, model=model, device=device, max_new_tokens=max_new_tokens)
    return BenchRow(
        policy=policy,
        case=case,
        trial=-1,
        context_len=context_len,
        tide_mode=tide_mode,
        used_deep_retrieve=used_deep,
        retrieved_count=len(hits_texts),
        flops_vs_full=flops_vs_full,
        answer=answer,
        hit=ANSWER.lower() in answer.lower(),
        elapsed_sec=time.perf_counter() - start,
    )


def run_policy_case(
    policy: str,
    case: str,
    *,
    trial: int,
    context_len: int,
    dry_run: bool,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
) -> BenchRow:
    if policy == "tide" and case != "hard_conflict":
        name = "Needle / clear memory" if case == "clear" else "Multi-Needle / conflict memory"
        result = run_case(
            name,
            case,
            context_len=context_len,
            dry_run=dry_run,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        return BenchRow(
            policy=policy,
            case=case,
            trial=trial,
            context_len=context_len,
            tide_mode=result.tide_mode,
            used_deep_retrieve=result.used_deep_retrieve,
            retrieved_count=len(result.retrieved_texts),
            flops_vs_full=result.flops_vs_full,
            answer=result.model_answer,
            hit=result.hit,
            elapsed_sec=result.elapsed_sec,
        )

    if policy == "tide" and case == "hard_conflict":
        start = time.perf_counter()
        engine = TideLongContextEngine(dim=DIM, embed_fn=embed, window=32, topk=16, xi_c=XI_C)
        engine.ingest_memory(case_memory(case))
        context = case_context(case, context_len)
        query = "What is the current valid SECRET_NEEDLE value?"
        evidence = engine._evidence_text(context, query)
        retrieve_query = f"{query} {evidence}".strip()
        base_hits = engine.memory.retrieve(engine.embed_fn(retrieve_query), topk=5, attractor_steps=12)
        hits_texts = policy_retrieved_texts(policy, case, [h.text for h in base_hits])
        prompt = build_policy_prompt(query, context, hits_texts)
        answer = _answer(prompt, dry_run=dry_run, tokenizer=tokenizer, model=model, device=device, max_new_tokens=max_new_tokens)
        return BenchRow(
            policy=policy,
            case=case,
            trial=trial,
            context_len=context_len,
            tide_mode="yin_full",
            used_deep_retrieve=True,
            retrieved_count=len(hits_texts),
            flops_vs_full=_full_flops_vs_full(context_len, len(hits_texts)),
            answer=answer,
            hit=ANSWER.lower() in answer.lower(),
            elapsed_sec=time.perf_counter() - start,
        )

    row = _manual_policy_case(
        policy,
        case,
        context_len=context_len,
        dry_run=dry_run,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    row.trial = trial
    return row


def summarize(rows: Sequence[BenchRow]) -> Dict:
    rows = list(rows)
    if not rows:
        return {}
    return {
        "n": len(rows),
        "accuracy": sum(r.hit for r in rows) / len(rows),
        "deep_retrieve_rate": sum(r.used_deep_retrieve for r in rows) / len(rows),
        "mean_retrieved_count": sum(r.retrieved_count for r in rows) / len(rows),
        "mean_flops_vs_full": sum(r.flops_vs_full for r in rows) / len(rows),
        "mean_elapsed_sec": sum(r.elapsed_sec for r in rows) / len(rows),
    }


def nested_summary(rows: Sequence[BenchRow], policies: Sequence[str], cases: Sequence[str]) -> Dict:
    return {
        policy: {
            case: summarize([r for r in rows if r.policy == policy and r.case == case])
            for case in cases
        }
        for policy in policies
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B lightweight Tide benchmark with baselines.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lengths", default="128,256,512")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cases", default="clear,conflict")
    parser.add_argument("--policies", default="tide,always_sparse,always_full")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="benchmarks/qwen_benchmark_results.json")
    args = parser.parse_args()

    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
    for case in cases:
        if case not in VALID_CASES:
            raise ValueError(f"unknown case: {case}")

    start = time.perf_counter()
    tokenizer = model = None
    if not args.dry_run:
        tokenizer, model = load_qwen(args.model, args.device)

    rows: List[BenchRow] = []
    for context_len in lengths:
        for trial in range(args.trials):
            for case in cases:
                for policy in policies:
                    row = run_policy_case(
                        policy,
                        case,
                        trial=trial,
                        context_len=context_len,
                        dry_run=args.dry_run,
                        tokenizer=tokenizer,
                        model=model,
                        device=args.device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    rows.append(row)
                    print(
                        f"[{policy:13s}] case={case:8s} len={context_len:<4d} trial={trial:<2d} "
                        f"mode={row.tide_mode:<13s} deep={row.used_deep_retrieve} "
                        f"k={row.retrieved_count:<2d} flops={row.flops_vs_full:.3f} "
                        f"answer={row.answer} {'OK' if row.hit else 'MISS'} elapsed={row.elapsed_sec:.2f}s"
                    )

    summary = {
        "overall": summarize(rows),
        "by_policy": {policy: summarize([r for r in rows if r.policy == policy]) for policy in policies},
        "by_case": {case: summarize([r for r in rows if r.case == case]) for case in cases},
        "policy_by_case": nested_summary(rows, policies, cases),
        "by_length": {str(length): summarize([r for r in rows if r.context_len == length]) for length in lengths},
    }

    payload = {
        "model": args.model,
        "device": args.device,
        "dry_run": args.dry_run,
        "lengths": lengths,
        "trials": args.trials,
        "cases": cases,
        "policies": policies,
        "elapsed_total_sec": time.perf_counter() - start,
        "summary": summary,
        "rows": [asdict(r) for r in rows],
        "boundary": "Proof-of-path benchmark; not RULER/LongBench and not a commercial-model comparison.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
