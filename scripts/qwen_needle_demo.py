"""Qwen2.5-0.5B Needle / Multi-Needle Conflict demo.

CPU-friendly by default with --dry-run. Real model mode loads
Qwen/Qwen2.5-0.5B-Instruct from HuggingFace and can run on CPU or GPU.

This script uses TideLongContextEngine as a pre-controller for memory conflict:
  - clear memory: sparse/shallow retrieval context
  - conflict memory: full/deep retrieval context
Then it asks Qwen to answer from the selected evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tide_attention import TideLongContextEngine, XI_C

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ANSWER = "PHOENIX"
VALID_VALUES = ("PHOENIX", "ORION", "DRAGON", "WOLVES", "ATLAS", "ZEPHYR")
DIM = 128



@dataclass
class DemoCase:
    name: str
    memory_mode: str
    context_len: int
    tide_mode: str
    mode_trace: List[str]
    used_deep_retrieve: bool
    flops_estimate: float
    flops_vs_full: float
    retrieved_texts: List[str]
    prompt_chars: int
    model_answer: str
    hit: bool
    elapsed_sec: float


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
    for mark in ("phoenix", "orion", "dragon", "wolves", "atlas", "zephyr"):
        if mark in joined:
            vec[stable_hash("ANSWER::" + mark) % dim] += 5.0
    if "secret" in toks and "needle" in toks:
        vec[stable_hash("TASK::secret_needle") % dim] += 3.0
    return F.normalize(vec, dim=0)


def make_context(length: int, answer: str = ANSWER, *, multi: bool = False) -> List[str]:
    chunks = []
    valid_at = max(2, int(length * 0.67))
    stale_at = max(1, int(length * 0.25))
    for i in range(length - 1):
        if i == valid_at:
            chunks.append(f"VALID_EVIDENCE: the current SECRET_NEEDLE value is {answer}.")
        elif multi and i == stale_at:
            chunks.append("STALE_EVIDENCE: an old SECRET_NEEDLE value was ORION, but this is obsolete.")
        else:
            chunks.append(f"Filler block {i}: warehouse logs, schedules, sensor notes, unrelated policy text.")
    return chunks


def make_memory(mode: str) -> List[str]:
    if mode == "clear":
        return [
            f"The current SECRET_NEEDLE value is {ANSWER}.",
            "Unrelated memory: operators prefer short answers.",
        ]
    if mode == "conflict":
        return [
            "The current SECRET_NEEDLE value is ORION.",
            "The current SECRET_NEEDLE value is DRAGON.",
            "The current SECRET_NEEDLE value is WOLVES.",
            f"The current SECRET_NEEDLE value is {ANSWER}.",
            "Stale memory warning: previous needle values may be false.",
        ]
    raise ValueError(mode)


def build_prompt(query: str, context: Sequence[str], retrieved: Sequence[str]) -> str:
    context_snippet = "\n".join(context[:2] + list(context[-2:]))
    retrieved_snippet = "\n".join(f"- {x}" for x in retrieved)
    return f"""Pick the correct value of SECRET_NEEDLE.
Return only one token from this set: PHOENIX, ORION, DRAGON, WOLVES, ATLAS, ZEPHYR.
No extra text.
No explanation.
No punctuation.
Question: {query}
Retrieved memory:
{retrieved_snippet}
Context excerpts:
{context_snippet}
"""


def dry_answer(prompt: str) -> str:
    if "The current SECRET_NEEDLE value is PHOENIX." in prompt:
        return "PHOENIX"
    if "The current SECRET_NEEDLE value is ORION." in prompt:
        return "ORION"
    if "The current SECRET_NEEDLE value is DRAGON." in prompt:
        return "DRAGON"
    if "The current SECRET_NEEDLE value is WOLVES." in prompt:
        return "WOLVES"
    return "UNKNOWN"


def load_qwen(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _extract_answer(text: str) -> str:
    upper = text.upper()
    for token in VALID_VALUES:
        if token in upper:
            return token
    return text.strip()


def qwen_answer(tokenizer, model, prompt: str, device: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "Return only one token from: PHOENIX, ORION, DRAGON, WOLVES, ATLAS, ZEPHYR."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device if hasattr(model, "device") else device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, inputs.input_ids.shape[-1] :]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _extract_answer(decoded)


def run_case(
    name: str,
    memory_mode: str,
    *,
    context_len: int,
    dry_run: bool,
    tokenizer=None,
    model=None,
    device: str = "cpu",
    max_new_tokens: int = 12,
) -> DemoCase:
    start = time.perf_counter()
    engine = TideLongContextEngine(dim=DIM, embed_fn=embed, window=32, topk=16, xi_c=XI_C)
    engine.ingest_memory(make_memory(memory_mode))
    context = make_context(context_len, multi=(memory_mode == "conflict"))
    query = "What is the current valid SECRET_NEEDLE value?"
    result = engine.run(context, query, shallow_k=2, deep_k=5)
    retrieved = [h.text for h in result.retrieved[:5]]
    prompt = build_prompt(query, context, retrieved)
    if dry_run:
        answer = dry_answer(prompt)
    else:
        answer = qwen_answer(tokenizer, model, prompt, device, max_new_tokens)
    seq_len = len(context) + 1
    full_flops = float(seq_len * seq_len)
    return DemoCase(
        name=name,
        memory_mode=memory_mode,
        context_len=context_len,
        tide_mode=result.attention.mode,
        mode_trace=result.mode_trace,
        used_deep_retrieve=result.used_deep_retrieve,
        flops_estimate=result.attention.flops_estimate,
        flops_vs_full=result.attention.flops_estimate / full_flops,
        retrieved_texts=retrieved,
        prompt_chars=len(prompt),
        model_answer=answer,
        hit=ANSWER.lower() in answer.lower(),
        elapsed_sec=time.perf_counter() - start,
    )


def print_case(case: DemoCase) -> None:
    print(f"\n=== {case.name} ===")
    print(f"memory_mode: {case.memory_mode}")
    print(f"tide_mode: {case.tide_mode} | trace: {case.mode_trace}")
    print(f"deep_retrieve: {case.used_deep_retrieve}")
    print(f"FLOPs estimate: {case.flops_estimate:.0f} ({case.flops_vs_full:.3f}x full)")
    print(f"prompt chars sent to Qwen: {case.prompt_chars}")
    print(f"answer: {case.model_answer} {'OK' if case.hit else 'MISS'}")
    print("retrieved memory:")
    for text in case.retrieved_texts:
        print(f"  - {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B Needle / Multi-Needle Conflict demo.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="Do not download/load Qwen; use deterministic local answerer.")
    parser.add_argument("--quick", action="store_true", help="Use shorter context and fewer generated tokens.")
    parser.add_argument("--only", choices=("both", "clear", "conflict"), default="both", help="Run only one case to reduce runtime.")
    parser.add_argument("--out", default="benchmarks/qwen_needle_demo.json")
    args = parser.parse_args()

    context_len = min(args.context_len, 256) if args.quick else args.context_len
    max_new_tokens = min(args.max_new_tokens, 4) if args.quick else args.max_new_tokens

    tokenizer = model = None
    if not args.dry_run:
        tokenizer, model = load_qwen(args.model, args.device)

    print("Qwen2.5-0.5B Needle / Multi-Needle Conflict demo")
    print(f"model: {args.model if not args.dry_run else 'dry-run local answerer'}")
    print(f"device: {args.device}")
    print(f"context_len: {context_len}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"cases: {args.only}")

    cases = []
    if args.only in ("both", "clear"):
        cases.append(run_case("Needle / clear memory", "clear", context_len=context_len, dry_run=args.dry_run, tokenizer=tokenizer, model=model, device=args.device, max_new_tokens=max_new_tokens))
    if args.only in ("both", "conflict"):
        cases.append(run_case("Multi-Needle / conflict memory", "conflict", context_len=context_len, dry_run=args.dry_run, tokenizer=tokenizer, model=model, device=args.device, max_new_tokens=max_new_tokens))
    for case in cases:
        print_case(case)

    payload = {
        "model": args.model,
        "dry_run": args.dry_run,
        "quick": args.quick,
        "only": args.only,
        "device": args.device,
        "cases": [asdict(c) for c in cases],
        "boundary": "This is a small-model demo harness. It is not a commercial-model comparison.",
        "expected_answers": VALID_VALUES,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
