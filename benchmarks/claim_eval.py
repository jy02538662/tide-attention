"""Claim-gate evaluation for Tide Attention.

Runs falsifiable gates from EVAL_PROTOCOL.md. Default is offline controlled
baselines; it must not be used to claim superiority over Kimi/Claude/GPT unless
real public frozen-LM suites are attached separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmark import run_conflict_suite, run_retrieval_trial, run_scaling_suite
from tide_attention.controller import XI_C


@dataclass
class Gate:
    name: str
    passed: bool
    score: float
    threshold: float
    details: Dict


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _variant(method: str, mode: str, seq_len: int, trials: int, xi_c: float | None = None) -> Dict:
    rows = []
    old = None
    old_rb = None
    if xi_c is not None:
        import tide_attention.controller as ctl
        import benchmarks.run_benchmark as rb
        old = ctl.XI_C
        old_rb = rb.XI_C
        ctl.XI_C = xi_c
        rb.XI_C = xi_c
    try:
        for i in range(trials):
            if method in ("full", "sparse", "tide"):
                rows.append(run_retrieval_trial(method, seq_len=seq_len, seed=7000 + i, memory_mode=mode))
            elif method == "always_sparse":
                rows.append(run_retrieval_trial("sparse", seq_len=seq_len, seed=7000 + i, memory_mode=mode))
            elif method == "always_full":
                rows.append(run_retrieval_trial("full", seq_len=seq_len, seed=7000 + i, memory_mode=mode))
            elif method == "no_attractor_memory":
                rows.append(run_retrieval_trial("full", seq_len=seq_len, seed=7000 + i, memory_mode=mode))
            else:
                raise ValueError(method)
    finally:
        if old is not None:
            import tide_attention.controller as ctl
            ctl.XI_C = old
            if old_rb is not None:
                import benchmarks.run_benchmark as rb
                rb.XI_C = old_rb
    return {
        "accuracy": _mean([1.0 if r.hit else 0.0 for r in rows]),
        "mean_flops": _mean([r.flops for r in rows]),
        "deep_rate": _mean([1.0 if ("yin" in r.mode or r.mode == "full") else 0.0 for r in rows]),
        "yang_rate": _mean([1.0 if ("yang" in r.mode or "sparse" in r.mode) else 0.0 for r in rows]),
        "modes": sorted(set(r.mode for r in rows)),
    }


def run_ablations(seq_len: int, trials: int) -> Dict:
    methods = ["tide", "always_sparse", "always_full", "no_attractor_memory"]
    out = {}
    for m in methods:
        out[m] = {
            "clear": _variant(m, "clear", seq_len, trials),
            "conflict": _variant(m, "conflict", seq_len, trials),
        }
    tide = out["tide"]
    out["no_phase_interference_proxy"] = {
        "clear": out["always_sparse"]["clear"],
        "conflict": out["always_sparse"]["conflict"],
        "note": "Proxy ablation: disabling phase defects degenerates to fixed sparse retrieval in this offline harness.",
    }
    out["summary"] = {
        "trigger_gap": tide["conflict"]["deep_rate"] - tide["clear"]["deep_rate"],
        "conflict_gain_vs_sparse": tide["conflict"]["accuracy"] - out["always_sparse"]["conflict"]["accuracy"],
        "conflict_gain_vs_nn": tide["conflict"]["accuracy"] - out["no_attractor_memory"]["conflict"]["accuracy"],
    }
    return out


def run_threshold_sweep(seq_len: int, trials: int) -> Dict:
    rows = []
    for xi in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        clear = _variant("tide", "clear", seq_len, trials, xi_c=xi)
        conflict = _variant("tide", "conflict", seq_len, trials, xi_c=xi)
        gap = conflict["deep_rate"] - clear["deep_rate"]
        score = 0.45 * conflict["accuracy"] + 0.25 * clear["accuracy"] + 0.30 * max(0.0, gap)
        rows.append({"xi_c": xi, "clear": clear, "conflict": conflict, "switch_gap": gap, "score": score})
    best = max(rows, key=lambda r: r["score"])
    return {"rows": rows, "best": best}


def gate_capability(conflict: Dict) -> Gate:
    c = conflict["summary"]["conflict"]
    tide, full, sparse = c["tide"], c["full"], c["sparse"]
    best = max(full["accuracy"], sparse["accuracy"])
    score = tide["accuracy"] - best
    return Gate("A_capability_conflict_win", score > 0 and tide["accuracy"] >= best - 0.005, score, 0.0, c)


def gate_cost(scaling: Dict) -> Gate:
    ratios = []
    for k, row in scaling.items():
        if int(k) >= 256:
            ratios.append(row["tide"]["mean_flops"] / max(1.0, row["full"]["mean_flops"]))
    ratio = _mean(ratios)
    return Gate("B_cost_advantage", ratio < 0.25, ratio, 0.25, {"ratios_256_plus": ratios})


def gate_ablation(ab: Dict) -> Gate:
    s = ab["summary"]
    passed = s["trigger_gap"] > 0.5 and max(s["conflict_gain_vs_sparse"], s["conflict_gain_vs_nn"]) > 0.1
    score = 0.5 * s["trigger_gap"] + 0.5 * max(s["conflict_gain_vs_sparse"], s["conflict_gain_vs_nn"])
    return Gate("C_ablation_isolates_modules", passed, score, 0.30, s)


def gate_threshold(sw: Dict) -> Gate:
    best = max(sw["rows"], key=lambda r: r["score"])
    near = [r for r in sw["rows"] if abs(r["xi_c"] - 2.5) <= 0.5]
    near_best = max(near, key=lambda r: r["score"])
    passed = near_best["score"] >= best["score"] - 0.03
    score_xi = near_best["xi_c"] if passed else best["xi_c"]
    return Gate("D_threshold_near_2_5", passed, score_xi, 2.5, {"best": best, "near_2_5_best": near_best})


def verdict(gates: List[Gate], public_lm: bool) -> str:
    if not public_lm:
        return "NO_MARKET_CLAIM: controlled offline gates only; run public frozen-LM Needle/RULER/LongBench before comparing to Kimi/Claude/GPT."
    if all(g.passed for g in gates):
        return "CONDITIONAL_MARKET_CLAIM_ALLOWED: all gates passed and public frozen-LM suites are attached."
    return "NO_MARKET_CLAIM: at least one required gate failed."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--out", default="benchmarks/claim_results.json")
    p.add_argument("--lengths", default="64,128,256,512", help="Comma-separated lengths for cost scaling.")
    p.add_argument("--public-lm", action="store_true")
    args = p.parse_args()

    t0 = time.perf_counter()
    lengths = tuple(int(x.strip()) for x in args.lengths.split(",") if x.strip())
    conflict = run_conflict_suite(seq_len=args.seq_len, trials=args.trials)
    scaling = run_scaling_suite(lengths=lengths, trials=max(2, args.trials // 4))
    ablations = run_ablations(args.seq_len, max(4, args.trials // 2))
    sweep = run_threshold_sweep(args.seq_len, max(4, args.trials // 2))
    gates = [gate_capability(conflict), gate_cost(scaling), gate_ablation(ablations), gate_threshold(sweep)]
    payload = {
        "protocol": "EVAL_PROTOCOL.md gates A-D",
        "public_frozen_lm_suites_run": bool(args.public_lm),
        "xi_c_default": XI_C,
        "elapsed_sec": time.perf_counter() - t0,
        "gates": [asdict(g) for g in gates],
        "verdict": verdict(gates, args.public_lm),
        "conflict_suite": conflict["summary"],
        "scaling_suite": scaling,
        "ablations": ablations,
        "threshold_sweep": sweep,
        "claim_boundary": "Synthetic/offline results cannot establish superiority over closed commercial long-context systems.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"gates": payload["gates"], "verdict": payload["verdict"]}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
