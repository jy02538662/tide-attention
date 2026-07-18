"""Attention kernels: full, condensate-sparse, and defect-gated Tide attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .condensate import select_condensate_indices
from .controller import BrainMode, TideController
from .defect import DefectDetector
from .phase import PhaseInterference


@dataclass
class TideAttentionOutput:
    context: torch.Tensor  # (B,H,Q,D) or (Q,D)
    probs: torch.Tensor
    mode: str
    mean_d: float
    mean_defect: float
    manifold_mass: float
    flops_estimate: float
    reason: str


def _ensure_bhqd(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0), True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"expected (Q,D)/(B,H,Q,D), got {tuple(x.shape)}")


def _causal_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
    qn, kn = scores.shape[-2], scores.shape[-1]
    causal = torch.triu(torch.ones(qn, kn, device=scores.device, dtype=torch.bool), diagonal=1)
    # For Q!=K lengths (memory concat), only mask within the self-attn prefix if square;
    # when kn != qn, apply causal only on the overlapping trailing region.
    if qn == kn:
        scores = scores.masked_fill(causal, float("-inf"))
    return scores


def full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, sq = _ensure_bhqd(q)
    k, _ = _ensure_bhqd(k)
    v, _ = _ensure_bhqd(v)
    scores = _causal_scores(q, k)
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, v)
    if sq:
        return ctx[0, 0], probs[0, 0]
    return ctx, probs


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: int = 64,
    topk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    q, sq = _ensure_bhqd(q)
    k, _ = _ensure_bhqd(k)
    v, _ = _ensure_bhqd(v)
    scores = _causal_scores(q, k)
    manifold = select_condensate_indices(scores, window=window, topk=topk, causal=(q.shape[-2] == k.shape[-2]))
    masked = scores.masked_fill(~manifold.mask, float("-inf"))
    probs = torch.softmax(masked, dim=-1)
    # Positions with all -inf (should not happen under causal) -> 0
    probs = torch.nan_to_num(probs, nan=0.0)
    ctx = torch.matmul(probs, v)
    mass = float(manifold.mass_on_manifold.mean().item()) if manifold.mass_on_manifold is not None else 0.0
    if sq:
        return ctx[0, 0], probs[0, 0], mass
    return ctx, probs, mass


def _estimate_flops(mode: BrainMode, q_len: int, k_len: int, window: int, topk: int) -> float:
    if mode in (BrainMode.YIN_FULL,):
        return float(q_len * k_len)
    # repair uses sparse path after recovery decision for this step
    sel = min(k_len, 1 + window + topk)
    return float(q_len * sel)


def tide_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    memory_keys: Optional[torch.Tensor] = None,
    controller: Optional[TideController] = None,
    detector: Optional[DefectDetector] = None,
    interferometer: Optional[PhaseInterference] = None,
    window: int = 64,
    topk: int = 32,
    memory_margin: float | None = None,
    memory_agreement: float | None = None,
    conflict_density: float | None = None,
) -> TideAttentionOutput:
    """Defect-gated attention: yang sparse vs yin full.

    Optional ``memory_keys`` (M,D) or (B,M,D) participate in phase interference
    against the mean query, modulating fringe clarity / defect density.
    """
    controller = controller or TideController()
    detector = detector or DefectDetector()
    interferometer = interferometer or PhaseInterference()

    q4, sq = _ensure_bhqd(q)
    k4, _ = _ensure_bhqd(k)
    v4, _ = _ensure_bhqd(v)
    scores = _causal_scores(q4, k4)
    causal = q4.shape[-2] == k4.shape[-2]
    manifold = select_condensate_indices(scores, window=window, topk=topk, causal=causal)

    fringe = None
    fringe_scalar = None
    if memory_keys is not None and memory_keys.numel() > 0:
        # Pool queries to (B,Q,D) from multi-head: average heads.
        q_pool = q4.mean(dim=1)  # (B,Q,D)
        mem = memory_keys
        if mem.ndim == 2:
            mem = mem.unsqueeze(0)
        report = interferometer.compare(q_pool, mem)
        fringe = report.fringe_clarity
        fringe_scalar = float(fringe[..., -1].mean().item())

    defect = detector.detect(scores, manifold, fringe_clarity=fringe, causal=causal)
    decision = controller.decide(
        defect,
        fringe_clarity=fringe_scalar,
        memory_margin=memory_margin,
        memory_agreement=memory_agreement,
        conflict_density=conflict_density,
    )

    if decision.mode == BrainMode.YIN_FULL:
        ctx, probs = full_attention(q4, k4, v4)
        mode_name = decision.mode.value
    else:
        # YANG_SPARSE and REPAIR both execute sparse condensate attention.
        masked = scores.masked_fill(~manifold.mask, float("-inf"))
        probs = torch.nan_to_num(torch.softmax(masked, dim=-1), nan=0.0)
        ctx = torch.matmul(probs, v4)
        mode_name = decision.mode.value

    mass = float(manifold.mass_on_manifold.mean().item()) if manifold.mass_on_manifold is not None else 0.0
    flops = _estimate_flops(decision.mode, q4.shape[-2], k4.shape[-2], window, topk)

    if sq:
        ctx = ctx[0, 0] if ctx.ndim == 4 else ctx
        probs = probs[0, 0] if probs.ndim == 4 else probs

    return TideAttentionOutput(
        context=ctx,
        probs=probs if isinstance(probs, torch.Tensor) else probs,
        mode=mode_name,
        mean_d=decision.mean_d,
        mean_defect=decision.mean_defect,
        manifold_mass=mass,
        flops_estimate=flops,
        reason=decision.reason,
    )
