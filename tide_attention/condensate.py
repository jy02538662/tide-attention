"""Condensate Manifold selection (reference implementation).

Inspired by the Condensate Theorem (Ruiz Williams, 2026):

    C_i = {anchor} ∪ {local window} ∪ {dynamic top-k}

This module only selects positions. It does not claim production Triton speedups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CondensateManifold:
    """Selected support for one or more queries."""

    indices: torch.Tensor  # (B, H, Q, K_sel) or (Q, K_sel)
    mask: torch.Tensor  # boolean mask over full key length
    n_anchor: int
    n_window: int
    n_topk: int
    mass_on_manifold: Optional[torch.Tensor] = None


def select_condensate_indices(
    scores: torch.Tensor,
    *,
    window: int = 64,
    topk: int = 32,
    include_anchor: bool = True,
    causal: bool = True,
) -> CondensateManifold:
    """Build condensate support from raw attention logits ``scores``.

    Args:
        scores: (B, H, Q, K) or (Q, K) attention logits before softmax.
        window: local window size ending at the query position.
        topk: dynamic top-k outside the forced set.
        include_anchor: always keep position 0 (attention sink / anchor).
        causal: if True, query i may only see keys j <= i.
    """
    squeezed = False
    if scores.ndim == 2:
        scores = scores.unsqueeze(0).unsqueeze(0)
        squeezed = True
    if scores.ndim != 4:
        raise ValueError(f"scores must be (B,H,Q,K) or (Q,K), got {tuple(scores.shape)}")

    b, h, q, k = scores.shape
    device = scores.device

    # Start from all-False mask; force anchor + window; add top-k of remainder.
    mask = torch.zeros(b, h, q, k, dtype=torch.bool, device=device)

    if include_anchor and k > 0:
        mask[..., 0] = True

    # Local window relative to each query position.
    q_idx = torch.arange(q, device=device).view(1, 1, q, 1)
    k_idx = torch.arange(k, device=device).view(1, 1, 1, k)
    if causal:
        in_window = (k_idx <= q_idx) & (k_idx > q_idx - window)
        causal_ok = k_idx <= q_idx
    else:
        in_window = (k_idx - q_idx).abs() <= window
        causal_ok = torch.ones_like(in_window)
    mask = mask | in_window

    # Dynamic top-k among remaining causal positions.
    fill = scores.masked_fill(mask | ~causal_ok, float("-inf"))
    actual_k = min(topk, k)
    if actual_k > 0:
        top_vals, top_idx = torch.topk(fill, k=actual_k, dim=-1)
        valid = top_vals > float("-inf")
        mask.scatter_(-1, top_idx, valid)

    # Enforce causality on final mask.
    mask = mask & causal_ok

    # Dense index tensor: gather selected positions (padded with -1).
    # For simplicity store the boolean mask; indices via nonzero per query is costly,
    # so we also materialize a packed index list of size n_sel = anchor+window+topk upper bound.
    n_sel = min(k, (1 if include_anchor else 0) + window + topk)
    # Re-rank selected scores to pack indices.
    packed_scores = scores.masked_fill(~mask, float("-inf"))
    packed_vals, packed_idx = torch.topk(packed_scores, k=n_sel, dim=-1)
    packed_idx = packed_idx.masked_fill(packed_vals == float("-inf"), -1)

    mass = None
    with torch.no_grad():
        probs = torch.softmax(scores.masked_fill(~causal_ok, float("-inf")), dim=-1)
        mass = (probs * mask.float()).sum(dim=-1)  # (B,H,Q)

    if squeezed:
        packed_idx = packed_idx[0, 0]
        mask = mask[0, 0]
        mass = mass[0, 0] if mass is not None else None

    return CondensateManifold(
        indices=packed_idx,
        mask=mask,
        n_anchor=1 if include_anchor else 0,
        n_window=window,
        n_topk=topk,
        mass_on_manifold=mass,
    )
