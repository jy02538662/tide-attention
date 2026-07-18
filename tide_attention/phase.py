"""Phase interference between input queries and memory keys.

Engineering translation of:

    input phase × memory phase -> interference fringes
    clear fringes  -> yang / intuition
    distorted      -> yin / defect
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class InterferenceReport:
    """Quantified interference between query and memory fields."""

    phase_delta: torch.Tensor  # (B, Q, M) absolute phase difference in [0, pi]
    fringe_clarity: torch.Tensor  # (B, Q) higher = clearer yang interference
    alignment: torch.Tensor  # (B, Q, M) cosine alignment in [-1, 1]
    constructive_mass: torch.Tensor  # (B, Q) mass of constructive interference
    destructive_mass: torch.Tensor  # (B, Q) mass of destructive interference
    best_memory_idx: torch.Tensor  # (B, Q)


def _complex_phase(x: torch.Tensor) -> torch.Tensor:
    """Map real vectors to a phase angle via first two PCA-like projected dims.

    Uses a fixed random orthonormal projection for stability without fitting.
    """
    d = x.shape[-1]
    # Deterministic projection axes from a hashed seed of dimension.
    g = torch.Generator(device="cpu")
    g.manual_seed(int(d) * 9973 + 17)
    basis = torch.randn(d, 2, generator=g)
    basis = basis / (basis.norm(dim=0, keepdim=True) + 1e-8)
    basis = basis.to(device=x.device, dtype=x.dtype)
    proj = x @ basis  # (..., 2)
    return torch.atan2(proj[..., 1], proj[..., 0])  # (-pi, pi]


class PhaseInterference:
    """Compute phase difference and fringe clarity between Q and memory."""

    def __init__(
        self,
        *,
        constructive_threshold: float = 0.55,
        destructive_threshold: float = -0.15,
        temperature: float = 0.07,
    ) -> None:
        self.constructive_threshold = constructive_threshold
        self.destructive_threshold = destructive_threshold
        self.temperature = temperature

    def compare(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> InterferenceReport:
        """Compare query tokens against a memory bank.

        Args:
            query:  (B, Q, D) or (Q, D)
            memory: (B, M, D) or (M, D)
            memory_mask: optional (B, M) True = valid memory slot
        """
        if query.ndim == 2:
            query = query.unsqueeze(0)
        if memory.ndim == 2:
            memory = memory.unsqueeze(0)
        if memory.shape[0] == 1 and query.shape[0] > 1:
            memory = memory.expand(query.shape[0], -1, -1)

        q = F.normalize(query, dim=-1)
        m = F.normalize(memory, dim=-1)
        align = torch.einsum("bqd,bmd->bqm", q, m)  # cosine

        if memory_mask is not None:
            if memory_mask.ndim == 1:
                memory_mask = memory_mask.unsqueeze(0).expand(query.shape[0], -1)
            align = align.masked_fill(~memory_mask.unsqueeze(1), -1.0)

        q_phase = _complex_phase(q)
        m_phase = _complex_phase(m)
        # Circular absolute difference wrapped into [0, pi]
        raw = (q_phase.unsqueeze(-1) - m_phase.unsqueeze(1)).abs()
        phase_delta = torch.remainder(raw, 2 * torch.pi)
        phase_delta = torch.where(phase_delta > torch.pi, 2 * torch.pi - phase_delta, phase_delta)

        weights = torch.softmax(align / self.temperature, dim=-1)
        if memory_mask is not None:
            weights = weights * memory_mask.unsqueeze(1).float()
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        constructive = (align >= self.constructive_threshold).float() * weights
        destructive = (align <= self.destructive_threshold).float() * weights
        constructive_mass = constructive.sum(dim=-1)
        destructive_mass = destructive.sum(dim=-1)

        # Competing basins: many near-equal high alignments => ambiguous fringes (yin).
        top2 = torch.topk(align, k=min(2, align.shape[-1]), dim=-1).values
        if top2.shape[-1] == 2:
            competition = (1.0 - (top2[..., 0] - top2[..., 1]).clamp(0, 1)).clamp(0, 1)
        else:
            competition = torch.zeros_like(constructive_mass)

        # Fringe clarity: concentrated constructive interference with low phase spread.
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        max_ent = torch.log(torch.tensor(float(max(memory.shape[1], 1)), device=query.device))
        concentration = 1.0 - (entropy / (max_ent + 1e-8))
        mean_phase = (weights * phase_delta).sum(dim=-1)
        phase_clarity = 1.0 - (mean_phase / torch.pi)
        fringe = (
            0.35 * constructive_mass
            + 0.25 * concentration
            + 0.15 * phase_clarity
            + 0.10 * (1.0 - destructive_mass)
            - 0.35 * competition
        ).clamp(0, 1)

        best = align.argmax(dim=-1)
        return InterferenceReport(
            phase_delta=phase_delta,
            fringe_clarity=fringe,
            alignment=align,
            constructive_mass=constructive_mass,
            destructive_mass=destructive_mass,
            best_memory_idx=best,
        )
