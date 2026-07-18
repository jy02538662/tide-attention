"""Defect detection on / off the Condensate Manifold.

Defects (阴) are quantified as:

    off-manifold attention mass
    attention entropy inflation
    low fringe clarity / high phase divergence
    rank dispersion of the attention pattern
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .condensate import CondensateManifold


@dataclass
class DefectReport:
    off_manifold_mass: torch.Tensor  # (B,H,Q) or (Q,)
    entropy: torch.Tensor
    entropy_norm: torch.Tensor
    fringe_defect: torch.Tensor  # 1 - fringe_clarity, broadcastable
    defect_density: torch.Tensor  # combined yin score in [0, 1]
    coherence: torch.Tensor  # yang score in [0, 1]
    noise: torch.Tensor


class DefectDetector:
    def __init__(
        self,
        *,
        w_off: float = 0.40,
        w_entropy: float = 0.25,
        w_fringe: float = 0.35,
    ) -> None:
        self.w_off = w_off
        self.w_entropy = w_entropy
        self.w_fringe = w_fringe

    def detect(
        self,
        scores: torch.Tensor,
        manifold: CondensateManifold,
        *,
        fringe_clarity: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> DefectReport:
        """Compute defect density from attention logits and manifold mask.

        Args:
            scores: (B,H,Q,K) or (Q,K)
            manifold: condensate selection for the same scores
            fringe_clarity: optional (B,Q) or (Q,) from PhaseInterference
        """
        squeezed = False
        if scores.ndim == 2:
            scores = scores.unsqueeze(0).unsqueeze(0)
            squeezed = True
        mask = manifold.mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        b, h, q, k = scores.shape
        device = scores.device
        k_idx = torch.arange(k, device=device).view(1, 1, 1, k)
        q_idx = torch.arange(q, device=device).view(1, 1, q, 1)
        causal_ok = k_idx <= q_idx if causal else torch.ones(b, h, q, k, dtype=torch.bool, device=device)

        logits = scores.masked_fill(~causal_ok, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        on = (probs * mask.float()).sum(dim=-1)
        off = (1.0 - on).clamp(0, 1)

        ent = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
        # Normalize by causal support size.
        support = causal_ok.float().sum(dim=-1).clamp_min(1.0)
        ent_norm = (ent / support.log().clamp_min(1e-8)).clamp(0, 1)

        if fringe_clarity is None:
            fringe_def = torch.zeros(b, q, device=device, dtype=scores.dtype)
        else:
            fc = fringe_clarity
            if fc.ndim == 1:
                fc = fc.unsqueeze(0)
            fringe_def = (1.0 - fc).clamp(0, 1)
        # Broadcast fringe defect over heads: (B,H,Q)
        fringe_bhq = fringe_def.unsqueeze(1).expand(b, h, q)

        coherence = (0.55 * on + 0.45 * (1.0 - fringe_bhq)).clamp(0, 1)
        noise = (0.50 * off + 0.30 * ent_norm + 0.20 * fringe_bhq).clamp(0, 1)
        defect = (
            self.w_off * off
            + self.w_entropy * ent_norm
            + self.w_fringe * fringe_bhq
        ).clamp(0, 1)

        if squeezed:
            return DefectReport(
                off_manifold_mass=off[0, 0],
                entropy=ent[0, 0],
                entropy_norm=ent_norm[0, 0],
                fringe_defect=fringe_def[0],
                defect_density=defect[0, 0],
                coherence=coherence[0, 0],
                noise=noise[0, 0],
            )
        return DefectReport(
            off_manifold_mass=off,
            entropy=ent,
            entropy_norm=ent_norm,
            fringe_defect=fringe_def,
            defect_density=defect,
            coherence=coherence,
            noise=noise,
        )
