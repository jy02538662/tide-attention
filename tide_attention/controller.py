"""Tide controller: 2.5-window yin/yang brain switching.

D_proxy = coherence / noise
mode = sparse (yang/intuition) when D >= xi_c else full (yin/logic)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from .defect import DefectReport

# Proposed critical constant from Quantum Tide papers — treated as a testable threshold.
XI_C = 2.5


class BrainMode(str, Enum):
    YANG_SPARSE = "yang_sparse"  # intuition: stay on condensate manifold
    YIN_FULL = "yin_full"  # logic: expand to full attention / deep retrieve
    REPAIR = "repair"  # transitional recovery morphism


@dataclass
class ControllerDecision:
    mode: BrainMode
    d_proxy: torch.Tensor  # (B,H,Q) or reduced
    mean_d: float
    mean_defect: float
    threshold: float
    reason: str


class TideController:
    """Defect-driven sparse/full attention switcher."""

    def __init__(
        self,
        *,
        xi_c: float = XI_C,
        hysteresis: float = 0.20,
        epsilon: float = 1e-3,
        d_gain: float = 6.5,
        defect_force_full: float = 0.78,
    ) -> None:
        # d_gain maps raw coherence/noise into the same numeric window as xi_c=2.5.
        # Without gain, typical raw ratios sit near ~1 and never reach the conjectured threshold.
        self.xi_c = xi_c
        self.hysteresis = hysteresis
        self.epsilon = epsilon
        self.defect_force_full = defect_force_full
        self.d_gain = d_gain
        self._last_mode: Optional[BrainMode] = None

    def decide(
        self,
        defect: DefectReport,
        *,
        focus_last_query: bool = True,
        fringe_clarity: float | None = None,
        memory_margin: float | None = None,
        memory_agreement: float | None = None,
        conflict_density: float | None = None,
    ) -> ControllerDecision:
        raw = defect.coherence / (defect.noise + self.epsilon)
        d = raw * self.d_gain

        # Long contexts have many filler tokens; decide on the query position
        # (last along Q), matching input↔memory interference at the question.
        if focus_last_query and d.ndim >= 1:
            d_focus = d[..., -1]
            defect_focus = defect.defect_density[..., -1]
        else:
            d_focus = d
            defect_focus = defect.defect_density

        _ = float(d_focus.mean().item())
        mean_defect = float(defect_focus.mean().item())
        entropy = defect.entropy_norm[..., -1].mean() if focus_last_query and defect.entropy_norm.ndim >= 1 else defect.entropy_norm.mean()
        entropy_scalar = float(entropy.item())
        fringe = 0.5 if fringe_clarity is None else float(fringe_clarity)
        margin = 0.0 if memory_margin is None else float(memory_margin)
        agreement = 0.5 if memory_agreement is None else float(memory_agreement)
        conflict = 0.0 if conflict_density is None else float(conflict_density)

        yang_score = (
            0.24 * (1.0 - mean_defect)
            + 0.22 * agreement
            + 0.18 * min(1.0, margin * 3.0)
            + 0.18 * fringe
            + 0.18 * (1.0 - entropy_scalar)
            - 0.35 * conflict
        )
        yang_score = max(0.0, min(1.0, yang_score))
        mean_d = self.xi_c * (yang_score / 0.62)

        clear_memory = agreement >= 0.70 and margin >= 0.03 and conflict <= 0.24
        strong_conflict = conflict >= 0.36 and agreement < 0.86

        # Hysteresis avoids chatter between brains.
        enter_yang = self.xi_c
        leave_yang = self.xi_c - self.hysteresis

        if clear_memory and mean_defect < 0.92:
            mode = BrainMode.YANG_SPARSE
            reason = (
                f"clear_memory: agreement={agreement:.3f}, margin={margin:.3f}, "
                f"conflict={conflict:.3f}, D={mean_d:.3f}"
            )
        elif strong_conflict:
            mode = BrainMode.YIN_FULL
            reason = (
                f"memory_conflict: conflict={conflict:.3f}, agreement={agreement:.3f}, "
                f"D={mean_d:.3f}"
            )
        elif mean_defect >= self.defect_force_full and mean_d < enter_yang:
            mode = BrainMode.YIN_FULL
            reason = (
                f"defect_density={mean_defect:.3f} >= force_full={self.defect_force_full} "
                f"and D={mean_d:.3f} < xi_c"
            )
        elif self._last_mode == BrainMode.YANG_SPARSE and mean_d >= leave_yang:
            mode = BrainMode.YANG_SPARSE
            reason = f"hysteresis hold yang: D={mean_d:.3f} >= leave_yang={leave_yang:.3f}"
        elif mean_d >= enter_yang:
            mode = BrainMode.YANG_SPARSE
            reason = f"D={mean_d:.3f} >= xi_c={enter_yang:.3f} (yang / intuition)"
        else:
            mode = BrainMode.YIN_FULL
            reason = f"D={mean_d:.3f} < xi_c={enter_yang:.3f} (yin / logic)"

        # Optional repair label when recovering toward yang.
        if (
            self._last_mode == BrainMode.YIN_FULL
            and mode == BrainMode.YANG_SPARSE
        ):
            mode = BrainMode.REPAIR
            reason = "defect repaired -> returning to yang sparse"

        self._last_mode = (
            BrainMode.YANG_SPARSE if mode == BrainMode.REPAIR else mode
        )
        return ControllerDecision(
            mode=mode,
            d_proxy=d,
            mean_d=mean_d,
            mean_defect=mean_defect,
            threshold=self.xi_c,
            reason=reason,
        )
