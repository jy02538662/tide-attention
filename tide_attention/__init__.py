"""Tide Attention: yin-yang condensate attention with defect-driven switching.

Maps Quantum Tide / Cognitive Tide ideas onto long-context attention:

    yang (阳)  = attention mass condensed on the Condensate Manifold
    yin  (阴)  = off-manifold mass / phase distortion / defects
    D_proxy    = coherence / noise  (testable analogue of the 2.5 window)
    switch     = sparse (intuition) <-> full (logic) when defects exceed threshold
"""

from .condensate import CondensateManifold, select_condensate_indices
from .phase import PhaseInterference, InterferenceReport
from .defect import DefectDetector, DefectReport
from .controller import TideController, ControllerDecision, XI_C
from .memory import TideMemoryBank, MemoryHit, MemoryDiagnostics
from .attention import (
    full_attention,
    sparse_attention,
    tide_attention,
    TideAttentionOutput,
)
from .long_context import TideLongContextEngine, LongContextResult

__all__ = [
    "CondensateManifold",
    "select_condensate_indices",
    "PhaseInterference",
    "InterferenceReport",
    "DefectDetector",
    "DefectReport",
    "TideController",
    "ControllerDecision",
    "XI_C",
    "TideMemoryBank",
    "MemoryHit",
    "full_attention",
    "sparse_attention",
    "tide_attention",
    "TideAttentionOutput",
    "TideLongContextEngine",
    "LongContextResult",
]
