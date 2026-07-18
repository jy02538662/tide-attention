"""Smoke tests for Tide Attention core modules."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tide_attention import (
    CondensateManifold,
    PhaseInterference,
    DefectDetector,
    TideController,
    select_condensate_indices,
    tide_attention,
    TideLongContextEngine,
    XI_C,
)
from tide_attention.condensate import select_condensate_indices as sel
from tide_attention.controller import BrainMode


def test_condensate_mask_causal():
    scores = torch.randn(8, 8)
    man = sel(scores, window=3, topk=2, causal=True)
    assert isinstance(man, CondensateManifold)
    assert man.mask.shape == (8, 8)
    # No future positions.
    for i in range(8):
        assert not man.mask[i, i + 1 :].any()


def test_phase_and_controller_switch():
    inter = PhaseInterference()
    q = torch.randn(1, 4, 64)
    m_clear = q[:, :2, :].clone()
    clear = inter.compare(q, m_clear)
    m_bad = torch.randn(1, 6, 64)
    bad = inter.compare(q, m_bad)
    assert float(clear.fringe_clarity.mean()) >= float(bad.fringe_clarity.mean()) - 1e-5

    ctrl = TideController(xi_c=XI_C)
    qkv = torch.randn(32, 64)
    out_clear = tide_attention(qkv, qkv, qkv, memory_keys=m_clear[0], controller=ctrl, window=8, topk=4)
    ctrl2 = TideController(xi_c=XI_C)
    out_bad = tide_attention(qkv, qkv, qkv, memory_keys=m_bad[0], controller=ctrl2, window=8, topk=4)
    assert out_clear.mean_d >= 0
    assert out_bad.mean_d >= 0
    assert out_clear.mode in {BrainMode.YANG_SPARSE.value, BrainMode.REPAIR.value, BrainMode.YIN_FULL.value}


def test_long_context_phoenix():
    engine = TideLongContextEngine(dim=64, window=16, topk=8)
    engine.ingest_memory(
        [
            "The secret project codename is PHOENIX.",
            "Conflicting rumor: codename is ORION.",
        ]
    )
    chunks = ["filler text about logistics"] * 30 + ["SECRET_NEEDLE PHOENIX END_NEEDLE"]
    result = engine.run(chunks, "What is the secret project codename?")
    assert result.retrieved
    assert "phoenix" in result.retrieved[0].text.lower()


if __name__ == "__main__":
    test_condensate_mask_causal()
    test_phase_and_controller_switch()
    test_long_context_phoenix()
    print("All smoke tests passed.")
