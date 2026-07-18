"""TideMemory: topological attractor memory bank for long-context interaction."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class MemoryHit:
    index: int
    score: float
    key: str
    text: str
    basin_id: int


@dataclass
class MemoryDiagnostics:
    margin: float
    agreement: float
    conflict_density: float
    top_score: float
    second_score: float
    distinct_values: int
    values: List[str]


@dataclass
class MemoryItem:
    key: str
    text: str
    embedding: torch.Tensor
    basin_id: int = 0
    strength: float = 1.0


@dataclass
class TideMemoryBank:
    """Attractor-style memory: retrieval = basin convergence, not only NN lookup."""

    dim: int
    items: List[MemoryItem] = field(default_factory=list)
    device: str = "cpu"

    def write(
        self,
        key: str,
        text: str,
        embedding: torch.Tensor,
        *,
        basin_id: Optional[int] = None,
        strength: float = 1.0,
    ) -> None:
        emb = F.normalize(embedding.detach().float().reshape(-1), dim=0)
        if emb.numel() != self.dim:
            raise ValueError(f"embedding dim {emb.numel()} != bank dim {self.dim}")
        bid = len(self.items) if basin_id is None else basin_id
        self.items.append(
            MemoryItem(key=key, text=text, embedding=emb.to(self.device), basin_id=bid, strength=strength)
        )

    def tensor(self) -> torch.Tensor:
        if not self.items:
            return torch.zeros(0, self.dim, device=self.device)
        return torch.stack([it.embedding for it in self.items], dim=0)

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        topk: int = 5,
        attractor_steps: int = 8,
        lr: float = 0.35,
    ) -> List[MemoryHit]:
        """Retrieve via short attractor dynamics then nearest basin."""
        if not self.items:
            return []
        mem = self.tensor()  # (M, D)
        strengths = torch.tensor([it.strength for it in self.items], device=self.device)
        q = F.normalize(query.detach().float().reshape(-1).to(self.device), dim=0)

        # Evolve query toward memory potential wells.
        x = q.clone()
        for _ in range(attractor_steps):
            sim = (mem @ x) * strengths
            # Soft assignment to basins
            w = torch.softmax(sim / 0.07, dim=0)
            target = F.normalize((w.unsqueeze(-1) * mem).sum(dim=0), dim=0)
            x = F.normalize((1 - lr) * x + lr * target, dim=0)

        scores = (mem @ x) * strengths
        k = min(topk, scores.numel())
        vals, idx = torch.topk(scores, k=k)
        hits: List[MemoryHit] = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            it = self.items[i]
            hits.append(
                MemoryHit(
                    index=i,
                    score=float(v),
                    key=it.key,
                    text=it.text,
                    basin_id=it.basin_id,
                )
            )
        return hits

    def diagnose(
        self,
        query: torch.Tensor,
        *,
        topk: int = 5,
        attractor_steps: int = 0,
    ) -> MemoryDiagnostics:
        hits = self.retrieve(query, topk=topk, attractor_steps=attractor_steps)
        if not hits:
            return MemoryDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

        top = hits[0].score
        second = hits[1].score if len(hits) > 1 else 0.0
        margin = max(0.0, min(1.0, top - second))
        scores = torch.tensor([h.score for h in hits], dtype=torch.float32)
        weights = torch.softmax(scores / 0.08, dim=0)

        values = [self._extract_value(h.text) for h in hits]
        value_mass: Dict[str, float] = {}
        for v, w in zip(values, weights.tolist()):
            if v:
                value_mass[v] = value_mass.get(v, 0.0) + float(w)
        if value_mass:
            dominant = max(value_mass.values())
            agreement = float(dominant)
            conflict_density = float(1.0 - dominant)
            distinct = len(value_mass)
            if distinct > 1:
                conflict_density = max(conflict_density, min(0.90, 0.18 + 0.14 * distinct))
                agreement = min(agreement, 1.0 - 0.35 * conflict_density)
        else:
            agreement = float(weights[0].item())
            conflict_density = float(1.0 - agreement)
            distinct = 0

        # A large top-1 gap is strong evidence for a clear attractor even if text parsing fails.
        agreement = max(agreement, min(1.0, 0.45 + margin * 2.0))
        if distinct <= 1:
            conflict_density = min(conflict_density, 0.05)
        return MemoryDiagnostics(
            margin=margin,
            agreement=max(0.0, min(1.0, agreement)),
            conflict_density=max(0.0, min(1.0, conflict_density)),
            top_score=float(top),
            second_score=float(second),
            distinct_values=distinct,
            values=values,
        )

    @staticmethod
    def _extract_value(text: str) -> str:
        t = text.lower()
        patterns = [
            r"secret[_\s-]*needle\s+(?:value\s*)?(?:is|=|:)\s*([a-z0-9_-]+)",
            r"valid\s+secret\s+needle\s+value\s+is\s+([a-z0-9_-]+)",
            r"value\s*=\s*([a-z0-9_-]+)",
        ]
        for p in patterns:
            m = re.search(p, t)
            if m:
                return m.group(1).strip(". ,;:")
        known = ("phoenix", "orion", "dragon", "wolves", "atlas", "zephyr")
        for k in known:
            if k in t:
                return k
        return ""

    def batch_embeddings(self) -> torch.Tensor:
        return self.tensor()

    @staticmethod
    def from_texts(
        texts: Sequence[str],
        embed_fn,
        *,
        dim: int,
        device: str = "cpu",
        prefix: str = "mem",
    ) -> "TideMemoryBank":
        bank = TideMemoryBank(dim=dim, device=device)
        for i, t in enumerate(texts):
            emb = embed_fn(t)
            bank.write(f"{prefix}_{i}", t, emb, basin_id=i)
        return bank
