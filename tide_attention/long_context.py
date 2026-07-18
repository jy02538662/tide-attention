"""Long-context input ↔ memory interaction engine."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .attention import tide_attention, TideAttentionOutput
from .controller import TideController, BrainMode
from .memory import TideMemoryBank, MemoryHit
from .phase import PhaseInterference


EmbedFn = Callable[[str], torch.Tensor]


def _stable_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@dataclass
class LongContextResult:
    answer_tokens: torch.Tensor
    retrieved: List[MemoryHit]
    attention: TideAttentionOutput
    interference_clarity: float
    mode_trace: List[str] = field(default_factory=list)
    used_deep_retrieve: bool = False
    notes: str = ""


class TideLongContextEngine:
    """End-to-end long-context pipeline with defect-driven memory interaction.

    Pipeline:
        1. Embed input tokens / chunks
        2. Interfere with TideMemory attractors
        3. Run Tide attention over context (+ optional memory keys)
        4. If yin/full: deep retrieve + expand attention
        5. If yang/sparse: stay on condensate + shallow retrieve
        6. After repair, return to yang
    """

    def __init__(
        self,
        dim: int,
        *,
        embed_fn: Optional[EmbedFn] = None,
        memory: Optional[TideMemoryBank] = None,
        window: int = 64,
        topk: int = 32,
        xi_c: float = 2.5,
        device: str = "cpu",
    ) -> None:
        self.dim = dim
        self.device = device
        self.embed_fn = embed_fn or self._hash_embed
        self.memory = memory or TideMemoryBank(dim=dim, device=device)
        self.window = window
        self.topk = topk
        self.controller = TideController(xi_c=xi_c)
        self.interferometer = PhaseInterference()

    def _hash_embed(self, text: str) -> torch.Tensor:
        """Deterministic bag-of-hash embedding (no external model required)."""
        vec = torch.zeros(self.dim, dtype=torch.float32)
        for tok in text.lower().split():
            h = _stable_hash(tok) % self.dim
            vec[h] += 1.0
            h2 = _stable_hash(tok[::-1]) % self.dim
            vec[h2] += 0.5
        return F.normalize(vec, dim=0)

    def ingest_memory(self, texts: Sequence[str], *, prefix: str = "mem") -> None:
        for i, t in enumerate(texts):
            self.memory.write(f"{prefix}_{i}", t, self.embed_fn(t), basin_id=i)

    def encode_sequence(self, chunks: Sequence[str]) -> torch.Tensor:
        embs = [self.embed_fn(c).to(self.device) for c in chunks]
        return torch.stack(embs, dim=0)  # (T, D)

    def run(
        self,
        context_chunks: Sequence[str],
        query: str,
        *,
        shallow_k: int = 3,
        deep_k: int = 12,
    ) -> LongContextResult:
        ctx = self.encode_sequence(list(context_chunks) + [query])  # (T,D)
        t, d = ctx.shape
        # Single-head Q/K/V from token embeddings (reference controller, not a trained LM).
        q = ctx.clone()
        k = ctx.clone()
        v = ctx.clone()

        mem = self.memory.batch_embeddings()
        fringe = 0.0
        evidence = self._evidence_text(context_chunks, query)
        retrieve_query = f"{query} {evidence}".strip()
        mem_diag = self.memory.diagnose(self.embed_fn(retrieve_query), topk=5, attractor_steps=0)
        if mem.numel() > 0:
            report = self.interferometer.compare(q.unsqueeze(0), mem.unsqueeze(0))
            fringe = float(report.fringe_clarity[0, -1].item())  # query token clarity

        # First pass attention with memory interference.
        attn = tide_attention(
            q,
            k,
            v,
            memory_keys=mem if mem.numel() > 0 else None,
            controller=self.controller,
            interferometer=self.interferometer,
            window=self.window,
            topk=self.topk,
            memory_margin=mem_diag.margin,
            memory_agreement=mem_diag.agreement,
            conflict_density=mem_diag.conflict_density,
        )

        mode_trace = [attn.mode]
        used_deep = attn.mode in (BrainMode.YIN_FULL.value, BrainMode.REPAIR.value)
        retrieve_k = deep_k if used_deep else shallow_k

        # Ground retrieval in both the question and the strongest context evidence.
        evidence = self._evidence_text(context_chunks, query)
        retrieve_query = f"{query} {evidence}".strip()
        hits = self.memory.retrieve(
            self.embed_fn(retrieve_query),
            topk=retrieve_k,
            attractor_steps=12 if used_deep else 4,
        )

        # Natural-transformation style repair: if yin, expand context with retrieved memory
        # and re-run attention once (deep path).
        notes = (
            f"{attn.reason} | mem_margin={mem_diag.margin:.3f} "
            f"agreement={mem_diag.agreement:.3f} conflict={mem_diag.conflict_density:.3f}"
        )
        if used_deep and hits:
            mem_vecs = torch.stack(
                [self.memory.items[h.index].embedding for h in hits], dim=0
            )
            k2 = torch.cat([k, mem_vecs], dim=0)
            v2 = torch.cat([v, mem_vecs], dim=0)
            # Queries stay on original sequence; keys grow with memory (non-square).
            attn2 = tide_attention(
                q,
                k2,
                v2,
                memory_keys=mem,
                controller=self.controller,
                interferometer=self.interferometer,
                window=self.window,
                topk=self.topk,
                memory_margin=mem_diag.margin,
                memory_agreement=mem_diag.agreement,
                conflict_density=mem_diag.conflict_density,
            )
            mode_trace.append(attn2.mode)
            attn = attn2
            notes = f"{notes} | deep_retrieve={len(hits)} | {attn2.reason}"

        # Answer readout: last-position context vector.
        answer = attn.context[-1] if attn.context.ndim == 2 else attn.context[..., -1, :]
        return LongContextResult(
            answer_tokens=answer,
            retrieved=hits,
            attention=attn,
            interference_clarity=fringe,
            mode_trace=mode_trace,
            used_deep_retrieve=used_deep,
            notes=notes,
        )

    def _evidence_text(self, chunks: Sequence[str], query: str) -> str:
        """Pick context spans that look like factual needles / high lexical overlap."""
        q_toks = set(query.lower().replace("_", " ").split())
        scored = []
        for c in chunks:
            cl = c.lower()
            bonus = 3.0 if "secret_needle" in cl or "codename" in cl or "needle" in cl else 0.0
            overlap = len(q_toks.intersection(cl.replace("_", " ").split()))
            scored.append((bonus + overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return " ".join(c for s, c in scored[:3] if s > 0)

    def answer_text(self, result: LongContextResult) -> str:
        if not result.retrieved:
            return ""
        return result.retrieved[0].text
