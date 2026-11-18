"""EXTENSION (not part of the original six stages): KAN-GRU with adaptive multi-scale fusion. SYNTHETIC data only.
Reuses the frozen KANLayer / CausalKANConv from src/models.py (not modified). Registered into models.MODELS at runtime."""
import torch
import torch.nn as nn

import models as md


class KANGRUNet(nn.Module):
    """x (B,L,C) -> 3 parallel causal KAN scales (k=3,5,7) -> (B,L,d) each -> LayerNorm -> fusion -> (B,L,d)
    -> GRU(d->hidden) -> last hidden (B,hidden) + static -> LayerNorm -> separate dep/anx heads (KANLayer + Linear), as in M1.
    fusion='adaptive': per-day softmax gate over the 3 scales (Linear(3d->3)); fusion='fixed': concat -> Linear(3d->d)."""

    def __init__(self, heads, n_out=None, n_ch=10, n_static=4, d=8, hidden=32, scales=(3, 5, 7), head_hidden=16,
                 fusion="adaptive"):
        super().__init__()
        assert fusion in ("adaptive", "fixed")
        self.fusion, self.n_scales = fusion, len(scales)
        self.convs = nn.ModuleList([md.CausalKANConv(n_ch, k, d) for k in scales])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in scales])
        if fusion == "adaptive":
            self.gate = nn.Linear(len(scales) * d, len(scales))
        else:
            self.proj = nn.Linear(len(scales) * d, d)
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.head_norm = nn.LayerNorm(hidden + n_static)
        self.head_kan = nn.ModuleList([md.KANLayer(hidden + n_static, head_hidden) for _ in range(2)])
        self.head_lin = nn.ModuleList([nn.Linear(head_hidden, len(md.COND_TASKS)) for _ in range(2)])
        idx = []
        for h in heads:
            c = 0 if "_dep_" in h else 1
            k = ("dep", "anx")[c]
            idx.append(c * len(md.COND_TASKS) + [t.format(k=k) for t in md.COND_TASKS].index(h))
        self.register_buffer("gather_idx", torch.tensor(idx, dtype=torch.long))

    def sequence(self, x):
        """Per-day quantities (all causal): branches (B,L,S,d), gate (B,L,S) or None, fused (B,L,d), gru_out (B,L,hidden)."""
        br = torch.stack([n(c(x)) for c, n in zip(self.convs, self.norms)], dim=2)       # (B,L,S,d)
        if self.fusion == "adaptive":
            gate = torch.softmax(self.gate(br.flatten(2)), dim=-1)                          # (B,L,S)
            fused = (gate.unsqueeze(-1) * br).sum(2)
        else:
            gate, fused = None, self.proj(br.flatten(2))
        out, _ = self.gru(fused)
        return br, gate, fused, out

    def forward(self, x, s, return_gate=False):
        _, gate, _, out = self.sequence(x)
        ctx = self.head_norm(torch.cat([out[:, -1], s], dim=-1))
        o = torch.cat([lin(kan(ctx)) for kan, lin in zip(self.head_kan, self.head_lin)], dim=1)[:, self.gather_idx]
        return (o, gate) if return_gate else o


def register():
    md.MODELS["kangru"] = lambda **kw: KANGRUNet(fusion="adaptive", **kw)
    md.MODELS["kangru_fixed"] = lambda **kw: KANGRUNet(fusion="fixed", **kw)
