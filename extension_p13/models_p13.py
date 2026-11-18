"""P1-3 EXTENSION (not part of the original six stages): matched-parameter MLP controls for the KAN components of B2 (MTLNetLite) and M1 (LatentKANNet), SYNTHETIC data.
src/models.py is NOT modified; the new classes are registered into models.MODELS at runtime only.

Control design (fixed a priori, not tuned): EVERY KANLayer(n_in, n_out) (grid 6, order 3 -> n_in*n_out*(1+6+3) = 10*n_in*n_out parameters) is replaced by a 2-layer MLP
n_in -> h -> n_out with SiLU, h = round((10*n_in*n_out - n_out) / (n_in + n_out + 1)) so that the layer has (almost exactly) the KAN layer's parameter count. Everything else
(multi-scale 3/5/7-day patches, mean+max pooling, LayerNorms, latent transition z_t = tanh(z_{t-1} + f([z_{t-1}, x_t])), separate dep/anx heads, linear output layers,
dropout positions, static inputs, initial-state map) is identical to the KAN models. The MLP is a deeper-but-smooth function class; it is a PARAMETER-matched, not a
structure-matched, control: a KAN edge is a learnable univariate function, an MLP unit is not."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import models as md

G_, K_ = 6, 3
KAN_PER_EDGE = 1 + G_ + K_


def matched_hidden(n_in, n_out):
    return max(1, round((KAN_PER_EDGE * n_in * n_out - n_out) / (n_in + n_out + 1)))


class MLPLayer(nn.Module):
    """Drop-in for KANLayer(n_in, n_out): Linear -> SiLU -> Linear with parameter count matched to the KAN layer."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.h = matched_hidden(n_in, n_out)
        self.fc1, self.fc2 = nn.Linear(n_in, self.h), nn.Linear(self.h, n_out)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class MLPConv(nn.Module):
    """B2 convolution analogue: MLP over every sliding k-day patch (all channels), then mean+max pooling over time."""

    def __init__(self, n_ch, k, d):
        super().__init__()
        self.k = k
        self.mlp = MLPLayer(n_ch * k, d)

    def forward(self, x):                                   # (B, L, C)
        p = x.unfold(1, self.k, 1)                          # (B, L-k+1, C, k)
        h = self.mlp(p.reshape(x.shape[0], p.shape[1], -1))
        return torch.cat([h.mean(1), h.amax(1)], dim=-1)


class CausalMLPConv(nn.Module):
    """M1 encoder analogue: per-day MLP of the k-day patch ending at that day (day 0 replicated on the left; causal)."""

    def __init__(self, n_ch, k, d):
        super().__init__()
        self.k = k
        self.mlp = MLPLayer(n_ch * k, d)

    def forward(self, x):
        B, L, _ = x.shape
        xp = torch.cat([x[:, :1].expand(-1, self.k - 1, -1), x], 1)
        return self.mlp(xp.unfold(1, self.k, 1).reshape(B, L, -1))


class MTLNetMLP(nn.Module):
    """B2 with every KAN component replaced by a parameter-matched MLP (same wiring as md.MTLNetLite)."""

    def __init__(self, n_ch=10, n_static=4, n_out=16, d=8, scales=(3, 5, 7), hidden=32, dropout=0.0):
        super().__init__()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.convs = nn.ModuleList([MLPConv(n_ch, k, d) for k in scales])
        n_feat = len(scales) * 2 * d + n_static
        self.norm = nn.LayerNorm(n_feat)
        self.kan = MLPLayer(n_feat, hidden)                 # attribute name kept ('kan') only so the wiring reads identically
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, s):
        z = torch.cat([c(x) for c in self.convs] + [s], dim=-1)
        return self.head(self.drop2(self.norm2(self.kan(self.drop1(self.norm(z))))))


class LatentMLPNet(md.LatentKANNet):
    """M1 with every KAN component (3 conv encoders, fusion, KAN transition, 2 head layers) replaced by a parameter-matched MLP; forward/encode are inherited unchanged."""

    def __init__(self, heads, n_out=None, n_ch=10, n_static=4, d=8, dx=12, dz=8, scales=(3, 5, 7), head_hidden=16, dropout=0.0, mode="transition"):
        super().__init__(heads, n_out=n_out, n_ch=n_ch, n_static=n_static, d=d, dx=dx, dz=dz, scales=scales, head_hidden=head_hidden, dropout=dropout, mode=mode)
        self.convs = nn.ModuleList([CausalMLPConv(n_ch, k, d) for k in scales])
        self.fuse = MLPLayer(len(scales) * d, dx)
        if mode == "transition":
            self.trans = MLPLayer(dz + dx, dz)
        else:
            self.pool = MLPLayer(2 * dx, dz)
        self.head_kan = nn.ModuleList([MLPLayer(dz + n_static, head_hidden) for _ in range(2)])


class MTLNetNoSpline(md.MTLNetLite):
    """ABLATION: B2 with the spline part of every KAN edge removed (spline weights zero and frozen) -> each edge is only w * silu(x); the architecture, init stream and
    optimiser are otherwise identical. Tells whether the learnable spline shapes contribute anything beyond a silu-linear unit."""

    def __init__(self, **kw):
        super().__init__(**kw)
        for m in self.modules():
            if isinstance(m, md.KANLayer):
                m.spline_weight.data.zero_()
                m.spline_weight.requires_grad_(False)


def n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def register():
    md.MODELS["mlp_b2"] = MTLNetMLP
    md.MODELS["kan_nospline"] = MTLNetNoSpline
    md.MODELS["mlp_m1"] = lambda **kw: LatentMLPNet(mode="transition", **kw)
