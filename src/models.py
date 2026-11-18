"""CPU-friendly neural baselines: B2 (static multi-scale KAN, 'MTLNet-lite') and B3 (GRU).
Our own re-implementations; NOT the authors' Interpretable-MTLNet code."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """Efficient KAN layer: y_o = sum_i [ w_oi * silu(x_i) + sum_g c_oig * B_g(x_i) ] with cubic B-splines on a
    fixed grid over [lo, hi] (x outside the grid gets only the base term)."""

    def __init__(self, n_in, n_out, grid_size=6, order=3, lo=-3.0, hi=3.0):
        super().__init__()
        self.n_in, self.n_out, self.G, self.k = n_in, n_out, grid_size, order
        h = (hi - lo) / grid_size
        self.register_buffer("grid", torch.arange(-order, grid_size + order + 1, dtype=torch.float32) * h + lo)
        self.base_weight = nn.Parameter(torch.empty(n_out, n_in))
        self.spline_weight = nn.Parameter(torch.empty(n_out, n_in * (grid_size + order)))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, 0.0, 0.1 / math.sqrt(n_in))

    def bases(self, x):
        g = self.grid
        x = x.unsqueeze(-1)
        b = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for k in range(1, self.k + 1):
            b = ((x - g[:-(k + 1)]) / (g[k:-1] - g[:-(k + 1)])) * b[..., :-1] + \
                ((g[k + 1:] - x) / (g[k + 1:] - g[1:-k])) * b[..., 1:]
        return b  # (..., n_in, G + k)

    def forward(self, x):
        return F.linear(F.silu(x), self.base_weight) + F.linear(self.bases(x).flatten(-2), self.spline_weight)


class KANConv(nn.Module):
    """KAN over every sliding patch of `k` days (all channels), then mean+max pooling over time.
    Mathematically a KANLayer on each flattened patch; the B-spline bases are computed once per (day, channel)
    and then unfolded (k times cheaper than recomputing them for every patch)."""

    def __init__(self, n_ch, k, d):
        super().__init__()
        self.k = k
        self.kan = KANLayer(n_ch * k, d)

    def forward(self, x):                                   # x (B, L, C)
        B = x.shape[0]
        p = x.unfold(1, self.k, 1)                          # (B, L-k+1, C, k)
        pb = self.kan.bases(x).unfold(1, self.k, 1)         # (B, L-k+1, C, G+k, k)
        h = F.linear(F.silu(p).reshape(B, p.shape[1], -1), self.kan.base_weight) +             F.linear(pb.reshape(B, p.shape[1], -1), self.kan.spline_weight)
        return torch.cat([h.mean(1), h.amax(1)], dim=-1)


class MTLNetLite(nn.Module):
    """B2: multi-scale (3/5/7-day) KAN convolutions -> pooled -> KAN head layer -> linear multitask heads.
    No recurrence / latent state: the 28-day window is summarized statically."""

    def __init__(self, n_ch=10, n_static=4, n_out=16, d=8, scales=(3, 5, 7), hidden=32, dropout=0.0):
        super().__init__()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()   # no RNG use / no change when 0
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.convs = nn.ModuleList([KANConv(n_ch, k, d) for k in scales])
        n_feat = len(scales) * 2 * d + n_static
        self.norm = nn.LayerNorm(n_feat)
        self.kan = KANLayer(n_feat, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, s):
        z = torch.cat([c(x) for c in self.convs] + [s], dim=-1)
        return self.head(self.drop2(self.norm2(self.kan(self.drop1(self.norm(z))))))


class GRUNet(nn.Module):
    """B3: GRU over the 28-day sequence; last hidden state + static -> small MLP -> multitask heads."""

    def __init__(self, n_ch=10, n_static=4, n_out=16, hidden=48, mlp=32):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden + n_static, mlp), nn.ReLU(), nn.Linear(mlp, n_out))

    def forward(self, x, s):
        _, h = self.gru(x)
        return self.mlp(torch.cat([h[-1], s], dim=-1))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class CausalKANConv(nn.Module):
    """Multi-scale temporal KAN, one feature vector PER DAY: day tau is a KAN of the k-day patch ending at tau
    (left edge-replicated), so day tau only sees days tau-k+1..tau (causal). Bases computed once per (day, channel)."""

    def __init__(self, n_ch, k, d):
        super().__init__()
        self.k = k
        self.kan = KANLayer(n_ch * k, d)

    def forward(self, x):                                            # (B, L, C) -> (B, L, d)
        B, L, _ = x.shape
        xp = torch.cat([x[:, :1].expand(-1, self.k - 1, -1), x], 1)  # replicate day 0 (a past value) on the left
        p = xp.unfold(1, self.k, 1)
        pb = self.kan.bases(xp).unfold(1, self.k, 1)
        return F.linear(F.silu(p).reshape(B, L, -1), self.kan.base_weight) + \
            F.linear(pb.reshape(B, L, -1), self.kan.spline_weight)


COND_TASKS = ("sev_{k}_t0", "status_{k}_t0", "sev_{k}_h7", "sev_{k}_h14", "sev_{k}_h21", "status_{k}_h7", "status_{k}_h14",
              "status_{k}_h21", "onset_sust_{k}_h7", "onset_sust_{k}_h14", "onset_sust_{k}_h21",
              "onset_cross_{k}_h7", "onset_cross_{k}_h14", "onset_cross_{k}_h21")


class LatentKANNet(nn.Module):
    """M1 (proposed): causal multi-scale (3/5/7-day) KAN encoder -> per-day feature x_tau -> latent participant state
    z_tau evolved by a discrete KAN transition  z_tau = tanh(z_{tau-1} + f_KAN([z_{tau-1}, x_tau]))  over the 28-day window
    (z_0 from static context) -> z at the origin day + static context -> SEPARATE depression / anxiety heads
    (KAN layer + linear), each emitting nowcast/forecast severity+status and sustained/first-crossing onset.
    mode='pooled' is the ablation: same encoder and heads, but z comes from pooling x_tau over the window (no transition)."""

    def __init__(self, heads, n_out=None, n_ch=10, n_static=4, d=8, dx=12, dz=8, scales=(3, 5, 7), head_hidden=16,
                 dropout=0.0, mode="transition"):
        super().__init__()
        self.mode = mode
        self.convs = nn.ModuleList([CausalKANConv(n_ch, k, d) for k in scales])
        self.norm_e = nn.LayerNorm(len(scales) * d)
        self.fuse = KANLayer(len(scales) * d, dx)
        self.norm_x = nn.LayerNorm(dx)
        self.z0 = nn.Linear(n_static, dz)
        if mode == "transition":
            self.trans = KANLayer(dz + dx, dz)
        else:
            self.pool = KANLayer(2 * dx, dz)
        self.drop_x = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_h = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head_norm = nn.LayerNorm(dz + n_static)
        self.head_kan = nn.ModuleList([KANLayer(dz + n_static, head_hidden) for _ in range(2)])
        self.head_lin = nn.ModuleList([nn.Linear(head_hidden, len(COND_TASKS)) for _ in range(2)])
        idx = []
        for h in heads:
            c = 0 if "_dep_" in h else 1
            k = ("dep", "anx")[c]
            idx.append(c * len(COND_TASKS) + [t.format(k=k) for t in COND_TASKS].index(h))
        self.register_buffer("gather_idx", torch.tensor(idx, dtype=torch.long))

    def encode(self, x, s):
        e = torch.cat([c(x) for c in self.convs], dim=-1)
        xt = self.drop_x(self.norm_x(self.fuse(self.norm_e(e))))        # (B, L, dx)
        if self.mode == "transition":
            z = torch.tanh(self.z0(s))
            traj = []
            for tau in range(xt.shape[1]):
                z = torch.tanh(z + self.trans(torch.cat([z, xt[:, tau]], dim=-1)))
                traj.append(z)
            return z, torch.stack(traj, 1)
        return torch.tanh(self.pool(torch.cat([xt.mean(1), xt.amax(1)], dim=-1)) + self.z0(s)), None

    def forward(self, x, s, return_latent=False):
        z, traj = self.encode(x, s)
        ctx = self.drop_h(self.head_norm(torch.cat([z, s], dim=-1)))
        out = torch.cat([lin(kan(ctx)) for kan, lin in zip(self.head_kan, self.head_lin)], dim=1)[:, self.gather_idx]
        return (out, z, traj) if return_latent else out


MODELS = {"kan": MTLNetLite, "gru": GRUNet,
          "m1": lambda **kw: LatentKANNet(mode="transition", **kw),
          "m1_static": lambda **kw: LatentKANNet(mode="pooled", **kw)}
