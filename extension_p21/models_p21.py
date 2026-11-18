"""P2-1 EXTENSION (not part of the original six stages): hybrid KAN-ODE and GRU-D-style decay-aware GRU, SYNTHETIC data.
src/models.py is NOT modified; the new classes are registered into models.MODELS at runtime only (as in P1-1/P1-3).

KAN-ODE (hybrid, 'ODE-RNN'-type). NOTE: this is NOT the input-conditioned KAN-ODE dz/dt = KAN([z, x(t)]) sketched in the roadmap.
Here the latent state evolves AUTONOMOUSLY between observation days (the vector field depends on z only, with the static-context
anchor zbar); wearable inputs act on the state ONLY through discrete, observation-gated updates on observed days:
    zbar          = tanh(W0 s + b0)                                         (M1's initial-state map)
    dz/dt         = f(z) = kappa * (zbar - z + 2 tanh(K_f(z)))              (K_f: KANLayer(dz, dz), grid [-3, 3])
    kappa         = KMIN + (KMAX - KMIN) * sigmoid(rho)  in [0.05, 2.0]
    between days  : S fixed RK4 steps of size h = 1/S (default S = 1)
    on day tau    : z+ = m_tau * tanh(z- + K_g([z-, xt_tau])) + (1 - m_tau) * z-      (m_tau = 1 - missing_day_tau; K_g = M1's transition)
Boundedness (exact flow): with u = z - zbar, du_j/dt = kappa_j (-u_j + 2 tanh(.)) so |u_j| can only decrease while |u_j| > 2; every jump
returns z into (-1, 1) and zbar is in (-1, 1), hence |u_j| < 2 and |z_j| < 3 = the spline grid edge. The RK4 discretisation can in principle
overshoot, so max|z| and the fraction of K_f / K_g latent inputs outside [-3, 3] (spline extrapolation) are logged, not assumed.
RK4 stability: the linear part has eigenvalues -kappa, so kappa*h <= 2 < 2.785 (RK4 real-axis limit) at S = 1; the K_f Jacobian adds to
this, so solver error (S = 1 vs S = 4) is measured during training.
Encoder, heads and everything else are M1's (models.LatentKANNet). With flow off and every day observed, the model reduces exactly to M1.

GRU-D-style GRU (after Che et al., 2018; own simplified re-implementation): computed from the 28-day window only (same information as every
other model). Two mask groups (whole day: 7 channels; HRV: channel 6). delta = days since the previous observation of the group (GRU-D
definition, delta_0 = 0); x' = last observed value inside the window (day 0: the frozen processed value); empirical mean = 0 (train mean
after scaling).  xhat = m x + (1-m) (gx x' + (1-gx) 0),  gx = exp(-relu(w_c delta + b_c)),  h <- gh * h,  gh = exp(-relu(W delta + b)),
GRUCell([xhat, missing flags]) with hidden 48 (as the GRU), same readout MLP as GRUNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import models as md

KMIN, KMAX = 0.05, 2.0
GRID = 3.0                      # KANLayer default grid [-3, 3]
HRV = 6                         # channel index of hrv
MD, HM = 8, 9                   # missing_day, hrv_missing channel indices (1 = missing)


def rk4_step(f, z, h):
    k1 = f(z)
    k2 = f(z + 0.5 * h * k1)
    k3 = f(z + 0.5 * h * k2)
    k4 = f(z + h * k3)
    return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class KANODENet(md.LatentKANNet):
    """Hybrid KAN-ODE (flow=True) or its no-flow ablation (flow=False: z held constant between observation days; gated updates only)."""

    def __init__(self, heads, n_out=None, flow=True, substeps=1, rho_init=-1.2, **kw):
        super().__init__(heads, n_out=n_out, mode="transition", **kw)
        self.flow, self.S = flow, substeps
        dz = self.z0.out_features
        if flow:
            self.kan_f = md.KANLayer(dz, dz)
            self.rho = nn.Parameter(torch.full((dz,), float(rho_init)))
        self.gate = True                 # False only in the reduction-to-M1 test
        self.reset_stats()

    # ---------------- diagnostics (not parameters, not in state_dict)
    def reset_stats(self):
        self.stats = {m: {"max_abs_z": 0.0, "n_f": 0, "out_f": 0, "n_g": 0, "out_g": 0, "nonfinite": 0} for m in ("train", "eval")}

    def _track(self, key, z):
        """Latent values entering K_f ('f') or K_g ('g'); 'out' = |z| > 3 = outside the spline grid (spline extrapolation)."""
        with torch.no_grad():
            st = self.stats["train" if self.training else "eval"]
            st["n_" + key] += z.numel()
            st["out_" + key] += int((z.abs() > GRID).sum())
            st["max_abs_z"] = max(st["max_abs_z"], float(z.abs().max()))
            st["nonfinite"] += int((~torch.isfinite(z)).sum())

    def kappa(self):
        return KMIN + (KMAX - KMIN) * torch.sigmoid(self.rho)

    def vf(self, z, zbar):
        self._track("f", z)
        return self.kappa() * (zbar - z + 2.0 * torch.tanh(self.kan_f(z)))

    def encode(self, x, s):
        e = torch.cat([c(x) for c in self.convs], dim=-1)
        xt = self.drop_x(self.norm_x(self.fuse(self.norm_e(e))))        # (B, L, dx), identical to M1
        m = (1.0 - x[..., MD]).unsqueeze(-1) if self.gate else torch.ones_like(x[..., :1])
        zbar = torch.tanh(self.z0(s))
        z, traj = zbar, []
        for tau in range(xt.shape[1]):
            if self.flow and tau > 0:
                for _ in range(self.S):
                    z = rk4_step(lambda u: self.vf(u, zbar), z, 1.0 / self.S)
            self._track("g", z)
            upd = torch.tanh(z + self.trans(torch.cat([z, xt[:, tau]], dim=-1)))
            z = m[:, tau] * upd + (1.0 - m[:, tau]) * z
            traj.append(z)
        return z, torch.stack(traj, 1)


class GRUDNet(nn.Module):
    def __init__(self, n_ch=10, n_static=4, n_out=16, hidden=48, mlp=32):
        super().__init__()
        self.H = hidden
        self.wx = nn.Parameter(torch.zeros(8))
        self.bx = nn.Parameter(torch.zeros(8))
        self.decay_h = nn.Linear(2, hidden)
        self.cell = nn.GRUCell(10, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden + n_static, mlp), nn.ReLU(), nn.Linear(mlp, n_out))

    @staticmethod
    def masks_deltas(x):
        """m (B,L,8) per-channel observed indicator; delta (B,L,2) days since previous observation of [day, hrv] groups."""
        md_, hm = x[..., MD], x[..., HM]
        m = (1.0 - md_).unsqueeze(-1).repeat(1, 1, 8)
        m[..., HRV] = 1.0 - hm
        g = torch.stack([1.0 - md_, 1.0 - hm], -1)                       # observed indicators per group
        ds = [torch.zeros_like(g[:, 0])]
        for t in range(1, x.shape[1]):
            ds.append(1.0 + ds[-1] * (1.0 - g[:, t - 1]))
        return m, torch.stack(ds, 1)

    def forward(self, x, s):
        m, delta = self.masks_deltas(x)
        v = x[..., :8]
        dch = torch.cat([delta[..., :1].expand(-1, -1, 8)[..., :HRV], delta[..., 1:2], delta[..., :1].expand(-1, -1, 1)], -1)
        gx = torch.exp(-F.relu(self.wx * dch + self.bx))
        h = x.new_zeros(x.shape[0], self.H)
        last = v[:, 0]
        for t in range(x.shape[1]):
            last = torch.where(m[:, t] > 0, v[:, t], last)              # x': last observed value up to day t (only used where m = 0)
            xhat = m[:, t] * v[:, t] + (1 - m[:, t]) * gx[:, t] * last
            h = torch.exp(-F.relu(self.decay_h(delta[:, t]))) * h
            h = self.cell(torch.cat([xhat, x[:, t, MD:]], -1), h)
        return self.mlp(torch.cat([h, s], -1))


LAST = {}


def register(substeps=1):
    def mk(flow):
        def f(**kw):
            m = KANODENet(flow=flow, substeps=substeps, **kw)
            LAST["model"] = m
            return m
        return f

    def g(**kw):
        m = GRUDNet(**kw)
        LAST["model"] = m
        return m
    md.MODELS["kanode"] = mk(True)
    md.MODELS["kanode_noflow"] = mk(False)
    md.MODELS["grud"] = g
