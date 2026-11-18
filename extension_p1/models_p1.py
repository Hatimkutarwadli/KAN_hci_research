"""P1-1 EXTENSION (not part of the original six stages): Transformer-encoder and CNN-LSTM baselines for the SYNTHETIC multitask problem.
Same interface as the frozen GRUNet (forward(x (B,L,C), s (B,S)) -> (B, n_out)) and the same input windows, so they run through the unchanged baselines.fit_nn.
Registered into models.MODELS at runtime only (src/models.py is not modified). Architecture sizes were fixed in advance to match the GRU's parameter budget; they are NOT tuned.
Our own small re-implementations; NOT the architectures of any published paper."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models as md


def sinusoidal(L, d):
    pos = torch.arange(L).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(L, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerNet(nn.Module):
    """Linear embedding of each day (10 channels) + fixed sinusoidal position -> 2 pre-LN self-attention encoder layers (4 heads, ff 48, no dropout)
    -> final LayerNorm -> representation of the LAST day (the origin) + static -> MLP -> multitask heads. Attention covers only days inside the 28-day window (all <= origin)."""

    def __init__(self, n_ch=10, n_static=4, n_out=16, d=24, heads=4, ff=48, layers=2, mlp=32, max_len=64):
        super().__init__()
        self.inp = nn.Linear(n_ch, d)
        self.register_buffer("pe", sinusoidal(max_len, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d + n_static, mlp), nn.ReLU(), nn.Linear(mlp, n_out))

    def forward(self, x, s):
        h = self.inp(x) + self.pe[:x.shape[1]]
        h = self.norm(self.enc(h))
        return self.mlp(torch.cat([h[:, -1], s], dim=-1))


class CNNLSTMNet(nn.Module):
    """CAUSAL 1-D convolution over days (kernel 3, 16 channels, left zero-padding, ReLU) -> LSTM(16 -> 38) over the 28-day sequence -> last hidden + static -> MLP -> multitask heads."""

    def __init__(self, n_ch=10, n_static=4, n_out=16, conv=16, k=3, hidden=38, mlp=32):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(n_ch, conv, k)
        self.lstm = nn.LSTM(conv, hidden, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden + n_static, mlp), nn.ReLU(), nn.Linear(mlp, n_out))

    def forward(self, x, s):
        c = F.relu(self.conv(F.pad(x.transpose(1, 2), (self.k - 1, 0))))        # (B, conv, L); position t sees days t-k+1..t only
        _, (h, _) = self.lstm(c.transpose(1, 2))
        return self.mlp(torch.cat([h[-1], s], dim=-1))


def register():
    md.MODELS["transformer"] = TransformerNet
    md.MODELS["cnnlstm"] = CNNLSTMNet
