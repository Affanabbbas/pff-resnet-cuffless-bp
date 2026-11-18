from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return F.silu(self.net(x) + self.shortcut(x))


class MambaStyleBlock(nn.Module):
    """
    Lightweight Mamba-style sequence block.

    This is a local implementation inspired by selective state-space models:
    projection/gating, depthwise causal mixing, a learned diagonal recurrence,
    and an output projection. It does not require the external mamba-ssm package.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_inner = int(d_model * expansion)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.depthwise = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_inner,
            bias=True,
        )

        self.log_a = nn.Parameter(torch.empty(d_inner))
        self.b = nn.Parameter(torch.empty(d_inner))
        self.c = nn.Parameter(torch.empty(d_inner))
        self.d = nn.Parameter(torch.ones(d_inner))

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.log_a, -4.0, -2.0)
        nn.init.normal_(self.b, mean=0.0, std=0.02)
        nn.init.normal_(self.c, mean=0.0, std=0.02)

    def _ssm_scan(self, x):
        # x: [batch, length, channels]
        a = torch.exp(-F.softplus(self.log_a)).view(1, -1)
        b = self.b.view(1, -1)
        c = self.c.view(1, -1)
        d = self.d.view(1, -1)

        state = torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.size(1)):
            xt = x[:, t, :]
            state = a * state + b * xt
            ys.append(c * state + d * xt)
        return torch.stack(ys, dim=1)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.depthwise(x)[..., : gate.size(1)]
        x = F.silu(x).transpose(1, 2)

        x = self._ssm_scan(x)
        x = x * torch.sigmoid(gate)
        x = self.out_norm(x)
        x = self.out_proj(x)
        return residual + self.drop(x)


class AttentionPool(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        hidden = max(32, d_model // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class CNNMambaAttention(nn.Module):
    """
    CNN + Mamba-style state-space + attention model for BP regression.

    Input:
        x: [batch, channels, length]
    Output:
        y: [batch, 2] = [SBP, DBP]
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        d_model: int = 128,
        mamba_layers: int = 3,
        mamba_expansion: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.model_name = "cnn_mamba_attention"

        c = int(base_channels)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(c),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ConvBlock1D(c, c, stride=1, dropout=dropout),
            ConvBlock1D(c, c * 2, stride=2, dropout=dropout),
            ConvBlock1D(c * 2, c * 2, stride=1, dropout=dropout),
            ConvBlock1D(c * 2, d_model, stride=2, dropout=dropout),
        )

        self.pos_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.mamba_blocks = nn.ModuleList([
            MambaStyleBlock(
                d_model=d_model,
                expansion=mamba_expansion,
                kernel_size=5,
                dropout=dropout,
            )
            for _ in range(int(mamba_layers))
        ])

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=int(attention_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = AttentionPool(d_model, dropout=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.cnn(x)
        x = x + self.pos_conv(x)
        x = x.transpose(1, 2)

        for block in self.mamba_blocks:
            x = block(x)

        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.attn_drop(attn_out)

        x = self.pool(x)
        return self.head(x)
