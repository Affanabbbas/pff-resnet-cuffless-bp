from __future__ import annotations

import torch.nn as nn

from models.cnn_mamba_attention import CNNMambaAttention
from models.resnet1d import (
    ResNet1D,
    ResNet1DDualHeadAvg,
    ResNet1DDualHeadSepAttn,
    ResNet1DDualHeadSharedAttn,
    ResNet1DPhysFusion,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name: str, in_channels: int, cfg: dict) -> nn.Module:
    model_name = str(model_name).lower()
    model_cfg = cfg.get("model", {})

    base_channels = int(model_cfg.get("base_channels", 64))
    dropout = float(model_cfg.get("dropout", 0.15))

    if model_name == "resnet1d":
        return ResNet1D(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
        )

    if model_name == "resnet1d_dualhead_sharedattn":
        return ResNet1DDualHeadSharedAttn(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
        )

    if model_name == "resnet1d_dualhead_avg":
        return ResNet1DDualHeadAvg(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
        )

    if model_name == "resnet1d_dualhead_sepattn":
        return ResNet1DDualHeadSepAttn(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
        )

    if model_name == "resnet1d_physfusion":
        return ResNet1DPhysFusion(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            physio_dim=int(model_cfg.get("physio_dim", 12)),
            physio_hidden=int(model_cfg.get("physio_hidden", 64)),
        )

    if model_name == "cnn_mamba_attention":
        return CNNMambaAttention(
            in_channels=in_channels,
            base_channels=base_channels,
            d_model=int(model_cfg.get("d_model", 128)),
            mamba_layers=int(model_cfg.get("mamba_layers", 3)),
            mamba_expansion=int(model_cfg.get("mamba_expansion", 2)),
            attention_heads=int(model_cfg.get("attention_heads", 4)),
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown model: {model_name}. Use one of: "
        f"'resnet1d', "
        f"'resnet1d_dualhead_sharedattn', "
        f"'resnet1d_dualhead_avg', "
        f"'resnet1d_dualhead_sepattn', "
        f"'resnet1d_physfusion', "
        f"'cnn_mamba_attention'."
    )
