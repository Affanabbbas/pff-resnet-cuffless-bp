from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class AttentionPool1D(nn.Module):
    """
    Learnable temporal pooling for [batch, channels, length] features.

    The module predicts one attention logit per time step and returns a
    weighted sum over time.
    """

    def __init__(self, channels: int, hidden_channels: int, dropout: float = 0.0):
        super().__init__()

        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class ScalarRegressionHead(nn.Module):
    """
    Small MLP that predicts one value: either SBP or DBP.
    """

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.15):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResNet1D(nn.Module):
    """
    Original ResNet1D regression model for BP estimation.

    Input:
        x: [batch, channels, length]

    Output:
        y: [batch, 2] = [SBP, DBP]
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        dropout: float = 0.15,
        blocks=(2, 2, 2, 2),
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.model_name = "resnet1d"

        c = int(base_channels)
        self.base_channels = c
        self.dropout_value = float(dropout)

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                c,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(c, c, blocks[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(c, c * 2, blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(c * 2, c * 4, blocks[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(c * 4, c * 8, blocks[3], stride=2, dropout=dropout)

        self.pool = AttentionPool1D(c * 8, c * 2, dropout=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(c * 8),
            nn.Dropout(dropout),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c * 4, 2),
        )

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, n_blocks, stride, dropout):
        layers = [
            BasicBlock1D(
                in_channels,
                out_channels,
                stride=stride,
                dropout=dropout,
            )
        ]

        for _ in range(1, n_blocks):
            layers.append(
                BasicBlock1D(
                    out_channels,
                    out_channels,
                    stride=1,
                    dropout=dropout,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_encoder(self, x):
        """
        Shared ResNet feature extractor.

        Output:
            [batch, channels, time]
        """

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        x = self.forward_encoder(x)
        x = self.pool(x)
        return self.head(x)


class ResNet1DDualHeadSharedAttn(ResNet1D):
    """
    Cleanest separate-head ablation.

    Same ResNet encoder as baseline.
    Same single attention pool as baseline.
    Separate SBP and DBP MLP heads.

    This tests:
        Does splitting only the final SBP/DBP heads help?
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        dropout: float = 0.15,
        blocks=(2, 2, 2, 2),
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            blocks=blocks,
        )

        self.model_name = "resnet1d_dualhead_sharedattn"

        c = int(base_channels)
        final_channels = c * 8
        hidden = c * 4

        # Remove original 2-output head and replace with two scalar heads
        self.head = nn.Identity()
        self.sbp_head = ScalarRegressionHead(final_channels, hidden, dropout=dropout)
        self.dbp_head = ScalarRegressionHead(final_channels, hidden, dropout=dropout)

        self._init_weights()

    def forward(self, x):
        x = self.forward_encoder(x)
        x = self.pool(x)

        sbp = self.sbp_head(x)
        dbp = self.dbp_head(x)

        return torch.cat([sbp, dbp], dim=1)


class ResNet1DDualHeadAvg(ResNet1D):
    """
    Dual-head model with simple average pooling.

    Same ResNet encoder.
    Average pooling instead of attention pooling.
    Separate SBP and DBP heads.

    This tests:
        Does dual-head work even without attention pooling?
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        dropout: float = 0.15,
        blocks=(2, 2, 2, 2),
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            blocks=blocks,
        )

        self.model_name = "resnet1d_dualhead_avg"

        c = int(base_channels)
        final_channels = c * 8
        hidden = c * 4

        # Replace attention pool with simple average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Remove original 2-output head and replace with two scalar heads
        self.head = nn.Identity()
        self.sbp_head = ScalarRegressionHead(final_channels, hidden, dropout=dropout)
        self.dbp_head = ScalarRegressionHead(final_channels, hidden, dropout=dropout)

        self._init_weights()

    def forward(self, x):
        x = self.forward_encoder(x)

        # [B, C, T] -> [B, C]
        x = self.pool(x).squeeze(-1)

        sbp = self.sbp_head(x)
        dbp = self.dbp_head(x)

        return torch.cat([sbp, dbp], dim=1)


class ResNet1DDualHeadSepAttn(ResNet1D):
    """
    Dual-head model with separate attention pooling.

    Same ResNet encoder.
    SBP has its own attention pool and head.
    DBP has its own attention pool and head.

    This tests:
        Do SBP and DBP need different temporal focus?
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        dropout: float = 0.15,
        blocks=(2, 2, 2, 2),
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            blocks=blocks,
        )

        self.model_name = "resnet1d_dualhead_sepattn"

        c = int(base_channels)
        final_channels = c * 8
        hidden = c * 4

        # Remove original pool/head and replace with target-specific pools/heads
        self.pool = nn.Identity()
        self.head = nn.Identity()

        self.sbp_pool = AttentionPool1D(final_channels, c * 2, dropout=dropout)
        self.dbp_pool = AttentionPool1D(final_channels, c * 2, dropout=dropout)

        self.sbp_head = ScalarRegressionHead(final_channels, hidden, dropout=dropout)
        self.dbp_head = ScalarRegressionHead(final_channels, hidden, dropout=dropout)

        self._init_weights()

    def forward(self, x):
        x = self.forward_encoder(x)

        sbp_features = self.sbp_pool(x)
        dbp_features = self.dbp_pool(x)

        sbp = self.sbp_head(sbp_features)
        dbp = self.dbp_head(dbp_features)

        return torch.cat([sbp, dbp], dim=1)



class ResNet1DPhysFusion(ResNet1D):
    """
    ResNet1D + physiological feature fusion.

    Signal branch:
        ECG/PPG/dPPG/ddPPG -> ResNet encoder -> AttentionPool -> deep vector

    Physiological branch:
        12 physiological features -> MLP -> physio vector

    Fusion:
        deep vector + physio vector -> MLP -> [SBP, DBP]
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        dropout: float = 0.15,
        physio_dim: int = 12,
        physio_hidden: int = 64,
        blocks=(2, 2, 2, 2),
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            blocks=blocks,
        )

        self.model_name = "resnet1d_physfusion"
        self.requires_physio_features = True

        c = int(base_channels)
        final_channels = c * 8

        self.head = nn.Identity()

        self.physio_branch = nn.Sequential(
            nn.BatchNorm1d(physio_dim),
            nn.Linear(physio_dim, physio_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(physio_hidden, physio_hidden),
            nn.ReLU(inplace=True),
        )

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(final_channels + physio_hidden),
            nn.Dropout(dropout),
            nn.Linear(final_channels + physio_hidden, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c * 4, 2),
        )

        self._init_weights()

    def forward(self, x, physio_features):
        x = self.forward_encoder(x)
        deep_features = self.pool(x)

        physio_features = torch.nan_to_num(
            physio_features.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        phys_features = self.physio_branch(physio_features)
        fused = torch.cat([deep_features, phys_features], dim=1)

        return self.fusion_head(fused)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
