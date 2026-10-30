"""Define the full static-dynamic gated fusion model."""

import torch
from torch import nn
from torch.nn import functional as F

SCALES = (
    ("short", 0, 3),
    ("mid", 3, 11),
    ("long", 11, 24),
)


class ResidualBlock(nn.Module):
    """Apply a two-layer residual transformation."""

    def __init__(self, dim, dropout=0.1):
        """Initialize the residual block.

        Args:
            dim (int): Input and output width.
            dropout (float): Hidden dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Transform features with a residual connection.

        Args:
            x (Tensor): Input shaped [batch, dim].

        Returns:
            Tensor: Activated residual features.
        """
        return self.relu(x + self.net(x))


class GatedFusionModule(nn.Module):
    """Share one gate across all lags within each scale."""

    def __init__(
        self, context_dim, num_ts_features, window_size
    ):
        """Initialize the sample-dependent gate network.

        Args:
            context_dim (int): Context vector width.
            num_ts_features (int): Number of variables.
            window_size (int): Number of lags in this scale.
        """
        super().__init__()
        self.num_ts_features = num_ts_features
        self.window_size = window_size
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_ts_features),
            nn.Sigmoid(),
        )

    def forward(
        self, static_logits, dynamic_logits, context
    ):
        """Mix static and dynamic logits before softmax.

        Args:
            static_logits (Tensor): Global logits [F, W].
            dynamic_logits (Tensor): Local logits [B, F, W].
            context (Tensor): Context features [B, C].

        Returns:
            Tensor: Mixed logits shaped [B, F, W].
        """
        gate = self.gate_net(context).view(
            -1, self.num_ts_features, 1
        )
        # Each gate broadcasts over its scale's lag axis.
        logits = (
            (1 - gate) * static_logits.unsqueeze(0)
            + gate * dynamic_logits
        )
        return logits


class FeatureAttention(nn.Module):
    """Reweight fused variables with residual attention."""

    def __init__(self, dim, reduction=4, dropout=0.1):
        """Initialize the attention bottleneck.

        Args:
            dim (int): Fused feature width.
            reduction (int): Bottleneck reduction factor.
            dropout (float): Hidden dropout probability.
        """
        super().__init__()
        bottleneck = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply attention with a residual connection.

        Args:
            x (Tensor): Concatenated fused features.

        Returns:
            Tensor: Residually reweighted feature vector.
        """
        return x * self.gate(x) + x


def _dynamic_network(context_dim, features, dims, dropout):
    """Construct the dynamic lag-logit generator.

    Args:
        context_dim (int): Context vector width.
        features (int): Time-series variable count.
        dims (tuple[int]): Hidden-layer dimensions.
        dropout (float): Hidden dropout probability.

    Returns:
        nn.Sequential: Network producing F * 24 logits.
    """
    layers = []
    in_dim = context_dim
    for hidden_dim in dims:
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, features * 24))
    return nn.Sequential(*layers)


class SDGFusionModel(nn.Module):
    """Forecast one step from history and known covariates.

    Sequence inputs are [batch, variables, 24], with the
    newest observation first. Known covariates refer to the
    forecast time and must be available before prediction.
    """

    def __init__(
        self, num_ts_features, num_static_features,
        dynamic_hidden_dims=(64, 32), dropout=0.1,
        use_feature_attention=True,
    ):
        """Initialize the complete SDGF architecture.

        Args:
            num_ts_features (int): Number of variables.
            num_static_features (int): Covariate count.
            dynamic_hidden_dims (tuple[int]): MLP widths.
            dropout (float): Dynamic/attention dropout.
            use_feature_attention (bool): Enable attention.
        """
        super().__init__()
        self.num_ts_features = num_ts_features
        self.num_static_features = num_static_features
        for name, start, end in SCALES:
            logits = torch.zeros(
                num_ts_features, end - start
            )
            setattr(
                self, name + "_static_logits",
                nn.Parameter(logits),
            )
        self.context_dim = (
            num_ts_features * 9 + num_static_features
        )
        self.dynamic_net = _dynamic_network(
            self.context_dim, num_ts_features,
            dynamic_hidden_dims, dropout,
        )
        for name, start, end in SCALES:
            module = GatedFusionModule(
                self.context_dim, num_ts_features,
                end - start,
            )
            setattr(self, name + "_gfu", module)
        fused_dim = (
            num_ts_features * 3 + num_static_features
        )
        self.use_feature_attention = use_feature_attention
        self.feature_attn = (
            FeatureAttention(fused_dim, dropout=dropout)
            if use_feature_attention else nn.Identity()
        )
        # Preserve the predictor's residual dropout of 0.1.
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, 128),
            ResidualBlock(128),
            nn.Linear(128, 1),
        )

    def build_context(self, x_seq, x_static):
        """Summarize history and known covariates.

        Args:
            x_seq (Tensor): Newest-first history [B, F, 24].
            x_static (Tensor): Known covariates [B, S].

        Returns:
            Tensor: Context with width F * 9 + S.
        """
        batch = x_seq.shape[0]
        return torch.cat([
            x_seq.mean(dim=2),
            x_seq.std(dim=2, unbiased=False),
            x_seq[:, :, 0] - x_seq[:, :, -1],
            x_seq[:, :, :6].reshape(batch, -1),
            x_static,
        ], dim=1)

    def _validate_inputs(self, x_seq, x_static):
        """Check input ranks and feature dimensions.

        Args:
            x_seq (Tensor): Historical inputs.
            x_static (Tensor): Known covariates.

        Raises:
            ValueError: An input has an unexpected shape.
        """
        expected = (self.num_ts_features, 24)
        if x_seq.ndim != 3 or x_seq.shape[1:] != expected:
            raise ValueError(
                "x_seq must have shape "
                "[batch, num_ts_features, 24]."
            )
        static_shape = (
            x_seq.shape[0], self.num_static_features
        )
        if x_static.shape != static_shape:
            raise ValueError(
                "x_static has an incompatible shape."
            )

    def _fuse_scales(self, x_seq, context):
        """Create three weighted historical summaries.

        Args:
            x_seq (Tensor): Historical inputs [B, F, 24].
            context (Tensor): Sample context [B, C].

        Returns:
            list[Tensor]: One fused vector per scale.
        """
        batch, features, _ = x_seq.shape
        dynamic = torch.split(
            self.dynamic_net(context),
            [features * 3, features * 8, features * 13],
            dim=1,
        )
        fused = []
        for (name, start, end), logits in zip(
            SCALES, dynamic
        ):
            fusion_logits = getattr(
                self, name + "_gfu"
            )(
                getattr(self, name + "_static_logits"),
                logits.view(batch, features, end - start),
                context,
            )
            # Normalize lags separately inside each scale.
            weight = F.softmax(fusion_logits, dim=2)
            fused.append(
                (x_seq[:, :, start:end] * weight).sum(dim=2)
            )
        return fused

    def forward(self, x_seq, x_static):
        """Compute the next-step load forecast.

        Args:
            x_seq (Tensor): Newest-first history [B, F, 24].
            x_static (Tensor): Known covariates [B, S].

        Returns:
            Tensor: Load forecasts shaped [B, 1].

        Raises:
            ValueError: Input shapes are incompatible.
        """
        self._validate_inputs(x_seq, x_static)
        context = self.build_context(x_seq, x_static)
        fused = self._fuse_scales(x_seq, context)
        combined = self.feature_attn(
            torch.cat([*fused, x_static], dim=1)
        )
        return self.predictor(combined)
