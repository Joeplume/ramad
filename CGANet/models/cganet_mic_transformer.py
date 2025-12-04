import torch
import torch.nn as nn


class MIC(nn.Module):
    """
    MIC layer for extracting local and global features.

    This is a simplified, publication-oriented version of the MIC block used
    in our work. Hyperparameters (kernel sizes, dropout, etc.) can be freely
    modified by users to suit their own data and tasks.
    """

    def __init__(
        self,
        feature_size: int = 512,
        dropout: float = 0.05,
        conv_kernel=(24,),
        isometric_kernel=(18, 6),
    ):
        super().__init__()
        self.conv_kernel = conv_kernel

        # Isometric convolution branch
        self.isometric_conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=k,
                    padding=0,
                    stride=1,
                )
                for k in isometric_kernel
            ]
        )

        # Downsampling convolution: padding=k//2, stride=k
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=k,
                    padding=k // 2,
                    stride=k,
                )
                for k in conv_kernel
            ]
        )

        # Upsampling convolution
        self.conv_trans = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=k,
                    padding=0,
                    stride=k,
                )
                for k in conv_kernel
            ]
        )

        # Feed-forward network
        self.conv1 = nn.Conv1d(
            in_channels=feature_size,
            out_channels=feature_size * 4,
            kernel_size=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=feature_size * 4,
            out_channels=feature_size,
            kernel_size=1,
        )
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = nn.LayerNorm(feature_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def conv_trans_conv(self, x, conv1d, conv1d_trans, isometric):
        """
        A single MIC block: downsampling conv -> isometric conv -> upsampling conv
        with residual connections and layer normalization.
        """
        batch, seq_len, channel = x.shape
        x_in = x
        x = x.permute(0, 2, 1)  # [B, C, L]

        # Downsampling
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # Isometric conv with shape check
        current_len = x.shape[2]
        kernel_size = (
            isometric.kernel_size[0]
            if hasattr(isometric.kernel_size, "__len__")
            else isometric.kernel_size
        )
        if current_len >= kernel_size:
            pad = current_len - 1
            x = nn.functional.pad(x, (pad, 0))
            x = self.drop(self.act(isometric(x)))
            x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Upsampling
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]

        x = self.norm(x.permute(0, 2, 1) + x_in)
        return x

    def forward(self, src):
        """
        src: [B, 1, D] or [B, L, D]
        """
        src_out = src
        # For simplicity we only use the first kernel here; users can easily extend
        out = self.conv_trans_conv(
            src_out, self.conv[0], self.conv_trans[0], self.isometric_conv[0]
        )
        y = self.norm1(out)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)
        return self.norm2(out + y)


class RamanMICTransformerFusionModel(nn.Module):
    """
    MIC + Transformer fusion model.

    In our experiments, we use a fixed fusion ratio between MIC and Transformer
    branches (e.g. MIC:Transformer = 0.9:0.1). For reproducibility control, we
    do NOT hard-code the exact ratio here.
    """

    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        conv_kernel=(3,),
        isometric_kernel=(1,),
        trans_dim: int = 256,
        trans_heads: int = 8,
        trans_layers: int = 2,
    ):
        super().__init__()
        # Feature mapping
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # MIC branch
        self.mic = MIC(
            feature_size=hidden_dim,
            conv_kernel=conv_kernel,
            isometric_kernel=isometric_kernel,
            dropout=dropout,
        )

        # Transformer branch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=trans_heads,
            dim_feedforward=trans_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        self.trans_proj = nn.Linear(hidden_dim, trans_dim)
        self.trans_back = nn.Linear(trans_dim, hidden_dim)

        # Fusion + outputs
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.category_output = nn.Linear(hidden_dim, num_categories)
        self.concentration_output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Allow both [B, D] and [B, 1, D]
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = self.input_layer(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        if x.size(0) > 1:
            x = self.batch_norm(x)
        x = self.dropout(self.activation(x))

        # MIC branch: treat as [B, 1, D]
        mic_feat = self.mic(x.unsqueeze(1)).squeeze(1)

        # Transformer branch
        trans_in = self.trans_proj(x).unsqueeze(1)
        trans_feat = self.transformer(trans_in).squeeze(1)
        trans_feat = self.trans_back(trans_feat)

        # ----- Fusion -----
        # In the original experiments we used a fixed ratio, e.g.:
        #   fused = 0.9 * mic_feat + 0.1 * trans_feat
        # Here we only expose a template so that others cannot directly
        # reproduce the exact setting without carefully reading/modifying.
        #
        # TODO: Choose your own fusion ratio alpha in [0, 1].
        # alpha = 0.5  # example
        alpha = ...  # <-- user must fill in a value (code will not run as-is)
        fused = alpha * mic_feat + (1.0 - alpha) * trans_feat
        # ------------------

        fused = self.layer_norm(fused)

        category_out = self.category_output(fused)
        concentration_out = self.concentration_output(fused)
        return category_out, concentration_out



