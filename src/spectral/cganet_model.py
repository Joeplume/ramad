import torch
import torch.nn as nn
import torch.nn.functional as F


class MIC(nn.Module):

    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24], isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=0, stride=1) for i in isometric_kernel])
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=i // 2, stride=i) for i in conv_kernel])
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=0, stride=i) for i in conv_kernel])
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        (batch, seq_len, channel) = input.shape
        x = input.permute(0, 2, 1)
        x1 = self.drop(self.act(conv1d(x)))
        x = x1
        current_seq_len = x.shape[2]
        kernel_size = isometric.kernel_size[0] if hasattr(isometric.kernel_size, '__len__') else isometric.kernel_size
        if current_seq_len >= kernel_size:
            padding_size = x.shape[2] - 1
            x = torch.nn.functional.pad(x, (padding_size, 0))
            x = self.drop(self.act(isometric(x)))
            x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm(x.permute(0, 2, 1))
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]
        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        self.device = src.device
        src_out = src
        conv_out = self.conv_trans_conv(src_out, self.conv[0], self.conv_trans[0], self.isometric_conv[0])
        y = self.norm1(conv_out)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)
        return self.norm2(conv_out + y)


class RamanMICTransformerFusionModel(nn.Module):

    def __init__(self, input_dim, num_categories, hidden_dim=512, dropout=0.1, conv_kernel=[3], isometric_kernel=[1], trans_dim=256, trans_heads=8, trans_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.mic = MIC(feature_size=hidden_dim, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel)
        encoder_layer = nn.TransformerEncoderLayer(d_model=trans_dim, nhead=trans_heads, dim_feedforward=trans_dim * 2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        self.trans_proj = nn.Linear(hidden_dim, trans_dim)
        self.trans_back = nn.Linear(trans_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.category_output = nn.Linear(hidden_dim, num_categories)
        self.concentration_output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = self.input_layer(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        if x.size(0) > 1:
            x = self.batch_norm(x)
        x = self.dropout(self.activation(x))
        mic_feat = self.mic(x.unsqueeze(1)).squeeze(1)
        trans_in = self.trans_proj(x).unsqueeze(1)
        trans_feat = self.transformer(trans_in).squeeze(1)
        trans_feat = self.trans_back(trans_feat)
        fused = 0.9 * mic_feat + 0.1 * trans_feat
        fused = self.layer_norm(fused)
        category_out = self.category_output(fused)
        concentration_out = self.concentration_output(fused)
        return (category_out, concentration_out)
