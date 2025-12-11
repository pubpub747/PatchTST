from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Keep original repo import style
from models.layers.pos_encoding import positional_encoding
from models.layers.basics import *                 # expects: Transpose, get_activation_fn, SigmoidRange, etc.
from models.layers.attention import *              # expects: MultiheadAttention, etc.
from models.layers.revin import RevIN              # explicit import for RevIN


# ----------------------------------------------------------------------
# Channel-Time Split Transformer Encoder (CT-PatchTST architecture)
# (Defined here to avoid "unresolved reference CTSTEncoder")
# ----------------------------------------------------------------------
class CTSTEncoder(nn.Module):
    """Channel-Time Split Transformer Encoder (CT-PatchTST architecture)."""

    def __init__(self, d_model, n_heads, n_heads_channel=1, d_ff=256, norm='BatchNorm',
                 attn_dropout=0., dropout=0., activation='gelu', res_attention=False,
                 n_layers=1, pre_norm=False, store_attn=False, bias=True):
        super().__init__()
        # n_heads_channel: number of heads for channel-wise attention (CT-PatchTST, default=1)
        self.layers = nn.ModuleList([
            CTSTEncoderLayer(
                d_model, n_heads=n_heads, n_heads_channel=n_heads_channel,
                d_ff=d_ff, store_attn=store_attn, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout, bias=bias,
                activation=activation, res_attention=res_attention,
                pre_norm=pre_norm
            ) for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, x: Tensor, W_pos: Optional[Tensor] = None) -> Tensor:
        """
        x: tensor [bs x n_vars x num_patch x d_model]
        """
        output = x
        scores = None
        if self.res_attention:
            for i, mod in enumerate(self.layers):
                if i == 0:
                    output, scores = mod(output, prev=scores, W_pos=W_pos)
                else:
                    output, scores = mod(output, prev=scores)
            return output
        else:
            for i, mod in enumerate(self.layers):
                if i == 0:
                    output = mod(output, W_pos=W_pos)
                else:
                    output = mod(output)
            return output


class CTSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_heads_channel=1, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0., dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads_channel, f"d_model ({d_model}) must be divisible by n_heads_channel ({n_heads_channel})"
        # Channel Multi-Head Attention
        d_k_c = d_model // n_heads_channel
        d_v_c = d_model // n_heads_channel
        self.res_attention = res_attention
        self.channel_attn = MultiheadAttention(
            d_model, n_heads_channel, d_k_c, d_v_c,
            attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention
        )
        # Time Multi-Head Attention
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        self.time_attn = MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention
        )

        # Dropouts for residual connections
        self.dropout_channel = nn.Dropout(dropout)
        self.dropout_time = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        # Optional dropout for positional encoding addition
        self.pos_dropout = nn.Dropout(dropout)

        # Layer Normalizations
        if "batch" in norm.lower():
            self.norm_channel = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm_time = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_channel = nn.LayerNorm(d_model)
            self.norm_time = nn.LayerNorm(d_model)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, W_pos: Optional[Tensor] = None):
        """
        src: tensor [bs x n_vars x num_patch x d_model]
        """
        bs, n_vars, num_patch, d_model = src.shape

        # Channel-wise attention sublayer
        # Flatten to [bs * num_patch, n_vars, d_model] to apply channel attention for each patch
        x_flat_ch = src.permute(0, 2, 1, 3).contiguous().view(bs * num_patch, n_vars, d_model)
        if self.pre_norm:
            x_flat_ch = self.norm_channel(x_flat_ch)
        if self.res_attention:
            out_flat_ch, attn_ch, scores_ch = self.channel_attn(x_flat_ch, x_flat_ch, x_flat_ch)
        else:
            out_flat_ch, attn_ch = self.channel_attn(x_flat_ch, x_flat_ch, x_flat_ch)
        if self.store_attn:
            self.attn_channel = attn_ch
        x_flat_ch = x_flat_ch + self.dropout_channel(out_flat_ch)
        if not self.pre_norm:
            x_flat_ch = self.norm_channel(x_flat_ch)
        # Reshape back to [bs, n_vars, num_patch, d_model]
        src_after_ch = x_flat_ch.view(bs, num_patch, n_vars, d_model).permute(0, 2, 1, 3).contiguous()

        # Time-wise attention sublayer
        # Flatten to [bs * n_vars, num_patch, d_model] for time attention on each channel's sequence
        x_flat_time = src_after_ch.contiguous().view(bs * n_vars, num_patch, d_model)
        # Add positional encoding (only in first layer, provided via W_pos)
        if W_pos is not None:
            x_flat_time = self.pos_dropout(x_flat_time + W_pos)
        if self.pre_norm:
            x_flat_time = self.norm_time(x_flat_time)
        if self.res_attention:
            out_flat_time, attn_time, scores_time = self.time_attn(x_flat_time, x_flat_time, x_flat_time, prev)
        else:
            out_flat_time, attn_time = self.time_attn(x_flat_time, x_flat_time, x_flat_time)
        if self.store_attn:
            self.attn = attn_time
        x_flat_time = x_flat_time + self.dropout_time(out_flat_time)
        if not self.pre_norm:
            x_flat_time = self.norm_time(x_flat_time)

        # Position-wise Feed-Forward sublayer
        if self.pre_norm:
            x_flat_time = self.norm_ffn(x_flat_time)
        out_ff = self.ff(x_flat_time)
        x_flat_time = x_flat_time + self.dropout_ffn(out_ff)
        if not self.pre_norm:
            x_flat_time = self.norm_ffn(x_flat_time)

        # Reshape output back to [bs, n_vars, num_patch, d_model] for consistency
        output = x_flat_time.view(bs, n_vars, num_patch, d_model)
        if self.res_attention:
            return output, scores_time
        else:
            return output


# ----------------------------------------------------------------------
# CT-PatchTST with Multi-Scale Patch + Adaptive Fusion + RevIN (end-to-end)
# ----------------------------------------------------------------------
class CTPatchTST(nn.Module):
    """
    Output dimension:
         [bs x pred_len x n_targets] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in: int, target_dim: int, seq_len: int, pred_len: int, patch_len, stride: Optional[int] = None,
                 n_layers: int = 3, d_model: int = 128, n_heads_time: int = 16, n_heads_channel: int = 1,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 head_dropout: float = 0., act: str = 'gelu', res_attention: bool = True, pre_norm: bool = False,
                 store_attn: bool = False, head_type: str = "prediction", individual: bool = False,
                 shared_embedding: bool = True, pe: str = 'zeros', learn_pe: bool = True,
                 revin: bool = False, affine: bool = False, subtract_last: bool = False,
                 y_range: Optional[tuple] = None, verbose: bool = False):
        super().__init__()
        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], \
            "head_type should be 'pretrain', 'prediction', 'regression', or 'classification'"

        # Process patch length and stride for multi-scale
        if isinstance(patch_len, (list, tuple)):
            patch_lens = list(patch_len)
        else:
            patch_lens = [patch_len]
        if stride is None:
            patch_strides = [L for L in patch_lens]
        elif isinstance(stride, (list, tuple)):
            patch_strides = list(stride)
        else:
            patch_strides = [stride] * len(patch_lens)
        assert len(patch_lens) == len(patch_strides), "patch_len and stride lists must have same length"

        self.n_vars = c_in
        self.head_type = head_type
        self.pred_len = pred_len

        # RevIN normalization
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Backbone encoder (multi-scale + fusion + CT encoder)
        self.backbone = CTPatchTSTEncoder(
            c_in, seq_len, patch_lens, patch_strides,
            n_layers=n_layers, d_model=d_model, n_heads_time=n_heads_time, n_heads_channel=n_heads_channel,
            d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            shared_embedding=shared_embedding, pe=pe, learn_pe=learn_pe
        )

        # Heads
        if head_type == "pretrain":
            patch_len_out = patch_lens[0]
            self.head = PretrainHead(d_model, patch_len_out, head_dropout)
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, self.backbone.num_patch, pred_len, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: tensor [bs x seq_len x n_vars]
        """
        x = x_enc

        # Apply RevIN normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # Backbone encoding
        z = self.backbone(x)  # [bs x n_vars x d_model x num_patch]

        # Task head
        out = self.head(z)

        # RevIN inverse for forecasting/pretrain
        if self.revin and self.head_type in ['prediction', 'pretrain']:
            if self.head_type == 'pretrain':
                # [bs x num_patch x n_vars x patch_len] -> [bs x L' x n_vars] -> denorm -> reshape back
                bs, num_patch, n_vars, patch_len = out.shape
                out_seq = out.permute(0, 1, 3, 2).reshape(bs, num_patch * patch_len, n_vars)
                out_seq = self.revin_layer(out_seq, 'denorm')
                out = out_seq.view(bs, num_patch, patch_len, n_vars).permute(0, 1, 3, 2)
            else:
                out = self.revin_layer(out, 'denorm')

        return out


# ----------------------------------------------------------------------
# Heads
# ----------------------------------------------------------------------
class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x n_vars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:, :, :, -1]          # [bs x n_vars x d_model]
        x = self.flatten(x)         # [bs x (n_vars * d_model)]
        x = self.dropout(x)
        y = self.linear(x)          # [bs x output_dim]
        if self.y_range:
            y = SigmoidRange(*self.y_range)(y)
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x n_vars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:, :, :, -1]          # [bs x n_vars x d_model]
        x = self.flatten(x)         # [bs x (n_vars * d_model)]
        x = self.dropout(x)
        y = self.linear(x)          # [bs x n_classes]
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.forecast_len = forecast_len
        self.head_dim = d_model * num_patch
        self.dropout = nn.Dropout(head_dropout)

        if self.individual:
            self.linear = nn.ModuleList([nn.Linear(self.head_dim, forecast_len) for _ in range(n_vars)])
        else:
            # shared linear applied per-channel
            self.linear = nn.Linear(self.head_dim, forecast_len)

    def forward(self, x):
        """
        x: [bs x n_vars x d_model x num_patch]
        output: [bs x forecast_len x n_vars]
        """
        bs, n_vars, d_model, num_patch = x.shape
        x = x.reshape(bs, n_vars, d_model * num_patch)  # [bs, n_vars, head_dim]
        x = self.dropout(x)

        if self.individual:
            outs = []
            for i in range(n_vars):
                o = self.linear[i](x[:, i, :])           # [bs, forecast_len]
                outs.append(o.unsqueeze(2))              # [bs, forecast_len, 1]
            y = torch.cat(outs, dim=2)                   # [bs, forecast_len, n_vars]
        else:
            xc = x.reshape(bs * n_vars, self.head_dim)   # [bs*n_vars, head_dim]
            yc = self.linear(xc)                         # [bs*n_vars, forecast_len]
            y = yc.view(bs, n_vars, self.forecast_len).transpose(1, 2)  # [bs, forecast_len, n_vars]

        return y


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x n_vars x d_model x num_patch]
        output: tensor [bs x num_patch x n_vars x patch_len]
        """
        x = x.transpose(2, 3)             # [bs x n_vars x num_patch x d_model]
        x = self.linear(self.dropout(x))  # [bs x n_vars x num_patch x patch_len]
        x = x.permute(0, 2, 1, 3)         # [bs x num_patch x n_vars x patch_len]
        return x


# ----------------------------------------------------------------------
# Encoder with Multi-Scale Patching + Adaptive Fusion
# ----------------------------------------------------------------------
class CTPatchTSTEncoder(nn.Module):
    def __init__(self, c_in, seq_len, patch_lens, patch_strides,
                 n_layers=3, d_model=128, n_heads_time=16, n_heads_channel=1,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0.,
                 act='gelu', res_attention=True, pre_norm=False, store_attn=False,
                 shared_embedding=True, pe='zeros', learn_pe=True):
        super().__init__()
        self.n_vars = c_in
        self.seq_len = seq_len
        self.d_model = d_model

        # Multi-scale parameters
        self.patch_lens = patch_lens if isinstance(patch_lens, list) else [patch_lens]
        self.patch_strides = patch_strides if isinstance(patch_strides, list) else [patch_strides]
        assert len(self.patch_lens) == len(self.patch_strides)
        self.N_scales = len(self.patch_lens)
        self.shared_embedding = shared_embedding

        # Patch projection layers
        if not self.shared_embedding:
            # One linear per variable for each scale
            self.W_P = nn.ModuleList([
                nn.ModuleList([nn.Linear(L, d_model) for _ in range(c_in)])
                for L in self.patch_lens
            ])
        else:
            # One linear per scale (shared across variables)
            self.W_P = nn.ModuleList([nn.Linear(L, d_model) for L in self.patch_lens])

        # Number of patches for each scale and global maximum
        self.num_patches = []
        for L, S in zip(self.patch_lens, self.patch_strides):
            if seq_len < L:
                self.num_patches.append(0)
            else:
                n_p = (seq_len - L) // S + 1
                self.num_patches.append(n_p)
        self.num_patch = max(self.num_patches) if self.num_patches else 0

        # Positional encoding for patch tokens
        self.W_pos = positional_encoding(pe, learn_pe, self.num_patch, d_model)

        # Fusion MLP (adaptive per-patch weights across scales)
        if self.N_scales > 1:
            if act.lower() == 'gelu':
                fuse_act = nn.GELU()
            elif act.lower() == 'relu':
                fuse_act = nn.ReLU()
            else:
                fuse_act = nn.GELU()
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.N_scales * d_model, d_model),
                fuse_act,
                nn.Linear(d_model, self.N_scales)
            )

        # Channel-Time Split Transformer Encoder
        self.encoder = CTSTEncoder(
            d_model=d_model, n_heads=n_heads_time, n_heads_channel=n_heads_channel,
            d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            activation=act, res_attention=res_attention, n_layers=n_layers,
            pre_norm=pre_norm, store_attn=store_attn
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: tensor [bs x seq_len x n_vars]
        returns: [bs x n_vars x d_model x num_patch]
        """
        bs, seq_len, n_vars = x.shape
        # [bs x n_vars x seq_len] for unfolding
        x = x.transpose(1, 2)

        scale_embeds = []
        for idx, (L, S) in enumerate(zip(self.patch_lens, self.patch_strides)):
            if seq_len < L or self.num_patches[idx] == 0:
                continue
            # Extract patches: [bs x n_vars x num_patch_i x L]
            patches = x.unfold(dimension=2, size=L, step=S)

            # Project patches to d_model
            if not self.shared_embedding:
                W_P_scale = self.W_P[idx]
                out_vars = []
                for i in range(n_vars):
                    out_i = W_P_scale[i](patches[:, i, :, :])  # [bs x num_patch_i x d_model]
                    out_vars.append(out_i)
                embed = torch.stack(out_vars, dim=1)          # [bs x n_vars x num_patch_i x d_model]
            else:
                embed = self.W_P[idx](patches)                # [bs x n_vars x num_patch_i x d_model]

            # Expand/replicate to match max number of patches
            num_patch_i = embed.size(2)
            if num_patch_i < self.num_patch:
                q = self.num_patch // num_patch_i
                r = self.num_patch % num_patch_i
                idxs = []
                for j in range(num_patch_i):
                    reps = q + (1 if j >= num_patch_i - r else 0)
                    idxs.extend([j] * reps)
                idxs = idxs[:self.num_patch]
                idx_tensor = torch.tensor(idxs, device=embed.device)
                embed = embed.index_select(2, idx_tensor)

            scale_embeds.append(embed)  # each: [bs x n_vars x num_patch x d_model]

        # Edge case: no patches
        if len(scale_embeds) == 0 or self.num_patch == 0:
            return torch.zeros(bs, n_vars, self.d_model, 0, device=x.device)

        # Fuse multi-scale features
        if len(scale_embeds) == 1:
            fused = scale_embeds[0]  # [bs x n_vars x num_patch x d_model]
        else:
            # [bs x n_vars x num_patch x N_scales x d_model]
            feat_stack = torch.stack(scale_embeds, dim=3)
            # [bs x n_vars x num_patch x (N_scales*d_model)]
            feat_concat = feat_stack.view(bs, n_vars, self.num_patch, -1)
            # Adaptive weights
            weights = self.fusion_mlp(feat_concat)         # [bs x n_vars x num_patch x N_scales]
            weights = F.softmax(weights, dim=-1)
            # Weighted sum over scales
            weights = weights.unsqueeze(-1)                 # [bs x n_vars x num_patch x N_scales x 1]
            fused = (weights * feat_stack).sum(dim=3)       # [bs x n_vars x num_patch x d_model]

        # Transformer encoder with positional encoding added inside (first layer)
        z = self.encoder(fused, self.W_pos)                 # [bs x n_vars x num_patch x d_model]
        z = z.permute(0, 1, 3, 2)                           # [bs x n_vars x d_model x num_patch]
        return z
