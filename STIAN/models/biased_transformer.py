# models/biased_transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Biased Multi-Head Attention
# =========================================================
class BiasedMultiheadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1
    ):
        super().__init__()

        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        bias=None,
        attn_mask=None
    ):

        # query shape:
        # (T, B, d_model)

        T, B, _ = query.shape

        # =========================================
        # Linear projection
        # =========================================
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # =========================================
        # reshape
        # =========================================
        Q = Q.view(
            T,
            B,
            self.nhead,
            self.d_k
        ).permute(1, 2, 0, 3)

        K = K.view(
            T,
            B,
            self.nhead,
            self.d_k
        ).permute(1, 2, 0, 3)

        V = V.view(
            T,
            B,
            self.nhead,
            self.d_k
        ).permute(1, 2, 0, 3)

        # =========================================
        # Attention scores
        # =========================================
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.d_k)

        # =========================================
        # Add temporal bias
        # =========================================
        if bias is not None:

            # bias shape:
            # (B, T, T)

            if bias.dim() == 2:
                bias = bias.unsqueeze(0).unsqueeze(0)

            elif bias.dim() == 3:
                bias = bias.unsqueeze(1)

            scores = scores + bias

        # =========================================
        # Attention mask
        # =========================================
        if attn_mask is not None:
            scores = scores + attn_mask

        # =========================================
        # Softmax
        # =========================================
        attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        # =========================================
        # Weighted sum
        # =========================================
        out = torch.matmul(attn, V)

        # =========================================
        # reshape back
        # =========================================
        out = out.permute(
            2,
            0,
            1,
            3
        ).contiguous()

        out = out.view(
            T,
            B,
            self.d_model
        )

        out = self.out_proj(out)

        return out


# =========================================================
# Biased Transformer Encoder Layer
# =========================================================
class BiasedTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()

        # =========================================
        # Self attention
        # =========================================
        self.self_attn = BiasedMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )

        # =========================================
        # Feed Forward
        # =========================================
        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward
        )

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model
        )

        # =========================================
        # LayerNorm
        # =========================================
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # =========================================
        # Dropout
        # =========================================
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src,
        bias=None,
        src_mask=None,
        src_key_padding_mask=None
    ):

        # =========================================
        # Self Attention
        # =========================================
        src2 = self.self_attn(
            src,
            src,
            src,
            bias=bias,
            attn_mask=src_mask
        )

        src = src + self.dropout1(src2)

        src = self.norm1(src)

        # =========================================
        # Feed Forward
        # =========================================
        src2 = self.linear2(
            self.dropout(
                F.relu(
                    self.linear1(src)
                )
            )
        )

        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src