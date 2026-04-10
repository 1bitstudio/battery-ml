import torch
import torch.nn as nn
import torch.nn.functional as F
from app.layers.Transformer_EncDec import Encoder, EncoderLayer
from app.layers.SelfAttention_Family import FullAttention, AttentionLayer
from app.layers.Embed import PositionalEmbedding


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        out = self.dropout(F.relu(self.fc1(x)))
        out = self.ln(self.dropout(self.fc2(out)) + x)
        return out


class Model(nn.Module):
    """
    CyclePatch-style Transformer for SOH prediction.

    Architecture (same two-stage design as CPTransformer):
      1. Intra-cycle: flatten each cycle's curve → linear → MLP blocks
      2. Inter-cycle: Transformer encoder with causal attention mask
      3. Projection: flatten encoder output → linear → SOH scalar
    """

    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.e_layers = configs.e_layers

        curve_features = configs.charge_discharge_length * 3

        # --- intra-cycle modelling ---
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(curve_features, self.d_model)
        self.intra_mlp = nn.ModuleList([
            MLPBlock(self.d_model, self.d_ff, self.d_model, configs.dropout)
            for _ in range(configs.e_layers)
        ])

        # --- inter-cycle modelling ---
        self.pe = PositionalEmbedding(self.d_model)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        True, configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=False,
                    ),
                    self.d_model, configs.n_heads,
                ),
                self.d_model, self.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.d_layers)
        ])
        self.dropout = nn.Dropout(configs.dropout)

        # --- output head ---
        self.projection = nn.Linear(
            self.d_model * self.early_cycle_threshold,
            configs.output_num,
        )

    def _set_output_attention(self, value):
        """Enable/disable returning attention from inner FullAttention (for interpretability)."""
        for layer in self.encoder.attn_layers:
            inner = getattr(layer.attention, "inner_attention", None)
            if inner is not None and hasattr(inner, "output_attention"):
                inner.output_attention = value

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False, return_attention=False):
        """
        cycle_curve_data : [B, L, 3, T]
        curve_attn_mask  : [B, L]
        return_attention : if True, return (out, attns) where attns is list of [B, n_heads, L, L]
        """
        # intra-cycle: per-cycle feature extraction
        x = self.intra_flatten(cycle_curve_data)      # [B, L, 3*T]
        x = self.intra_embed(x)                       # [B, L, d_model]
        for mlp in self.intra_mlp:
            x = mlp(x)                                # [B, L, d_model]

        # positional encoding
        x = x + self.pe(x)

        # build causal attention mask from curve_attn_mask
        attn_mask = curve_attn_mask.unsqueeze(1)                    # [B, 1, L]
        attn_mask = attn_mask.repeat(1, attn_mask.shape[-1], 1)    # [B, L, L]
        attn_mask = (attn_mask == 0).unsqueeze(1)                  # [B, 1, L, L]

        if return_attention:
            self._set_output_attention(True)
        x, attns = self.encoder(x, attn_mask=attn_mask)
        if return_attention:
            self._set_output_attention(False)

        x = self.dropout(x)

        # flatten & project
        embedding = x.reshape(x.shape[0], -1)         # [B, L * d_model]
        out = self.projection(embedding)               # [B, output_num]

        if return_attention and attns:
            return out, attns
        if return_embedding:
            return out, embedding
        return out
