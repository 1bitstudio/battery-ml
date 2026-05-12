import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
from models.soh_feature_fusion import SOHFeatureFusion, curve_feature_channels


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
    Transformer V4 for SOH prediction.

    Improvements over V3:
      1. Local cycle mixing with a depthwise separable Conv1d branch
      2. Recency-aware summary pooling to bias toward the latest observed cycles
      3. Observation-ratio embedding to inform the head how much history is available
    """

    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff

        curve_features = configs.charge_discharge_length * curve_feature_channels(configs)

        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(curve_features, self.d_model)
        self.intra_mlp = nn.ModuleList([
            MLPBlock(self.d_model, self.d_ff, self.d_model, configs.dropout)
            for _ in range(configs.e_layers)
        ])
        self.feature_fusion = SOHFeatureFusion(configs)

        self.pe = PositionalEmbedding(self.d_model)
        self.local_dw = nn.Conv1d(
            self.d_model,
            self.d_model,
            kernel_size=3,
            padding=1,
            groups=self.d_model,
        )
        self.local_pw = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        self.local_dropout = nn.Dropout(configs.dropout)
        self.local_norm = nn.LayerNorm(self.d_model)

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        True,
                        configs.factor,
                        attention_dropout=configs.dropout,
                        output_attention=False,
                    ),
                    self.d_model,
                    configs.n_heads,
                ),
                self.d_model,
                self.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.d_layers)
        ])
        self.dropout = nn.Dropout(configs.dropout)

        self.pool_key = nn.Linear(self.d_model, self.d_model)
        self.pool_query = nn.Linear(self.d_model, self.d_model)
        self.pool_score = nn.Linear(self.d_model, 1)
        self.obs_ratio_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.recency_scale = nn.Parameter(torch.tensor(2.0))

        summary_dim = self.d_model * 5
        self.summary_proj = nn.Sequential(
            nn.Linear(summary_dim, self.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.d_ff, self.d_model),
        )
        self.residual_proj = nn.Linear(self.d_model * 2, self.d_model)
        self.summary_gate = nn.Linear(summary_dim, self.d_model)
        self.summary_norm = nn.LayerNorm(self.d_model)
        self.projection = nn.Linear(self.d_model, configs.output_num)

    def _set_output_attention(self, value):
        for layer in self.encoder.attn_layers:
            inner = getattr(layer.attention, "inner_attention", None)
            if inner is not None and hasattr(inner, "output_attention"):
                inner.output_attention = value

    def _local_mix(self, x, observed_mask):
        local = x.transpose(1, 2)
        local = self.local_dw(local)
        local = self.local_pw(local)
        local = F.gelu(local).transpose(1, 2)
        local = self.local_dropout(local) * observed_mask
        return self.local_norm(x + local)

    def _build_summary(self, x, observed_mask):
        lengths = observed_mask.sum(dim=1).clamp(min=1)

        mean_embedding = x.sum(dim=1) / lengths

        last_idx = lengths.long().squeeze(-1) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
        last_embedding = x.gather(1, gather_idx).squeeze(1)

        attn_hidden = torch.tanh(
            self.pool_key(x) + self.pool_query(last_embedding).unsqueeze(1)
        )
        attn_scores = self.pool_score(attn_hidden).squeeze(-1)
        attn_scores = attn_scores.masked_fill(observed_mask.squeeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pool = (x * attn_weights.unsqueeze(-1)).sum(dim=1)

        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0)
        distances = (last_idx.to(x.dtype).unsqueeze(1) - positions).clamp(min=0.0)
        normalizer = lengths.squeeze(-1).clamp(min=1.0)
        recency_logits = -self.recency_scale.abs() * distances / normalizer.unsqueeze(1)
        recency_logits = recency_logits.masked_fill(observed_mask.squeeze(-1) == 0, -1e9)
        recency_weights = torch.softmax(recency_logits, dim=1)
        recency_pool = (x * recency_weights.unsqueeze(-1)).sum(dim=1)

        obs_ratio = lengths / x.shape[1]
        obs_ratio_embedding = self.obs_ratio_embed(obs_ratio)

        summary_input = torch.cat(
            [
                last_embedding,
                mean_embedding,
                attn_pool,
                recency_pool,
                obs_ratio_embedding,
            ],
            dim=-1,
        )
        gated_summary = self.summary_proj(summary_input)
        residual_summary = self.residual_proj(
            torch.cat([last_embedding, attn_pool], dim=-1)
        )
        gate = torch.sigmoid(self.summary_gate(summary_input))
        embedding = self.summary_norm(residual_summary + gate * gated_summary)
        return embedding

    def forward(
        self,
        cycle_curve_data,
        curve_attn_mask,
        temperature_features=None,
        temperature_mask=None,
        static_numeric=None,
        static_categorical=None,
        return_embedding=False,
        return_attention=False,
    ):
        x = self.intra_flatten(cycle_curve_data)
        x = self.intra_embed(x)
        for mlp in self.intra_mlp:
            x = mlp(x)
        x = self.feature_fusion(
            x,
            curve_attn_mask,
            temperature_features=temperature_features,
            temperature_mask=temperature_mask,
            static_numeric=static_numeric,
            static_categorical=static_categorical,
        )

        observed_mask = curve_attn_mask.unsqueeze(-1)
        x = x * observed_mask
        x = x + self.pe(x) * observed_mask
        x = self._local_mix(x, observed_mask)

        attn_mask = curve_attn_mask.unsqueeze(1)
        attn_mask = attn_mask.repeat(1, attn_mask.shape[-1], 1)
        attn_mask = (attn_mask == 0).unsqueeze(1)

        if return_attention:
            self._set_output_attention(True)
        x, attns = self.encoder(x, attn_mask=attn_mask)
        if return_attention:
            self._set_output_attention(False)

        x = self.dropout(x)
        x = x * observed_mask

        embedding = self._build_summary(x, observed_mask)
        out = self.projection(embedding)

        if return_attention and attns:
            return out, attns
        if return_embedding:
            return out, embedding
        return out
