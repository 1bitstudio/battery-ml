import torch
import torch.nn as nn


def curve_feature_channels(configs):
    return int(getattr(configs, "curve_feature_channels", 3))


class SOHFeatureFusion(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.use_metadata = bool(getattr(configs, "use_metadata", False))
        self.use_temperature = bool(getattr(configs, "use_temperature", False))
        self.temperature_feature_mode = getattr(configs, "temperature_feature_mode", "summary")

        self.metadata_numeric_dim = int(getattr(configs, "metadata_numeric_dim", 0))
        self.metadata_categorical_keys = list(getattr(configs, "metadata_categorical_keys", []))
        self.metadata_vocab_sizes = getattr(configs, "metadata_vocab_sizes", {})
        self.metadata_embedding_dim = int(
            getattr(configs, "metadata_embedding_dim", max(4, configs.d_model // 4))
        )
        self.temperature_summary_dim = int(getattr(configs, "temperature_summary_dim", 5))

        self.temperature_encoder = None
        self.temperature_gate = None
        if self.use_temperature and self.temperature_feature_mode == "summary":
            self.temperature_encoder = nn.Sequential(
                nn.Linear(self.temperature_summary_dim, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
            )
            self.temperature_gate = nn.Linear(self.d_model * 2, self.d_model)

        self.numeric_encoder = None
        self.categorical_embeddings = None
        self.categorical_projection = None
        self.metadata_projection = None
        self.metadata_gate = None

        if self.use_metadata:
            if self.metadata_numeric_dim > 0:
                self.numeric_encoder = nn.Sequential(
                    nn.Linear(self.metadata_numeric_dim, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.d_model),
                )

            if self.metadata_categorical_keys:
                self.categorical_embeddings = nn.ModuleDict()
                for key in self.metadata_categorical_keys:
                    vocab_size = int(self.metadata_vocab_sizes.get(key, 2))
                    self.categorical_embeddings[key] = nn.Embedding(
                        num_embeddings=max(vocab_size, 2),
                        embedding_dim=self.metadata_embedding_dim,
                    )
                categorical_dim = self.metadata_embedding_dim * len(self.metadata_categorical_keys)
                self.categorical_projection = nn.Sequential(
                    nn.Linear(categorical_dim, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.d_model),
                )

            metadata_input_dim = 0
            if self.numeric_encoder is not None:
                metadata_input_dim += self.d_model
            if self.categorical_projection is not None:
                metadata_input_dim += self.d_model

            if metadata_input_dim > 0:
                self.metadata_projection = nn.Sequential(
                    nn.Linear(metadata_input_dim, self.d_model),
                    nn.ReLU(),
                    nn.LayerNorm(self.d_model),
                )
                self.metadata_gate = nn.Linear(self.d_model * 2, self.d_model)

        self.output_norm = nn.LayerNorm(self.d_model)

    def _encode_metadata(self, static_numeric, static_categorical):
        if self.metadata_projection is None:
            return None

        parts = []
        if self.numeric_encoder is not None and static_numeric is not None:
            parts.append(self.numeric_encoder(static_numeric.float()))

        if self.categorical_projection is not None and static_categorical is not None:
            embeddings = []
            for idx, key in enumerate(self.metadata_categorical_keys):
                embeddings.append(self.categorical_embeddings[key](static_categorical[:, idx].long()))
            parts.append(self.categorical_projection(torch.cat(embeddings, dim=-1)))

        if not parts:
            return None
        return self.metadata_projection(torch.cat(parts, dim=-1))

    def forward(
        self,
        x,
        curve_attn_mask,
        temperature_features=None,
        temperature_mask=None,
        static_numeric=None,
        static_categorical=None,
    ):
        fused = x
        observed_mask = curve_attn_mask.unsqueeze(-1).float()
        has_extra_features = False

        if self.temperature_encoder is not None and temperature_features is not None:
            temp_emb = self.temperature_encoder(temperature_features.float())
            if temperature_mask is None:
                valid_temperature_mask = observed_mask
            else:
                valid_temperature_mask = temperature_mask.unsqueeze(-1).float() * observed_mask
            temp_emb = temp_emb * valid_temperature_mask
            temp_gate = torch.sigmoid(self.temperature_gate(torch.cat([fused, temp_emb], dim=-1)))
            fused = fused + temp_gate * temp_emb
            has_extra_features = True

        metadata_emb = self._encode_metadata(static_numeric, static_categorical)
        if metadata_emb is not None:
            metadata_emb = metadata_emb.unsqueeze(1).expand(-1, fused.shape[1], -1) * observed_mask
            metadata_gate = torch.sigmoid(
                self.metadata_gate(torch.cat([fused, metadata_emb], dim=-1))
            )
            fused = fused + metadata_gate * metadata_emb
            has_extra_features = True

        if has_extra_features:
            fused = self.output_norm(fused)
        return fused
