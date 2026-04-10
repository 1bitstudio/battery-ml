import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Linear regression baseline for SOH prediction.

    Per-cycle curves are flattened and embedded, then masked-mean-pooled
    over observed cycles, and projected to a single SOH value.
    """

    def __init__(self, configs):
        super().__init__()
        curve_features = configs.charge_discharge_length * 3
        self.flatten = nn.Flatten(start_dim=2)
        self.embed = nn.Linear(curve_features, configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.output_num)

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False, return_cycle_embedding=False):
        """
        cycle_curve_data : [B, L, 3, T]
        curve_attn_mask  : [B, L]   (1 = observed, 0 = masked)
        return_cycle_embedding : if True, return (out, x_per_cycle [B,L,d_model]) for interpretability.
        """
        x = self.flatten(cycle_curve_data)        # [B, L, 3*T]
        x = self.embed(x)                         # [B, L, d_model]
        x_per_cycle = x

        mask = curve_attn_mask.unsqueeze(-1)       # [B, L, 1]
        lengths = mask.sum(dim=1).clamp(min=1)     # [B, 1]
        x_pooled = (x * mask).sum(dim=1) / lengths  # [B, d_model]

        out = self.projection(x_pooled)            # [B, output_num]
        if return_cycle_embedding:
            return out, x_per_cycle
        if return_embedding:
            return out, x_pooled
        return out
