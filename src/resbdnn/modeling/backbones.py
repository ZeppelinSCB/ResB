import math

import torch
from torch import nn
import torch.nn.functional as F

from resbdnn.simulation.candidates import candidate_index_data


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(value))


class ConditionedSelfAttentionBlock(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.cond1 = nn.Linear(token_dim, token_dim * 2)
        self.cond2 = nn.Linear(token_dim, token_dim * 2)
        self.attn = nn.MultiheadAttention(token_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _condition(self, x: torch.Tensor, query: torch.Tensor, projector: nn.Linear) -> torch.Tensor:
        gamma, beta = projector(query).chunk(2, dim=1)
        return x * (1.0 + gamma[:, None, :]) + beta[:, None, :]

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        attn_input = self._condition(self.norm1(tokens), query, self.cond1)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + self.dropout(attn_output)
        ffn_input = self._condition(self.norm2(tokens), query, self.cond2)
        return tokens + self.dropout(self.ffn(ffn_input))


class TMCNet(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        n_t: int = 4,
        n_ris: int = 64,
        s: int = 4,
        candidate_strategy: str = "prefix",
        csi_conditioned: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.n_t = n_t
        self.n_ris = n_ris
        self.s = s
        self.candidate_strategy = candidate_strategy
        self.csi_conditioned = bool(csi_conditioned)
        self.num_candidates = (n_t // 2) * s

        combo_idx, combo_mask, attrs = candidate_index_data(n_t, s, candidate_strategy)
        self.register_buffer("candidate_combo_idx", torch.as_tensor(combo_idx, dtype=torch.long))
        self.register_buffer("candidate_combo_mask", torch.as_tensor(combo_mask, dtype=torch.float32))
        self.register_buffer("candidate_attrs", torch.as_tensor(attrs, dtype=torch.float32))

        ris_feat_dim = 2 * n_t + 2
        self.input_proj = nn.Sequential(
            nn.LayerNorm(ris_feat_dim),
            nn.Linear(ris_feat_dim, token_dim),
        )
        cond_in_dim = 2 if self.csi_conditioned else 1
        self.snr_embed = nn.Sequential(
            nn.Linear(cond_in_dim, token_dim // 2),
            nn.SiLU(),
            nn.Linear(token_dim // 2, token_dim),
        )
        self.blocks = nn.ModuleList(
            ConditionedSelfAttentionBlock(
                token_dim=token_dim,
                hidden_dim=token_dim * 4,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.candidate_query = nn.Embedding(self.num_candidates, token_dim)
        self.candidate_attr_proj = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, token_dim),
            nn.SiLU(),
        )
        self.phase_proj = nn.Sequential(
            nn.LayerNorm(n_ris),
            nn.Linear(n_ris, token_dim),
            nn.SiLU(),
        )
        self.ideal_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.practical_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.mismatch_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.active_candidate_proj = nn.Sequential(
            nn.LayerNorm(11),
            nn.Linear(11, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.head_condition_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(token_dim * 9, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.mismatch_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, 2),
        )

    def forward_parts(
        self,
        h_hat: torch.Tensor,
        g_hat: torch.Tensor,
        sigma_n: torch.Tensor,
        phi_config: torch.Tensor,
        mu_ideal: torch.Tensor,
        mu_base: torch.Tensor | None = None,
        csi_error_var: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_real = h_hat.real.permute(0, 2, 1).to(torch.float32)
        h_imag = h_hat.imag.permute(0, 2, 1).to(torch.float32)
        g_real = g_hat.real.unsqueeze(-1).to(torch.float32)
        g_imag = g_hat.imag.unsqueeze(-1).to(torch.float32)
        ris_features = torch.cat([h_real, h_imag, g_real, g_imag], dim=-1)

        ris_tokens = self.input_proj(ris_features)
        sigma_features = torch.log1p(sigma_n.to(torch.float32).reshape(h_hat.size(0), 1))
        if self.csi_conditioned:
            if csi_error_var is None:
                raise ValueError("csi_conditioned=True but csi_error_var was not provided")
            csi_features = torch.log1p(
                csi_error_var.to(torch.float32).reshape(h_hat.size(0), 1).clamp_min(0.0)
            )
            cond_features = torch.cat([sigma_features, csi_features], dim=-1)
        else:
            cond_features = sigma_features
        snr_condition = self.snr_embed(cond_features)
        for block in self.blocks:
            ris_tokens = block(ris_tokens, snr_condition)
        ris_tokens = self.output_norm(ris_tokens)
        ris_repr = ris_tokens.mean(dim=1)

        mu_base = mu_ideal if mu_base is None else mu_base
        ideal_features = torch.stack(
            [
                mu_ideal.real.to(torch.float32),
                mu_ideal.imag.to(torch.float32),
                mu_ideal.abs().to(torch.float32),
                (torch.angle(mu_ideal) / math.pi).to(torch.float32),
            ],
            dim=-1,
        )
        practical_features = torch.stack(
            [
                mu_base.real.to(torch.float32),
                mu_base.imag.to(torch.float32),
                mu_base.abs().to(torch.float32),
                (torch.angle(mu_base) / math.pi).to(torch.float32),
            ],
            dim=-1,
        )
        mismatch = mu_ideal - mu_base
        mismatch_features = torch.stack(
            [
                mismatch.real.to(torch.float32),
                mismatch.imag.to(torch.float32),
                mismatch.abs().to(torch.float32),
                (torch.angle(mismatch) / math.pi).to(torch.float32),
            ],
            dim=-1,
        )
        candidate_count = mu_ideal.size(1)
        candidate_query = self.candidate_query.weight.unsqueeze(0).expand(h_hat.size(0), -1, -1)
        candidate_attr = self.candidate_attr_proj(self.candidate_attrs.unsqueeze(0).expand(h_hat.size(0), -1, -1))
        combo_idx = self.candidate_combo_idx.to(h_hat.device)
        combo_mask = self.candidate_combo_mask.to(h_hat.device, dtype=torch.float32)
        h_active = h_hat[:, combo_idx, :]
        if phi_config.ndim == 3:
            active_phase = phi_config[:, :, None, :].expand(-1, -1, combo_idx.size(1), -1)
        else:
            active_phase = phi_config
        active_phase = active_phase.to(torch.float32)
        g_active = g_hat[:, None, None, :].expand_as(h_active)
        steering = torch.polar(torch.ones_like(active_phase), active_phase)
        effective = h_active * g_active * steering
        active_ris_features = torch.stack(
            [
                h_active.real.to(torch.float32),
                h_active.imag.to(torch.float32),
                h_active.abs().to(torch.float32),
                g_active.real.to(torch.float32),
                g_active.imag.to(torch.float32),
                g_active.abs().to(torch.float32),
                torch.cos(active_phase),
                torch.sin(active_phase),
                effective.real.to(torch.float32),
                effective.imag.to(torch.float32),
                effective.abs().to(torch.float32),
            ],
            dim=-1,
        )
        active_features = active_ris_features.mean(dim=3)
        active_mask = combo_mask[None, :, :, None]
        active_features = (active_features * active_mask).sum(dim=2)
        active_features = active_features / combo_mask.sum(dim=1).clamp_min(1.0)[None, :, None]
        active_features = self.active_candidate_proj(active_features)
        phase_input = (phi_config / (2.0 * math.pi)).to(torch.float32)
        phase_features = self.phase_proj(phase_input)
        if phase_features.ndim == 4:
            max_active = phase_features.size(2)
            attr_na = self.candidate_attrs[:, 0].to(phase_features.device)
            active_counts = torch.round(attr_na * max(max_active - 1, 1) + 1.0).long().clamp(1, max_active)
            active_mask = (
                torch.arange(max_active, device=phase_features.device)[None, :] < active_counts[:, None]
            ).to(phase_features.dtype)
            phase_features = (phase_features * active_mask[None, :, :, None]).sum(dim=2)
            phase_features = phase_features / active_counts.to(phase_features.dtype)[None, :, None]
        ideal_features = self.ideal_proj(ideal_features)
        practical_features = self.practical_proj(practical_features)
        mismatch_features = self.mismatch_proj(mismatch_features)
        global_repr = ris_repr.unsqueeze(1).expand(-1, candidate_count, -1)
        head_condition = self.head_condition_proj(snr_condition).unsqueeze(1).expand(-1, candidate_count, -1)
        candidate_context = self.fusion_proj(
            torch.cat(
                [
                    candidate_query,
                    candidate_attr,
                    ideal_features,
                    practical_features,
                    mismatch_features,
                    active_features,
                    phase_features,
                    global_repr,
                    head_condition,
                ],
                dim=-1,
            )
        )
        delta = self.mismatch_head(candidate_context)
        pred_residual = torch.complex(delta[..., 0].float(), delta[..., 1].float())
        residual_scale = mu_base.abs().square().mean(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        delta_mu = pred_residual * residual_scale.to(pred_residual.real.dtype)
        mu_corrected = mu_base.to(torch.complex64) + delta_mu
        return {
            "mu_base": mu_base,
            "mu_ideal": mu_ideal,
            "mu_corrected": mu_corrected,
            "delta_mu": delta_mu,
            "pred_residual": pred_residual,
            "residual_scale": residual_scale,
            "ris_tokens": ris_tokens,
            "ris_repr": ris_repr,
            "snr_condition": snr_condition,
        }

    def forward(
        self,
        h_hat: torch.Tensor,
        g_hat: torch.Tensor,
        sigma_n: torch.Tensor,
        phi_config: torch.Tensor,
        mu_ideal: torch.Tensor,
        mu_base: torch.Tensor | None = None,
        csi_error_var: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_parts(
            h_hat, g_hat, sigma_n, phi_config, mu_ideal, mu_base, csi_error_var=csi_error_var
        )["mu_corrected"]
