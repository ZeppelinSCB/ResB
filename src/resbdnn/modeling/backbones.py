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
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.n_t = n_t
        self.n_ris = n_ris
        self.s = s
        self.num_candidates = (n_t // 2) * s

        _, _, attrs = candidate_index_data(n_t, s)
        self.register_buffer("candidate_attrs", torch.as_tensor(attrs, dtype=torch.float32))

        ris_feat_dim = 2 * n_t + 2
        self.input_proj = nn.Sequential(
            nn.LayerNorm(ris_feat_dim),
            nn.Linear(ris_feat_dim, token_dim),
        )
        self.snr_embed = nn.Sequential(
            nn.Linear(1, token_dim // 2),
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
        self.head_condition_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(token_dim * 6, token_dim),
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
    ) -> dict[str, torch.Tensor]:
        h_real = h_hat.real.permute(0, 2, 1).to(torch.float32)
        h_imag = h_hat.imag.permute(0, 2, 1).to(torch.float32)
        g_real = g_hat.real.unsqueeze(-1).to(torch.float32)
        g_imag = g_hat.imag.unsqueeze(-1).to(torch.float32)
        ris_features = torch.cat([h_real, h_imag, g_real, g_imag], dim=-1)

        ris_tokens = self.input_proj(ris_features)
        sigma_features = torch.log1p(sigma_n.to(torch.float32).reshape(h_hat.size(0), 1))
        snr_condition = self.snr_embed(sigma_features)
        for block in self.blocks:
            ris_tokens = block(ris_tokens, snr_condition)
        ris_tokens = self.output_norm(ris_tokens)
        ris_repr = ris_tokens.mean(dim=1)

        ideal_features = torch.stack(
            [
                mu_ideal.real.to(torch.float32),
                mu_ideal.imag.to(torch.float32),
                mu_ideal.abs().to(torch.float32),
                (torch.angle(mu_ideal) / math.pi).to(torch.float32),
            ],
            dim=-1,
        )
        candidate_count = mu_ideal.size(1)
        candidate_query = self.candidate_query.weight.unsqueeze(0).expand(h_hat.size(0), -1, -1)
        candidate_attr = self.candidate_attr_proj(self.candidate_attrs.unsqueeze(0).expand(h_hat.size(0), -1, -1))
        phase_features = self.phase_proj((phi_config / (2.0 * math.pi)).to(torch.float32))
        ideal_features = self.ideal_proj(ideal_features)
        global_repr = ris_repr.unsqueeze(1).expand(-1, candidate_count, -1)
        head_condition = self.head_condition_proj(snr_condition).unsqueeze(1).expand(-1, candidate_count, -1)
        candidate_context = self.fusion_proj(
            torch.cat(
                [candidate_query, candidate_attr, ideal_features, phase_features, global_repr, head_condition],
                dim=-1,
            )
        )
        delta = self.mismatch_head(candidate_context)
        delta_mu = torch.complex(delta[..., 0].float(), delta[..., 1].float())
        mu_base = mu_ideal if mu_base is None else mu_base
        mu_corrected = mu_base.to(torch.complex64) + delta_mu
        return {
            "mu_base": mu_base,
            "mu_ideal": mu_ideal,
            "mu_corrected": mu_corrected,
            "delta_mu": delta_mu,
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
    ) -> torch.Tensor:
        return self.forward_parts(h_hat, g_hat, sigma_n, phi_config, mu_ideal, mu_base)["mu_corrected"]
