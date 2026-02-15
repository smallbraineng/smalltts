from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import RMSNorm
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch import Tensor


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.complex(torch.cos(freqs), torch.sin(freqs))
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:3], -1, 2))
    x_ = x_ * freqs_cis[..., None, :]
    x_ = torch.view_as_real(x_).reshape(x.shape)
    return x_.type_as(x)


class SelfAttention(nn.Module):
    def __init__(
        self, model_size: int, num_heads: int, is_causal: bool, norm_eps: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)
        self.wo = nn.Linear(model_size, model_size, bias=False)
        self.gate = nn.Linear(model_size, model_size, bias=False)
        assert model_size % num_heads == 0
        self.q_norm = RMSNorm((num_heads, model_size // num_heads), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, model_size // num_heads), eps=norm_eps)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        xq = self.wq(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xk = self.wk(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xv = self.wv(x).reshape(batch_size, seq_len, self.num_heads, -1)
        gate = self.gate(x)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq = apply_rotary_emb(xq, freqs_cis[:seq_len])
        xk = apply_rotary_emb(xk, freqs_cis[:seq_len])
        if mask is not None:
            mask = mask[:, None, None]
        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            attn_mask=mask,
            is_causal=self.is_causal,
        ).transpose(1, 2)
        output = output.reshape(batch_size, seq_len, -1) * torch.sigmoid(gate)
        return self.wo(output)


class MLP(nn.Module):
    def __init__(self, model_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, model_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderTransformerBlock(nn.Module):
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        intermediate_size: int,
        is_causal: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.attention = SelfAttention(
            model_size=model_size,
            num_heads=num_heads,
            is_causal=is_causal,
            norm_eps=norm_eps,
        )
        self.mlp = MLP(model_size=model_size, intermediate_size=intermediate_size)
        self.attention_norm = RMSNorm(model_size, norm_eps)
        self.mlp_norm = RMSNorm(model_size, norm_eps)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask, freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


LATENT_SIZE = 64
PATCH_SIZE = 1
MODEL_SIZE = 512
NUM_LAYERS = 12
NUM_HEADS = 8
INTERMEDIATE_SIZE = 1536
NORM_EPS = 1e-5
STYLE_DIM = 512


class StyleEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.in_proj = nn.Linear(LATENT_SIZE * PATCH_SIZE, MODEL_SIZE, bias=True)
        self.blocks = nn.ModuleList(
            [
                EncoderTransformerBlock(
                    model_size=MODEL_SIZE,
                    num_heads=NUM_HEADS,
                    intermediate_size=INTERMEDIATE_SIZE,
                    is_causal=False,
                    norm_eps=NORM_EPS,
                )
                for _ in range(NUM_LAYERS)
            ]
        )
        self.head_dim = MODEL_SIZE // NUM_HEADS
        self.log_scale = nn.Parameter(torch.tensor(-1.8))
        self.norm = RMSNorm(MODEL_SIZE, NORM_EPS)
        self.out_proj = nn.Linear(MODEL_SIZE, out_dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(self.head_dim, 4096), persistent=False
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        latent: Float[Tensor, "batch seq latent_dim"],
        lengths: Int64[Tensor, " batch"] | None = None,
    ) -> Tuple[
        Float[Tensor, "batch patched_seq out_dim"], Bool[Tensor, "batch patched_seq"]
    ]:
        b, t, d = latent.shape
        t_trim = (t // self.patch_size) * self.patch_size
        latent = latent[:, :t_trim, :]
        patched_len = t_trim // self.patch_size
        if lengths is not None:
            lengths_patch = (
                lengths.clamp(max=t_trim) + self.patch_size - 1
            ) // self.patch_size
            mask = (
                torch.arange(patched_len, device=latent.device)[None, :]
                < lengths_patch[:, None]
            )
        else:
            mask = torch.ones(b, patched_len, dtype=torch.bool, device=latent.device)
        x = latent.reshape(b, patched_len, latent.shape[-1] * self.patch_size)
        x = self.in_proj(x)
        x = x * self.log_scale.exp()
        freqs_cis = self.freqs_cis[: x.shape[1]]
        for block in self.blocks:
            x = block(x, mask, freqs_cis)
        x = self.norm(x)
        x = self.out_proj(x)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x, mask
