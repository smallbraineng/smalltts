# Adapted from https://github.com/lucasnewman/nanospeech/blob/main/nanospeech/nanospeech_torch.py

import torch
import torch.nn as nn

from .dit import RMSNorm


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 block, https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/modules.py
class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/modules.py
def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0
):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


# https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/modules.py
def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(
        start, dtype=torch.float32
    )  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (
            torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0)
            * scale.unsqueeze(1)
        ).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


def precompute_freqs_cis_complex(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
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
        self,
        model_size: int,
        num_heads: int,
        is_causal: bool,
        norm_eps: float,
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
            assert mask.ndim == 2
            mask = mask[:, None, None]
        output = torch.nn.functional.scaled_dot_product_attention(
            query=xq.transpose(1, 2),
            key=xk.transpose(1, 2),
            value=xv.transpose(1, 2),
            attn_mask=mask,
            is_causal=self.is_causal,
        ).transpose(1, 2)
        output = output.reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate)
        output = self.wo(output)
        return output


class MLP(nn.Module):
    def __init__(self, model_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, model_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


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


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, model_size)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = EncoderTransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                is_causal=False,
                norm_eps=norm_eps,
            )
            self.blocks.append(block)
        self.head_dim = model_size // num_heads

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.text_embedding(input_ids)
        freqs_cis = precompute_freqs_cis_complex(self.head_dim, input_ids.shape[1]).to(
            x.device
        )
        for block in self.blocks:
            x = block(x, mask, freqs_cis)
        return x
