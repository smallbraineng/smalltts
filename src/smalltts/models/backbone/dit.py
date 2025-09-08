from typing import Tuple

import torch
import torch.nn as nn
from beartype import beartype
from einops import rearrange
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor, autocast
from torch.types import Device


# Adapted from https://github.com/lucasnewman/nanospeech/blob/main/nanospeech/nanospeech_torch.py
class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            emb, 6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(
        self,
        x: Float[Tensor, "batch sequence_length hidden_dim"],
        mask: Bool[Tensor, "batch sequence_length"],  # True = attending to
        rope,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # rope
        freqs, xpos_scale = rope
        q_xpos_scale, k_xpos_scale = (
            (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        )
        query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)  # type: ignore
        key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)  # type: ignore

        # attention
        query = rearrange(query, "b n (h d) -> b h n d", h=self.heads)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.heads)

        attn_mask = rearrange(mask, "b n -> b 1 1 n")
        attn_mask = attn_mask.expand(
            batch_size, self.heads, query.shape[-2], key.shape[-2]
        )

        x = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        x = rearrange(x, "b h n d -> b n (h d)")
        x = x.to(query.dtype)

        x = self.to_out(x)

        mask = mask.unsqueeze(-1)
        x = x.masked_fill(~mask, 0.0)

        return x


class CrossAttention(nn.Module):
    def __init__(
        self, dim_q: int, dim_kv: int, heads: int, dim_head: int, dropout: float
    ):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.to_q = nn.Linear(dim_q, inner)
        self.to_k = nn.Linear(dim_kv, inner)
        self.to_v = nn.Linear(dim_kv, inner)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim_q, bias=False), nn.Dropout(dropout)
        )

    def forward(
        self,
        x_q: Float[Tensor, "b n_q d_q"],  # latent hidden states
        x_kv: Float[Tensor, "b n_kv d_kv"],  # phoneme memory
        key_padding_mask: Bool[
            Tensor, "b n_kv"
        ],  # True = attending to (same convention as your code)
        rope_q: Tuple[Tensor, Tensor],  # RoPE for queries only
    ) -> Tensor:
        b, h = x_q.size(0), self.heads
        q = self.to_q(x_q)
        k = self.to_k(x_kv)
        v = self.to_v(x_kv)

        # RoPE on queries only (text already has its own positions from encoder)
        freqs, xpos_scale = rope_q
        q_xpos_scale = xpos_scale if isinstance(xpos_scale, torch.Tensor) else 1.0
        q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)  # type: ignore  # [b, n_q, h*d]

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        # broadcastable mask: [b, 1, n_q, n_kv]
        attn_mask = rearrange(key_padding_mask, "b n -> b 1 1 n").expand(
            b, h, q.size(-2), k.size(-2)
        )

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# RoPE
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        inv_freq = 1.0 / (1e4 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward_from_seq_len(self, seq_len: int):
        device: Device = self.inv_freq.device  # type: ignore
        t = torch.arange(seq_len, device=device)
        return self.forward(t)

    @autocast("cuda", enabled=False)
    @jaxtyped(typechecker=beartype)
    def forward(self, t: Int[Tensor, " n"]):
        t = rearrange(t, "n -> 1 n")
        freqs = torch.einsum(
            "b i , j -> b i j",
            t.type_as(self.inv_freq),  # type: ignore
            self.inv_freq,
        )
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")
        return freqs, 1.0


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim=-1)

    return out.type(orig_dtype)


class FF(nn.Module):
    def __init__(self, dim: int, dim_out: int, mlp_ratio: int, dropout: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(approximate="tanh")),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, x):
        return self.ff(x)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, enable_xattn: bool):
        super().__init__()

        self.xattn_enabled = enable_xattn
        if self.xattn_enabled:
            self.xattn_norm = AdaLayerNormZero(dim)
            self.xattn = CrossAttention(
                dim_q=dim, dim_kv=dim, heads=8, dim_head=dim // 8, dropout=0.1
            )

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            dim=dim,
            heads=8,
            dim_head=dim // 8,
            dropout=0.1,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FF(dim, dim, 4, 0.1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence_length hidden_dim"],
        emb: Float[Tensor, "batch hidden_dim"],
        mask: Bool[Tensor, "batch sequence_length"],
        phoneme_mem: Float[Tensor, "batch phoneme_length phoneme_dim"],
        phonemes_mask: Bool[Tensor, "batch phoneme_length"],
        rope,
    ):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=emb)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        if self.xattn_enabled:
            norm_x, gate_xattn, _, _, _ = self.xattn_norm(x, emb=emb)
            xattn_output = self.xattn(
                x_q=norm_x,
                x_kv=phoneme_mem,
                key_padding_mask=phonemes_mask,
                rope_q=rope,
            )
            x = x + gate_xattn.unsqueeze(1) * xattn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, 31, groups=16, padding=15),
            nn.Mish(),
            nn.Conv1d(dim, dim, 31, groups=16, padding=15),
            nn.Mish(),
        )

    def forward(
        self,
        x: Float[Tensor, "batch sequence_length hidden_dim"],
        mask: Bool[Tensor, "batch sequence_length"],
    ):
        mask = mask[..., None]
        x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)
        out = out.masked_fill(~mask, 0.0)
        return out


class InputEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=hidden_dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence_length hidden_dim"],
        cond: Float[Tensor, "batch sequence_length hidden_dim"],
        mask: Bool[Tensor, "batch sequence_length"],
    ):
        x = self.proj(torch.cat((x, cond), dim=-1))
        x = self.conv_pos_embed(x, mask) + x
        return x


def default_cross_indices(n: int) -> list[int]:
    idx = [0] + list(range(1, n, 2))  # 0 plus all odd layers
    return idx[: len(idx) // 2] + idx[len(idx) // 2 + 1 :]  # remove the middle one


class DiT(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        phoneme_dim: int,
        hidden_dim: int,
        n_blocks: int,
    ):
        super().__init__()

        self.input_embed = InputEmbedding(latent_dim * 2, hidden_dim)
        self.rotary_embed = RotaryEmbedding(64)
        self.phoneme_proj = nn.Linear(phoneme_dim, hidden_dim)
        cross_indices = set(default_cross_indices(n_blocks))
        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, i in cross_indices) for i in range(n_blocks)]
        )
        self.norm_out = AdaLayerNormZero_Final(hidden_dim)

        for i, block in enumerate(self.transformer_blocks):
            nn.init.constant_(block.attn_norm.linear.weight, 0)  # type: ignore
            nn.init.constant_(block.attn_norm.linear.bias, 0)  # type: ignore
            if i in cross_indices:
                nn.init.constant_(block.xattn_norm.linear.weight, 0)  # type: ignore
                nn.init.constant_(block.xattn_norm.linear.bias, 0)  # type: ignore

        # zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence_length latent_dim"],  # noised input audio
        cond: Float[Tensor, "batch sequence_length latent_dim"],  # masked true audio
        phoneme_embedding: Float[
            Tensor, "batch phoneme_length phoneme_embedding_dim"
        ],  # text
        phonemes_mask: Bool[Tensor, "batch phoneme_length"],  # phoneme mask
        time_embedding: Float[Tensor, "batch hidden_dim"],  # time
        mask: Bool[Tensor, "batch sequence_length"],  # regular mask
    ):
        seq_len = x.shape[1]
        x = self.input_embed(x, cond, mask)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        phoneme_mem = self.phoneme_proj(phoneme_embedding)
        phoneme_mem = phoneme_mem.masked_fill_(
            ~phonemes_mask.unsqueeze(-1).expand(-1, -1, phoneme_mem.size(-1)), 0
        )

        stacked_transformer_features = []
        for block in self.transformer_blocks:
            x = block(
                x=x,
                emb=time_embedding,
                mask=mask,
                phoneme_mem=phoneme_mem,
                phonemes_mask=phonemes_mask,
                rope=rope,
            )
            stacked_transformer_features.append(x)
        x = self.norm_out(x, time_embedding)
        return x, torch.stack(stacked_transformer_features, dim=1)
