import torch
import torch.nn as nn
import torch.nn.functional as F
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


class RMSNorm(nn.Module):
    def __init__(self, model_size: int | tuple, eps: float):
        super().__init__()
        self.eps = eps
        if isinstance(model_size, int):
            model_size = (model_size,)
        self.weight = nn.Parameter(torch.ones(model_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dim() == 1:
            return F.rms_norm(x, self.weight.shape, self.weight, self.eps)
        return F.rms_norm(x, (self.weight.shape[-1],), eps=self.eps) * self.weight


class JointAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head

        self.to_q = nn.Linear(dim, inner)
        self.to_k_self = nn.Linear(dim, inner)
        self.to_v_self = nn.Linear(dim, inner)
        self.gate = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim, bias=False), nn.Dropout(dropout)
        )

        self.q_norm = RMSNorm((heads, dim_head), eps=1e-6)
        self.k_norm = RMSNorm((heads, dim_head), eps=1e-6)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch seq_len hidden_dim"],
        k_ref: Float[Tensor, "batch heads ref_len dim_head"],
        v_ref: Float[Tensor, "batch heads ref_len dim_head"],
        k_text: Float[Tensor, "batch heads text_len dim_head"],
        v_text: Float[Tensor, "batch heads text_len dim_head"],
        mask: Bool[Tensor, "batch seq_len"],
        ref_mask: Bool[Tensor, "batch ref_len"],
        text_mask: Bool[Tensor, "batch text_len"],
        rope,
    ) -> Float[Tensor, "batch seq_len hidden_dim"]:
        b, n_q, h = x.shape[0], x.shape[1], self.heads

        gate = self.gate(x)

        q = self.to_q(x)
        k_self, v_self = self.to_k_self(x), self.to_v_self(x)

        q = self.q_norm(q.reshape(b, n_q, h, self.dim_head))
        q = rearrange(q, "b n h d -> b h n d")

        k_self = k_self.reshape(b, n_q, h, self.dim_head)
        k_self = self.k_norm(k_self)
        k_self = rearrange(k_self, "b n h d -> b h n d")
        v_self = rearrange(v_self, "b n (h d) -> b h n d", h=h)

        freqs, xpos_scale = rope
        q_scale, k_scale = (
            (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        )
        q = apply_rotary_pos_emb(q, freqs, q_scale)
        k_self = apply_rotary_pos_emb(k_self, freqs, k_scale)

        k = torch.cat([k_self, k_ref, k_text], dim=2)
        v = torch.cat([v_self, v_ref, v_text], dim=2)

        joint_mask = torch.cat([mask, ref_mask, text_mask], dim=1)
        float_mask = torch.where(joint_mask, 0.0, float("-inf"))
        attn_mask = rearrange(float_mask, "b n -> b 1 1 n").expand(
            b, h, n_q, joint_mask.shape[1]
        )

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = rearrange(out, "b h n d -> b n (h d)").to(q.dtype)
        out = out * torch.sigmoid(gate)
        out = self.to_out(out)
        out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out


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
def apply_rotary_pos_emb(t, freqs, scale: float | torch.Tensor = 1):
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
    def __init__(self, dim: int, dim_out: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class DiTBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = JointAttention(dim=dim, heads=8, dim_head=dim // 8, dropout=0.1)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FF(dim, dim, 8 / 3, 0.1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence_length hidden_dim"],
        emb: Float[Tensor, "batch hidden_dim"],
        mask: Bool[Tensor, "batch sequence_length"],
        k_ref: Float[Tensor, "batch heads ref_len dim_head"],
        v_ref: Float[Tensor, "batch heads ref_len dim_head"],
        ref_mask: Bool[Tensor, "batch ref_len"],
        k_text: Float[Tensor, "batch heads text_len dim_head"],
        v_text: Float[Tensor, "batch heads text_len dim_head"],
        phonemes_mask: Bool[Tensor, "batch text_len"],
        rope,
    ) -> Float[Tensor, "batch sequence_length hidden_dim"]:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=emb)
        attn_output = self.attn(
            x=norm,
            k_ref=k_ref,
            v_ref=v_ref,
            k_text=k_text,
            v_text=v_text,
            mask=mask,
            ref_mask=ref_mask,
            text_mask=phonemes_mask,
            rope=rope,
        )
        x = x + gate_msa.unsqueeze(1) * attn_output
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
        mask: Bool[Tensor, "batch sequence_length"],
    ):
        x = self.proj(x)
        x = self.conv_pos_embed(x, mask) + x
        return x


class DiT(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        phoneme_dim: int,
        hidden_dim: int,
        n_blocks: int,
    ):
        super().__init__()
        self.heads = 8
        self.dim_head = hidden_dim // self.heads
        inner = hidden_dim

        self.input_embed = InputEmbedding(latent_dim, hidden_dim)
        self.rotary_embed = RotaryEmbedding(64)
        self.phoneme_proj = nn.Linear(phoneme_dim, hidden_dim)

        self.to_k_ref = nn.Linear(hidden_dim, inner)
        self.to_v_ref = nn.Linear(hidden_dim, inner)
        self.to_k_text = nn.Linear(hidden_dim, inner)
        self.to_v_text = nn.Linear(hidden_dim, inner)
        self.k_norm_cross = RMSNorm((self.heads, self.dim_head), eps=1e-6)

        self.emb_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(hidden_dim) for _ in range(n_blocks)]
        )
        self.norm_out = AdaLayerNormZero_Final(hidden_dim)

        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)  # type: ignore
            nn.init.constant_(block.attn_norm.linear.bias, 0)  # type: ignore
        nn.init.constant_(self.norm_out.linear.weight, 0)  # type: ignore
        nn.init.constant_(self.norm_out.linear.bias, 0)  # type: ignore

    def _project_cross_kv(self, seq: Tensor, to_k: nn.Linear, to_v: nn.Linear):
        b, n = seq.shape[:2]
        h, d = self.heads, self.dim_head
        k = to_k(seq).reshape(b, n, h, d)
        k = self.k_norm_cross(k)
        k = rearrange(k, "b n h d -> b h n d")
        v = rearrange(to_v(seq), "b n (h d) -> b h n d", h=h)
        return k, v

    def encode_cross_kv(
        self, ref_seq, ref_mask, phoneme_embedding, phonemes_mask, seq_len
    ):
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        phoneme_mem = self.phoneme_proj(phoneme_embedding)
        phoneme_mem = phoneme_mem.masked_fill_(
            ~phonemes_mask.unsqueeze(-1).expand(-1, -1, phoneme_mem.size(-1)), 0
        )
        k_ref, v_ref = self._project_cross_kv(ref_seq, self.to_k_ref, self.to_v_ref)
        k_text, v_text = self._project_cross_kv(
            phoneme_mem, self.to_k_text, self.to_v_text
        )
        return {
            "k_ref": k_ref,
            "v_ref": v_ref,
            "ref_mask": ref_mask,
            "k_text": k_text,
            "v_text": v_text,
            "phonemes_mask": phonemes_mask,
            "rope": rope,
        }

    def forward_cached(self, x, time_embedding, mask, cached):
        x = self.input_embed(x, mask)
        emb = self.emb_proj(time_embedding)
        for block in self.transformer_blocks:
            x = block(
                x=x,
                emb=emb,
                mask=mask,
                k_ref=cached["k_ref"],
                v_ref=cached["v_ref"],
                ref_mask=cached["ref_mask"],
                k_text=cached["k_text"],
                v_text=cached["v_text"],
                phonemes_mask=cached["phonemes_mask"],
                rope=cached["rope"],
            )
        x = self.norm_out(x, emb)
        return x

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence_length latent_dim"],
        ref_seq: Float[Tensor, "batch ref_len hidden_dim"],
        ref_mask: Bool[Tensor, "batch ref_len"],
        phoneme_embedding: Float[Tensor, "batch phoneme_length phoneme_dim"],
        phonemes_mask: Bool[Tensor, "batch phoneme_length"],
        time_embedding: Float[Tensor, "batch hidden_dim"],
        mask: Bool[Tensor, "batch sequence_length"],
        get_stacked_transformer_features: bool = False,
    ):
        x = self.input_embed(x, mask)
        rope = self.rotary_embed.forward_from_seq_len(x.size(1))
        phoneme_mem = self.phoneme_proj(phoneme_embedding)
        phoneme_mem = phoneme_mem.masked_fill_(
            ~phonemes_mask.unsqueeze(-1).expand(-1, -1, phoneme_mem.size(-1)), 0
        )

        k_ref, v_ref = self._project_cross_kv(ref_seq, self.to_k_ref, self.to_v_ref)
        k_text, v_text = self._project_cross_kv(
            phoneme_mem, self.to_k_text, self.to_v_text
        )

        emb = self.emb_proj(time_embedding)

        stacked_transformer_features = [] if get_stacked_transformer_features else None
        for block in self.transformer_blocks:
            x = block(
                x=x,
                emb=emb,
                mask=mask,
                k_ref=k_ref,
                v_ref=v_ref,
                ref_mask=ref_mask,
                k_text=k_text,
                v_text=v_text,
                phonemes_mask=phonemes_mask,
                rope=rope,
            )
            if stacked_transformer_features is not None:
                stacked_transformer_features.append(x)
        x = self.norm_out(x, emb)
        if stacked_transformer_features is None:
            return x, None
        return x, torch.stack(stacked_transformer_features, dim=1)
