# Adapted from https://github.com/lucasnewman/nanospeech/blob/main/nanospeech/nanospeech_torch.py

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped
from torch import Tensor


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


class PhonemeEmbedding(nn.Module):
    def __init__(self, phoneme_length: int, dim: int, conv_layers: int):
        super().__init__()
        self.phoneme_embed = nn.Embedding(phoneme_length, dim)  # use 0 as filler token
        self.extra_modeling = True
        self.precompute_max_pos = 2_048
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim, self.precompute_max_pos),
            persistent=False,
        )
        self.convnextv2 = nn.Sequential(
            *[ConvNeXtV2Block(dim, dim * 2) for _ in range(conv_layers)]
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        phonemes: Int[Tensor, "batch phoneme_length"],
        phonemes_mask: Bool[Tensor, "batch phoneme_length"],
    ):
        phonemes = self.phoneme_embed(phonemes)
        batch = phonemes.size(0)
        seq_len = phonemes.size(1)
        batch_start = torch.zeros((batch,), dtype=torch.long)
        pos_idx = get_pos_embed_indices(
            batch_start, seq_len, max_pos=self.precompute_max_pos
        )
        pos_embed = self.freqs_cis[pos_idx]  # type: ignore
        phonemes = phonemes + pos_embed
        phonemes = phonemes.masked_fill_(
            ~phonemes_mask.unsqueeze(-1).expand(-1, -1, phonemes.size(-1)), 0.0
        )
        for block in self.convnextv2:
            phonemes = block(phonemes)
            phonemes = phonemes.masked_fill_(
                ~phonemes_mask.unsqueeze(-1).expand(-1, -1, phonemes.size(-1)), 0.0
            )
        return phonemes
