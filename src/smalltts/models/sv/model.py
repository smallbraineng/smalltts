import torch.nn as nn
from jaxtyping import Float, Int64
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from torch import Tensor

from smalltts.models.asr import ASR


class SV(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        asr: ASR,
    ):
        super().__init__()
        self.encoder = asr.encoder
        self.hidden_dim = 64
        self.ecapa = ECAPA_TDNN(input_size=self.hidden_dim, lin_neurons=emb_dim)

    def forward(
        self,
        latents: Float[Tensor, "batch latent_len latent_dim"],
        lengths: Int64[Tensor, " batch"],
    ) -> Float[Tensor, "batch emb_dim"]:
        feats, lengths = self.encoder(latents, lengths)  # [B, T, 64]
        audio_lengths = lengths.float()
        max_length = audio_lengths.max()
        audio_lengths = audio_lengths / max_length
        trunk = self.ecapa(feats, audio_lengths).squeeze(1)  # [B, emb_dim]
        return trunk
