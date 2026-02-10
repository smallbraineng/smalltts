import torch.nn as nn
from jaxtyping import Float, Int64
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from torch import Tensor


class SV(nn.Module):
    def __init__(
        self,
        emb_dim: int,
    ):
        super().__init__()
        self.ecapa = ECAPA_TDNN(
            input_size=64,
            lin_neurons=emb_dim,
            channels=[768, 768, 768, 768, 2_304],
            kernel_sizes=[3, 3, 3, 3, 1],
            dilations=[1, 2, 3, 5, 1],
            attention_channels=192,
            res2net_scale=12,
            se_channels=192,
            global_context=True,
            dropout=0.1,
        )

    def forward(
        self,
        latents: Float[Tensor, "batch latent_len latent_dim"],
        lengths: Int64[Tensor, " batch"],
    ) -> Float[Tensor, "batch emb_dim"]:
        audio_lengths = lengths.float()
        max_length = audio_lengths.max()
        audio_lengths = audio_lengths / max_length
        trunk = self.ecapa(latents, audio_lengths).squeeze(1)  # [B, emb_dim]
        return trunk
