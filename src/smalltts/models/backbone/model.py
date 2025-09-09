import math

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch import Tensor, nn

from smalltts.data.phonemization.phonemes import phoneme_len

from .dit import DiT
from .phonemes import PhonemeEmbedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = 256
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

    @jaxtyped(typechecker=beartype)
    def forward(self, t: Float[Tensor, " batch"]) -> Float[Tensor, "batch length"]:
        # sinusoidal
        half = self.dim // 2
        emb = math.log(1e4) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device).float() * -emb)
        emb = 1e3 * t.unsqueeze(1) * emb.unsqueeze(0)  # scale by 1_000
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # mlp
        emb = self.mlp(emb)
        return emb


class Backbone(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        hidden_dim = 896
        phoneme_dim = 512
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.phoneme_embedding = PhonemeEmbedding(phoneme_len, phoneme_dim, 8)  # 4, 8
        self.dit = DiT(latent_dim, phoneme_dim, hidden_dim, 18)  # 8, 18
        self.velocity = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.velocity.weight, 0)
        nn.init.constant_(self.velocity.bias, 0)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        noised: Float[Tensor, "batch sequence_length latent_dim"],
        cond: Float[Tensor, "batch sequence_length latent_dim"],
        mask: Bool[Tensor, "batch sequence_length"],
        phonemes: Int64[Tensor, "batch phoneme_length"],
        phonemes_mask: Bool[Tensor, "batch phoneme_length"],
        t: Float[Tensor, " batch"],
        get_stacked_transformer_features: bool = False,
    ):
        # [batch, latent_length, 64]
        phoneme_embedding = self.phoneme_embedding(phonemes, phonemes_mask)
        time_embedding = self.time_embedding(t)

        decoded, stacked_transformer_features = self.dit(
            noised,
            cond,
            phoneme_embedding,
            phonemes_mask,
            time_embedding,
            mask,
        )
        velocity = self.velocity(decoded)  # [batch, latent_length, latent_dim]

        if get_stacked_transformer_features:
            return velocity, stacked_transformer_features
        return velocity


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    seq_len = 40

    model = Backbone(64).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_params:,}")

    latents = torch.randn(batch_size, seq_len, 64).to(device)
    noised = torch.randn_like(latents)
    phonemes = torch.randint(0, 50, (batch_size, phoneme_len)).to(device)
    phonemes[1, phoneme_len // 2 :] = 0
    phonemes_mask = torch.ones_like(phonemes, dtype=torch.bool).to(device)
    phonemes_mask[1, phoneme_len // 2 :] = False

    cond_length = torch.randint(
        0, latents.shape[1] // 2 + 1, (latents.shape[0],), device=device
    )
    cond = torch.zeros_like(latents, device=device)  # [batch, latent_length]

    for i in range(cond.shape[0]):
        cond[i, : cond_length[i]] = latents[i, : cond_length[i]]

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    mask[1, seq_len // 2 :] = False

    t = torch.rand(batch_size).to(device)

    model.train()

    start_time = time.time()
    output = model(noised, cond, mask, phonemes, phonemes_mask, t)
    end_time = time.time()
    print(f"took {end_time - start_time:.4f} seconds")

    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print(f"output shape: {output.shape}")
    print("fw and bw passes successful")
