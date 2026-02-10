import math
import time

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch import Tensor, nn

from smalltts.data.phonemization.phonemes import phoneme_len

from .dit import DiT
from .phonemes import TextEncoder
from .style import StyleEncoder


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


class DiTModel(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        hidden_dim = 768
        phoneme_dim = 384

        self.time_embedding = TimeEmbedding(hidden_dim)
        self.phoneme_embedding = TextEncoder(
            vocab_size=phoneme_len,
            model_size=phoneme_dim,
            num_layers=6,
            num_heads=6,
            intermediate_size=768,
            norm_eps=1e-6,
        )

        self.style_encoder = StyleEncoder(out_dim=hidden_dim)
        self.dit = DiT(latent_dim, phoneme_dim, hidden_dim, 18)
        self.velocity = nn.Linear(hidden_dim, latent_dim)

        nn.init.constant_(self.velocity.weight, 0)
        nn.init.constant_(self.velocity.bias, 0)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        noised: Float[Tensor, "batch sequence_length latent_dim"],
        ref_latents: Float[Tensor, "batch ref_seq latent_dim"],
        ref_latents_lengths: Int64[Tensor, " batch"],
        mask: Bool[Tensor, "batch sequence_length"],
        phonemes: Int64[Tensor, "batch phoneme_length"],
        phonemes_mask: Bool[Tensor, "batch phoneme_length"],
        t: Float[Tensor, " batch"],
        get_stacked_transformer_features: bool = False,
    ):
        ref_seq, ref_mask = self.style_encoder(ref_latents, ref_latents_lengths)
        phoneme_embedding = self.phoneme_embedding(phonemes, phonemes_mask)
        time_embedding = self.time_embedding(t)

        decoded, stacked_transformer_features = self.dit(
            noised,
            ref_seq,
            ref_mask,
            phoneme_embedding,
            phonemes_mask,
            time_embedding,
            mask,
            get_stacked_transformer_features=get_stacked_transformer_features,
        )
        velocity = self.velocity(decoded)
        if get_stacked_transformer_features:
            assert stacked_transformer_features is not None
            return velocity, stacked_transformer_features
        return velocity

    def encode_conditions(
        self, ref_latents, ref_latents_lengths, phonemes, phonemes_mask, seq_len
    ):
        ref_seq, ref_mask = self.style_encoder(ref_latents, ref_latents_lengths)
        phoneme_embedding = self.phoneme_embedding(phonemes, phonemes_mask)
        return self.dit.encode_cross_kv(
            ref_seq, ref_mask, phoneme_embedding, phonemes_mask, seq_len
        )

    def denoise_step(self, noised, mask, t, cached):
        time_embedding = self.time_embedding(t)
        decoded = self.dit.forward_cached(noised, time_embedding, mask, cached)
        return self.velocity(decoded)


# Keep backward compat alias
Backbone = DiTModel


def test_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    seq_len = 40

    model = DiTModel(64).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_params:,}")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {n:,}")

    latents = torch.randn(batch_size, seq_len, 64).to(device)
    noised = torch.randn_like(latents)
    phonemes = torch.randint(0, 50, (batch_size, phoneme_len)).to(device)
    phonemes[1, phoneme_len // 2 :] = 0
    phonemes_mask = torch.ones_like(phonemes, dtype=torch.bool).to(device)
    phonemes_mask[1, phoneme_len // 2 :] = False

    ref_seq = 24
    ref_latents = torch.randn(batch_size, ref_seq, 64).to(device)
    ref_latents_lengths = torch.full((batch_size,), ref_seq, dtype=torch.int64).to(
        device
    )

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    mask[1, seq_len // 2 :] = False

    t = torch.rand(batch_size).to(device)

    model.train()

    start_time = time.time()
    output = model(
        noised,
        ref_latents,
        ref_latents_lengths,
        mask,
        phonemes,
        phonemes_mask,
        t,
    )
    end_time = time.time()
    print(f"time taken for model forward pass: {end_time - start_time:.4f} seconds")

    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print(f"forward pass successful! output shape: {output.shape}")
    print(f"loss: {loss.item():.4f}")
    print("backward pass completed successfully!")

    return output, loss


if __name__ == "__main__":
    test_forward_backward()
