import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch import Tensor, nn
from torchaudio.models.conformer import ConformerLayer

from smalltts.data.phonemization.phonemes import phoneme_len


# slightly modified torchaudio conformer, to specify a mask
class Conformer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    # change here, directly pass encoder_padding_mask
    def forward(
        self, input: torch.Tensor, encoder_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, key_padding_mask=encoder_padding_mask)
        return x.transpose(0, 1)


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.transformer_dim = 896
        self.layers_proj = nn.Linear(3 * self.transformer_dim, 512)  # last three layers
        self.audio_proj = nn.Linear(latent_dim, 512)
        self.input_fuse_proj = nn.Sequential(nn.Linear(3 * 512, 512), nn.GELU())
        self.phoneme_embed = nn.Embedding(phoneme_len, 512)
        self.c_audio_proj = nn.Linear(latent_dim + 1 + 1, 512)
        self.post_proj_norm = nn.LayerNorm(512)

        self.enc_a = Conformer(
            input_dim=512,
            num_layers=6,
            num_heads=8,
            ffn_dim=1_024,
            depthwise_conv_kernel_size=7,
            use_group_norm=True,
        )

        self.enc_b = Conformer(
            input_dim=512,
            num_layers=2,
            num_heads=8,
            ffn_dim=1_024,
            depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.out = nn.Conv1d(512, 1, kernel_size=1)

    @jaxtyped(typechecker=beartype)
    def _build_audio_cond_proj(
        self,
        cond: Float[Tensor, "batch sequence_length latent_dim"],
        mask: Bool[Tensor, "batch sequence_length"],
        t: Float[Tensor, " batch"],
        phonemes: Int64[Tensor, "batch phoneme_length"],
    ) -> Float[Tensor, "batch sequence_length_plus_phoneme_length 512"]:
        batch_size, sequence_length, _ = cond.shape
        mask_f = mask.float().unsqueeze(-1)
        t_f = t.view(batch_size, 1, 1).expand(-1, sequence_length, 1)
        c_audio = torch.cat([cond, mask_f, t_f], dim=-1)
        c_audio_proj = self.c_audio_proj(c_audio)
        return c_audio_proj

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        stacked_transformer_layers: Float[
            Tensor, "batch num_layers sequence_length 896"
        ],
        noised: Float[Tensor, "batch sequence_length latent_dim"],
        cond: Float[Tensor, "batch sequence_length latent_dim"],
        mask: Bool[Tensor, "batch sequence_length"],
        phonemes: Int64[Tensor, "batch phoneme_length"],
        t: Float[Tensor, " batch"],
    ):
        stacked_transformer_layers = stacked_transformer_layers[:, -3:, :, :]
        batch_size, num_layers, sequence_length, _ = stacked_transformer_layers.shape
        transformer_layers_flat = stacked_transformer_layers.permute(
            0, 2, 1, 3
        ).reshape(batch_size, sequence_length, num_layers * self.transformer_dim)
        layers_proj = self.layers_proj(transformer_layers_flat)
        noised_proj = self.audio_proj(noised)
        c_audio_proj = self._build_audio_cond_proj(cond, mask, t, phonemes)

        x_fused = torch.cat([layers_proj, noised_proj, c_audio_proj], dim=-1)
        x_fused = self.input_fuse_proj(x_fused)
        x_fused = self.post_proj_norm(x_fused)

        c_phonemes = self.phoneme_embed(phonemes)

        feats = torch.cat([x_fused, c_phonemes], dim=1)

        phoneme_mask = phonemes != 0
        encoder_key_padding_mask = ~torch.cat([mask, phoneme_mask], dim=1)

        enc = self.enc_a(feats, encoder_key_padding_mask)
        enc = self.enc_b(enc, encoder_key_padding_mask)
        valid = (~encoder_key_padding_mask).float()
        y = self.out(enc.transpose(1, 2)).squeeze(1)  # [B, L]
        logits = (y * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    seq_len = 20
    phoneme_len = 5
    n_layers = 3
    latent_dim = 128
    phoneme_dim = 177

    discriminator = Discriminator(latent_dim).to(device)

    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"total parameters: {total_params:,}")
    stacked_transformer_layers = torch.randn(batch_size, n_layers, seq_len, 896).to(
        device
    )
    noised = torch.randn(batch_size, seq_len, latent_dim).to(device)
    cond = torch.randn(batch_size, seq_len, latent_dim).to(device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    mask[1, seq_len // 2 :] = False
    phonemes = torch.randint(0, phoneme_dim, (batch_size, phoneme_len)).to(device)
    phonemes[1, phoneme_len // 2 :] = 0
    t = torch.rand(batch_size).to(device)

    discriminator.train()
    output = discriminator(stacked_transformer_layers, noised, cond, mask, phonemes, t)

    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)

    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print("fw and bw pass completed")
