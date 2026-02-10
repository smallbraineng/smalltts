import torch
from jaxtyping import Bool, Float, Int64
from torch import Tensor, nn
from torchaudio.models.conformer import ConformerLayer

from smalltts.data.phonemization.phonemes import phoneme_len


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

    def forward(
        self, input: torch.Tensor, encoder_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, key_padding_mask=encoder_padding_mask)
        return x.transpose(0, 1)


class Discriminator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        transformer_dim: int = 768,
        ref_dim: int = 768,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.layers_proj = nn.Linear(3 * transformer_dim, 512)
        self.audio_proj = nn.Linear(latent_dim, 512)
        self.phoneme_embed = nn.Embedding(phoneme_len, 512)
        self.ref_proj = nn.Linear(ref_dim, 512)
        self.cond_proj = nn.Linear(2, 512)

        self.enc_a = Conformer(
            input_dim=512,
            num_layers=6,
            num_heads=8,
            ffn_dim=1_024,
            depthwise_conv_kernel_size=7,
            use_group_norm=True,
        )

        self.out = nn.Conv1d(512, 1, kernel_size=1)

    def forward(
        self,
        stacked_transformer_layers: Float[
            Tensor, "batch num_layers sequence_length hidden_dim"
        ],
        noised: Float[Tensor, "batch sequence_length latent_dim"],
        ref_seq: Float[Tensor, "batch ref_len ref_dim"],
        ref_mask: Bool[Tensor, "batch ref_len"],
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

        mask_f = mask.float().unsqueeze(-1)
        t_f = t.view(batch_size, 1, 1).expand(-1, noised.shape[1], 1)
        cond = self.cond_proj(torch.cat([mask_f, t_f], dim=-1))

        ref_proj = self.ref_proj(ref_seq)
        c_phonemes = self.phoneme_embed(phonemes)

        feats = torch.cat([layers_proj, noised_proj, cond, ref_proj, c_phonemes], dim=1)

        phoneme_mask = phonemes != 0
        encoder_key_padding_mask = ~torch.cat(
            [mask, mask, mask, ref_mask, phoneme_mask], dim=1
        )

        enc = self.enc_a(feats, encoder_key_padding_mask)
        valid = (~encoder_key_padding_mask).float()
        y = self.out(enc.transpose(1, 2)).squeeze(1)
        logits = (y * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return logits


def test_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    seq_len = 20
    phon_len = 5
    ref_len = 6
    n_layers = 18
    latent_dim = 64
    transformer_dim = 768
    ref_dim = 768

    discriminator = Discriminator(latent_dim, transformer_dim, ref_dim).to(device)

    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"total parameters: {total_params:,}")

    stacked_transformer_layers = torch.randn(
        batch_size, n_layers, seq_len, transformer_dim
    ).to(device)
    noised = torch.randn(batch_size, seq_len, latent_dim).to(device)
    ref_seq = torch.randn(batch_size, ref_len, ref_dim).to(device)
    ref_mask = torch.ones(batch_size, ref_len, dtype=torch.bool).to(device)
    ref_mask[1, ref_len // 2 :] = False
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    mask[1, seq_len // 2 :] = False
    phonemes = torch.randint(0, phoneme_len, (batch_size, phon_len)).to(device)
    phonemes[1, phon_len // 2 :] = 0
    t = torch.rand(batch_size).to(device)

    discriminator.train()
    output = discriminator(
        stacked_transformer_layers, noised, ref_seq, ref_mask, mask, phonemes, t
    )

    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)

    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print(f"forward pass successful! output shape: {output.shape}")
    print(f"loss: {loss.item():.4f}")
    print("backward pass completed successfully!")

    return output, loss


if __name__ == "__main__":
    test_forward_backward()
