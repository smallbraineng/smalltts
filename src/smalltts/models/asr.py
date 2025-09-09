from typing import Tuple

import torch.nn as nn
from jaxtyping import Float, Int64
from torch import Tensor
from torchaudio.models import Conformer

from smalltts.data.phonemization.phonemes import phoneme_len


class TimeDeconvUpsample(nn.Module):
    def __init__(self, d_model: int, r: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=r, stride=r, groups=d_model
        )

    def forward(
        self, x: Float[Tensor, "batch time dim"], lengths: Int64[Tensor, " batch"]
    ) -> Tuple[Float[Tensor, "batch upsampled_time dim"], Int64[Tensor, " batch"]]:
        y = self.deconv(x.transpose(1, 2)).transpose(1, 2)
        return y, lengths * y.size(1) // x.size(1)


class ASR(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = Conformer(
            input_dim=input_dim,
            num_heads=16,
            ffn_dim=1_024,
            num_layers=7,
            depthwise_conv_kernel_size=9,
            use_group_norm=False,
            convolution_first=False,
        )
        self.proj = nn.Linear(input_dim, phoneme_len)
        self.upsample = TimeDeconvUpsample(input_dim, 4)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        x: Float[Tensor, "batch latent_len latent_dim"],
        lengths: Int64[Tensor, " batch"],
    ) -> Tuple[
        Float[Tensor, "batch latent_len_expanded classes_len"], Int64[Tensor, " batch"]
    ]:
        x, lengths = self.upsample(x, lengths)
        out, out_lengths = self.encoder(x, lengths)
        logits = self.proj(out)
        log_probs = self.log_softmax(logits)
        return log_probs, out_lengths


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    latent_len = 100
    latent_dim = 64
    num_classes = phoneme_len

    model = ASR(latent_dim).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_params:,}")

    x = torch.randn(batch_size, latent_len, latent_dim).to(device)
    lengths = torch.randint(50, latent_len, (batch_size,)).to(device)
    lengths[0] = x.shape[1]
    lengths, _ = torch.sort(lengths, descending=True)

    log_probs, out_lengths = model(x, lengths)

    target_lengths = torch.randint(10, 30, (batch_size,)).to(device)
    max_target_len = target_lengths.max().item()
    targets = torch.randint(1, num_classes, (batch_size, int(max_target_len))).to(
        device
    )

    targets_flat = []
    for i in range(batch_size):
        targets_flat.append(targets[i, : target_lengths[i]])
    targets_flat = torch.cat(targets_flat)

    log_probs_transposed = log_probs.transpose(0, 1)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    loss = ctc_loss(log_probs_transposed, targets_flat, out_lengths, target_lengths)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("fw and bw pass completed")
