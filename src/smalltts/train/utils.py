import math
from typing import Tuple

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, Int64, jaxtyped
from torch import Tensor


# look at "diffusion noise schedule" in SimpleTTS: https://openreview.net/pdf?id=m4mwbPjOwb
def get_alpha_sigma(
    t: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    t = t.clamp(min=eps, max=1 - 1e-5)
    alpha_t_sq = torch.cos(math.pi / 2 * t) ** 2
    log_snr = torch.log(alpha_t_sq / (1 - alpha_t_sq))
    log_snr_s = log_snr + 2 * np.log(0.5)
    alpha_sq = torch.sigmoid(log_snr_s)
    sigma = torch.sqrt(1 - alpha_sq)
    alpha = torch.sqrt(alpha_sq)
    return alpha, sigma


@jaxtyped(typechecker=beartype)
def get_random_cond(
    data: Float[Tensor, "batch sequence_length mel_dim"],
    lengths: Int64[Tensor, " batch"],
    device: torch.device,
):
    cond_mask = torch.zeros_like(data).to(dtype=torch.bool).to(device)
    for i, length in enumerate(lengths):
        mask_len = int(torch.randint(0, int(length.item()) // 2, (1,)).item())
        mask_start = torch.randint(0, int(length.item()) - mask_len, (1,)).item()
        cond_mask[i, mask_start : mask_start + mask_len, :] = True
    cond = data * cond_mask
    return cond, cond_mask


@jaxtyped(typechecker=beartype)
def get_mask(
    batch_size: int,
    max_sequence_length: int,
    latents_lengths: Int64[Tensor, " batch"],
    device: torch.device,
):
    mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.bool).to(device)
    for i, length in enumerate(latents_lengths):
        mask[i, :length] = True
    return mask


@jaxtyped(typechecker=beartype)
def apply_noise(
    mels: Float[Tensor, "batch sequence_length mel_dim"],
    timesteps: Float[Tensor, " batch"],
) -> Tuple[
    Float[Tensor, "batch sequence_length mel_dim"],
    Float[Tensor, "batch sequence_length mel_dim"],
]:
    alpha, sigma = get_alpha_sigma(timesteps)
    noise = torch.randn_like(mels)
    alpha = alpha.view(-1, 1, 1)
    sigma = sigma.view(-1, 1, 1)
    noised = alpha * mels + sigma * noise
    true_velocity = alpha * noise - sigma * mels
    return noised, true_velocity
