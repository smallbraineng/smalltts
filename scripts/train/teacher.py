# Launch with accelerate, e.g. uv run accelerate launch scripts/train/teacher.py
from typing import Tuple

import torch
from accelerate import Accelerator
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss
from tqdm import tqdm

from smalltts.data.dummy import get_dummy_dataloader
from smalltts.models.backbone.model import DiTModel
from smalltts.train.utils import get_alpha_sigma, get_mask

BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_STEPS = 600_000
NUM_SAVE_STEPS = 3_000
LOAD_FROM_CHECKPOINT: str | None = None


def get_noised_latents(
    latents: Float[Tensor, "batch sequence_length latent_dim"],
    timesteps: Float[Tensor, "batch"],
) -> Tuple[
    Float[Tensor, "batch sequence_length latent_dim"],
    Float[Tensor, "batch sequence_length latent_dim"],
]:
    alpha, sigma = get_alpha_sigma(timesteps)
    noise = torch.randn_like(latents)
    alpha = alpha.view(-1, 1, 1)
    sigma = sigma.view(-1, 1, 1)
    noised = alpha * latents + sigma * noise
    true_velocity = alpha * noise - sigma * latents
    return noised, true_velocity


if __name__ == "__main__":
    train_loader = get_dummy_dataloader(BATCH_SIZE, NUM_WORKERS)
    accelerator = Accelerator()
    model = DiTModel(64).to(accelerator.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=4_000,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_STEPS - 4_000,
        eta_min=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[4_000],
    )

    train_loader, model, scheduler, optimizer = accelerator.prepare(
        train_loader, model, scheduler, optimizer
    )

    train = iter(train_loader)

    if LOAD_FROM_CHECKPOINT is not None:
        print("loading from checkpoint", LOAD_FROM_CHECKPOINT)
        accelerator.load_state(LOAD_FROM_CHECKPOINT)

    pbar = tqdm(range(0, NUM_STEPS), desc="training")

    for step in pbar:
        batch = next(train)

        phonemes = batch["phonemes"].to(accelerator.device)
        phonemes_lengths = batch["phonemes_lengths"].to(accelerator.device)
        batch_size = phonemes.shape[0]
        phonemes_mask = get_mask(
            batch_size, phonemes.shape[1], phonemes_lengths, accelerator.device
        )

        latent_lengths = batch["latents_lengths"].to(accelerator.device)

        # cfg prob is 10%
        cfg_mask = torch.rand(phonemes.shape[0], device=accelerator.device) < 0.09
        phonemes[cfg_mask] = 0
        phonemes_mask[cfg_mask] = False

        latents = batch["latents"].to(accelerator.device)
        ref_latents = batch["ref_latents"].to(accelerator.device)
        ref_latents_lengths = batch["ref_latents_lengths"].to(accelerator.device)

        timesteps = torch.rand(batch_size).to(accelerator.device)
        noised, true_velocity = get_noised_latents(latents, timesteps)

        mask = get_mask(
            batch_size, latents.shape[1], latent_lengths, accelerator.device
        )

        velocity = model(
            noised,
            ref_latents,
            ref_latents_lengths,
            mask,
            phonemes,
            phonemes_mask,
            timesteps,
        )

        valid = mask.unsqueeze(-1).expand(-1, -1, 64)

        velocity = velocity * valid
        true_velocity = true_velocity * valid

        optimizer.zero_grad()
        loss = mse_loss(velocity, true_velocity, reduction="sum") / valid.sum()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        gathered_loss = accelerator.gather(loss).mean().item()  # type: ignore
        accelerator.log({"teacher_loss": gathered_loss}, step=step)
        pbar.set_postfix(loss=f"{gathered_loss:.3f}")

        if accelerator.is_main_process and step % NUM_SAVE_STEPS == 0 and step > 1:
            print("saving checkpoint")
            accelerator.save_state("assets/teacher_checkpoints/checkpoint_latest")
            torch.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                },
                "assets/teacher_checkpoints/checkpoint_latest.pt",
            )

        del batch, noised, mask, velocity, true_velocity, loss
