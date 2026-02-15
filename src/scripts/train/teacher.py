from typing import Tuple

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import jaxtyping
import torch
from accelerate import Accelerator
from jaxtyping import Float

jaxtyping.config.update("jaxtyping_disable", True)
from torch import Tensor
from torch.nn.functional import mse_loss
from tqdm import tqdm

from smalltts.data.dummy import get_dummy_dataloader
from smalltts.models.backbone.model import DiTModel
from smalltts.train.utils import get_alpha_sigma, get_mask

BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_STEPS = 330_000
NUM_SAVE_STEPS = 1_500
LOAD_FROM_CHECKPOINT: str | None = None
PRETRAINED_WEIGHTS: str | None = None
TEXT_CFG_DROP = 0.10
SPEAKER_CFG_DROP = 0.10


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
    if PRETRAINED_WEIGHTS is not None:
        print("loading pretrained weights", PRETRAINED_WEIGHTS)
        ckpt = torch.load(PRETRAINED_WEIGHTS, map_location=accelerator.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        cleaned = {}
        for k, v in ckpt.items():
            if k in ("initted", "step"):
                continue
            for prefix in ("module.", "_orig_mod.", "ema_model.", "online_model."):
                while k.startswith(prefix):
                    k = k[len(prefix) :]
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)
    model.dit = torch.compile(model.dit, dynamic=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), weight_decay=1e-2
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=1_500
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_STEPS - 1_500, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[1_500]
    )

    train_loader, model, scheduler, optimizer = accelerator.prepare(
        train_loader, model, scheduler, optimizer
    )

    from ema_pytorch import EMA

    ema = EMA(model, beta=0.9999, update_every=1)

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

        latents_lengths = batch["latents_lengths"].to(accelerator.device)

        text_cfg_mask = (
            torch.rand(batch_size, device=accelerator.device) < TEXT_CFG_DROP
        )
        phonemes[text_cfg_mask] = 0
        phonemes_mask[text_cfg_mask] = False

        latents = batch["latents"].to(accelerator.device)
        ref_latents = batch["ref_latents"].to(accelerator.device)
        ref_latents_lengths = batch["ref_latents_lengths"].to(accelerator.device)

        speaker_cfg_mask = (
            torch.rand(batch_size, device=accelerator.device) < SPEAKER_CFG_DROP
        )
        ref_latents[speaker_cfg_mask] = 0
        ref_latents_lengths[speaker_cfg_mask] = 0

        timesteps = torch.sigmoid(torch.randn(batch_size)).to(accelerator.device)
        noised, true_velocity = get_noised_latents(latents, timesteps)

        mask = get_mask(
            batch_size, latents.shape[1], latents_lengths, accelerator.device
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

        diff = (velocity - true_velocity) ** 2
        loss = (diff * valid).sum() / valid.sum()

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ema.update()

        if step % 100 == 0:
            gathered_loss = accelerator.gather(loss).mean().item()  # type: ignore
            log_dict = {"teacher_loss": gathered_loss}
            accelerator.log(log_dict, step=step)
            pbar.set_postfix(loss=f"{gathered_loss:.3f}")

        if accelerator.is_main_process and step % NUM_SAVE_STEPS == 0 and step > 1:
            print("saving checkpoint")
            raw = model
            while hasattr(raw, "module"):
                raw = raw.module
            torch.save(
                raw.state_dict(), "assets/teacher_checkpoints/checkpoint_latest.pt"
            )
            torch.save(
                ema.ema_model.state_dict(),
                "assets/teacher_checkpoints/checkpoint_ema.pt",
            )

        del batch, noised, mask, velocity, true_velocity, loss
