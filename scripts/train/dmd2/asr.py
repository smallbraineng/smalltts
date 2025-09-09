# Note: If you are on a mac, use the --cpu flag in accelerate `uv run accelerate launch --cpu scripts/train/asr.py`
import torch
from accelerate.accelerator import Accelerator
from torch import nn
from tqdm import tqdm

from smalltts.data.dummy import get_dummy_dataloader
from smalltts.models.asr import ASR

BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_STEPS = 200_000
NUM_SAVE_STEPS = 2_000
LOAD_FROM_CHECKPOINT: str | None = None

if __name__ == "__main__":
    accelerator = Accelerator()

    train_loader = get_dummy_dataloader(BATCH_SIZE, NUM_WORKERS)

    asr = ASR(64).to(accelerator.device)

    optimizer = torch.optim.AdamW(
        asr.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2
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

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    train_loader, asr, scheduler, optimizer = accelerator.prepare(
        train_loader, asr, scheduler, optimizer
    )

    if LOAD_FROM_CHECKPOINT is not None:
        print(f"loading from checkpoint {LOAD_FROM_CHECKPOINT}")
        accelerator.load_state(LOAD_FROM_CHECKPOINT)

    train = iter(train_loader)
    pbar = tqdm(range(0, NUM_STEPS), desc="asr train")

    for step in pbar:
        batch = next(train)

        latents = batch["latents"].to(accelerator.device)
        batch_size = latents.shape[0]
        latents_lengths = batch["latents_lengths"].to(accelerator.device)

        phonemes = batch["phonemes"]

        lp, lp_lens = asr(latents, latents_lengths)
        target_lens = (phonemes != 0).sum(dim=1)

        lp = lp.permute(1, 0, 2)
        loss = ctc_loss(lp, phonemes, lp_lens, target_lens)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        gathered_loss = accelerator.gather(loss).mean().item()  # type: ignore

        accelerator.log({"loss": gathered_loss}, step=step)
        pbar.set_postfix(
            {"loss": f"{gathered_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
        )

        if accelerator.is_main_process and step % NUM_SAVE_STEPS == 0 and step > 0:
            print("saving checkpoint")
            accelerator.save_state("conformer_checkpoints/checkpoint_latest")
            torch.save(
                {
                    "model": accelerator.unwrap_model(asr).state_dict(),
                },
                "conformer_checkpoints/checkpoint_latest.pt",
            )

        del batch
