# Speaker verification model on latent representations used during DMD2 training, note: true model needs 16khz input
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator
from tqdm import tqdm

from smalltts.codec.onnx import Decoder
from smalltts.data.dummy import get_dummy_dataloader
from smalltts.models.asr import ASR
from smalltts.models.sv.model import SV
from smalltts.models.sv.true import get_embedding_model, get_true_embeddings

BATCH_SIZE = 2
NUM_WORKERS = 0
ASR_CKPT = "assets/asr_checkpoints/checkpoint_latest.pt"
NUM_STEPS = 200_000
LOAD_FROM_CHECKPOINT: str | None = None
SAVE_STEPS = 1_000


if __name__ == "__main__":
    accelerator = Accelerator()

    dataloader = get_dummy_dataloader(BATCH_SIZE, NUM_WORKERS)
    decoder = Decoder()

    true_model = get_embedding_model(accelerator)

    asr = ASR(64).to(accelerator.device)
    asr.load_state_dict(torch.load(ASR_CKPT, map_location=accelerator.device)["model"])
    asr.encoder.requires_grad_(True)

    sv = SV(192, asr)

    optimizer = torch.optim.AdamW(
        sv.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2
    )

    sv, optimizer, dataloader, true_model, decoder = accelerator.prepare(
        sv, optimizer, dataloader, true_model, decoder
    )

    pbar = tqdm(range(NUM_STEPS), desc="training")
    train_iter = iter(dataloader)

    if LOAD_FROM_CHECKPOINT is not None:
        print("loading from checkpoint", LOAD_FROM_CHECKPOINT)
        accelerator.load_state(LOAD_FROM_CHECKPOINT)

    resampler = torchaudio.transforms.Resample(
        orig_freq=24_000,
        new_freq=16_000,
    )

    for step in pbar:
        batch = next(train_iter)

        latents = batch["latents"].to(accelerator.device)
        latents_lengths = batch["latents_lengths"].to(accelerator.device)

        audio = decoder.decode(latents)
        audio = resampler(audio).to(accelerator.device)
        audio_lengths = [
            length * 21_333 for length in latents_lengths
        ]  # since we resample to 16khz

        audio_lengths = torch.tensor(audio_lengths, device=accelerator.device)

        spk = sv(latents, latents_lengths)

        true_speaker = get_true_embeddings(true_model, audio, audio_lengths)
        if torch.isnan(true_speaker).any():
            print(
                f"warning: NaNs detected in true_speaker at step {step}, replacing with zeros"
            )
            true_speaker = torch.nan_to_num(true_speaker, nan=0.0)

        spk = F.normalize(spk, dim=-1)
        true_speaker = F.normalize(true_speaker, dim=-1)

        true_speaker = true_speaker.to(accelerator.device)

        loss = (1.0 - F.cosine_similarity(spk, true_speaker, dim=-1)).mean()

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(sv.parameters(), 5.0)
        optimizer.step()

        loss = loss.item()

        pbar.set_postfix(
            loss=f"{loss:.6f}",
        )
        accelerator.log(
            {
                "loss": loss,
            },
            step,
        )

        if accelerator.is_main_process and step % SAVE_STEPS == 0 and step > 0:
            print("saving checkpoint")
            accelerator.save_state("assets/sv_checkpoints/checkpoint_latest")
            checkpoint = {
                "step": step,
                "model": accelerator.unwrap_model(sv).state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
            }
            torch.save(checkpoint, "assets/sv_checkpoints/checkpoint_latest.pt")
