import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from accelerate import Accelerator
from jaxtyping import Bool, Float, Int64
from torch import Tensor
from torch.nn import CTCLoss
from torch.nn.functional import mse_loss
from tqdm import tqdm

from smalltts.data.dummy import get_dummy_dataloader
from smalltts.models.asr import ASR
from smalltts.models.backbone.model import DiTModel
from smalltts.models.discriminator import Discriminator
from smalltts.models.sv.model import SV
from smalltts.train.utils import apply_noise, get_alpha_sigma, get_mask

BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_STEPS = 40_000
SCORER_UPDATES = 5
NUM_SAVE_STEPS = 800
TIMESTEPS = [1.0, 1.0, 0.75, 0.50, 0.25]
TEACHER_CHECKPOINT = "assets/teacher_checkpoints/checkpoint_ema.pt"
ASR_CHECKPOINT = "assets/asr_checkpoints/checkpoint_latest.pt"
SV_CHECKPOINT = "assets/sv_checkpoints/checkpoint_latest.pt"
LOAD_FROM_CHECKPOINT: str | None = None


def cosine_loss(x, y):
    return 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=-1)


def set_grad(model, set: bool):
    for p in model.parameters():
        p.requires_grad = set


def load_from_checkpoint(model, checkpoint_path: str, device: torch.device, key=None):
    print("loading checkpoint", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and key is not None and key in ckpt:
        ckpt = ckpt[key]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    cleaned = {}
    for k, v in ckpt.items():
        if k in ("initted", "step"):
            continue
        for prefix in ("module.", "_orig_mod.", "ema_model.", "online_model."):
            while k.startswith(prefix):
                k = k[len(prefix):]
        k = k.replace("._orig_mod.", ".")
        cleaned[k] = v
    sd = model.state_dict()
    filtered = {k: v for k, v in cleaned.items() if k in sd}
    model.load_state_dict(filtered)


def get_x_pred(
    model: DiTModel,
    x_t: Float[Tensor, "batch sequence_length latent_dim"],
    ref_latents: Float[Tensor, "batch ref_seq latent_dim"],
    ref_latents_lengths: Int64[Tensor, " batch"],
    mask: Bool[Tensor, "batch sequence_length"],
    phonemes: Int64[Tensor, "batch phoneme_length"],
    phonemes_mask: Bool[Tensor, "batch phoneme_length"],
    t: Float[Tensor, " batch"],
    cfg: bool,
    get_stacked_transformer_features: bool,
    cfg_scale_text: float = 2.0,
    cfg_scale_speaker: float = 1.5,
):
    stacked_features = None
    if cfg and not get_stacked_transformer_features:
        x_t_3 = x_t.repeat(3, 1, 1)
        ref_3 = torch.cat([ref_latents, ref_latents, torch.zeros_like(ref_latents)], dim=0)
        ref_len_3 = torch.cat([ref_latents_lengths, ref_latents_lengths, torch.zeros_like(ref_latents_lengths)], dim=0)
        mask_3 = mask.repeat(3, 1)
        ph_3 = torch.cat([phonemes, torch.zeros_like(phonemes), phonemes], dim=0)
        ph_mask_3 = torch.cat([phonemes_mask, torch.zeros_like(phonemes_mask).to(dtype=torch.bool), phonemes_mask], dim=0)
        t_3 = t.repeat(3)
        vel_3 = model(x_t_3, ref_3, ref_len_3, mask_3, ph_3, ph_mask_3, t_3)
        v_cond, v_uncond_text, v_uncond_spk = vel_3.chunk(3, dim=0)
        velocity = v_cond + cfg_scale_text * (v_cond - v_uncond_text) + cfg_scale_speaker * (v_cond - v_uncond_spk)
    elif get_stacked_transformer_features:
        velocity, stacked_features = model(
            x_t, ref_latents, ref_latents_lengths, mask,
            phonemes, phonemes_mask, t,
            get_stacked_transformer_features=True,
        )
    else:
        velocity = model(
            x_t, ref_latents, ref_latents_lengths, mask,
            phonemes, phonemes_mask, t,
        )
    alpha_t, sigma_t = get_alpha_sigma(t)
    alpha_t = alpha_t.view(-1, 1, 1)
    sigma_t = sigma_t.view(-1, 1, 1)
    x_pred = alpha_t * x_t - sigma_t * velocity
    if get_stacked_transformer_features:
        return x_pred, stacked_features
    else:
        return x_pred


if __name__ == "__main__":
    train_loader = get_dummy_dataloader(BATCH_SIZE, NUM_WORKERS)
    accelerator = Accelerator()

    student = DiTModel(64).to(accelerator.device)
    student.dit.gradient_checkpointing = True
    student_scorer = DiTModel(64).to(accelerator.device)
    teacher = DiTModel(64).to(accelerator.device)
    discriminator = Discriminator(64, transformer_dim=1024, ref_dim=1024).to(accelerator.device)
    asr = ASR(64).to(accelerator.device)
    sv = SV(192).to(accelerator.device)
    ctc_loss = CTCLoss(blank=0, zero_infinity=True)

    load_from_checkpoint(teacher, TEACHER_CHECKPOINT, accelerator.device)
    load_from_checkpoint(asr, ASR_CHECKPOINT, accelerator.device)
    load_from_checkpoint(sv, SV_CHECKPOINT, accelerator.device)

    if not LOAD_FROM_CHECKPOINT:
        print("initializing student & student scorer from teacher checkpoint")
        load_from_checkpoint(student, TEACHER_CHECKPOINT, accelerator.device)
        load_from_checkpoint(student_scorer, TEACHER_CHECKPOINT, accelerator.device)

    student_optimizer = torch.optim.AdamW(
        student.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2, fused=True
    )
    student_scorer_optimizer = torch.optim.AdamW(
        student_scorer.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        fused=True,
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        fused=True,
    )

    teacher.eval()
    set_grad(teacher, False)
    set_grad(asr, False)
    set_grad(sv, False)

    (
        train_loader,
        student,
        student_scorer,
        teacher,
        discriminator,
        asr,
        sv,
        ctc_loss,
        student_optimizer,
        student_scorer_optimizer,
        discriminator_optimizer,
    ) = accelerator.prepare(
        train_loader,
        student,
        student_scorer,
        teacher,
        discriminator,
        asr,
        sv,
        ctc_loss,
        student_optimizer,
        student_scorer_optimizer,
        discriminator_optimizer,
    )

    train = iter(train_loader)  # type: ignore

    current_step = 0
    if LOAD_FROM_CHECKPOINT is not None:
        print("loading from checkpoint", LOAD_FROM_CHECKPOINT)
        accelerator.load_state(LOAD_FROM_CHECKPOINT)
        current_step = 0
        for group in student_optimizer.state_dict()["state"].values():
            if "step" in group:
                current_step = max(current_step, int(group["step"].item()))

    pbar = tqdm(range(current_step, NUM_STEPS), desc="training")

    for step in pbar:
        batch = next(train)

        latents = batch["latents"].to(accelerator.device)
        latents_lengths = batch["latents_lengths"].to(accelerator.device)
        batch_size = latents.shape[0]
        mask = get_mask(
            batch_size, latents.shape[1], latents_lengths, accelerator.device
        )
        phonemes = batch["phonemes"].to(accelerator.device)
        phonemes_lengths = batch["phonemes_lengths"].to(accelerator.device)
        phonemes_mask = get_mask(
            batch_size, phonemes.shape[1], phonemes_lengths, accelerator.device
        )
        ref_latents = batch["ref_latents"].to(accelerator.device)
        ref_latents_lengths = batch["ref_latents_lengths"].to(accelerator.device)

        with torch.inference_mode():
            _teacher = accelerator.unwrap_model(teacher)
            ref_seq, ref_mask = _teacher.style_encoder(ref_latents, ref_latents_lengths)
        ref_seq = ref_seq.clone()
        ref_mask = ref_mask.clone()

        valid = mask.unsqueeze(-1).expand(-1, -1, 64)

        student_timestep_indices = torch.randint(
            0, len(TIMESTEPS) - 1, (batch_size,), device=accelerator.device
        )
        student_timesteps_prev = torch.tensor(
            [TIMESTEPS[int(i.item())] for i in student_timestep_indices],
            device=accelerator.device,
        )
        z_prev, _ = apply_noise(latents, student_timesteps_prev)
        student.eval()
        with torch.inference_mode():
            x_0_prev: Tensor = get_x_pred(
                student,
                z_prev,
                ref_latents,
                ref_latents_lengths,
                mask,
                phonemes,
                phonemes_mask,
                student_timesteps_prev,
                False,
                False,
            )  # type: ignore
        student.train()
        student_timesteps = torch.tensor(
            [TIMESTEPS[int(i.item() + 1)] for i in student_timestep_indices],
            device=accelerator.device,
        )
        z, _ = apply_noise(x_0_prev, student_timesteps)
        x_0: Tensor = get_x_pred(  # type: ignore[assignment]
            student,
            z,
            ref_latents,
            ref_latents_lengths,
            mask,
            phonemes,
            phonemes_mask,
            student_timesteps,
            cfg=False,
            get_stacked_transformer_features=False,
        )

        timesteps = torch.rand(batch_size).to(accelerator.device)
        x_t, _ = apply_noise(x_0, timesteps)

        with torch.inference_mode():
            p_real = x_0 - get_x_pred(
                teacher,
                x_t,
                ref_latents,
                ref_latents_lengths,
                mask,
                phonemes,
                phonemes_mask,
                timesteps,
                cfg=True,
                get_stacked_transformer_features=False,
            )  # type: ignore

            x_pred_fake, stacked_transformer_features_fake = get_x_pred(
                student_scorer,
                x_t,
                ref_latents,
                ref_latents_lengths,
                mask,
                phonemes,
                phonemes_mask,
                timesteps,
                cfg=False,
                get_stacked_transformer_features=True,
            )
            p_fake = x_0 - x_pred_fake  # type: ignore
            p_real = p_real * valid
            p_fake = p_fake * valid
            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2], keepdim=True)
            grad = torch.nan_to_num(grad)
            grad_magnitude = torch.norm(grad, p=2, dim=[1, 2])
        student_pseudo_loss = (
            0.5
            * mse_loss(x_0.float(), (x_0 - grad).detach().float(), reduction="sum")
            / valid.sum().clamp_min(1.0)
        )

        discriminator.eval()
        set_grad(discriminator, False)
        discriminator_logits = discriminator(
            stacked_transformer_features_fake,
            x_t,
            ref_seq,
            ref_mask,
            mask,
            phonemes,
            timesteps,
        )
        student_discriminator_loss = ((discriminator_logits - 1) ** 2).mean()

        latents_lengths = batch["latents_lengths"].to(accelerator.device)
        transcribed_lps, transcribed_lengths = asr(x_0, latents_lengths)
        transcribed_lps = transcribed_lps.permute(1, 0, 2)
        student_asr_loss = ctc_loss(
            transcribed_lps, phonemes, transcribed_lengths, phonemes_lengths
        )

        with torch.inference_mode():
            true_sv = sv(latents.clone().detach(), latents_lengths.clone().detach())
        student_sv = sv(x_0, latents_lengths)
        student_sv_loss = cosine_loss(student_sv, true_sv).mean()

        lambda_asr = 1 if step > 5_000 else 0.0
        lambda_sv = 1 if step > 7_000 else 0.0

        student_optimizer.zero_grad()
        accelerator.backward(
            student_pseudo_loss
            + 1e-3 * student_discriminator_loss
            + lambda_asr * student_asr_loss
            + lambda_sv * student_sv_loss
        )
        student_optimizer.step()
        torch.cuda.empty_cache()

        set_grad(discriminator, True)
        x_real, _ = apply_noise(latents, timesteps)
        with torch.inference_mode():
            _, stacked_transformer_features_real = student_scorer(
                x_real,
                ref_latents,
                ref_latents_lengths,
                mask,
                phonemes,
                phonemes_mask,
                timesteps,
                get_stacked_transformer_features=True,
            )

        discriminator.train()
        stacked_features = torch.cat(
            [
                stacked_transformer_features_real,
                stacked_transformer_features_fake.detach(),  # type: ignore
            ],
            dim=0,
        )
        x_t_det = x_t.detach()
        stacked_x = torch.cat([x_real, x_t_det], dim=0)
        stacked_ref_seq = torch.cat([ref_seq, ref_seq], dim=0)
        stacked_ref_mask = torch.cat([ref_mask, ref_mask], dim=0)
        stacked_mask = torch.cat([mask, mask], dim=0)
        stacked_phonemes = torch.cat([phonemes, phonemes], dim=0)
        stacked_timesteps = torch.cat([timesteps, timesteps], dim=0)
        stacked_logits = discriminator(
            stacked_features,
            stacked_x,
            stacked_ref_seq,
            stacked_ref_mask,
            stacked_mask,
            stacked_phonemes,
            stacked_timesteps,
        )
        discriminator_logits_real, discriminator_logits_fake = torch.chunk(
            stacked_logits, 2, dim=0
        )
        discriminator_loss = (
            discriminator_logits_fake**2 + (discriminator_logits_real - 1) ** 2
        ).mean()
        discriminator_optimizer.zero_grad()
        accelerator.backward(discriminator_loss)
        discriminator_optimizer.step()
        torch.cuda.empty_cache()

        loss = None
        for _ in range(SCORER_UPDATES):
            z, _ = apply_noise(x_0_prev, student_timesteps)
            with torch.inference_mode():
                x_0_scorer: Tensor = get_x_pred(  # type: ignore[assignment]
                    student,
                    z,
                    ref_latents,
                    ref_latents_lengths,
                    mask,
                    phonemes,
                    phonemes_mask,
                    student_timesteps,
                    False,
                    False,
                )

            timesteps = torch.rand(batch_size).to(accelerator.device)
            noised, v_target = apply_noise(x_0_scorer, timesteps)

            v_pred = student_scorer(
                noised,
                ref_latents,
                ref_latents_lengths,
                mask,
                phonemes,
                phonemes_mask,
                timesteps,
            )
            v_target = v_target * valid
            v_pred = v_pred * valid
            loss = mse_loss(v_pred, v_target, reduction="sum") / valid.sum()
            student_scorer_optimizer.zero_grad()
            accelerator.backward(loss)
            student_scorer_optimizer.step()

        metric_values = {
            "st_pseudo": student_pseudo_loss.item(),
            "st_gan": student_discriminator_loss.item(),
            "st_asr": student_asr_loss.item(),
            "st_sv": student_sv_loss.item(),
            "disc_loss": discriminator_loss.item(),
            "scorer_loss": loss.item(),  # type: ignore
            "dmd_grad_mag": grad_magnitude.mean().item(),
        }
        pbar.set_postfix({k: f"{v:.5f}" for k, v in metric_values.items()})
        accelerator.log(metric_values, step)

        if accelerator.is_main_process and step % NUM_SAVE_STEPS == 0 and step > 1:
            print("saving checkpoint")
            accelerator.save_state("assets/dmd_checkpoints/checkpoint_latest")
            torch.save(
                {
                    "student_model": accelerator.unwrap_model(student).state_dict(),
                    "student_scorer_model": accelerator.unwrap_model(
                        student_scorer
                    ).state_dict(),
                    "discriminator_model": accelerator.unwrap_model(
                        discriminator
                    ).state_dict(),
                },
                "assets/dmd_checkpoints/checkpoint_latest.pt",
            )
