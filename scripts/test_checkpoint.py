"""Test loading the dmdspeech teacher checkpoint into the new DiTModel architecture."""

import os

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak.dylib")

import torch

from smalltts.data.phonemization.phonemes import phoneme_len
from smalltts.models.backbone.model import DiTModel


def load_from_checkpoint(checkpoint_path: str, device: torch.device):
    print(f"loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    cleaned = {}
    for k, v in state_dict.items():
        if k in ("initted", "step"):
            continue
        for prefix in ("module.", "_orig_mod.", "ema_model.", "online_model."):
            k = k.replace(prefix, "")
        cleaned[k] = v
    return cleaned


if __name__ == "__main__":
    device = torch.device("cpu")
    checkpoint_path = "assets/teacher_checkpoints_large/checkpoint_latest.pt"

    print("instantiating DiTModel(64)...")
    model = DiTModel(64).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_params:,}")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {n:,}")

    state_dict = load_from_checkpoint(checkpoint_path, device)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"\nmissing keys ({len(missing)}):")
        for k in sorted(missing):
            print(f"  {k}")
    if unexpected:
        print(f"\nunexpected keys ({len(unexpected)}):")
        for k in sorted(unexpected):
            print(f"  {k}")
    if not missing and not unexpected:
        print("\nall keys match!")

    shape_mismatches = []
    for k in model_keys & ckpt_keys:
        ms = model.state_dict()[k].shape
        cs = state_dict[k].shape
        if ms != cs:
            shape_mismatches.append((k, ms, cs))

    if shape_mismatches:
        print(f"\nshape mismatches ({len(shape_mismatches)}):")
        for k, ms, cs in shape_mismatches:
            print(f"  {k}: model={ms}, ckpt={cs}")
    else:
        print("all shapes match!")

    model.load_state_dict(state_dict)
    print("\ncheckpoint loaded successfully!")

    print("\nrunning forward pass...")
    model.eval()
    batch_size = 2
    seq_len = 40
    ref_seq_len = 24

    noised = torch.randn(batch_size, seq_len, 64, device=device)
    ref_latents = torch.randn(batch_size, ref_seq_len, 64, device=device)
    ref_latents_lengths = torch.full((batch_size,), ref_seq_len, dtype=torch.int64, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    phonemes = torch.randint(0, 50, (batch_size, phoneme_len), device=device)
    phonemes_mask = torch.ones_like(phonemes, dtype=torch.bool)
    t = torch.rand(batch_size, device=device)

    with torch.inference_mode():
        velocity = model(noised, ref_latents, ref_latents_lengths, mask, phonemes, phonemes_mask, t)

    print(f"output shape: {velocity.shape}")
    assert velocity.shape == (batch_size, seq_len, 64)
    print("forward pass successful!")

    print("\ntesting cached inference...")
    with torch.inference_mode():
        cached = model.encode_conditions(ref_latents, ref_latents_lengths, phonemes, phonemes_mask, seq_len)
        velocity_cached = model.denoise_step(noised, mask, t, cached)

    print(f"cached output shape: {velocity_cached.shape}")
    assert velocity_cached.shape == (batch_size, seq_len, 64)
    print("cached inference successful!")

    print("\ntesting stacked transformer features...")
    with torch.inference_mode():
        velocity, stacked = model(noised, ref_latents, ref_latents_lengths, mask, phonemes, phonemes_mask, t, get_stacked_transformer_features=True)

    print(f"velocity shape: {velocity.shape}")
    print(f"stacked features shape: {stacked.shape}")
    assert stacked.shape[0] == batch_size
    assert stacked.shape[1] == 12
    assert stacked.shape[2] == seq_len
    assert stacked.shape[3] == 960
    print("stacked features test successful!")

    print("\n=== all tests passed! ===")
