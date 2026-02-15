import random

import torch
from torch.nn.utils.rnn import pad_sequence

from .phonemization.phonemes import phoneme_len


def dummy_collate_fn(
    batch_size: int,
):
    texts = [f"dummy text {i}" for i in range(batch_size)]

    phonemes = []
    for _ in range(batch_size):
        length = random.randint(5, phoneme_len)
        seq = torch.randint(low=1, high=phoneme_len, size=(length,))
        phonemes.append(seq)
    phonemes_lengths = torch.tensor([len(p) for p in phonemes], dtype=torch.int64)
    phonemes = pad_sequence(phonemes, batch_first=True, padding_value=0)

    latents = []
    for _ in range(batch_size):
        length = random.randint(20, 256)
        seq = torch.randn(length, 64)
        latents.append(seq)
    latents_lengths = torch.tensor([len(lat) for lat in latents], dtype=torch.int64)
    latents = pad_sequence(latents, batch_first=True, padding_value=0.0)

    ref_latents = []
    for _ in range(batch_size):
        length = random.randint(8, 64)
        seq = torch.randn(length, 64)
        ref_latents.append(seq)
    ref_latents_lengths = torch.tensor([len(r) for r in ref_latents], dtype=torch.int64)
    ref_latents = pad_sequence(ref_latents, batch_first=True, padding_value=0.0)

    return {
        "texts": texts,
        "phonemes": phonemes,
        "phonemes_lengths": phonemes_lengths,
        "latents": latents,
        "latents_lengths": latents_lengths,
        "ref_latents": ref_latents,
        "ref_latents_lengths": ref_latents_lengths,
    }


def get_dummy_dataloader(batch_size: int, num_workers: int):
    print("warn: using dummy data, you probably want to use real data")
    while True:
        yield dummy_collate_fn(batch_size)


if __name__ == "__main__":
    dataloader = get_dummy_dataloader(batch_size=2, num_workers=3)
    for batch in dataloader:
        print(batch)
