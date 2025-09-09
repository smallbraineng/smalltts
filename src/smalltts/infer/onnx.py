import math
from typing import Any, Iterable, List, Optional

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from smalltts.codec.onnx import _default_providers as default_providers
from smalltts.data.phonemization.phonemes import get_token_ids
from smalltts.train.utils import get_mask


class SmallTTS:
    def __init__(
        self,
        path: str = "assets/e2e/e2e.onnx",
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        provider = list(providers) if providers is not None else default_providers()
        self.session = ort.InferenceSession(path, providers=provider)
        self.out_name = self.session.get_outputs()[0].name

    def forward(
        self,
        conditionings: List[Tensor],
        transcriptions: List[Any],
        texts: List[Any],
    ) -> List[Tensor]:
        batch_size = len(conditionings)
        assert batch_size == len(transcriptions) and batch_size == len(texts)
        latent_dim = conditionings[0].shape[1]

        def to_tokens(xs):
            if len(xs) == 0:
                return []
            if isinstance(xs[0], str):
                return [get_token_ids(s) for s in xs]
            return [list(map(int, t)) for t in xs]

        cond_tokens = to_tokens(transcriptions)
        new_tokens = to_tokens(texts)
        estimated_lengths = []
        expanded_conds: List[Tensor] = []
        for i, cond in enumerate(conditionings):
            num_phonemes = len(cond_tokens[i])
            latents_per_phoneme = (
                float(cond.shape[0]) / float(num_phonemes) * 1.25
            )  # experimental
            estimated_latents_length = int(
                math.ceil(latents_per_phoneme * len(new_tokens[i]))
            )
            expanded_cond = torch.cat(
                [
                    cond,
                    torch.zeros(
                        (estimated_latents_length, latent_dim), device=cond.device
                    ),
                ],
                dim=0,
            )
            estimated_lengths.append(estimated_latents_length)
            expanded_conds.append(expanded_cond)

        latents_lengths = torch.tensor(
            [len(cond) for cond in expanded_conds], dtype=torch.int64
        )
        padded_conditionings = pad_sequence(
            expanded_conds, batch_first=True, padding_value=0
        )
        sequence_length = padded_conditionings.shape[1]
        mask = get_mask(
            batch_size, sequence_length, latents_lengths, latents_lengths.device
        )

        phonemes = [
            torch.tensor(cond_tokens[i] + new_tokens[i], dtype=torch.int64)
            for i in range(batch_size)
        ]
        phonemes_lengths = torch.tensor([len(p) for p in phonemes], dtype=torch.int64)
        padded_phonemes = pad_sequence(phonemes, batch_first=True, padding_value=0)
        phonemes_mask = get_mask(
            batch_size,
            padded_phonemes.shape[1],
            phonemes_lengths,
            phonemes_lengths.device,
        )

        noise = torch.randn(
            (4, batch_size, padded_conditionings.shape[1], latent_dim),
            dtype=torch.float32,
        )

        out = self.session.run(
            [self.out_name],
            {
                "cond": padded_conditionings.numpy().astype(np.float32, copy=False),
                "mask": mask.cpu().numpy().astype(np.bool_, copy=False),
                "phonemes": padded_phonemes.numpy().astype(np.int32, copy=False),
                "phonemes_mask": phonemes_mask.cpu()
                .numpy()
                .astype(np.bool_, copy=False),
                "noise": noise.numpy().astype(np.float32, copy=False),
            },
        )[0]

        out = torch.from_numpy(out)

        audios: List[Tensor] = []

        for i in range(0, batch_size):
            audio = out[i]
            cond_start = conditionings[i].shape[0] * 3_200
            audios.append(
                audio[:, cond_start : cond_start + estimated_lengths[i] * 3_200]
            )

        return audios

    __call__ = forward


if __name__ == "__main__":
    tts = SmallTTS()
    x = [torch.randn(50, 64, dtype=torch.float32)]
    y = tts(x, ["hello world"], ["this is a test"])
    print(y, y[0].shape)
