import torch
from accelerate import Accelerator
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor


def get_embedding_model(accelerator: Accelerator) -> EncoderClassifier:
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(accelerator.device)},
    )
    if model is None:
        raise Exception("failed to load embedding model")
    return model


@torch.inference_mode()
def get_true_embeddings(
    spk_model: EncoderClassifier,
    batch_audio: Tensor,
    audio_lengths: Tensor,
) -> Tensor:
    audio_lengths = audio_lengths.float()
    max_length = audio_lengths.max()
    audio_lengths = audio_lengths / max_length
    assert batch_audio.dim() == 3 and batch_audio.size(1) == 1, "expect (B,1,T)"
    return spk_model.encode_batch(
        batch_audio.squeeze(1), wav_lens=audio_lengths, normalize=False
    ).squeeze(1)
