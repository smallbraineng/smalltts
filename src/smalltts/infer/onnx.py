"""ONNX inference using condition_encoder + denoiser split (DMD 4-step, no CFG)."""

from typing import Iterable, List, Optional

import numpy as np
import onnxruntime as ort
from torch import Tensor

from smalltts.codec.onnx import _default_providers as default_providers

SAMPLE_RATE = 24_000
HOP_SIZE = 3_200
NUM_STEPS = 4
CHARS_PER_SECOND = 11.5


def estimate_duration(text: str, min_sec: float = 0.5, max_sec: float = 30.0) -> float:
    return max(min_sec, min(len(text) / CHARS_PER_SECOND, max_sec))


def _make_session(path: str, providers: list[str]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def _input_names(sess: ort.InferenceSession) -> list[str]:
    return [i.name for i in sess.get_inputs()]


def _get_alpha_sigma(t: float, eps: float = 1e-5):
    t = np.clip(t, eps, 1 - eps)
    alpha_t_sq = np.cos(np.pi / 2 * t) ** 2
    log_snr = np.log(alpha_t_sq / (1 - alpha_t_sq))
    log_snr_s = log_snr + 2 * np.log(0.5)
    alpha_sq = 1.0 / (1.0 + np.exp(-log_snr_s))
    return np.sqrt(alpha_sq).astype(np.float32), np.sqrt(1 - alpha_sq).astype(
        np.float32
    )


def _compute_rope_freqs(seq_len: int, dim: int = 64) -> np.ndarray:
    inv_freq = 1.0 / (1e4 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(seq_len, dtype=np.float32).reshape(1, -1)
    freqs = np.einsum("bi,j->bij", t, inv_freq)
    freqs = np.stack([freqs, freqs], axis=-1).reshape(1, seq_len, dim)
    return freqs


class SmallTTS:
    """DMD 4-step inference using condition_encoder + denoiser ONNX split (no CFG)."""

    def __init__(
        self,
        cond_encoder_path: str = "assets/dmd/condition_encoder.onnx",
        denoiser_path: str = "assets/dmd/denoiser.onnx",
        codec_decoder_path: str = "assets/codec/decoder.onnx",
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        prov = list(providers) if providers is not None else default_providers()
        self.cond_enc = _make_session(cond_encoder_path, prov)
        self.denoiser = _make_session(denoiser_path, prov)
        self.codec_dec = _make_session(codec_decoder_path, prov)
        self.cond_enc_names = _input_names(self.cond_enc)
        self.den_names = _input_names(self.denoiser)
        self.dec_names = _input_names(self.codec_dec)

    def synthesize(
        self,
        ref_latents: np.ndarray,
        phoneme_ids: list[int],
        duration_sec: float,
    ) -> np.ndarray:
        """Run the full DMD pipeline.

        Args:
            ref_latents: Reference audio latents, shape (T, 64), float32.
            phoneme_ids: Phoneme token IDs for the text to speak.
            duration_sec: Desired output duration in seconds.

        Returns:
            Audio samples, shape (1, samples), float32, 24kHz.
        """
        seq_len = max(1, int(duration_sec * SAMPLE_RATE / HOP_SIZE))
        ref = ref_latents[np.newaxis].astype(np.float32)
        ref_len = np.array([ref.shape[1]], dtype=np.int64)

        phonemes = np.array([phoneme_ids], dtype=np.int64)
        phonemes_mask = np.ones_like(phonemes, dtype=np.bool_)

        cond_feed = dict(
            zip(self.cond_enc_names, [ref, ref_len, phonemes, phonemes_mask])
        )
        all_k_ref, all_v_ref, ref_mask, all_k_text, all_v_text = self.cond_enc.run(
            None, cond_feed
        )

        rope = _compute_rope_freqs(seq_len)
        mask = np.ones((1, seq_len), dtype=np.bool_)
        x_pred = np.zeros((1, seq_len, 64), dtype=np.float32)

        for t_val in np.linspace(1, 0, NUM_STEPS, dtype=np.float32):
            alpha, sigma = _get_alpha_sigma(float(t_val))
            noise = np.random.randn(1, seq_len, 64).astype(np.float32)
            x_t = (alpha * x_pred + sigma * noise).astype(np.float32)

            den_feed = dict(
                zip(
                    self.den_names,
                    [
                        x_t,
                        mask,
                        np.array([t_val], dtype=np.float32),
                        all_k_ref,
                        all_v_ref,
                        ref_mask,
                        all_k_text,
                        all_v_text,
                        phonemes_mask,
                        rope,
                    ],
                )
            )
            velocity = self.denoiser.run(None, den_feed)[0]
            x_pred = (alpha * x_t - sigma * velocity).astype(np.float32)

        dec_feed = {self.dec_names[0]: x_pred}
        audio = self.codec_dec.run(None, dec_feed)[0]
        return audio[0]

    def forward(
        self,
        conditionings: List[Tensor],
        transcriptions: list,
        texts: list,
        duration_sec: float = 3.0,
    ) -> List[Tensor]:
        import torch

        from smalltts.data.phonemization.phonemes import get_token_ids

        results = []
        for cond, trans, text in zip(conditionings, transcriptions, texts):
            trans_tok = (
                get_token_ids(trans)
                if isinstance(trans, str)
                else list(map(int, trans))
            )
            text_tok = (
                get_token_ids(text) if isinstance(text, str) else list(map(int, text))
            )
            all_tokens = trans_tok + text_tok
            audio = self.synthesize(
                cond.numpy().astype(np.float32), all_tokens, duration_sec
            )
            results.append(torch.from_numpy(audio))
        return results

    __call__ = forward
