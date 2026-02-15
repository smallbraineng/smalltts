import argparse
from pathlib import Path

import soundfile as sf
import torch

from smalltts.assets.ensure import ensure_assets
from smalltts.codec.onnx import Encoder
from smalltts.data.phonemization.phonemes import get_token_ids
from smalltts.infer.onnx import SmallTTS, estimate_duration
from smalltts.infer.utils import resample_hq

if __name__ == "__main__":
    ensure_assets(["codec", "dmd"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="reference audio file")
    ap.add_argument("--text", required=True, help="text to speak")
    ap.add_argument("--duration", type=float, default=None, help="duration in seconds (auto if omitted)")
    ap.add_argument("--out", default="out/clone.wav")
    args = ap.parse_args()

    Path("out").mkdir(exist_ok=True)
    print("loading")
    y, sr = sf.read(args.wav, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    x = torch.from_numpy(y).view(1, -1)
    x = resample_hq(x, sr, 24_000)

    print("encoding reference audio")
    ref_latents = Encoder().encode(x.unsqueeze(0))[0].cpu().numpy()

    tts = SmallTTS()
    tokens = get_token_ids(args.text)
    duration = args.duration or estimate_duration(args.text)

    print(f"generating ({duration:.1f}s)")
    audio = tts.synthesize(ref_latents, tokens, duration)
    sf.write(args.out, audio.squeeze(), 24_000, subtype="PCM_16")
    print(args.out)
