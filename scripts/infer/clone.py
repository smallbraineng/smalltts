import argparse
from pathlib import Path

import soundfile as sf
import torch

from smalltts.assets.ensure import ensure_assets
from smalltts.codec.onnx import Encoder
from smalltts.infer.onnx import SmallTTS
from smalltts.infer.utils import resample_hq

if __name__ == "__main__":
    ensure_assets(["codec", "e2e"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--transcription", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out/clone.wav")
    args = ap.parse_args()

    Path("out").mkdir(exist_ok=True)
    print("loading")
    y, sr = sf.read(args.wav, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    x = torch.from_numpy(y).view(1, -1)
    x = resample_hq(x, sr, 24_000)

    print("encoding")
    lat = Encoder().encode(x.unsqueeze(0))[0].cpu()
    tts = SmallTTS()
    print("inference")
    y = tts([lat], [args.transcription], [args.text])[0]
    sf.write(args.out, y.squeeze(0).t().numpy(), 24_000, subtype="PCM_16")
    print(args.out)
