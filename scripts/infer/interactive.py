import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from smalltts.codec.onnx import Encoder
from smalltts.data.phonemization.phonemes import get_token_ids
from smalltts.infer.onnx import SmallTTS
from smalltts.infer.utils import resample_hq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str)
    ap.add_argument("--transcription", type=str)
    args = ap.parse_args()
    Path("out").mkdir(exist_ok=True)
    print("loading model")
    t0 = time.perf_counter()
    model = SmallTTS()
    if args.wav and args.transcription:
        y, sr = sf.read(args.wav, dtype="float32")
        if y.ndim == 2:
            y = y.mean(axis=1)
        x = torch.from_numpy(y).view(1, -1)
        x = resample_hq(x, sr, 24_000)

        lat = Encoder().encode(x.unsqueeze(0))[0].cpu()
        toks = get_token_ids(args.transcription)
    else:
        lat = torch.from_numpy(np.load("assets/tryme/latents.npy")).float()
        toks = np.load("assets/tryme/tokens.npy").tolist()
    i = 1
    first = True
    while True:
        try:
            s = input(">> ").strip()
            if not s:
                continue
            st = time.perf_counter()
            y = model([lat], [toks], [s])[0]
            dt = time.perf_counter() - st
            dur = y.shape[-1] / 24_000.0
            rtf = dur / dt if dt > 0 else 0.0
            p = Path(f"out/interactive_{i}.wav")
            sf.write(str(p), y.squeeze(0).t().numpy(), 24_000, subtype="PCM_16")
            if first:
                print(
                    f"gen {dt:.2f}s (+{time.perf_counter() - t0 - dt:.2f}s warmup), {rtf:.1f}x rt"
                )
                first = False
            else:
                print(f"gen {dt:.2f}s, {rtf:.1f}x rt")
            try:
                subprocess.run(["afplay", str(p)], check=False)
            except Exception:
                pass
            i += 1
        except (EOFError, KeyboardInterrupt):
            print()
            break


if __name__ == "__main__":
    main()
