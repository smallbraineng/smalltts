import argparse
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from rich.console import Console
from rich.prompt import Prompt

from smalltts.assets.ensure import ensure_assets
from smalltts.codec.onnx import Encoder
from smalltts.data.phonemization.phonemes import get_token_ids
from smalltts.infer.onnx import SmallTTS
from smalltts.infer.utils import resample_hq

if __name__ == "__main__":
    ensure_assets(["codec", "e2e", "tryme"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str)
    ap.add_argument("--transcription", type=str)
    args = ap.parse_args()
    console = Console()
    console.print("smalltts interactive", style="bold magenta")
    console.print("type and press enter. ctrl-c to exit.", style="dim")
    console.print("loading model", style="yellow")
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
    first = True
    while True:
        s = Prompt.ask("[bold cyan]>>[/]").strip()
        if not s:
            continue
        st = time.perf_counter()
        with console.status("[bold green]generating...", spinner="dots"):
            y = model([lat], [toks], [s])[0]
        dt = time.perf_counter() - st
        dur = y.shape[-1] / 24_000.0
        rtf = dur / dt if dt > 0 else 0.0
        if first:
            console.print(
                f"gen {dt:.2f}s (+{time.perf_counter() - t0 - dt:.2f}s warmup), {rtf:.1f}x rt",
                style="green",
            )
            first = False
        else:
            console.print(f"gen {dt:.2f}s, {rtf:.1f}x rt", style="green")
        a = y.squeeze().cpu().numpy()
        sd.play(a, 24_000)
        sd.wait()
