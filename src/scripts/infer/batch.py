import json
from pathlib import Path

import soundfile as sf
import torch

from smalltts.assets.ensure import ensure_assets
from smalltts.codec.onnx import Encoder
from smalltts.data.phonemization.phonemes import get_token_ids
from smalltts.infer.onnx import SmallTTS, estimate_duration
from smalltts.infer.utils import resample_hq

if __name__ == "__main__":
    ensure_assets(["codec", "dmd", "test_audio"])
    td = Path("assets/test_audio")
    with open(td / "transcriptions.json") as f:
        items = json.load(f)
    files = [td / it["filename"] for it in items]
    texts = [
        "Hello world, I am small tts, and I am talking!",
        "I can clone any voice and emotion.",
        "I have an ONNX export and run very fast.",
        "Woah, this is awesome I can do any character!",
    ]
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    enc = Encoder()
    tts = SmallTTS()

    for i, (fpath, text) in enumerate(zip(files, texts)):
        print(f"[{i + 1}/{len(files)}] {fpath.name}")
        y, sr = sf.read(str(fpath), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        x = torch.from_numpy(y).view(1, -1)
        x = resample_hq(x, sr, 24_000)
        ref_latents = enc.encode(x.unsqueeze(0))[0].cpu().numpy()

        tokens = get_token_ids(text)
        duration = estimate_duration(text)
        audio = tts.synthesize(ref_latents, tokens, duration)

        out_path = outdir / f"{fpath.stem}_gen.wav"
        sf.write(str(out_path), audio.squeeze(), 24_000, subtype="PCM_16")
        print(f"  -> {out_path}")
